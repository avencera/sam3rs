//! Vision backbone: patch embedding, position embedding and windowed attention layout
//!
//! Every kernel here is one thread per output element (or per output pair, for the RoPE
//! split) with the thread's index as its only write location. Kernels whose natural loop
//! is over the input, like the window partition, are written as gathers over the output
//! index so the writes stay disjoint. Index math is 32-bit: the host launches one thread
//! per output element and keeps every buffer under 2^32 elements

use cuda_device::{DisjointSlice, LinearTiles, kernel, launch_contract, thread};

use crate::common::{Bf16, Elem, pack_bf16};

/// `pixels [3, size, size] -> patches [(size / patch)^2, 3 * patch * patch]`, columns in
/// the conv weight's order
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn patchify(mut dst: DisjointSlice<f32>, src: &[f32], size: u32, patch: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let grid = size / patch;
    let k = 3 * patch * patch;
    let (tok, col) = (i / k, i % k);
    let (c, ky, kx) = (col / (patch * patch), (col / patch) % patch, col % patch);
    let (ty, tx) = (tok / grid, tok % grid);
    let s = c * size * size + (ty * patch + ky) * size + (tx * patch + kx);
    if let Some(mut dst) = dst.element_thread32(t) {
        dst.write(src[s as usize]);
    }
}

/// `x[y * grid + x, :] += pos[(y % tile) * tile + x % tile, :]` in place: the absolute
/// position embedding tiled from the 24x24 pretraining grid
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn add_pos_tiled(mut x: DisjointSlice<f32>, pos: &[f32], grid: u32, tile: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    // `x` is exactly grid * grid rows; `pos`'s row count is not guaranteed to be tile * tile
    let dim = x.len() as u32 / (grid * grid);
    let (d, tok) = (i % dim, i / dim);
    let (y, xx) = (tok / grid, tok % grid);
    let p = ((y % tile) * tile + xx % tile) * dim + d;
    if let Some(mut x) = x.element_thread32(t) {
        x.write(x.read() + pos[p as usize]);
    }
}

/// The rotated q and k pairs and the copied v pair of one output pair
struct Rotated {
    q: [f32; 2],
    k: [f32; 2],
    v: [f32; 2],
}

/// `qkv [grid * grid, 3 * heads * hd] -> q, k, v [windows * heads, win * win, hd]`, for
/// output pair `t` of the window-partitioned layout
///
/// Partitions into windows (`win == grid` for global attention) and applies interleaved
/// axial RoPE to q and k. The cos/sin tables are `[win * win, hd / 2]`. The port looped
/// over input pairs and scattered the window layout; this inverts the window mapping so
/// each thread gathers its source token and the outputs are written at `t` alone
#[inline(always)]
fn split_rope_pair<In: Elem>(
    t: u32,
    qkv: &[In],
    cos_t: &[f32],
    sin_t: &[f32],
    grid: u32,
    win: u32,
    heads: u32,
    hd: u32,
) -> Rotated {
    let half = hd / 2;
    let nwx = grid / win;
    let dim = heads * hd;
    let p = t % half;
    let (wtok, bh) = ((t / half) % (win * win), t / (half * win * win));
    let (h, w) = (bh % heads, bh / heads);
    let (y, x) = ((w / nwx) * win + wtok / win, (w % nwx) * win + wtok % win);
    let src = ((y * grid + x) * 3 * dim + h * hd + 2 * p) as usize;
    let tab = (wtok * half + p) as usize;
    let (c, s) = (cos_t[tab], sin_t[tab]);
    let (q0, q1) = (qkv[src].widen(), qkv[src + 1].widen());
    let (k0, k1) = (qkv[src + dim as usize].widen(), qkv[src + dim as usize + 1].widen());
    let vs = src + 2 * dim as usize;
    Rotated {
        q: [q0 * c - q1 * s, q0 * s + q1 * c],
        k: [k0 * c - k1 * s, k0 * s + k1 * c],
        v: [qkv[vs].widen(), qkv[vs + 1].widen()],
    }
}

/// FP32 RoPE split; each thread writes one pair of each of q, k and v. The launch is one
/// thread per pair, `grid * grid * heads * hd / 2` threads
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn vit_split_rope(
    mut q: DisjointSlice<f32, LinearTiles<2>>,
    mut k: DisjointSlice<f32, LinearTiles<2>>,
    mut v: DisjointSlice<f32, LinearTiles<2>>,
    qkv: &[f32],
    cos_t: &[f32],
    sin_t: &[f32],
    grid: u32,
    win: u32,
    heads: u32,
    hd: u32,
) {
    // one witness per output buffer: each is unique to this thread in its own slice
    let (tq, tk, tv) = (thread::index_1d_u32(lc), thread::index_1d_u32(lc), thread::index_1d_u32(lc));
    let t = tq.get();
    // the tail of the last block has no pair to write and must not read
    if let (Some(mut q), Some(mut k), Some(mut v)) = (q.tile_thread32(tq), k.tile_thread32(tk), v.tile_thread32(tv)) {
        let r = split_rope_pair(t, qkv, cos_t, sin_t, grid, win, heads, hd);
        q.at_const::<0>().write(r.q[0]);
        q.at_const::<1>().write(r.q[1]);
        k.at_const::<0>().write(r.k[0]);
        k.at_const::<1>().write(r.k[1]);
        v.at_const::<0>().write(r.v[0]);
        v.at_const::<1>().write(r.v[1]);
    }
}

/// BF16 RoPE split, as [`vit_split_rope`] with BF16 qkv in and q, k, v out; the pair is
/// rounded to nearest even with one conversion
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn vit_split_rope_bf16(
    mut q: DisjointSlice<Bf16, LinearTiles<2>>,
    mut k: DisjointSlice<Bf16, LinearTiles<2>>,
    mut v: DisjointSlice<Bf16, LinearTiles<2>>,
    qkv: &[Bf16],
    cos_t: &[f32],
    sin_t: &[f32],
    grid: u32,
    win: u32,
    heads: u32,
    hd: u32,
) {
    let (tq, tk, tv) = (thread::index_1d_u32(lc), thread::index_1d_u32(lc), thread::index_1d_u32(lc));
    let t = tq.get();
    if let (Some(mut q), Some(mut k), Some(mut v)) = (q.tile_thread32(tq), k.tile_thread32(tk), v.tile_thread32(tv)) {
        let r = split_rope_pair(t, qkv, cos_t, sin_t, grid, win, heads, hd);
        let (pq, pk, pv) = (pack_bf16(r.q[0], r.q[1]), pack_bf16(r.k[0], r.k[1]), pack_bf16(r.v[0], r.v[1]));
        q.at_const::<0>().write(Bf16(pq as u16));
        q.at_const::<1>().write(Bf16((pq >> 16) as u16));
        k.at_const::<0>().write(Bf16(pk as u16));
        k.at_const::<1>().write(Bf16((pk >> 16) as u16));
        v.at_const::<0>().write(Bf16(pv as u16));
        v.at_const::<1>().write(Bf16((pv >> 16) as u16));
    }
}

/// Index into `o [windows * heads, win * win, hd]` of merged output element `i` of
/// `[grid * grid, heads * hd]`, undoing the window partition
#[inline(always)]
fn merge_source(i: u32, grid: u32, win: u32, heads: u32, hd: u32) -> usize {
    let nwx = grid / win;
    let (d, h, tok) = (i % hd, (i / hd) % heads, i / (heads * hd));
    let (y, x) = (tok / grid, tok % grid);
    let (wy, wx, ly, lx) = (y / win, x / win, y % win, x % win);
    ((((wy * nwx + wx) * heads + h) * win * win + ly * win + lx) * hd + d) as usize
}

/// `o [windows * heads, win * win, hd] -> out [grid * grid, heads * hd]`, undoing the
/// window partition
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn vit_merge_heads(mut out: DisjointSlice<f32>, o: &[f32], grid: u32, win: u32, heads: u32, hd: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    if let Some(mut out) = out.element_thread32(t) {
        out.write(o[merge_source(i, grid, win, heads, hd)]);
    }
}

/// As [`vit_merge_heads`], rounding to BF16 for the projection GEMM
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn vit_merge_heads_bf16(mut out: DisjointSlice<Bf16>, o: &[f32], grid: u32, win: u32, heads: u32, hd: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    if let Some(mut out) = out.element_thread32(t) {
        out.write(Bf16(pack_bf16(o[merge_source(i, grid, win, heads, hd)], 0.0) as u16));
    }
}
