//! Vision backbone: patch embedding, position embedding and windowed attention layout
//!
//! Every kernel here is one thread per output element (or per output pair, for the key
//! rotation) with the thread's index as its only write location. Kernels whose natural loop
//! is over the input, like the window partition, are written as gathers over the output
//! index so the writes stay disjoint. Index math is 32-bit: the host launches one thread
//! per output element and keeps every buffer under 2^32 elements

use cuda_device::{DisjointSlice, LinearTiles, kernel, launch_contract, thread};

use crate::attention::{VIT_GRID, VIT_WIN, VIT_WIN_TOKENS};
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

/// The raster index of row `r` in window-major order, where each `VIT_WIN`-square window's
/// tokens are contiguous and in raster order inside it
#[inline(always)]
fn window_to_raster(r: u32) -> u32 {
    const NWX: u32 = VIT_GRID / VIT_WIN;
    let (w, l) = (r / (VIT_WIN * VIT_WIN), r % (VIT_WIN * VIT_WIN));
    ((w / NWX) * VIT_WIN + l / VIT_WIN) * VIT_GRID + (w % NWX) * VIT_WIN + l % VIT_WIN
}

/// The inverse of [`window_to_raster`]
#[inline(always)]
fn raster_to_window(tok: u32) -> u32 {
    const NWX: u32 = VIT_GRID / VIT_WIN;
    let (y, x) = (tok / VIT_GRID, tok % VIT_GRID);
    ((y / VIT_WIN) * NWX + x / VIT_WIN) * VIT_WIN * VIT_WIN + (y % VIT_WIN) * VIT_WIN + x % VIT_WIN
}

/// Reorders the rows of `x [VIT_GRID^2, cols]` from raster to window-major order with
/// `to_window` nonzero, and back otherwise, for the fused ViT attention's token order
/// (see [`crate::attention::vit_attn_bf16_d64`])
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn vit_window_order(mut out: DisjointSlice<f32>, x: &[f32], cols: u32, to_window: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut out) = out.element_thread32(t) else { return };
    let (row, c) = (i / cols, i % cols);
    let src = if to_window != 0 { window_to_raster(row) } else { raster_to_window(row) };
    out.write(x[(src * cols + c) as usize]);
}

/// Rotates the k columns of the qkv GEMM's output `qkv [tokens, 3 * dim]` by RoPE into `k
/// [tokens, dim]`, one adjacent pair per thread, for [`crate::attention::vit_attn_bf16_d64`].
/// Tokens are window-major; with `windowed` nonzero `cos`/`sin` are the window tables
/// `[VIT_WIN_TOKENS, hd / 2]`, otherwise the global ones `[tokens, hd / 2]`
///
/// A pass of its own because the attention stages every key tile once per query tile:
/// rotating while staging repeated the work and its table loads 9 times per window and 81
/// times in a global block, which cost more than this pass
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn vit_rope_k_bf16(
    mut k: DisjointSlice<Bf16, LinearTiles<2>>,
    qkv: &[Bf16],
    cos: &[f32],
    sin: &[f32],
    dim: u32,
    hd: u32,
    windowed: u32,
) {
    let t = thread::index_1d_u32(lc);
    let p = t.get();
    let Some(mut k) = k.tile_thread32(t) else { return };
    let (row, c) = (p / (dim / 2), (p % (dim / 2)) * 2);
    let src = (row * 3 * dim + dim + c) as usize;
    let pos = if windowed != 0 { row % VIT_WIN_TOKENS } else { row };
    let tab = (pos * (hd / 2) + (c % hd) / 2) as usize;
    let (a, b, cs, sn) = (qkv[src].widen(), qkv[src + 1].widen(), cos[tab], sin[tab]);
    // FP32 rotation and one rounding of the pair, as the q side does while staging
    let y = pack_bf16(a * cs - b * sn, a * sn + b * cs);
    k.at_const::<0>().write(Bf16(y as u16));
    k.at_const::<1>().write(Bf16((y >> 16) as u16));
}
