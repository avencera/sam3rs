//! Vision backbone: patch embedding, position embedding and windowed attention layout

use cuda_device::kernel;

use crate::common::{Bf16, Elem, grid_stride};

/// `pixels [3, size, size] -> patches [(size / patch)^2, 3 * patch * patch]`, columns in
/// the conv weight's order
#[kernel]
pub unsafe fn patchify(dst: *mut f32, src: *const f32, size: i32, patch: i32) {
    let grid = size / patch;
    let k = 3 * patch * patch;
    let n = grid as i64 * grid as i64 * k as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let col = (i % k as i64) as i32;
        let tok = i / k as i64;
        let (c, ky, kx) = (col / (patch * patch), (col / patch) % patch, col % patch);
        let (ty, tx) = ((tok / grid as i64) as i32, (tok % grid as i64) as i32);
        let s =
            c as i64 * size as i64 * size as i64 + (ty * patch + ky) as i64 * size as i64 + (tx * patch + kx) as i64;
        unsafe { *dst.add(i as usize) = *src.add(s as usize) };
        i += stride;
    }
}

/// `x[y * grid + x, :] += pos[(y % tile) * tile + x % tile, :]`: the absolute position
/// embedding tiled from the 24x24 pretraining grid
#[kernel]
pub unsafe fn add_pos_tiled(x: *mut f32, pos: *const f32, grid: i32, tile: i32, dim: i32) {
    let n = grid as i64 * grid as i64 * dim as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let d = i % dim as i64;
        let tok = i / dim as i64;
        let (y, xx) = ((tok / grid as i64) as i32, (tok % grid as i64) as i32);
        let p = ((y % tile) * tile + xx % tile) as i64 * dim as i64 + d;
        unsafe { *x.add(i as usize) += *pos.add(p as usize) };
        i += stride;
    }
}

/// `qkv [grid * grid, 3 * heads * hd] -> q, k, v [windows * heads, win * win, hd]`
///
/// Partitions into windows (`win == grid` for global attention) and applies interleaved
/// axial RoPE to q and k. The cos/sin tables are `[win * win, hd / 2]`
#[inline(always)]
unsafe fn split_rope<In: Elem, Out: Elem>(
    q: *mut Out,
    k: *mut Out,
    v: *mut Out,
    qkv: *const In,
    cos_t: *const f32,
    sin_t: *const f32,
    grid: i32,
    win: i32,
    heads: i32,
    hd: i32,
) {
    let nwx = grid / win;
    let dim = heads * hd;
    let n = grid as i64 * grid as i64 * dim as i64 / 2;
    let (mut i, stride) = grid_stride();
    while i < n {
        let pair = (i % (dim / 2) as i64) as i32;
        let tok = i / (dim / 2) as i64;
        let (h, p) = (pair / (hd / 2), pair % (hd / 2));
        let (y, x) = ((tok / grid as i64) as i32, (tok % grid as i64) as i32);
        let (wy, wx, ly, lx) = (y / win, x / win, y % win, x % win);
        let wtok = ly * win + lx;
        let dst = ((((wy * nwx + wx) as i64 * heads as i64 + h as i64) * win as i64 * win as i64 + wtok as i64)
            * hd as i64
            + 2 * p as i64) as usize;
        unsafe {
            let src = qkv.add((tok * 3 * dim as i64 + (h * hd) as i64 + 2 * p as i64) as usize);
            let t = (wtok * (hd / 2) + p) as usize;
            let (c, s) = (*cos_t.add(t), *sin_t.add(t));
            let (q0, q1) = (In::load(src), In::load(src.add(1)));
            Out::store(q.add(dst), q0 * c - q1 * s);
            Out::store(q.add(dst + 1), q0 * s + q1 * c);
            let ks = src.add(dim as usize);
            let (k0, k1) = (In::load(ks), In::load(ks.add(1)));
            Out::store(k.add(dst), k0 * c - k1 * s);
            Out::store(k.add(dst + 1), k0 * s + k1 * c);
            let vs = src.add(2 * dim as usize);
            Out::store(v.add(dst), In::load(vs));
            Out::store(v.add(dst + 1), In::load(vs.add(1)));
        }
        i += stride;
    }
}

#[kernel]
pub unsafe fn vit_split_rope(
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    qkv: *const f32,
    cos_t: *const f32,
    sin_t: *const f32,
    grid: i32,
    win: i32,
    heads: i32,
    hd: i32,
) {
    unsafe { split_rope(q, k, v, qkv, cos_t, sin_t, grid, win, heads, hd) };
}

#[kernel]
pub unsafe fn vit_split_rope_bf16(
    q: *mut Bf16,
    k: *mut Bf16,
    v: *mut Bf16,
    qkv: *const Bf16,
    cos_t: *const f32,
    sin_t: *const f32,
    grid: i32,
    win: i32,
    heads: i32,
    hd: i32,
) {
    unsafe { split_rope(q, k, v, qkv, cos_t, sin_t, grid, win, heads, hd) };
}

/// `o [windows * heads, win * win, hd] -> [grid * grid, heads * hd]`, undoing the window
/// partition
#[inline(always)]
unsafe fn merge_heads<T: Elem>(out: *mut T, o: *const f32, grid: i32, win: i32, heads: i32, hd: i32) {
    let nwx = grid / win;
    let dim = heads * hd;
    let n = grid as i64 * grid as i64 * dim as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let d = (i % hd as i64) as i32;
        let h = ((i / hd as i64) % heads as i64) as i32;
        let tok = i / dim as i64;
        let (y, x) = ((tok / grid as i64) as i32, (tok % grid as i64) as i32);
        let (wy, wx, ly, lx) = (y / win, x / win, y % win, x % win);
        let s = (((wy * nwx + wx) as i64 * heads as i64 + h as i64) * win as i64 * win as i64 + (ly * win + lx) as i64)
            * hd as i64
            + d as i64;
        unsafe { T::store(out.add(i as usize), *o.add(s as usize)) };
        i += stride;
    }
}

#[kernel]
pub unsafe fn vit_merge_heads(out: *mut f32, o: *const f32, grid: i32, win: i32, heads: i32, hd: i32) {
    unsafe { merge_heads(out, o, grid, win, heads, hd) };
}

#[kernel]
pub unsafe fn vit_merge_heads_bf16(out: *mut Bf16, o: *const f32, grid: i32, win: i32, heads: i32, hd: i32) {
    unsafe { merge_heads(out, o, grid, win, heads, hd) };
}
