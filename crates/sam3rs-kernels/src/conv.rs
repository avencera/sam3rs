//! Convolutions as GEMMs: the layout kernels around cuBLAS
//!
//! Each kernel writes one output element per thread (one 16-byte group of eight BF16
//! values for the im2col kernels), so outputs are `DisjointSlice`s indexed by the thread
//! index and index math is 32-bit, as in the elementwise kernels. Map heights come from
//! slice lengths: a source pixel below the last row lies past the end of its slice, so
//! the checked read that keeps it in bounds is also the bottom edge test

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{Bf16, Packed8, activate, ldg8, math, pack_bf16};

/// Eight floats rounded to nearest-even BF16
#[inline(always)]
fn pack8_bf16(v: [f32; 8]) -> Packed8<Bf16> {
    Packed8::new([pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7])])
}

/// `ConvTranspose2d(k=2, s=2)`: `g [h * w, o * 4]` holds `x @ W` for `W [C, o, 2, 2]`;
/// scatters to `y [2h * 2w, o]` and adds `bias` (none if empty)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn convt2x2_scatter(mut y: DisjointSlice<f32>, g: &[f32], bias: &[f32], w: u32, o: u32) {
    let t = thread::index_1d_u32(lc);
    let (oc, p) = (t.get() % o, t.get() / o);
    let (yy, xx) = (p / (2 * w), p % (2 * w));
    let src = ((yy >> 1) * w + (xx >> 1)) * o * 4 + oc * 4 + (yy & 1) * 2 + (xx & 1);
    let Some(mut y) = y.element_thread32(t) else { return };
    let b = if bias.is_empty() { 0.0 } else { bias[oc as usize] };
    y.write(g[src as usize] + b);
}

/// `x [h * w, c] -> xp [(h + 2) * (w + 2) + 2, c]` zero-padded, one thread per element of
/// `xp`. With the padded row width, the input of every 3x3 tap is one contiguous matrix
/// at a fixed offset
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn pad1(mut xp: DisjointSlice<f32>, x: &[f32], w: u32, c: u32) {
    let t = thread::index_1d_u32(lc);
    let (ch, p) = (t.get() % c, t.get() / c);
    let (py, px) = (p / (w + 2), p % (w + 2));
    let Some(mut xp) = xp.element_thread32(t) else { return };
    let mut v = 0.0;
    if py >= 1 && px >= 1 && px <= w {
        // the bottom padding rows and the two trailing pixels read past the end of x.
        // The load stays inside this branch: a pointer merged with the null of the
        // other side loses its global address space and becomes a generic load
        if let Some(&x) = x.get((((py - 1) * w + px - 1) * c + ch) as usize) {
            v = x;
        }
    }
    xp.write(v);
}

/// Pixel `p` of a `w`-wide map shifted by 3x3 tap `t` (offsets `t / 3 - 1`, `t % 3 - 1`),
/// or `None` left, right or above the map; below it is the caller's slice check
#[inline(always)]
fn tap_pixel(p: u32, t: u32, w: u32) -> Option<(u32, u32)> {
    let yy = (p / w) as i32 + (t / 3) as i32 - 1;
    let xx = (p % w) as i32 + (t % 3) as i32 - 1;
    (yy >= 0 && xx >= 0 && xx < w as i32).then_some((yy as u32, xx as u32))
}

/// 3x3, padding 1 im2col to BF16 for output pixels `[p0, p0 + n)` of a `w`-wide map
///
/// `out[p - p0, t * c + ic]` is `x` at pixel `p` shifted by tap `t = dy * 3 + dx`, zero
/// outside the image, so the conv is one GEMM with `K = 9c`. One thread per 8 channels:
/// `out` holds `n * 9 * c / 8` groups, and `c` must be a multiple of 8
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn im2col3x3_bf16(mut out: DisjointSlice<Packed8<Bf16>>, x: &[f32], w: u32, c: u32, p0: u32) {
    let t = thread::index_1d_u32(lc);
    let c8 = c / 8;
    let (ch, r) = ((t.get() % c8) * 8, t.get() / c8);
    let (tap, p) = (r % 9, p0 + r / 9);
    let Some(mut out) = out.element_thread32(t) else { return };
    let v = tap_pixel(p, tap, w).and_then(|(yy, xx)| ldg8(x, ((yy * w + xx) * c + ch) as usize));
    out.write(v.map_or(Packed8::new([0; 4]), pack8_bf16));
}

/// [`im2col3x3_bf16`] of `fine + up2(coarse)` without materializing that sum
///
/// The input at fine pixel `(y, x)` is `fine[y, x] + coarse[y / 2, x / 2]`, as
/// `upsample2_add` computes it. With a non-empty `stats` (`[groups, 2]` sums and sums of
/// squares), coarse first gets the group norm with weights `gw`, `gb` and the activation
/// `group_norm_apply` would apply, so neither the coarse map's normalized copy nor the
/// upsampled sum is ever written. `w` is the fine map's (even) width; `groups` must
/// divide `c` into multiples of 8
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn im2col3x3_up_bf16(
    mut out: DisjointSlice<Packed8<Bf16>>,
    fine: &[f32],
    coarse: &[f32],
    stats: &[f32],
    gw: &[f32],
    gb: &[f32],
    groups: u32,
    eps: f32,
    act: i32,
    w: u32,
    c: u32,
    p0: u32,
) {
    let t = thread::index_1d_u32(lc);
    let c8 = c / 8;
    let (ch, r) = ((t.get() % c8) * 8, t.get() / c8);
    let (tap, p) = (r % 9, p0 + r / 9);
    let Some(mut out) = out.element_thread32(t) else { return };
    let Some((yy, xx)) = tap_pixel(p, tap, w) else { return out.write(Packed8::new([0; 4])) };
    let Some(a) = ldg8(fine, ((yy * w + xx) * c + ch) as usize) else { return out.write(Packed8::new([0; 4])) };

    let q = ((yy >> 1) * (w >> 1) + (xx >> 1)) * c + ch;
    // a fine pixel inside the map has its coarse pixel inside too, unless the host's
    // shapes disagree
    let Some(mut u) = ldg8(coarse, q as usize) else { panic!() };
    if !stats.is_empty() {
        // group norm over each group's coarse values: (h / 2) * (w / 2) pixels of c /
        // groups channels
        let count = (coarse.len() as u32 / groups) as f32;
        let g = (ch / (c / groups)) as usize;
        let mean = stats[2 * g] / count;
        let var = math::fmaxf(stats[2 * g + 1] / count - mean * mean, 0.0);
        let (gw, gb) = (&gw[ch as usize..ch as usize + 8], &gb[ch as usize..ch as usize + 8]);
        let mut k = 0;
        #[unroll]
        while k < 8 {
            u[k] = activate((u[k] - mean) * math::rsqrtf(var + eps) * gw[k] + gb[k], act);
            k += 1;
        }
    }
    let s = |k: usize| a[k] + u[k];
    out.write(pack8_bf16([s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7)]));
}

/// `yp [h * (w + 2), o]` (a 3x3 conv over padded rows) `-> y [h * w, o]` plus `bias` (none
/// if empty)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn crop_bias(mut y: DisjointSlice<f32>, yp: &[f32], bias: &[f32], w: u32, o: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    // each padded row has 2 more pixels than its output row
    let src = i + 2 * o * (i / (w * o));
    let Some(mut y) = y.element_thread32(t) else { return };
    let b = if bias.is_empty() { 0.0 } else { bias[(i % o) as usize] };
    y.write(yp[src as usize] + b);
}

/// `out [2h * 2w, c] = fine + nearest-upsampled coarse [h * w, c]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn upsample2_add(mut out: DisjointSlice<f32>, fine: &[f32], coarse: &[f32], w: u32, c: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (ch, p) = (i % c, i / c);
    let (y, x) = (p / (2 * w), p % (2 * w));
    let q = ((y >> 1) * w + (x >> 1)) * c + ch;
    let Some(mut out) = out.element_thread32(t) else { return };
    out.write(fine[i as usize] + coarse[q as usize]);
}

/// Non-overlapping 2x2 patches of channels-last `x [h * w, c] -> [(h / 2) * (w / 2), c * 4]`
/// in `(c, ky, kx)` order, so a stride-2 2x2 conv is one GEMM with the `[O, C, 2, 2]`
/// weight
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn patchify2x2(mut out: DisjointSlice<f32>, x: &[f32], w: u32, c: u32) {
    let t = thread::index_1d_u32(lc);
    let (col, p) = (t.get() % (c * 4), t.get() / (c * 4));
    let (ch, ky, kx) = (col / 4, (col / 2) % 2, col % 2);
    let ow = w / 2;
    let (y, xx) = (p / ow, p % ow);
    let s = ((2 * y + ky) * w + 2 * xx + kx) * c + ch;
    let Some(mut out) = out.element_thread32(t) else { return };
    out.write(x[s as usize]);
}
