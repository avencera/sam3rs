//! Video tracker kernels: axial RoPE for the memory attention, the memory encoder's
//! convolutions and its mask transform
//!
//! One thread per output element, so outputs are `DisjointSlice`s indexed by the thread
//! index, and index math is 32-bit as in the elementwise kernels. Feature maps are
//! channels-last `[h * w, c]`; map heights come from slice lengths

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{activate, math, pack_bf16, pack_f16};

/// Axial RoPE over rows of `dim` values, out of place: rows below `rope_rows` rotate each
/// pair `(2j, 2j + 1)` by the angle at position `row % positions`, and the rows after
/// them (the memory attention's object pointer tokens) are copied as they are
///
/// `cos` and `sin` are `[positions, dim / 2]`. Keys repeat the 72x72 grid's positions
/// once per memory frame, which the row index taken mod `positions` gives
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn rope_rows(mut out: DisjointSlice<f32>, x: &[f32], cos: &[f32], sin: &[f32], dim: u32, rope_rows: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (row, c) = (i / dim, i % dim);
    let Some(mut out) = out.element_thread32(t) else { return };
    if row >= rope_rows {
        out.write(x[i as usize]);
        return;
    }
    let half = dim / 2;
    let positions = cos.len() as u32 / half;
    let k = ((row % positions) * half + c / 2) as usize;
    let base = (row * dim + (c & !1)) as usize;
    let (re, im) = (x[base], x[base + 1]);
    // the complex product (re + i im)(cos + i sin), as `apply_rotary_enc` computes it
    let v = if c & 1 == 0 { re * cos[k] - im * sin[k] } else { re * sin[k] + im * cos[k] };
    out.write(v);
}

/// [`rope_rows`] into FP16, one thread per pair, for the fused d256 attention; with
/// `rope_rows = 0` it only converts, as for values
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn rope_rows_f16(mut out: DisjointSlice<u32>, x: &[f32], cos: &[f32], sin: &[f32], dim: u32, rope_rows: u32) {
    let t = thread::index_1d_u32(lc);
    let base = t.get() * 2;
    let (row, c) = (base / dim, base % dim);
    let Some(mut out) = out.element_thread32(t) else { return };
    let (re, im) = (x[base as usize], x[base as usize + 1]);
    if row >= rope_rows {
        out.write(pack_f16(re, im));
        return;
    }
    let half = dim / 2;
    let positions = cos.len() as u32 / half;
    let k = ((row % positions) * half + c / 2) as usize;
    // the same products as rope_rows, rounded once
    out.write(pack_f16(re * cos[k] - im * sin[k], re * sin[k] + im * cos[k]));
}

/// 3x3 conv, stride 2, padding 1: `x [ih * iw, cin] -> out [(ih / 2) * (iw / 2), cout]`
/// with PyTorch's `[cout, cin, 3, 3]` weights; `cout` is `bias.len()`
///
/// The mask downsampler's layers have 1 to 64 input channels, too few for a GEMM to pay
/// for an im2col copy at the early 576x576 and 288x288 maps
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn conv3x3_s2(mut out: DisjointSlice<f32>, x: &[f32], w: &[f32], bias: &[f32], iw: u32, cin: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let cout = bias.len() as u32;
    let ih = x.len() as u32 / (iw * cin);
    let ow = iw / 2;
    let (oc, p) = (i % cout, i / cout);
    let (oy, ox) = (p / ow, p % ow);
    let Some(mut out) = out.element_thread32(t) else { return };
    let mut acc = bias[oc as usize];
    let mut ky = 0;
    while ky < 3 {
        let iy = (2 * oy + ky) as i32 - 1;
        if iy >= 0 && (iy as u32) < ih {
            let mut kx = 0;
            while kx < 3 {
                let ix = (2 * ox + kx) as i32 - 1;
                if ix >= 0 && (ix as u32) < iw {
                    let src = (iy as u32 * iw + ix as u32) * cin;
                    let mut ic = 0;
                    while ic < cin {
                        acc += w[(((oc * cin + ic) * 3 + ky) * 3 + kx) as usize] * x[(src + ic) as usize];
                        ic += 1;
                    }
                }
                kx += 1;
            }
        }
        ky += 1;
    }
    out.write(acc);
}

/// The im2col of a 3x3 conv with stride 2 and padding 1: `x [ih * iw, cin] -> out [(ih /
/// 2) * (iw / 2), cin * 9]`, columns in PyTorch's weight order `(ic, ky, kx)`, so the conv
/// is one GEMM against the `[cout, cin * 9]` weights
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn im2col3x3_s2(mut out: DisjointSlice<f32>, x: &[f32], iw: u32, cin: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let k = cin * 9;
    let ih = x.len() as u32 / (iw * cin);
    let ow = iw / 2;
    let (p, col) = (i / k, i % k);
    let (ic, tap) = (col / 9, col % 9);
    let (oy, ox) = (p / ow, p % ow);
    let Some(mut out) = out.element_thread32(t) else { return };
    let iy = (2 * oy + tap / 3) as i32 - 1;
    let ix = (2 * ox + tap % 3) as i32 - 1;
    let inside = iy >= 0 && (iy as u32) < ih && ix >= 0 && (ix as u32) < iw;
    out.write(if inside { x[((iy as u32 * iw + ix as u32) * cin + ic) as usize] } else { 0.0 });
}

/// LayerNorm, then the activation `act`, over each row of `x [rows, cols]` in place, one
/// thread per row: the mask downsampler's maps have 4 to 64 channels, which would leave a
/// block per row nearly idle (331,776 rows of 4 at the first layer)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn layernorm_act_rows(mut x: DisjointSlice<f32>, w: &[f32], b: &[f32], cols: u32, eps: f32, act: i32) {
    let t = thread::index_1d_u32(lc);
    let r = t.get() as usize;
    let n = cols as usize;
    if n == 0 || (r + 1) * n > x.len() || w.len() < n || b.len() < n {
        return;
    }
    let row = x.as_mut_ptr().wrapping_add(r * n);
    // SAFETY, for every access below: row r lies inside x (checked above) and no other
    // thread touches it; c < n <= w.len(), b.len()
    let mut s = 0.0;
    for c in 0..n {
        s += unsafe { *row.add(c) };
    }
    let mean = s / cols as f32;
    let mut v = 0.0;
    for c in 0..n {
        let d = unsafe { *row.add(c) } - mean;
        v += d * d;
    }
    // two passes and rsqrt, as the block-per-row LayerNorm computes it
    let rstd = math::rsqrtf(v / cols as f32 + eps);
    for c in 0..n {
        unsafe {
            let y = (*row.add(c) - mean) * rstd * *w.get_unchecked(c) + *b.get_unchecked(c);
            *row.add(c) = activate(y, act);
        }
    }
}

/// Depthwise 7x7 conv, padding 3, of a `w`-wide channels-last map `x [h * w, c]`, with
/// PyTorch's `[c, 1, 7, 7]` weights; `c` is `bias.len()`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn dwconv7x7(mut out: DisjointSlice<f32>, x: &[f32], w: &[f32], bias: &[f32], width: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let c = bias.len() as u32;
    let h = x.len() as u32 / (width * c);
    let (ch, p) = (i % c, i / c);
    let (y, xx) = (p / width, p % width);
    let Some(mut out) = out.element_thread32(t) else { return };
    let mut acc = bias[ch as usize];
    let mut ky = 0;
    while ky < 7 {
        let iy = (y + ky) as i32 - 3;
        if iy >= 0 && (iy as u32) < h {
            let mut kx = 0;
            while kx < 7 {
                let ix = (xx + kx) as i32 - 3;
                if ix >= 0 && (ix as u32) < width {
                    acc += w[(ch * 49 + ky * 7 + kx) as usize] * x[((iy as u32 * width + ix as u32) * c + ch) as usize];
                }
                kx += 1;
            }
        }
        ky += 1;
    }
    out.write(acc);
}

/// The memory encoder's mask input: `sigmoid(x) * 20 - 10`, or with `binarize` (frames
/// the user clicked on) `(x > 0) * 20 - 10`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn mask_for_mem(mut out: DisjointSlice<f32>, x: &[f32], binarize: i32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    let Some(mut out) = out.element_thread32(t) else { return };
    let v = x[i];
    let m = if binarize != 0 { (v > 0.0) as u32 as f32 } else { 1.0 / (1.0 + math::expf(-v)) };
    out.write(m * 20.0 - 10.0);
}

/// `x = f32(bf16(x))` in place, round to nearest even: upstream stores the memory bank in
/// BF16 even in FP32 mode
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn round_bf16(mut x: DisjointSlice<f32>) {
    let t = thread::index_1d_u32(lc);
    let Some(mut x) = x.element_thread32(t) else { return };
    let bits = pack_bf16(x.read(), 0.0) & 0xffff;
    x.write(f32::from_bits(bits << 16));
}

/// Upstream's `_apply_non_overlapping_constraints` over `objects` planes `x [objects, n]`:
/// at each pixel the object with the highest logit (the first on ties) keeps it, and the
/// others are clamped to at most -10
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn nonoverlap(mut out: DisjointSlice<f32>, x: &[f32], objects: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let n = x.len() as u32 / objects;
    let (o, p) = (i / n, i % n);
    let Some(mut out) = out.element_thread32(t) else { return };
    let mut best = 0;
    let mut k = 1;
    while k < objects {
        if x[(k * n + p) as usize] > x[(best * n + p) as usize] {
            best = k;
        }
        k += 1;
    }
    let v = x[i as usize];
    out.write(if best == o { v } else { math::fminf(v, -10.0) });
}
