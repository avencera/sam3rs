//! Image preprocessing: the processor's antialiased resize and normalization
//!
//! Every kernel writes only the element at its own thread index, so outputs are
//! `DisjointSlice`s; `quantize_normalize` works in place and takes the buffer once. Image
//! dimensions come from slice lengths, and index math is 32-bit: the host launches one
//! thread per output element and keeps every buffer under 2^32 elements

use cuda_device::view::MatrixView32;
use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::math;

/// Antialiased bilinear resize along rows (PyTorch's `_upsample_bilinear2d_aa`)
///
/// `src [channels, in_h, w] -> dst [channels, out_h, w]`, with taps precomputed on the
/// host per output row: `starts[y]`, `counts[y]` and `weights[y * max_taps + k]`. `out_h`
/// is `starts.len()`, `max_taps` is `weights.len() / out_h` and `channels` follows from
/// the lengths of `dst` and `w`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn resample_rows(
    mut dst: DisjointSlice<f32>,
    src: &[f32],
    starts: &[i32],
    counts: &[i32],
    weights: &[f32],
    w: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let out_h = starts.len() as u32;
    let max_taps = weights.len() as u32 / out_h;
    let channels = dst.len() as u32 / (out_h * w);
    let Some(mut dst) = dst.element_thread32(t) else { return };
    let (x, y, c) = (i % w, (i / w) % out_h, i / (w * out_h));
    let (start, cnt) = (starts[y as usize] as u32, counts[y as usize] as u32);

    // this channel's plane from the tap window's first row down: column `x` of it holds
    // the inputs of the taps
    let plane = src.len() as u32 / channels;
    let window = &src[(c * plane + start * w) as usize..((c + 1) * plane) as usize];
    let taps = MatrixView32::new(weights, max_taps).row(y, cnt);
    let col = MatrixView32::new(window, w).col(x, cnt);
    let mut acc = 0.0;
    // both views are `None` only for a row without taps, which sums to zero
    if let (Some(taps), Some(col)) = (taps, col) {
        let mut k = 0;
        while k < cnt {
            acc += taps.get(k).unwrap_or(0.0) * col.get(k).unwrap_or(0.0);
            k += 1;
        }
    }
    dst.write(acc);
}

/// [`resample_rows`] along columns: `src [rows, in_w] -> dst [rows, out_w]`
///
/// `out_w` is `starts.len()`, `rows` is `dst.len() / out_w` and `in_w` is `src.len() / rows`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn resample_cols(mut dst: DisjointSlice<f32>, src: &[f32], starts: &[i32], counts: &[i32], weights: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let out_w = starts.len() as u32;
    let max_taps = weights.len() as u32 / out_w;
    let rows = dst.len() as u32 / out_w;
    let Some(mut dst) = dst.element_thread32(t) else { return };
    let (x, r) = (i % out_w, i / out_w);
    let (start, cnt) = (starts[x as usize] as u32, counts[x as usize] as u32);

    let in_w = src.len() as u32 / rows;
    let row = &src[(r * in_w + start) as usize..][..cnt as usize];
    let taps = &weights[(x * max_taps) as usize..][..cnt as usize];
    let mut acc = 0.0;
    for (wt, v) in taps.iter().zip(row) {
        acc += wt * v;
    }
    dst.write(acc);
}

/// uint8 HWC image into float CHW planes
///
/// One thread per output element, so the writes are contiguous and neighbouring threads'
/// stride-3 reads share cache lines
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn hwc_u8_to_chw(mut dst: DisjointSlice<f32>, src: &[u8]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let plane = dst.len() as u32 / 3;
    let Some(mut dst) = dst.element_thread32(t) else { return };
    let (c, p) = (i / plane, i % plane);
    dst.write(src[(p * 3 + c) as usize] as f32);
}

/// Rounds to uint8 and scales to [-1, 1] in place: the processor resizes a uint8
/// tensor, so values are rounded back to uint8 before scaling
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn quantize_normalize(mut x: DisjointSlice<f32>) {
    let t = thread::index_1d_u32(lc);
    let Some(mut x) = x.element_thread32(t) else { return };
    let v = math::fminf(math::fmaxf(math::rintf(x.read()), 0.0), 255.0);
    x.write((v / 255.0 - 0.5) / 0.5);
}
