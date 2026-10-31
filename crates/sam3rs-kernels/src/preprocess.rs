//! Image preprocessing: the processor's antialiased resize and normalization

use cuda_device::kernel;

use crate::common::{grid_stride, math};

/// Antialiased bilinear resize along rows (PyTorch's `_upsample_bilinear2d_aa`)
///
/// `src [channels, in_h, w] -> dst [channels, out_h, w]`, with taps precomputed on the
/// host per output row: `starts[y]`, `counts[y]` and `weights[y * max_taps + k]`
#[kernel]
pub unsafe fn resample_rows(
    dst: *mut f32,
    src: *const f32,
    channels: i32,
    in_h: i32,
    w: i32,
    out_h: i32,
    starts: *const i32,
    counts: *const i32,
    weights: *const f32,
    max_taps: i32,
) {
    let n = channels as i64 * out_h as i64 * w as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let x = (i % w as i64) as i32;
        let y = ((i / w as i64) % out_h as i64) as i32;
        let c = (i / (w as i64 * out_h as i64)) as i32;
        let (s, cnt) = unsafe { (*starts.add(y as usize), *counts.add(y as usize)) };
        let col = unsafe { src.add((c as i64 * in_h as i64 * w as i64 + x as i64) as usize) };
        let mut acc = 0.0;
        for k in 0..cnt {
            acc +=
                unsafe { *weights.add((y * max_taps + k) as usize) * *col.add(((s + k) as i64 * w as i64) as usize) };
        }
        unsafe { *dst.add(i as usize) = acc };
        i += stride;
    }
}

/// [`resample_rows`] along columns: `src [rows, in_w] -> dst [rows, out_w]`
#[kernel]
pub unsafe fn resample_cols(
    dst: *mut f32,
    src: *const f32,
    rows: i32,
    in_w: i32,
    out_w: i32,
    starts: *const i32,
    counts: *const i32,
    weights: *const f32,
    max_taps: i32,
) {
    let n = rows as i64 * out_w as i64;
    let (mut i, stride) = grid_stride();
    while i < n {
        let x = (i % out_w as i64) as i32;
        let r = i / out_w as i64;
        let (s, cnt) = unsafe { (*starts.add(x as usize), *counts.add(x as usize)) };
        let row = unsafe { src.add((r * in_w as i64) as usize) };
        let mut acc = 0.0;
        for k in 0..cnt {
            acc += unsafe { *weights.add((x * max_taps + k) as usize) * *row.add((s + k) as usize) };
        }
        unsafe { *dst.add(i as usize) = acc };
        i += stride;
    }
}

/// uint8 HWC image into float CHW planes
#[kernel]
pub unsafe fn hwc_u8_to_chw(dst: *mut f32, src: *const u8, h: i32, w: i32) {
    let plane = h as i64 * w as i64;
    let n = plane * 3;
    let (mut i, stride) = grid_stride();
    while i < n {
        let (c, p) = (i % 3, i / 3);
        unsafe { *dst.add((c * plane + p) as usize) = *src.add(i as usize) as f32 };
        i += stride;
    }
}

/// Rounds to uint8 and scales to [-1, 1] in place: the processor resizes a uint8
/// tensor, so values are rounded back to uint8 before scaling
#[kernel]
pub unsafe fn quantize_normalize(x: *mut f32, n: i64) {
    let (mut i, stride) = grid_stride();
    while i < n {
        let v = math::rintf(unsafe { *x.add(i as usize) }).max(0.0).min(255.0);
        unsafe { *x.add(i as usize) = (v / 255.0 - 0.5) / 0.5 };
        i += stride;
    }
}
