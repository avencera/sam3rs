//! Image decoding and the SAM 3 processor transform: antialiased bilinear resize to
//! 1008x1008 in uint8 precision, then scale to [-1, 1]

use std::path::Path;

use eyre::{Result, WrapErr};

use crate::gpu::{Gpu, Tensor};
use crate::launch;
use crate::ops::{B256, ew, upload_i32};
use crate::vision::IMAGE_SIZE;

/// An 8-bit RGB image in HWC order
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn open(path: &Path) -> Result<Self> {
        let img = image::open(path).wrap_err_with(|| format!("decode {}", path.display()))?.to_rgb8();
        Ok(Self { width: img.width() as usize, height: img.height() as usize, data: img.into_raw() })
    }
}

/// Filter taps for one axis, as `_upsample_bilinear2d_aa` computes them on CUDA (in f32)
struct Taps {
    starts: Vec<i32>,
    counts: Vec<i32>,
    weights: Vec<f32>,
    max_taps: usize,
}

fn aa_taps(input: usize, output: usize) -> Taps {
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_taps = (support.ceil() as usize) * 2 + 1;
    let mut taps = Taps { starts: vec![], counts: vec![], weights: vec![0.0; output * max_taps], max_taps };
    for i in 0..output {
        let center = scale * (i as f32 + 0.5);
        let xmin = ((center - support + 0.5) as i32).max(0);
        let xsize = ((center + support + 0.5) as i32).min(input as i32) - xmin;
        let w = &mut taps.weights[i * max_taps..(i + 1) * max_taps];
        let mut total = 0f32;
        for j in 0..xsize as usize {
            let x = ((j as f32 + (xmin as f32 - center) + 0.5) * invscale).abs();
            let v = if x < 1.0 { 1.0 - x } else { 0.0 };
            w[j] = v;
            total += v;
        }
        if total != 0.0 {
            for v in &mut w[..xsize as usize] {
                *v /= total;
            }
        }
        taps.starts.push(xmin);
        taps.counts.push(xsize);
    }
    taps
}

/// RGB image -> normalized pixels `[3, 1008, 1008]`
pub fn preprocess(gpu: &Gpu, img: &RgbImage) -> Result<Tensor> {
    let (h, w) = (img.height, img.width);
    let raw = gpu.upload_bytes(&img.data)?;
    let planes = gpu.empty(&[3, h, w])?;
    launch!(gpu, "hwc_u8_to_chw", ew(h * w * 3), B256, 0; &planes, raw.ptr(), h, w)?;

    // separable passes; rounding back to uint8 absorbs the summation-order difference
    // from the fused PyTorch kernel
    let tx = aa_taps(w, IMAGE_SIZE);
    let (xs, xc) = (upload_i32(gpu, &tx.starts)?, upload_i32(gpu, &tx.counts)?);
    let xw = gpu.upload(&tx.weights, &[tx.weights.len()])?;
    let wide = gpu.empty(&[3, h, IMAGE_SIZE])?;
    launch!(gpu, "resample_cols", ew(wide.len()), B256, 0;
        &wide, &planes, 3 * h, w, IMAGE_SIZE, xs.ptr(), xc.ptr(), &xw, tx.max_taps)?;

    let ty = aa_taps(h, IMAGE_SIZE);
    let (ys, yc) = (upload_i32(gpu, &ty.starts)?, upload_i32(gpu, &ty.counts)?);
    let yw = gpu.upload(&ty.weights, &[ty.weights.len()])?;
    let out = gpu.empty(&[3, IMAGE_SIZE, IMAGE_SIZE])?;
    launch!(gpu, "resample_rows", ew(out.len()), B256, 0;
        &out, &wide, 3usize, h, IMAGE_SIZE, IMAGE_SIZE, ys.ptr(), yc.ptr(), &yw, ty.max_taps)?;
    launch!(gpu, "quantize_normalize", ew(out.len()), B256, 0; &out, out.len() as u64)?;
    Ok(out)
}
