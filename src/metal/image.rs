//! The SAM 3 processor transform on the GPU: antialiased bilinear resize to 1008x1008 in
//! uint8 precision, then scale to [-1, 1]

use eyre::Result;

use crate::image::{RgbImage, Taps, aa_taps};
use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::vision::IMAGE_SIZE;

/// One axis's filter taps on the GPU, and the most taps any output reads
struct AxisTaps {
    starts: Tensor,
    counts: Tensor,
    weights: Tensor,
    max: usize,
}

impl AxisTaps {
    fn upload(gpu: &Gpu, t: Taps) -> Result<Self> {
        let n = t.starts.len();
        let max = t.weights.len() / n;
        Ok(Self {
            starts: gpu.upload(&t.starts, &[n], DType::I32)?,
            counts: gpu.upload(&t.counts, &[n], DType::I32)?,
            weights: gpu.upload_f32(&t.weights, &[t.weights.len()])?,
            max,
        })
    }
}

/// RGB image -> normalized pixels `[3, 1008, 1008]`
pub fn preprocess(gpu: &Gpu, img: &RgbImage) -> Result<Tensor> {
    let (h, w) = (img.height(), img.width());
    let raw = gpu.upload(img.data(), &[h * w * 3], DType::U8)?;
    let planes = gpu.empty(&[3, h, w], DType::F32)?;
    gpu.dispatch_1d("hwc_u8_to_chw", planes.len(), &[planes.arg(), raw.arg(), (h * w).into()])?;

    // separable passes; rounding back to uint8 absorbs the summation-order difference
    // from the fused PyTorch kernel
    let tx = AxisTaps::upload(gpu, aa_taps(w, IMAGE_SIZE))?;
    let wide = gpu.empty(&[3, h, IMAGE_SIZE], DType::F32)?;
    let args = [
        wide.arg(),
        planes.arg(),
        tx.starts.arg(),
        tx.counts.arg(),
        tx.weights.arg(),
        w.into(),
        IMAGE_SIZE.into(),
        tx.max.into(),
    ];
    gpu.dispatch_1d("resample_cols", wide.len(), &args)?;

    let ty = AxisTaps::upload(gpu, aa_taps(h, IMAGE_SIZE))?;
    let out = gpu.empty(&[3, IMAGE_SIZE, IMAGE_SIZE], DType::F32)?;
    let args = [
        out.arg(),
        wide.arg(),
        ty.starts.arg(),
        ty.counts.arg(),
        ty.weights.arg(),
        IMAGE_SIZE.into(),
        h.into(),
        IMAGE_SIZE.into(),
        ty.max.into(),
    ];
    gpu.dispatch_1d("resample_rows", out.len(), &args)?;
    gpu.dispatch_1d("quantize_normalize", out.len(), &[out.arg()])?;
    Ok(out)
}
