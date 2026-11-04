//! The SAM 3 processor transform on the GPU: antialiased bilinear resize to 1008x1008 in
//! uint8 precision, then scale to [-1, 1]

use eyre::Result;

use crate::cuda::gpu::{Buf, Gpu, Tensor, launch};
use crate::cuda::ops::{B256, ew, upload_i32};
use crate::cuda::vision::IMAGE_SIZE;
use crate::image::{RgbImage, Taps, aa_taps};

/// One antialiased bilinear resize (`_upsample_bilinear2d_aa`) of planes from one size to
/// another, with the taps uploaded once
pub struct AaResize {
    rows: (Buf, Buf, Tensor),
    cols: (Buf, Buf, Tensor),
    input: (usize, usize),
    output: (usize, usize),
}

impl AaResize {
    /// Resizes `(height, width)` planes to `output`
    pub fn new(gpu: &Gpu, input: (usize, usize), output: (usize, usize)) -> Result<Self> {
        let upload = |t: Taps| -> Result<(Buf, Buf, Tensor)> {
            let n = t.weights.len();
            Ok((upload_i32(gpu, t.starts)?, upload_i32(gpu, t.counts)?, gpu.upload(t.weights, &[n])?))
        };
        Ok(Self { rows: upload(aa_taps(input.0, output.0))?, cols: upload(aa_taps(input.1, output.1))?, input, output })
    }

    /// `x [planes, in_h, in_w]` (any shape of that length) -> `[planes, out_h, out_w]`
    pub fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        let ((ih, iw), (oh, ow)) = (self.input, self.output);
        let planes = x.len() / (ih * iw);
        assert_eq!(x.len(), planes * ih * iw, "resize input of {} values", x.len());
        let wide = gpu.empty(&[planes, ih, ow])?;
        let (s, c, w) = &self.cols;
        launch!(gpu, "resample_cols", ew(wide.len()), B256, 0; &wide, x, s.slice::<i32>(), c.slice::<i32>(), w)?;
        let out = gpu.empty(&[planes, oh, ow])?;
        let (s, c, w) = &self.rows;
        launch!(gpu, "resample_rows", ew(out.len()), B256, 0;
            &out, &wide, s.slice::<i32>(), c.slice::<i32>(), w, ow)?;
        Ok(out)
    }
}

/// RGB image -> normalized pixels `[3, 1008, 1008]`
pub fn preprocess(gpu: &Gpu, img: &RgbImage) -> Result<Tensor> {
    let (h, w) = (img.height(), img.width());
    let raw = gpu.upload_slice(img.data())?;
    let planes = gpu.empty(&[3, h, w])?;
    launch!(gpu, "hwc_u8_to_chw", ew(planes.len()), B256, 0; &planes, raw.slice::<u8>())?;

    // separable passes; rounding back to uint8 absorbs the summation-order difference
    // from the fused PyTorch kernel
    let tx = aa_taps(w, IMAGE_SIZE);
    let w_taps = tx.weights.len();
    let (xs, xc) = (upload_i32(gpu, tx.starts)?, upload_i32(gpu, tx.counts)?);
    let xw = gpu.upload(tx.weights, &[w_taps])?;
    let wide = gpu.empty(&[3, h, IMAGE_SIZE])?;
    launch!(gpu, "resample_cols", ew(wide.len()), B256, 0;
        &wide, &planes, xs.slice::<i32>(), xc.slice::<i32>(), &xw)?;

    let ty = aa_taps(h, IMAGE_SIZE);
    let h_taps = ty.weights.len();
    let (ys, yc) = (upload_i32(gpu, ty.starts)?, upload_i32(gpu, ty.counts)?);
    let yw = gpu.upload(ty.weights, &[h_taps])?;
    let out = gpu.empty(&[3, IMAGE_SIZE, IMAGE_SIZE])?;
    launch!(gpu, "resample_rows", ew(out.len()), B256, 0;
        &out, &wide, ys.slice::<i32>(), yc.slice::<i32>(), &yw, IMAGE_SIZE)?;
    launch!(gpu, "quantize_normalize", ew(out.len()), B256, 0; &out)?;
    Ok(out)
}
