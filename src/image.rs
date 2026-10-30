//! Image decoding and the SAM 3 processor transform: antialiased bilinear resize to
//! 1008x1008 in uint8 precision, then scale to [-1, 1]

use std::path::Path;

use eyre::{Result, WrapErr, ensure, eyre};

use crate::gpu::{Gpu, Tensor, launch};
use crate::ops::{B256, ew, upload_i32};
use crate::vision::IMAGE_SIZE;

/// Longest side accepted, which keeps every pixel index within the kernels' 32-bit range
pub const MAX_IMAGE_SIDE: usize = 16384;

/// An 8-bit RGB image in HWC order
#[derive(Clone, Debug)]
pub struct RgbImage {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl RgbImage {
    /// Wraps `width * height * 3` bytes of interleaved RGB rows
    ///
    /// Fails when a side is zero or above [`MAX_IMAGE_SIDE`], or when `data` has the wrong length
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        ensure!(
            (1..=MAX_IMAGE_SIDE).contains(&width) && (1..=MAX_IMAGE_SIDE).contains(&height),
            "image is {width}x{height}; each side must be 1 to {MAX_IMAGE_SIDE} pixels"
        );
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| eyre!("image size {width}x{height} overflows"))?;
        ensure!(data.len() == expected, "a {width}x{height} RGB image needs {expected} bytes, got {}", data.len());
        Ok(Self { width, height, data })
    }

    /// Decodes a JPEG or PNG file
    pub fn open(path: &Path) -> Result<Self> {
        let img = image::open(path).wrap_err_with(|| format!("decode {}", path.display()))?.to_rgb8();
        Self::new(img.width() as usize, img.height() as usize, img.into_raw())
    }

    /// Width in pixels
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in pixels
    pub fn height(&self) -> usize {
        self.height
    }

    /// `height` rows of `width * 3` bytes
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// A copy with each mask's pixels blended halfway toward its own color, cycling
    /// through six colors; each mask is `height * width` bytes, nonzero = inside
    pub fn overlay(&self, masks: &[&[u8]]) -> Result<Self> {
        const COLORS: [[u8; 3]; 6] =
            [[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240]];
        let pixels = self.width * self.height;
        let mut data = self.data.clone();
        for (m, mask) in masks.iter().enumerate() {
            ensure!(mask.len() == pixels, "mask {m} has {} pixels, the image {pixels}", mask.len());
            let color = COLORS[m % COLORS.len()];
            for (px, _) in data.as_chunks_mut::<3>().0.iter_mut().zip(mask.iter()).filter(|(_, v)| **v != 0) {
                for (p, c) in px.iter_mut().zip(color) {
                    *p = ((*p as u16 + c as u16) / 2) as u8;
                }
            }
        }
        Ok(Self { data, ..*self })
    }

    /// Writes the image as PNG or JPEG, chosen by the file extension
    pub fn save(&self, path: &Path) -> Result<()> {
        image::save_buffer(path, &self.data, self.width as u32, self.height as u32, image::ExtendedColorType::Rgb8)
            .wrap_err_with(|| format!("write {}", path.display()))
    }

    /// The RGB bytes, giving up the image
    pub fn into_data(self) -> Vec<u8> {
        self.data
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
        for (j, tap) in w[..xsize as usize].iter_mut().enumerate() {
            let x = ((j as f32 + (xmin as f32 - center) + 0.5) * invscale).abs();
            *tap = if x < 1.0 { 1.0 - x } else { 0.0 };
            total += *tap;
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
    let raw = gpu.upload_slice(&img.data)?;
    let planes = gpu.empty(&[3, h, w])?;
    launch!(gpu, "hwc_u8_to_chw", ew(h * w * 3), B256, 0; &planes, raw.ptr(), h, w)?;

    // separable passes; rounding back to uint8 absorbs the summation-order difference
    // from the fused PyTorch kernel
    let tx = aa_taps(w, IMAGE_SIZE);
    let w_taps = tx.weights.len();
    let (xs, xc) = (upload_i32(gpu, tx.starts)?, upload_i32(gpu, tx.counts)?);
    let xw = gpu.upload(tx.weights, &[w_taps])?;
    let wide = gpu.empty(&[3, h, IMAGE_SIZE])?;
    launch!(gpu, "resample_cols", ew(wide.len()), B256, 0;
        &wide, &planes, 3 * h, w, IMAGE_SIZE, xs.ptr(), xc.ptr(), &xw, tx.max_taps)?;

    let ty = aa_taps(h, IMAGE_SIZE);
    let h_taps = ty.weights.len();
    let (ys, yc) = (upload_i32(gpu, ty.starts)?, upload_i32(gpu, ty.counts)?);
    let yw = gpu.upload(ty.weights, &[h_taps])?;
    let out = gpu.empty(&[3, IMAGE_SIZE, IMAGE_SIZE])?;
    launch!(gpu, "resample_rows", ew(out.len()), B256, 0;
        &out, &wide, 3usize, h, IMAGE_SIZE, IMAGE_SIZE, ys.ptr(), yc.ptr(), &yw, ty.max_taps)?;
    launch!(gpu, "quantize_normalize", ew(out.len()), B256, 0; &out, out.len() as u64)?;
    Ok(out)
}
