//! Image decoding, and the filter taps of the SAM 3 processor's antialiased bilinear
//! resize that each backend's preprocessing kernels apply

use std::path::Path;

use eyre::{Result, WrapErr, ensure, eyre};

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
pub(crate) struct Taps {
    pub(crate) starts: Vec<i32>,
    pub(crate) counts: Vec<i32>,
    pub(crate) weights: Vec<f32>,
}

pub(crate) fn aa_taps(input: usize, output: usize) -> Taps {
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_taps = (support.ceil() as usize) * 2 + 1;
    let mut taps = Taps { starts: vec![], counts: vec![], weights: vec![0.0; output * max_taps] };
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
