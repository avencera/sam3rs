//! Mask postprocessing shared by the detector and the point decoder: low-resolution
//! logits to full-size binary masks

use eyre::{Result, ensure};

use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::vision::GRID;

/// Side of the low-resolution mask logits
pub const LOW_RES: usize = 4 * GRID;

/// Most output pixels per launch, so the kernel's 32-bit thread index can't overflow; a
/// mask at the largest image size is 2^28 pixels, so a launch holds at least one
const LAUNCH_PIXELS: usize = 1 << 31;

/// Masks `first..first + n` of `low_res [*, 288 * 288]` upsampled to `width` x `height`
/// and thresholded, as `[n, height * width]` bytes of 0 or 1
///
/// The bilinear resize runs on the logits, as upstream's postprocessing does, and
/// thresholds at 0 in the same pass, so no FP32 full-size plane is ever stored
pub fn upsample_masks(
    gpu: &Gpu,
    low_res: &Tensor,
    first: usize,
    n: usize,
    width: usize,
    height: usize,
) -> Result<Tensor> {
    let side = LOW_RES * LOW_RES;
    ensure!((first + n) * side <= low_res.len(), "masks {first}..{} of {:?}", first + n, low_res.shape());
    let plane = width * height;
    let out = gpu.empty(&[n, plane], DType::U8)?;
    let per_launch = (LAUNCH_PIXELS / plane.max(1)).max(1);
    for start in (0..n).step_by(per_launch) {
        let count = per_launch.min(n - start);
        let args = [
            out.at(start * plane),
            low_res.at((first + start) * side),
            LOW_RES.into(),
            LOW_RES.into(),
            height.into(),
            width.into(),
        ];
        gpu.dispatch_1d("resize_threshold_u8", count * plane, &args)?;
    }
    Ok(out)
}
