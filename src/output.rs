//! What prompts and the video tracker return

use crate::masks::MaskBuffer;

/// `Sam3Processor.confidence_threshold`
pub const SCORE_THRESHOLD: f32 = 0.5;

/// Text used for box-only prompts, as in upstream and osam
pub const VISUAL_PROMPT: &str = "visual";

/// One detected instance; its mask is [`Detections::mask`]
#[derive(Clone, Debug)]
pub struct Detection {
    /// sigmoid(logit) * sigmoid(presence)
    pub score: f32,
    /// `[x0, y0, x1, y1]` in pixels, not clamped to the image
    pub bbox: [f32; 4],
    /// Index of the DETR query (0 to 199) that produced it
    pub query: usize,
}

/// The instances [`crate::Sam3::detect`] found, in DETR query order, with all their masks in
/// one buffer
///
/// One buffer lets callers take every mask without copying it again: at 2560 px a mask
/// is 5 MB, and copying a stack of them into fresh memory cost more than the detection
#[derive(Clone, Debug, Default)]
pub struct Detections {
    pub(crate) instances: Vec<Detection>,
    pub(crate) masks: MaskBuffer,
}

impl Detections {
    /// Number of instances
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether nothing scored above [`SCORE_THRESHOLD`]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// The instances, without their masks
    pub fn iter(&self) -> std::slice::Iter<'_, Detection> {
        self.instances.iter()
    }

    /// Mask of instance `i`: `height * width` bytes, 1 = foreground
    ///
    /// Panics if `i` is out of range
    pub fn mask(&self, i: usize) -> &[u8] {
        assert!(i < self.len(), "mask {i} of {} detections", self.len());
        let pixels = self.masks.len() / self.len();
        &self.masks[i * pixels..][..pixels]
    }

    /// Every mask in instance order: `len * height * width` bytes, 1 = foreground
    pub fn masks(&self) -> &[u8] {
        &self.masks
    }

    /// Splits into the instances and the mask buffer of [`Detections::masks`]
    pub fn into_parts(self) -> (Vec<Detection>, MaskBuffer) {
        (self.instances, self.masks)
    }
}

impl IntoIterator for Detections {
    type Item = Detection;
    type IntoIter = std::vec::IntoIter<Detection>;

    fn into_iter(self) -> Self::IntoIter {
        self.instances.into_iter()
    }
}

impl<'a> IntoIterator for &'a Detections {
    type Item = &'a Detection;
    type IntoIter = std::slice::Iter<'a, Detection>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// The best of the three candidate masks for a point prompt
#[derive(Clone, Debug)]
pub struct PointMask {
    /// Predicted IoU of the chosen mask
    pub score: f32,
    /// `height * width` bytes, 1 = foreground
    pub mask: MaskBuffer,
}

/// Largest background region of a point mask that hole filling turns into foreground,
/// in low-res pixels, as upstream's `postprocess_masks`
pub(crate) const HOLE_AREA: usize = 256;

/// Most prompts of equal click counts [`crate::Sam3::segment_points_batch`] decodes together.
/// Batches of 32 measured no faster per prompt than 16 on an RTX 5090, and every
/// decoder activation grows with the batch
pub const POINT_BATCH: usize = 16;

/// One mask per prompt of [`crate::Sam3::segment_points_batch`], in prompt order, with all the
/// masks in one buffer as in [`Detections`]
#[derive(Clone, Debug, Default)]
pub struct PointMasks {
    pub(crate) scores: Vec<f32>,
    pub(crate) masks: MaskBuffer,
}

impl PointMasks {
    /// Number of prompts
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether there were no prompts
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Predicted IoU of each prompt's chosen mask
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Mask of prompt `i`: `height * width` bytes, 1 = foreground
    ///
    /// Panics if `i` is out of range
    pub fn mask(&self, i: usize) -> &[u8] {
        assert!(i < self.len(), "mask {i} of {} prompts", self.len());
        let pixels = self.masks.len() / self.len();
        &self.masks[i * pixels..][..pixels]
    }

    /// Every mask in prompt order: `len * height * width` bytes, 1 = foreground
    pub fn masks(&self) -> &[u8] {
        &self.masks
    }

    /// Splits into the scores and the mask buffer of [`PointMasks::masks`]
    pub fn into_parts(self) -> (Vec<f32>, MaskBuffer) {
        (self.scores, self.masks)
    }
}

/// One tracked object on one frame
#[derive(Clone, Debug)]
pub struct TrackedObject {
    /// Object score logit; the object counts as present above 0
    pub score: f32,
    /// `height * width` bytes, 1 = foreground
    pub mask: Vec<u8>,
    /// `288 * 288` mask logits before upsampling, `-1024` everywhere for an absent object
    pub low_res: Vec<f32>,
    /// The decoder's predicted IoU of each candidate mask, the single-mask output first
    pub ious: Vec<f32>,
    /// Index into `ious` of the candidate kept
    pub chosen: usize,
}

/// Every object's mask on one frame, in the order the objects were given
pub struct TrackedFrame {
    /// Frame index from 0, the frame the objects were clicked on
    pub frame: usize,
    /// One entry per object
    pub objects: Vec<TrackedObject>,
    /// Memory tokens the frame attended to, object pointer tokens included: 0 on the
    /// clicked frame, then growing to a constant once the bank is full
    pub memory_tokens: usize,
    /// Of [`Self::memory_tokens`], those from object pointers (4 per pointer)
    pub pointer_tokens: usize,
}
