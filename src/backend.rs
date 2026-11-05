//! The internal interface between the public API and a GPU backend
//!
//! [`crate::Sam3`], [`crate::Embedding`] and [`crate::VideoTracker`] are written once
//! against [`Backend`]. A backend owns its GPU, its weights and every stage of the model
//! graph, and answers the coarse steps here: encode an image, run the detector, decode
//! point prompts, step the video tracker and download masks into host buffers. The
//! public layer owns everything else: input checks, the tokenizer and the cached visual
//! prompt, score thresholds and box conversion, the batching of point prompts, host mask
//! memory, and the tracker's memory bank and memory selection.
//!
//! [`Active`] picks the backend for the target. A build that chooses its backend at run
//! time would make `Active` an enum over the backends (`Cuda(..)`, `Vulkan(..)`) that
//! implements [`Backend`] by matching, with enums for the associated types, and the
//! public types would not change

use eyre::Result;

use crate::image::RgbImage;
use crate::masks::MaskSink;
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::tokenizer::CONTEXT;
use crate::tracker::Selection;
use crate::weights::Weights;

/// The backend this build runs on
#[cfg(not(target_os = "macos"))]
pub(crate) type Active = crate::cuda::Cuda;
/// The backend this build runs on
#[cfg(target_os = "macos")]
pub(crate) type Active = crate::metal::Metal;

/// The detector's scores and boxes for every query, downloaded
pub(crate) struct DetectionScores {
    /// The presence logit, before the sigmoid
    pub(crate) presence: f32,
    /// `[200]` query logits, before the sigmoid
    pub(crate) logits: Vec<f32>,
    /// `[200, 4]` normalized cxcywh boxes
    pub(crate) boxes: Vec<f32>,
}

/// One decoder pass over one object of a video frame, with its scores downloaded
pub(crate) struct ObjectPass<D> {
    pub(crate) decoded: D,
    /// Object score logit; the object counts as present above 0
    pub(crate) score: f32,
    /// Predicted IoU of all four mask tokens
    pub(crate) ious: Vec<f32>,
}

/// One GPU backend: the loaded model on one device, and the steps the public API runs
///
/// Sizes are `(width, height)` of the input image or video in pixels. Every method runs
/// on the model's one stream or queue, so calls are ordered as they are made
pub(crate) trait Backend: Sized {
    /// The opened GPU, before any weights load
    type Device;
    /// Features of one encoded image
    type Image;
    /// Text features of one prompt
    type Text;
    /// Raw detector outputs of one prompt, on the device
    type Detected;
    /// Both point decoder passes over a batch of prompts, on the device
    type PointPass;
    /// Preprocessed pixels `[3, 1008, 1008]` of one video frame
    type Pixels;
    /// Features of one video frame
    type Frame;
    /// Low-resolution mask logits `[objects, 288^2]` of one tracked frame
    type Logits;
    /// One decoder pass over one object of a video frame, on the device
    type Decoded;
    /// The memory-conditioned image embedding of one object on one frame
    type Embed;
    /// One object's pointer on one frame
    type Pointer;
    /// One object's memory on one frame
    type Memory;
    /// What the memory bank keeps of one frame for every object: memories, pointers and
    /// whatever the backend derives from them
    type FrameMemory;
    /// The memory attention's state for one frame, which every object's attention reads
    type Attention;

    /// Opens GPU `device` in `precision`, failing when this machine can't run the model
    fn open(device: usize, precision: Precision) -> Result<Self::Device>;

    /// Loads every stage's weights onto `device` and waits for the uploads
    fn load(device: Self::Device, wts: &Weights) -> Result<Self>;

    fn precision(&self) -> Precision;

    /// Makes the model's GPU current on the calling thread
    fn bind(&self) -> Result<()>;

    /// The vision backbone and both necks, waited for
    fn encode(&self, img: &RgbImage) -> Result<Self::Image>;

    fn encode_text(&self, tokens: &[u32; CONTEXT]) -> Result<Self::Text>;

    /// The detector over one prompt: text features plus any boxes
    fn detect(&self, img: &Self::Image, text: &Self::Text, boxes: &[BoxPrompt]) -> Result<Self::Detected>;

    fn detection_scores(&self, det: &Self::Detected) -> Result<DetectionScores>;

    /// The full-size masks of queries `kept`, in order, into `sink`
    fn detection_masks(
        &self,
        det: &Self::Detected,
        kept: &[usize],
        size: (usize, usize),
        sink: &mut MaskSink,
    ) -> Result<()>;

    /// Both decoder passes over prompts with equal click counts: a first pass without a
    /// mask, then a second with the first pass's best mask as the mask prompt
    fn decode_points(
        &self,
        img: &Self::Image,
        prompts: &[&[PointPrompt]],
        size: (usize, usize),
    ) -> Result<Self::PointPass>;

    /// The best second-pass mask of each prompt after hole filling, at full size into
    /// `sink`, and its predicted IoU
    fn point_masks(&self, pass: Self::PointPass, size: (usize, usize), sink: &mut MaskSink) -> Result<Vec<f32>>;

    fn preprocess(&self, frame: &RgbImage) -> Result<Self::Pixels>;

    /// The backbone and tracker neck of one video frame
    fn video_frame(&self, pixels: &Self::Pixels) -> Result<Self::Frame>;

    /// Room for the chosen low-res masks of `objects` objects
    fn frame_logits(&self, objects: usize) -> Result<Self::Logits>;

    /// The decoder over `frame` for one object, with masks for mask tokens `first..4`
    /// only: from `points` and the frame's own embedding on the clicked frame, or from no
    /// points and the memory-conditioned `embed` on a tracked one
    fn decode_object(
        &self,
        frame: &Self::Frame,
        embed: Option<&Self::Embed>,
        points: &[PointPrompt],
        size: (usize, usize),
        first: usize,
    ) -> Result<ObjectPass<Self::Decoded>>;

    /// The pass's first low-res mask `[288^2]`, downloaded
    fn first_mask(&self, dec: &Self::Decoded) -> Result<Vec<f32>>;

    /// Writes mask token `index`'s mask of a pass into row `b` of `logits`, or
    /// `NO_OBJ_SCORE` everywhere for an absent object
    fn keep_mask(
        &self,
        dec: &Self::Decoded,
        index: usize,
        present: bool,
        logits: &Self::Logits,
        b: usize,
    ) -> Result<()>;

    /// The object pointer from mask token `token`, or the no-object pointer for an absent
    /// object
    fn pointer(&self, dec: &Self::Decoded, token: usize, present: bool) -> Result<Self::Pointer>;

    /// Every object's memory on the clicked frame, from its kept mask in `logits` with
    /// overlaps between objects resolved, as upstream's preflight consolidation does
    fn prompt_memories(
        &self,
        frame: &Self::Frame,
        logits: &Self::Logits,
        present: &[bool],
    ) -> Result<Vec<Self::Memory>>;

    /// Object `b`'s memory on a tracked frame, from its kept mask in `logits`
    fn object_memory(
        &self,
        frame: &Self::Frame,
        logits: &Self::Logits,
        b: usize,
        present: bool,
    ) -> Result<Self::Memory>;

    /// What the bank keeps of a frame, one memory and pointer per object
    fn frame_memory(&self, memories: Vec<Self::Memory>, pointers: Vec<Self::Pointer>) -> Result<Self::FrameMemory>;

    /// Called on a bank frame that memory selection can no longer pick as a memory, though
    /// its pointers can still be picked, so the backend can free what only the memory
    /// attention would read
    fn retire(frame: &mut Self::FrameMemory) {
        let _ = frame;
    }

    /// The memory attention's state for `frame` over the bank's selection `sel`
    fn attention(&self, frame: &Self::Frame, sel: &Selection<'_, Self::FrameMemory>) -> Result<Self::Attention>;

    /// Object `b`'s features on `frame` conditioned on its memories and pointers in `sel`
    fn attend(
        &self,
        frame: &Self::Frame,
        attention: &mut Self::Attention,
        sel: &Selection<'_, Self::FrameMemory>,
        b: usize,
    ) -> Result<Self::Embed>;

    /// The thresholded full-size mask of row `b` of `logits` into `out[b]`, `width *
    /// height` bytes each, for every row in `out`
    fn frame_masks(&self, logits: &Self::Logits, size: (usize, usize), out: &mut [Vec<u8>]) -> Result<()>;

    fn download_logits(&self, logits: &Self::Logits) -> Result<Vec<f32>>;
}
