//! SAM 3's architecture constants, the same for every backend, one module per stage

/// The Perception Encoder ViT-L backbone and its SimpleFPN necks
pub(crate) mod vision {
    /// Side of the square frame every image is resized to
    pub(crate) const IMAGE_SIZE: usize = 1008;
    /// Side of one patch
    pub(crate) const PATCH: usize = 14;
    /// Side of the patch grid, 72
    pub(crate) const GRID: usize = IMAGE_SIZE / PATCH;
    /// Tokens of the patch grid
    pub(crate) const TOKENS: usize = GRID * GRID;
    /// Width of the backbone's blocks
    pub(crate) const DIM: usize = 1024;
    pub(crate) const HEADS: usize = 16;
    pub(crate) const HEAD_DIM: usize = DIM / HEADS;
    /// Side of the windowed blocks' attention windows
    pub(crate) const WINDOW: usize = 24;
    /// The blocks that attend over the whole grid rather than within windows
    pub(crate) const GLOBAL_BLOCKS: [usize; 4] = [7, 15, 23, 31];

    /// The raster index of row `r` of the window-major token order, where each window's
    /// tokens are contiguous rows in raster order inside it; the host twin of the
    /// kernels' `window_to_raster`
    pub(crate) fn window_to_raster(r: usize) -> usize {
        let nwx = GRID / WINDOW;
        let (w, l) = (r / (WINDOW * WINDOW), r % (WINDOW * WINDOW));
        ((w / nwx) * WINDOW + l / WINDOW) * GRID + (w % nwx) * WINDOW + l % WINDOW
    }
}

/// The CLIP-style text encoder
pub(crate) mod text {
    pub(crate) const WIDTH: usize = 1024;
    pub(crate) const HEADS: usize = 16;
}

/// The detector: geometry and fusion encoders, DETR decoder and segmentation head
pub(crate) mod detector {
    pub(crate) const D: usize = 256;
    pub(crate) const HEADS: usize = 8;
    /// Object queries of the DETR decoder, one score, box and mask each
    pub(crate) const QUERIES: usize = 200;
    /// Side of the ROI-aligned box features
    pub(crate) const ROI: usize = 7;
}

/// The SAM 2 style point decoder
pub(crate) mod point {
    use super::vision::GRID;

    pub(crate) const D: usize = 256;
    pub(crate) const HEADS: usize = 8;
    /// Side of the low-resolution mask logits
    pub(crate) const LOW_RES: usize = 4 * GRID;
    /// Row of mask token 0 in each prompt's decoder output tokens, after the object score
    /// and IoU tokens
    pub(crate) const MASK_TOKEN_ROW: usize = 2;
}

/// The video tracker, as upstream's `Sam3TrackerPredictor` configures it
pub(crate) mod tracker {

    pub(crate) const D: usize = 256;
    /// Channels of a memory frame
    pub(crate) const MEM_D: usize = 64;
    /// Tokens of one frame's 72x72 feature map
    pub(crate) const TOKENS: usize = super::vision::TOKENS;
    /// `num_maskmem`: memory frames per step, the conditioning frame's position plus 6
    pub(crate) const NUM_MASKMEM: usize = 7;
    /// `max_cond_frames_in_attn`
    pub(crate) const MAX_COND_FRAMES: usize = 4;
    /// `max_obj_ptrs_in_encoder`
    pub(crate) const MAX_OBJ_PTRS: usize = 16;
    /// Clicks per object the prompt encoder sees: the first 8 and the last 8 of more
    /// (`max_point_num_in_prompt_enc`, to stay near the 8-point training prompts)
    pub(crate) const MAX_CLICKS: usize = 16;
    /// `mf_threshold`: earlier frames whose score is at most this are skipped as memories
    pub(crate) const MEMORY_SCORE_THRESHOLD: f32 = 0.01;
    /// `NO_OBJ_SCORE`, the logit upstream gives every pixel of an absent object's mask
    pub(crate) const NO_OBJ_SCORE: f32 = -1024.0;
    /// Side of the memory encoder's mask input, upsampled from the 1008 frame
    pub(crate) const MASK_SIDE: usize = 1152;
    /// Base of the memory attention's axial RoPE frequencies
    pub(crate) const ROPE_THETA: f32 = 10000.0;
}
