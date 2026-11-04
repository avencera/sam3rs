//! Perception Encoder ViT-L backbone and the two SimpleFPN necks (SAM 3 detector and
//! SAM 2 tracker) on Metal
//!
//! Both precisions run one block structure in window-major token order, where each
//! 24x24 window's tokens are contiguous rows, so the fused attention reads a window as
//! one run of rows. The residual stream is FP32; in BF16 mode the norms, GEMMs and
//! attention read and write BF16, and each branch output is rounded to BF16 before the
//! residual add, as the reference's autocast does

use std::f32::consts::LOG2_E;

use eyre::Result;

use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::ops::{Act, Conv3x3, ConvT2x2, LayerNorm, Linear, add_assign, convert, weight_dtype};
use crate::weights::Weights;

pub const IMAGE_SIZE: usize = 1008;
pub const PATCH: usize = 14;
pub const GRID: usize = IMAGE_SIZE / PATCH;
const TOKENS: usize = GRID * GRID;
const DIM: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = DIM / HEADS;
const WINDOW: usize = 24;
const GLOBAL_BLOCKS: [usize; 4] = [7, 15, 23, 31];
/// Query rows of each attention SIMD group, and SIMD groups a threadgroup (the
/// `vit_attn_*` instantiation's BQ and SGS)
const ATTN_ROWS: usize = 16;
const ATTN_SIMDGROUPS: usize = 4;

/// The raster index of row `r` of the window-major token order; the host twin of the
/// kernels' `window_to_raster`
fn window_to_raster(r: usize) -> usize {
    let nwx = GRID / WINDOW;
    let (w, l) = (r / (WINDOW * WINDOW), r % (WINDOW * WINDOW));
    ((w / nwx) * WINDOW + l / WINDOW) * GRID + (w % nwx) * WINDOW + l % WINDOW
}

/// Axial RoPE tables `[side * side, HEAD_DIM / 2]` for a `side` x `side` token grid, rows
/// in raster order, or in window-major order with `window_major`
///
/// Matches `compute_axial_cis`: the first 16 complex pairs rotate by x, the last 16 by y,
/// and positions are scaled by `scale` (1/3 for global blocks, interpolated from the
/// 24x24 pretraining grid). Angles are formed in f32 like PyTorch before cos/sin
fn rope_tables(gpu: &Gpu, side: usize, scale: f32, window_major: bool) -> Result<(Tensor, Tensor)> {
    let quarter = HEAD_DIM / 4;
    let freqs: Vec<f32> = (0..quarter).map(|i| 1.0 / 10000f32.powf((4 * i) as f32 / HEAD_DIM as f32)).collect();
    let half = HEAD_DIM / 2;
    let mut cos = vec![0f32; side * side * half];
    let mut sin = vec![0f32; side * side * half];
    for row in 0..side * side {
        let tok = if window_major { window_to_raster(row) } else { row };
        let tx = (tok % side) as f32 * scale;
        let ty = (tok / side) as f32 * scale;
        for p in 0..half {
            let angle = if p < quarter { tx * freqs[p] } else { ty * freqs[p - quarter] };
            cos[row * half + p] = (angle as f64).cos() as f32;
            sin[row * half + p] = (angle as f64).sin() as f32;
        }
    }
    let shape = [side * side, half];
    Ok((gpu.upload_f32(&cos, &shape)?, gpu.upload_f32(&sin, &shape)?))
}

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    global: bool,
}

pub struct Backbone {
    patch: Linear,
    pos: Tensor,
    ln_pre: LayerNorm,
    blocks: Vec<Block>,
    /// Activation type of the blocks: BF16 in BF16 mode, FP32 otherwise
    dtype: DType,
    rope_window: (Tensor, Tensor),
    /// The global blocks' table in window-major token order
    rope_global: (Tensor, Tensor),
}

impl Backbone {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "detector_model.vision_encoder.backbone";
        let dtype = weight_dtype(gpu);
        let patch_w = wts.get(gpu, &format!("{p}.embeddings.patch_embeddings.projection.weight"))?;
        // the patch embedding stays FP32: its K of 588 is 0.6% of the FLOPs
        let patch = Linear::new(patch_w.reshape(&[DIM, 3 * PATCH * PATCH]), None);
        let pos = wts.get(gpu, &format!("{p}.embeddings.position_embeddings"))?;
        let ln_pre = LayerNorm::load(gpu, wts, &format!("{p}.layer_norm"), 1e-5)?;
        let mut blocks = Vec::new();
        for i in 0..32 {
            let l = format!("{p}.layers.{i}");
            let a = format!("{l}.attention");
            let qkv = [format!("{a}.q_proj"), format!("{a}.k_proj"), format!("{a}.v_proj")];
            blocks.push(Block {
                norm1: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm1"), 1e-5)?,
                qkv: Linear::load_concat(gpu, wts, &qkv, dtype)?,
                proj: Linear::load(gpu, wts, &format!("{a}.o_proj"), dtype)?,
                norm2: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm2"), 1e-5)?,
                fc1: Linear::load(gpu, wts, &format!("{l}.mlp.fc1"), dtype)?,
                fc2: Linear::load(gpu, wts, &format!("{l}.mlp.fc2"), dtype)?,
                global: GLOBAL_BLOCKS.contains(&i),
            });
        }
        Ok(Self {
            patch,
            pos,
            ln_pre,
            blocks,
            dtype,
            rope_window: rope_tables(gpu, WINDOW, 1.0, false)?,
            rope_global: rope_tables(gpu, GRID, WINDOW as f32 / GRID as f32, true)?,
        })
    }

    /// Pixels `[3, 1008, 1008]` -> FP32 tokens `[72 * 72, 1024]` in raster order
    pub fn forward(&self, gpu: &Gpu, pixels: &Tensor) -> Result<Tensor> {
        let k = 3 * PATCH * PATCH;
        let patches = gpu.empty(&[TOKENS, k], DType::F32)?;
        gpu.dispatch_1d("patchify", patches.len(), &[patches.arg(), pixels.arg(), IMAGE_SIZE.into(), PATCH.into()])?;
        let x = self.patch.forward(gpu, &patches, Act::None)?;
        drop(patches);
        let args = [x.arg(), self.pos.arg(), GRID.into(), WINDOW.into(), DIM.into()];
        gpu.dispatch_1d("add_pos_tiled", x.len(), &args)?;
        self.ln_pre.forward_inplace(gpu, &x)?;

        let x = window_order(gpu, &x, true)?;
        let mut pending = None;
        for block in &self.blocks {
            pending = Some(self.block(gpu, block, &x, pending)?);
        }
        if let Some(fc2) = pending {
            add_assign(gpu, &x, &fc2)?;
        }
        window_order(gpu, &x, false)
    }

    /// One block over window-major FP32 `x` in place
    ///
    /// Each residual add is folded into the LayerNorm that reads the sum, so the block
    /// returns its fc2 output for the next block's `norm1` (or the caller) to add;
    /// `pending` is the previous one. The attention reads RoPE-rotated copies of q and k
    /// and v straight from the qkv GEMM's output
    fn block(&self, gpu: &Gpu, b: &Block, x: &Tensor, pending: Option<Tensor>) -> Result<Tensor> {
        let h = match &pending {
            Some(fc2) => b.norm1.add_forward(gpu, x, fc2)?,
            None => b.norm1.forward_to(gpu, x, self.dtype)?,
        };
        drop(pending);
        let qkv = b.qkv.forward(gpu, &h, Act::None)?;
        drop(h);

        let (cos, sin) = if b.global { &self.rope_global } else { &self.rope_window };
        let windowed = u32::from(!b.global);
        let merged = gpu.empty(&[TOKENS, DIM], self.dtype)?;
        let scale = LOG2_E / (HEAD_DIM as f32).sqrt();
        let (rope, attn) = match self.dtype {
            DType::Bf16 => ("vit_rope_qk_bf16", "vit_attn_bf16"),
            _ => ("vit_rope_qk_f32", "vit_attn_f32"),
        };
        let q = gpu.empty(&[TOKENS, DIM], self.dtype)?;
        let k = gpu.empty(&[TOKENS, DIM], self.dtype)?;
        let args = [q.arg(), k.arg(), qkv.arg(), cos.arg(), sin.arg(), DIM.into(), HEAD_DIM.into(), windowed.into()];
        gpu.dispatch_1d(rope, TOKENS * DIM / 2, &args)?;
        let args = [merged.arg(), q.arg(), k.arg(), qkv.arg(), windowed.into(), TOKENS.into(), scale.into()];
        let rows = ATTN_ROWS * ATTN_SIMDGROUPS;
        gpu.dispatch(attn, [TOKENS / rows, HEADS, 1], [32 * ATTN_SIMDGROUPS, 1, 1], &args)?;
        drop((qkv, q, k));

        let attn = b.proj.forward(gpu, &merged, Act::None)?;
        drop(merged);
        let h = b.norm2.add_forward(gpu, x, &attn)?;
        drop(attn);
        let hidden = b.fc1.forward(gpu, &h, Act::GeluTanh)?;
        drop(h);
        b.fc2.forward(gpu, &hidden, Act::None)
    }
}

/// `x [GRID * GRID, cols]` reordered from raster to window-major rows with `to_window`,
/// and back otherwise
fn window_order(gpu: &Gpu, x: &Tensor, to_window: bool) -> Result<Tensor> {
    let out = gpu.empty(x.shape(), DType::F32)?;
    let args = [out.arg(), x.arg(), x.cols().into(), u32::from(to_window).into()];
    gpu.dispatch_1d("vit_window_order", x.len(), &args)?;
    Ok(out)
}

/// The three SimpleFPN levels, channels-last FP32 with 256 channels
pub struct Fpn {
    /// 4x level `[288^2, 256]`
    pub fine: Tensor,
    /// 2x level `[144^2, 256]`
    pub mid: Tensor,
    /// 1x level `[72^2, 256]`, at the backbone's resolution
    pub coarse: Tensor,
}

/// How a level resizes the trunk before its projections
enum Scale {
    /// Two 2x transposed convs with a GELU between
    Up4(ConvT2x2, ConvT2x2),
    Up2(ConvT2x2),
    Same,
}

struct Level {
    scale: Scale,
    proj1: Linear,
    proj2: Conv3x3,
}

impl Level {
    fn load(gpu: &Gpu, wts: &Weights, prefix: &str, scale: Scale) -> Result<Self> {
        let dtype = weight_dtype(gpu);
        Ok(Self {
            scale,
            proj1: Linear::load(gpu, wts, &format!("{prefix}.proj1"), dtype)?,
            proj2: Conv3x3::load(gpu, wts, &format!("{prefix}.proj2"), dtype)?,
        })
    }

    /// The level from the trunk in the necks' activation type; every layer up to `proj1`
    /// reads and writes that type, and `proj1` writes FP32 for the 3x3 conv
    fn forward(&self, gpu: &Gpu, trunk: &Tensor) -> Result<Tensor> {
        let (h, side) = match &self.scale {
            Scale::Up4(a, b) => {
                let h = a.forward(gpu, trunk, GRID, GRID, Act::Gelu)?;
                let y = b.forward(gpu, &h, 2 * GRID, 2 * GRID, Act::None)?;
                drop(h);
                (self.proj1.forward_to(gpu, &y, Act::None, DType::F32)?, 4 * GRID)
            }
            Scale::Up2(a) => {
                let y = a.forward(gpu, trunk, GRID, GRID, Act::None)?;
                (self.proj1.forward_to(gpu, &y, Act::None, DType::F32)?, 2 * GRID)
            }
            Scale::Same => (self.proj1.forward_to(gpu, trunk, Act::None, DType::F32)?, GRID),
        };
        self.proj2.forward(gpu, &h, side, side)
    }
}

/// One SimpleFPN neck: 4x, 2x and 1x levels (the 0.5x level is dropped upstream)
pub struct Neck {
    fine: Level,
    mid: Level,
    coarse: Level,
}

impl Neck {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let dtype = weight_dtype(gpu);
        let level = |i: usize| format!("{prefix}.fpn_layers.{i}");
        let up = |i: usize, j: usize| ConvT2x2::load(gpu, wts, &format!("{}.scale_layers.{j}", level(i)), dtype);
        Ok(Self {
            fine: Level::load(gpu, wts, &level(0), Scale::Up4(up(0, 0)?, up(0, 2)?))?,
            mid: Level::load(gpu, wts, &level(1), Scale::Up2(up(1, 0)?))?,
            coarse: Level::load(gpu, wts, &level(2), Scale::Same)?,
        })
    }

    /// Trunk -> the three levels
    pub fn forward(&self, gpu: &Gpu, trunk: &Trunk) -> Result<Fpn> {
        // struct fields evaluate in the order written, so the levels run fine to coarse
        Ok(Fpn {
            fine: self.fine.forward(gpu, &trunk.input)?,
            mid: self.mid.forward(gpu, &trunk.input)?,
            coarse: self.coarse.forward(gpu, &trunk.input)?,
        })
    }
}

/// The backbone's output tokens `[72 * 72, 1024]` as the necks read them: converted once
/// to the necks' activation type for both necks' six levels
pub struct Trunk {
    input: Tensor,
}

impl Trunk {
    pub fn new(gpu: &Gpu, f32: Tensor) -> Result<Self> {
        Ok(Self { input: convert(gpu, f32, weight_dtype(gpu))? })
    }
}
