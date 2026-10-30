//! Perception Encoder ViT-L backbone and the two SimpleFPN necks (SAM 3 detector and
//! SAM 2 tracker)

use eyre::Result;

use crate::gpu::{Gpu, Mat, Precision, Tensor};
use crate::launch;
use crate::ops::{ACT_GELU, ACT_GELU_TANH, B256, Bf16, Conv3x3, ConvT2x2, LayerNorm, Linear, Mask, attention, bias_act, ew};
use crate::weights::Weights;

pub const IMAGE_SIZE: usize = 1008;
pub const PATCH: usize = 14;
pub const GRID: usize = IMAGE_SIZE / PATCH;
const DIM: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = DIM / HEADS;
const WINDOW: usize = 24;
const GLOBAL_BLOCKS: [usize; 4] = [7, 15, 23, 31];

/// Axial RoPE tables `[side * side, HEAD_DIM / 2]` for a `side` x `side` token grid
///
/// Matches `compute_axial_cis`: the first 16 complex pairs rotate by x, the last 16 by y,
/// and positions are scaled by `scale` (1/3 for global blocks, interpolated from the
/// 24x24 pretraining grid). Angles are formed in f32 like PyTorch before cos/sin
fn rope_tables(gpu: &Gpu, side: usize, scale: f32) -> Result<(Tensor, Tensor)> {
    let quarter = HEAD_DIM / 4;
    let freqs: Vec<f32> =
        (0..quarter).map(|i| 1.0 / 10000f32.powf((4 * i) as f32 / HEAD_DIM as f32)).collect();
    let half = HEAD_DIM / 2;
    let mut cos = vec![0f32; side * side * half];
    let mut sin = vec![0f32; side * side * half];
    for tok in 0..side * side {
        let tx = (tok % side) as f32 * scale;
        let ty = (tok / side) as f32 * scale;
        for p in 0..half {
            let angle = if p < quarter { tx * freqs[p] } else { ty * freqs[p - quarter] };
            cos[tok * half + p] = (angle as f64).cos() as f32;
            sin[tok * half + p] = (angle as f64).sin() as f32;
        }
    }
    let shape = [side * side, half];
    Ok((gpu.upload(&cos, &shape)?, gpu.upload(&sin, &shape)?))
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
    rope_window: (Tensor, Tensor),
    rope_global: (Tensor, Tensor),
}

impl Backbone {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "detector_model.vision_encoder.backbone";
        let patch_w = wts.get(gpu, &format!("{p}.embeddings.patch_embeddings.projection.weight"))?;
        let patch = Linear::new(patch_w.reshape(&[DIM, 3 * PATCH * PATCH]), None);
        let pos = wts.get(gpu, &format!("{p}.embeddings.position_embeddings"))?;
        let ln_pre = LayerNorm::load(gpu, wts, &format!("{p}.layer_norm"), 1e-5)?;
        let mut blocks = Vec::new();
        for i in 0..32 {
            let l = format!("{p}.layers.{i}");
            let a = format!("{l}.attention");
            blocks.push(Block {
                norm1: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm1"), 1e-5)?,
                qkv: Linear::load_concat(gpu, wts, &[format!("{a}.q_proj"), format!("{a}.k_proj"), format!("{a}.v_proj")])?.bf16(gpu)?,
                proj: Linear::load(gpu, wts, &format!("{a}.o_proj"))?.bf16(gpu)?,
                norm2: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm2"), 1e-5)?,
                fc1: Linear::load(gpu, wts, &format!("{l}.mlp.fc1"))?.bf16(gpu)?,
                fc2: Linear::load(gpu, wts, &format!("{l}.mlp.fc2"))?.bf16(gpu)?,
                global: GLOBAL_BLOCKS.contains(&i),
            });
        }
        Ok(Self {
            patch,
            pos,
            ln_pre,
            blocks,
            rope_window: rope_tables(gpu, WINDOW, 1.0)?,
            rope_global: rope_tables(gpu, GRID, WINDOW as f32 / GRID as f32)?,
        })
    }

    /// pixels `[3, 1008, 1008]` -> tokens `[72 * 72, 1024]`
    pub fn forward(&self, gpu: &Gpu, pixels: &Tensor) -> Result<Tensor> {
        let k = 3 * PATCH * PATCH;
        let patches = gpu.empty(&[GRID * GRID, k])?;
        launch!(gpu, "patchify", ew(patches.len()), B256, 0; &patches, pixels, IMAGE_SIZE, PATCH)?;
        let x = self.patch.forward(gpu, &patches)?;
        drop(patches);
        launch!(gpu, "add_pos_tiled", ew(x.len()), B256, 0; &x, &self.pos, GRID, 24usize, DIM)?;
        self.ln_pre.forward_inplace(gpu, &x)?;
        for block in &self.blocks {
            self.block(gpu, block, &x)?;
        }
        Ok(x)
    }

    fn block(&self, gpu: &Gpu, b: &Block, x: &Tensor) -> Result<()> {
        let tokens = GRID * GRID;
        // BF16 mode keeps the GEMM inputs in BF16 from the kernel that produces them
        let bf16 = gpu.precision() == Precision::Bf16;
        let qkv = if bf16 {
            b.qkv.forward16(gpu, &b.norm1.forward16(gpu, x)?)?
        } else {
            b.qkv.forward(gpu, &b.norm1.forward(gpu, x)?)?
        };

        let (win, rope) = if b.global { (GRID, &self.rope_global) } else { (WINDOW, &self.rope_window) };
        let batch = (GRID / win) * (GRID / win) * HEADS;
        let l = win * win;
        let q = gpu.empty(&[batch, l, HEAD_DIM])?;
        let k = gpu.empty(&[batch, l, HEAD_DIM])?;
        let v = gpu.empty(&[batch, l, HEAD_DIM])?;
        launch!(gpu, "vit_split_rope", ew(tokens * DIM / 2), B256, 0;
            &q, &k, &v, &qkv, &rope.0, &rope.1, GRID, win, HEADS, HEAD_DIM)?;
        drop(qkv);

        let o = gpu.empty(&[batch, l, HEAD_DIM])?;
        let view = |t: &Tensor| Mat::new(t.ptr(), HEAD_DIM).stride(l * HEAD_DIM);
        attention(gpu, view(&q), view(&k), view(&v), view(&o), batch, l, l, HEAD_DIM, Mask::default())?;
        drop((q, k, v));

        if bf16 {
            let merged = Bf16::empty(gpu, tokens, DIM)?;
            launch!(gpu, "vit_merge_heads_bf16", ew(tokens * DIM), B256, 0;
                merged.buf.ptr(), &o, GRID, win, HEADS, HEAD_DIM)?;
            drop(o);
            b.proj.forward16_residual(gpu, &merged, x)?;
            drop(merged);
            let hidden = b.fc1.forward16_act16(gpu, &b.norm2.forward16(gpu, x)?, ACT_GELU_TANH)?;
            return b.fc2.forward16_residual(gpu, &hidden, x);
        }

        let merged = gpu.empty(&[tokens, DIM])?;
        launch!(gpu, "vit_merge_heads", ew(merged.len()), B256, 0; &merged, &o, GRID, win, HEADS, HEAD_DIM)?;
        drop(o);
        b.proj.forward_residual(gpu, &merged, x)?;
        drop(merged);

        let h = b.norm2.forward(gpu, x)?;
        // upstream runs fc1 through `addmm_act`, whose CUDA GELU is the tanh approximation
        let hidden = b.fc1.forward_act(gpu, &h, ACT_GELU_TANH)?;
        drop(h);
        b.fc2.forward_residual(gpu, &hidden, x)
    }
}

enum Scale {
    Up4(ConvT2x2, ConvT2x2),
    Up2(ConvT2x2),
    Same,
}

struct Level {
    scale: Scale,
    proj1: Linear,
    proj2: Conv3x3,
}

/// One SimpleFPN neck: 4x, 2x and 1x levels (the 0.5x level is dropped upstream)
pub struct Neck {
    levels: Vec<Level>,
}

impl Neck {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let mut levels = Vec::new();
        for i in 0..3 {
            let l = format!("{prefix}.fpn_layers.{i}");
            let s = |j: usize| ConvT2x2::load(gpu, wts, &format!("{l}.scale_layers.{j}"));
            let scale = match i {
                0 => Scale::Up4(s(0)?, s(2)?),
                1 => Scale::Up2(s(0)?),
                _ => Scale::Same,
            };
            levels.push(Level {
                scale,
                proj1: Linear::load(gpu, wts, &format!("{l}.proj1"))?,
                proj2: Conv3x3::load(gpu, wts, &format!("{l}.proj2"))?,
            });
        }
        Ok(Self { levels })
    }

    /// trunk `[72 * 72, 1024]` -> `[288^2, 256]`, `[144^2, 256]`, `[72^2, 256]`
    pub fn forward(&self, gpu: &Gpu, trunk: &Tensor) -> Result<Vec<Tensor>> {
        let mut out = Vec::new();
        for level in &self.levels {
            let (h, side) = match &level.scale {
                Scale::Up4(a, b) => {
                    let h = a.forward(gpu, trunk, GRID, GRID)?;
                    bias_act(gpu, &h, None, ACT_GELU)?;
                    (level.proj1.forward(gpu, &b.forward(gpu, &h, 2 * GRID, 2 * GRID)?)?, 4 * GRID)
                }
                Scale::Up2(a) => (level.proj1.forward(gpu, &a.forward(gpu, trunk, GRID, GRID)?)?, 2 * GRID),
                Scale::Same => (level.proj1.forward(gpu, trunk)?, GRID),
            };
            out.push(level.proj2.forward(gpu, &h, side, side)?);
        }
        Ok(out)
    }
}
