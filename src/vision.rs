//! Perception Encoder ViT-L backbone and the two SimpleFPN necks (SAM 3 detector and
//! SAM 2 tracker)

use eyre::Result;

use crate::gpu::{Arg, Gpu, Mat, Precision, Tensor, launch};
use crate::ops::{
    ACT_GELU, ACT_GELU_TANH, ACT_NONE, B256, Bf16, Conv3x3, ConvT2x2, LayerNorm, Linear, Mask, add_assign_bf16,
    attention, bias_act, ew,
};
use crate::weights::Weights;

pub const IMAGE_SIZE: usize = 1008;
pub const PATCH: usize = 14;
pub const GRID: usize = IMAGE_SIZE / PATCH;
const DIM: usize = 1024;
const HEADS: usize = 16;
const HEAD_DIM: usize = DIM / HEADS;
const WINDOW: usize = 24;
const GLOBAL_BLOCKS: [usize; 4] = [7, 15, 23, 31];

/// The raster index of row `r` of the BF16 blocks' window-major token order, where each
/// window's tokens are contiguous rows in raster order inside it; the host twin of the
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
    Ok((gpu.upload(cos, &shape)?, gpu.upload(sin, &shape)?))
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
    /// `rope_global` in the BF16 blocks' window-major token order
    rope_global_wm: (Tensor, Tensor),
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
                qkv: Linear::load_concat(
                    gpu,
                    wts,
                    &[format!("{a}.q_proj"), format!("{a}.k_proj"), format!("{a}.v_proj")],
                )?
                .bf16(gpu)?,
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
            rope_window: rope_tables(gpu, WINDOW, 1.0, false)?,
            rope_global: rope_tables(gpu, GRID, WINDOW as f32 / GRID as f32, false)?,
            rope_global_wm: rope_tables(gpu, GRID, WINDOW as f32 / GRID as f32, true)?,
        })
    }

    /// pixels `[3, 1008, 1008]` -> tokens `[72 * 72, 1024]`
    pub fn forward(&self, gpu: &Gpu, pixels: &Tensor) -> Result<Tensor> {
        let k = 3 * PATCH * PATCH;
        let patches = gpu.empty(&[GRID * GRID, k])?;
        launch!(gpu, "patchify", ew(patches.len()), B256, 0; &patches, pixels, IMAGE_SIZE, PATCH)?;
        let x = self.patch.forward(gpu, &patches)?;
        drop(patches);
        launch!(gpu, "add_pos_tiled", ew(x.len()), B256, 0; &x, &self.pos, GRID, 24usize)?;
        self.ln_pre.forward_inplace(gpu, &x)?;
        if gpu.precision() != Precision::Bf16 {
            for block in &self.blocks {
                self.block(gpu, block, &x)?;
            }
            return Ok(x);
        }

        // the BF16 blocks run in window-major token order, so each window is a run of
        // contiguous rows for the fused attention; LayerNorm and the MLP are per token
        let x = window_order(gpu, &x, true)?;
        let mut pending = None;
        for block in &self.blocks {
            pending = Some(self.block16(gpu, block, &x, pending)?);
        }
        if let Some(fc2) = pending {
            add_assign_bf16(gpu, &x, &fc2)?;
        }
        window_order(gpu, &x, false)
    }

    /// One BF16 block over window-major `x` in place
    ///
    /// The residual GEMMs write BF16 and each sum is folded into the LayerNorm that reads
    /// it, so the block returns its fc2 output for the next block's `norm1` (or the
    /// caller) to add; `pending` is the previous one. The branch outputs match the
    /// reference's autocast, which rounds each linear's output to BF16 before the residual
    /// add. The attention reads q, k and v from the qkv GEMM's output and writes the merged
    /// heads, with RoPE applied as it stages q and k
    fn block16(&self, gpu: &Gpu, b: &Block, x: &Tensor, pending: Option<Bf16>) -> Result<Bf16> {
        let tokens = GRID * GRID;
        let h = match &pending {
            Some(fc2) => b.norm1.add_forward16(gpu, x, fc2)?,
            None => b.norm1.forward16(gpu, x)?,
        };
        drop(pending);
        let qkv = b.qkv.forward16_act16(gpu, &h, ACT_NONE)?;
        drop(h);

        let (cos, sin) = if b.global { &self.rope_global_wm } else { &self.rope_window };
        let windowed = usize::from(!b.global);
        let k = Bf16::empty(gpu, tokens, DIM)?;
        launch!(gpu, "vit_rope_k_bf16", ew(tokens * DIM / 2), B256, 0;
            &k, &qkv, cos, sin, DIM, HEAD_DIM, windowed)?;

        let merged = Bf16::empty(gpu, tokens, DIM)?;
        // q and v are column blocks of qkv's rows and k a column block of the rotated
        // keys; the output is BF16 pairs as 4-byte slots, so its strides count pairs
        let view = |ptr: u64, ld: usize| {
            let extent = (HEADS - 1) * HEAD_DIM + (tokens - 1) * ld + HEAD_DIM;
            [Arg::slice(ptr, extent), Arg::from(ld), Arg::from(HEAD_DIM)]
        };
        let qkv_col = |col: usize| qkv.buf.ptr() + (col * 2) as u64;
        let mut args =
            vec![Arg::slice(merged.buf.ptr(), tokens * DIM / 2), Arg::from(DIM / 2), Arg::from(HEAD_DIM / 2)];
        args.extend(view(qkv_col(0), 3 * DIM));
        args.extend(view(k.buf.ptr(), DIM));
        args.extend(view(qkv_col(2 * DIM), 3 * DIM));
        args.extend([
            Arg::from(tokens),
            Arg::from(tokens),
            Arg::from(std::f32::consts::LOG2_E / (HEAD_DIM as f32).sqrt()),
            Arg::from(cos),
            Arg::from(sin),
            Arg::from(windowed),
            Arg::EMPTY,
        ]);
        gpu.launch("vit_attn_bf16_d64", ((tokens / 64) as u32, HEADS as u32, 1), (128, 1, 1), 0, args)?;
        drop((qkv, k));

        let attn = b.proj.forward16_act16(gpu, &merged, ACT_NONE)?;
        drop(merged);
        let hidden = b.fc1.forward16_act16(gpu, &b.norm2.add_forward16(gpu, x, &attn)?, ACT_GELU_TANH)?;
        drop(attn);
        b.fc2.forward16_act16(gpu, &hidden, ACT_NONE)
    }

    /// One FP32 block over raster-order `x` in place
    fn block(&self, gpu: &Gpu, b: &Block, x: &Tensor) -> Result<()> {
        let tokens = GRID * GRID;
        let (win, rope) = if b.global { (GRID, &self.rope_global) } else { (WINDOW, &self.rope_window) };
        let batch = (GRID / win) * (GRID / win) * HEADS;
        let l = win * win;
        let o = gpu.empty(&[batch, l, HEAD_DIM])?;
        let out = Mat::new(o.ptr(), HEAD_DIM).stride(l * HEAD_DIM);
        let qkv = b.qkv.forward(gpu, &b.norm1.forward(gpu, x)?)?;
        let q = gpu.empty(&[batch, l, HEAD_DIM])?;
        let k = gpu.empty(&[batch, l, HEAD_DIM])?;
        let v = gpu.empty(&[batch, l, HEAD_DIM])?;
        launch!(gpu, "vit_split_rope", ew(tokens * DIM / 2), B256, 0;
            &q, &k, &v, &qkv, &rope.0, &rope.1, GRID, win, HEADS, HEAD_DIM)?;
        drop(qkv);
        let view = |t: &Tensor| Mat::new(t.ptr(), HEAD_DIM).stride(l * HEAD_DIM);
        attention(gpu, view(&q), view(&k), view(&v), out, batch, l, l, HEAD_DIM, Mask::default())?;

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

/// `x [GRID * GRID, cols]` reordered from raster to window-major rows with `to_window`,
/// and back otherwise
fn window_order(gpu: &Gpu, x: &Tensor, to_window: bool) -> Result<Tensor> {
    let out = gpu.empty(x.shape())?;
    launch!(gpu, "vit_window_order", ew(x.len()), B256, 0; &out, x, x.cols(), usize::from(to_window))?;
    Ok(out)
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
                proj1: Linear::load(gpu, wts, &format!("{l}.proj1"))?.bf16(gpu)?,
                proj2: Conv3x3::load(gpu, wts, &format!("{l}.proj2"))?,
            });
        }
        Ok(Self { levels })
    }

    /// trunk `[72 * 72, 1024]` -> `[288^2, 256]`, `[144^2, 256]`, `[72^2, 256]`. In BF16
    /// mode pass the trunk's BF16 copy too ([`Trunk::new`]): every GEMM up to `proj1`
    /// then reads BF16 its producer wrote, rather than an FP32 map converted again per level
    pub fn forward(&self, gpu: &Gpu, trunk: &Trunk) -> Result<Vec<Tensor>> {
        let mut out = Vec::new();
        for level in &self.levels {
            let (h, side) = match (&level.scale, &trunk.bf16) {
                (Scale::Up4(a, b), Some(t16)) => {
                    let h16 = a.forward16(gpu, t16, GRID, GRID, ACT_GELU)?;
                    let y16 = b.forward16(gpu, &h16, 2 * GRID, 2 * GRID, ACT_NONE)?;
                    (level.proj1.forward_from16(gpu, &y16)?, 4 * GRID)
                }
                (Scale::Up2(a), Some(t16)) => {
                    (level.proj1.forward_from16(gpu, &a.forward16(gpu, t16, GRID, GRID, ACT_NONE)?)?, 2 * GRID)
                }
                (Scale::Same, Some(t16)) => (level.proj1.forward_from16(gpu, t16)?, GRID),
                (Scale::Up4(a, b), None) => {
                    let h = a.forward(gpu, &trunk.f32, GRID, GRID)?;
                    bias_act(gpu, &h, None, ACT_GELU)?;
                    (level.proj1.forward(gpu, &b.forward(gpu, &h, 2 * GRID, 2 * GRID)?)?, 4 * GRID)
                }
                (Scale::Up2(a), None) => {
                    (level.proj1.forward(gpu, &a.forward(gpu, &trunk.f32, GRID, GRID)?)?, 2 * GRID)
                }
                (Scale::Same, None) => (level.proj1.forward(gpu, &trunk.f32)?, GRID),
            };
            out.push(level.proj2.forward(gpu, &h, side, side)?);
        }
        Ok(out)
    }
}

/// The backbone's output tokens `[72 * 72, 1024]` as the necks read them
pub struct Trunk {
    pub f32: Tensor,
    /// BF16 copy in BF16 mode, converted once for every neck level
    pub bf16: Option<Bf16>,
}

impl Trunk {
    pub fn new(gpu: &Gpu, f32: Tensor) -> Result<Self> {
        let bf16 = match gpu.precision() {
            Precision::Bf16 => Some(Bf16::from_f32(gpu, &f32)?),
            Precision::Fp32 | Precision::Tf32 => None,
        };
        Ok(Self { f32, bf16 })
    }
}
