//! SAM 3 video tracking on Metal (the tracker-only "SAM 3 for the SAM 2 video task"
//! path): objects clicked on the first frame, followed forward through the video with
//! the tracker's memory bank
//!
//! Per frame: the image backbone and tracker neck, memory attention from the frame's
//! features to earlier frames' memories and object pointers, the point decoder on the
//! result, then the memory encoder on the chosen mask, whose output joins the bank. This
//! follows upstream's `Sam3TrackerPredictor` (`sam3_tracker_base.py` and
//! `sam3_tracking_predictor.py` at f1647e8): all objects share one state, so the memory
//! selection uses their mean score
//!
//! Everything after the tracker neck runs in FP32 in both precisions, as on the rest of
//! the Metal backend, so the memory attention assembles each frame's memory and projects
//! it in every layer rather than caching reduced-precision keys per frame

use std::collections::VecDeque;

use eyre::{Result, ensure};

use crate::image::{RgbImage, Taps, aa_taps};
use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::image::preprocess;
use crate::metal::masks::{LOW_RES, upsample_masks};
use crate::metal::ops::{Act, LayerNorm, Linear, Mask, Mlp, View, attention};
use crate::metal::pipeline::Sam3;
use crate::metal::point::{Decoded, MASK_TOKEN_ROW, PointImage, add, add_tiled, argmax, clone, row};
use crate::metal::vision::{GRID, IMAGE_SIZE, Trunk};
use crate::output::{TrackedFrame, TrackedObject};
use crate::posenc::sine_2d;
use crate::prompt::PointPrompt;
use crate::weights::Weights;

const D: usize = 256;
/// Channels of a memory frame
const MEM_D: usize = 64;
/// Tokens of one frame's 72x72 feature map
const TOKENS: usize = GRID * GRID;
/// `num_maskmem`: memory frames per step, the conditioning frame's position plus 6
const NUM_MASKMEM: usize = 7;
/// `max_cond_frames_in_attn`
const MAX_COND_FRAMES: usize = 4;
/// `max_obj_ptrs_in_encoder`
const MAX_OBJ_PTRS: usize = 16;
/// Clicks per object the prompt encoder sees: the first 8 and the last 8 of more
/// (`max_point_num_in_prompt_enc`, to stay near the 8-point training prompts)
const MAX_CLICKS: usize = 16;
/// `mf_threshold`: earlier frames whose score is at most this are skipped as memories
const MEMORY_SCORE_THRESHOLD: f32 = 0.01;
/// `NO_OBJ_SCORE`, the logit upstream gives every pixel of an absent object's mask
const NO_OBJ_SCORE: f32 = -1024.0;
/// Side of the memory encoder's mask input, upsampled from the 1008 frame
const MASK_SIDE: usize = 1152;
const ROPE_THETA: f32 = 10000.0;

/// `n` FP32 elements of `src` from element `from` into `dst` from element `to`
fn copy(gpu: &Gpu, dst: &Tensor, to: usize, src: &Tensor, from: usize, n: usize) -> Result<()> {
    ensure!(from + n <= src.len() && to + n <= dst.len(), "copy of {n} from {:?} into {:?}", src.shape(), dst.shape());
    gpu.dispatch_1d("copy_f32", n, &[dst.at(to), src.at(from)])
}

/// `RoPEAttention` with one head of 256: projections, then axial RoPE on q and on the
/// first `rope_keys` keys
struct RopeAttention {
    q: Linear,
    k: Linear,
    out: ValueOut,
}

/// What follows the attention probabilities P
enum ValueOut {
    /// `o_proj(P (v_in Wv^T + bv))`, the self-attention's path
    Separate { v: Linear, o: Linear },
    /// `(P memory) (Wo Wv)^T + Wo bv + bo` as one 64 -> 256 linear after the attention:
    /// softmax rows sum to 1, so the value bias passes through P unchanged, and the
    /// attention runs on the memory's 64-wide values rather than 256-wide projections
    /// (a quarter of the P @ V work and accumulator registers)
    Folded(Linear),
}

impl RopeAttention {
    fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let l = |n: &str| Linear::load(gpu, wts, &format!("{prefix}.{n}"), DType::F32);
        Ok(Self { q: l("q_proj")?, k: l("k_proj")?, out: ValueOut::Separate { v: l("v_proj")?, o: l("o_proj")? } })
    }

    /// [`RopeAttention::load`] for the cross-attention to the 64-channel memory, with
    /// `v_proj` folded into `o_proj` (see [`ValueOut::Folded`])
    fn load_cross(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let l = |n: &str| Linear::load(gpu, wts, &format!("{prefix}.{n}"), DType::F32);
        let (wv, sv) = wts.host(&format!("{prefix}.v_proj.weight"))?;
        let (bv, _) = wts.host(&format!("{prefix}.v_proj.bias"))?;
        let (wo, so) = wts.host(&format!("{prefix}.o_proj.weight"))?;
        let (bo, _) = wts.host(&format!("{prefix}.o_proj.bias"))?;
        ensure!(sv == [D, MEM_D] && so == [D, D], "{prefix}: v_proj {sv:?}, o_proj {so:?}");
        // f64 sums, rounded once
        let mut w = vec![0f32; D * MEM_D];
        let mut b = vec![0f32; D];
        for i in 0..D {
            let row = &wo[i * D..(i + 1) * D];
            for j in 0..MEM_D {
                w[i * MEM_D + j] =
                    row.iter().enumerate().map(|(k, &x)| x as f64 * wv[k * MEM_D + j] as f64).sum::<f64>() as f32;
            }
            b[i] = (row.iter().zip(&bv).map(|(&x, &y)| x as f64 * y as f64).sum::<f64>() + bo[i] as f64) as f32;
        }

        let vo = Linear::new(gpu.upload_f32(&w, &[D, MEM_D])?, Some(gpu.upload_f32(&b, &[D])?));
        Ok(Self { q: l("q_proj")?, k: l("k_proj")?, out: ValueOut::Folded(vo) })
    }

    /// `residual += o_proj(attn(rope(q_in Wq), rope(k_in Wk), v_in Wv))`, where `v_in` is
    /// the memory itself on the folded path
    #[allow(clippy::too_many_arguments)]
    fn forward_residual(
        &self,
        gpu: &Gpu,
        rope: &Rope,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        rope_keys: usize,
        residual: &Tensor,
    ) -> Result<()> {
        let q = rope.apply(gpu, &self.q.forward(gpu, q_in, Act::None)?, q_in.rows())?;
        let k = rope.apply(gpu, &self.k.forward(gpu, k_in, Act::None)?, rope_keys)?;
        let (lq, lk) = (q.rows(), k.rows());
        let attend = |v: &Tensor| -> Result<Tensor> {
            let dv = v.cols();
            let attn = gpu.empty(&[lq, dv], DType::F32)?;
            let (qv, kv, vv, ov) = (View::rows(&q), View::rows(&k), View::rows(v), View::rows(&attn));
            attention(gpu, qv, kv, vv, ov, 1, lq, lk, (D, dv), Mask::default())?;
            Ok(attn)
        };
        match &self.out {
            ValueOut::Separate { v, o } => {
                o.forward_residual(gpu, &attend(&v.forward(gpu, v_in, Act::None)?)?, residual)
            }
            ValueOut::Folded(vo) => vo.forward_residual(gpu, &attend(v_in)?, residual),
        }
    }
}

/// Axial RoPE angles for the 72x72 grid at head dim 256: pairs 0 to 63 rotate with the
/// column, 64 to 127 with the row (`compute_axial_cis`)
struct Rope {
    /// `[TOKENS, 128]`
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(gpu: &Gpu) -> Result<Self> {
        let quarter = D / 4;
        // f32 as upstream computes them: `1 / theta ** (4i / dim)`, then position * freq
        let freqs: Vec<f32> = (0..quarter).map(|i| 1.0 / ROPE_THETA.powf((4 * i) as f32 / D as f32)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(TOKENS * D / 2), Vec::with_capacity(TOKENS * D / 2));
        for t in 0..TOKENS {
            let (x, y) = ((t % GRID) as f32, (t / GRID) as f32);
            for (pos, f) in freqs.iter().map(|f| (x, f)).chain(freqs.iter().map(|f| (y, f))) {
                let angle = (pos * f) as f64;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Ok(Self { cos: gpu.upload_f32(&cos, &[TOKENS, D / 2])?, sin: gpu.upload_f32(&sin, &[TOKENS, D / 2])? })
    }

    /// `x [n, 256]` with its first `rows` rows rotated at positions `row % 5184`
    fn apply(&self, gpu: &Gpu, x: &Tensor, rows: usize) -> Result<Tensor> {
        ensure!(x.cols() == D, "RoPE over {:?}, expected rows of {D}", x.shape());
        let out = gpu.empty(x.shape(), DType::F32)?;
        let args = [out.arg(), x.arg(), self.cos.arg(), self.sin.arg(), D.into(), rows.into(), TOKENS.into()];
        gpu.dispatch_1d("rope_rows", out.len(), &args)?;
        Ok(out)
    }
}

struct MemoryLayer {
    norm1: LayerNorm,
    self_attn: RopeAttention,
    norm2: LayerNorm,
    cross_attn: RopeAttention,
    norm3: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

/// The memory attention (`TransformerEncoderCrossAttention` of 4 pre-norm
/// `TransformerDecoderLayerv2`s): the frame's features attend to themselves, then to the
/// memory bank
pub struct MemoryAttention {
    layers: Vec<MemoryLayer>,
    norm: LayerNorm,
    rope: Rope,
    /// `0.1 * pos` of the tracker neck's 72x72 level, added to the input once
    pub(crate) src_pos: Tensor,
}

impl MemoryAttention {
    fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "tracker_model.memory_attention";
        let layers = (0..4)
            .map(|i| {
                let l = format!("{p}.layers.{i}");
                let ln = |n: &str| LayerNorm::load(gpu, wts, &format!("{l}.{n}"), 1e-5);
                let lin = |n: &str| Linear::load(gpu, wts, &format!("{l}.{n}"), DType::F32);
                Ok(MemoryLayer {
                    norm1: ln("layer_norm1")?,
                    self_attn: RopeAttention::load(gpu, wts, &format!("{l}.self_attn"))?,
                    norm2: ln("layer_norm2")?,
                    cross_attn: RopeAttention::load_cross(gpu, wts, &format!("{l}.cross_attn_image"))?,
                    norm3: ln("layer_norm3")?,
                    linear1: lin("linear1")?,
                    linear2: lin("linear2")?,
                })
            })
            .collect::<Result<_>>()?;
        // f32 products, as `output + 0.1 * src_pos` rounds them
        let src_pos: Vec<f32> = sine_2d(GRID, D).into_iter().map(|v| 0.1 * v).collect();
        Ok(Self {
            layers,
            norm: LayerNorm::load(gpu, wts, &format!("{p}.layer_norm"), 1e-5)?,
            rope: Rope::new(gpu)?,
            src_pos: gpu.upload_f32(&src_pos, &[TOKENS, D])?,
        })
    }

    /// The part of [`Self::forward`] every object shares: frame features `src [5184, 256]`
    /// plus position, through the first layer's self-attention, which sees no memory
    pub fn prepare(&self, gpu: &Gpu, src: &Tensor) -> Result<Tensor> {
        ensure!(src.shape() == [TOKENS, D], "memory attention over {:?}", src.shape());
        let x = add(gpu, src, &self.src_pos)?;
        let l = &self.layers[0];
        let h = l.norm1.forward_to(gpu, &x, DType::F32)?;
        l.self_attn.forward_residual(gpu, &self.rope, &h, &h, &h, TOKENS, &x)?;
        Ok(x)
    }

    /// The frame's features from [`Self::prepare`] conditioned on `memory [L, 64]` (its
    /// positional encoding `memory_pos` alongside), whose last `ptr_tokens` rows are
    /// object pointer tokens without RoPE
    pub fn forward(
        &self,
        gpu: &Gpu,
        prepared: &Tensor,
        memory: &Tensor,
        memory_pos: &Tensor,
        ptr_tokens: usize,
    ) -> Result<Tensor> {
        ensure!(prepared.shape() == [TOKENS, D], "memory attention over {:?}", prepared.shape());
        ensure!(memory.shape() == memory_pos.shape(), "memory {:?}, position {:?}", memory.shape(), memory_pos.shape());
        ensure!(memory.cols() == MEM_D && memory.rows() >= ptr_tokens, "memory {:?}", memory.shape());
        // keys see the memory plus its position, values the memory alone
        let keys = add(gpu, memory, memory_pos)?;
        let rope_keys = memory.rows() - ptr_tokens;
        let rope = &self.rope;
        // each object updates its own copy
        let x = clone(gpu, prepared)?;
        for (i, l) in self.layers.iter().enumerate() {
            if i > 0 {
                let h = l.norm1.forward_to(gpu, &x, DType::F32)?;
                l.self_attn.forward_residual(gpu, rope, &h, &h, &h, TOKENS, &x)?;
            }
            let h = l.norm2.forward_to(gpu, &x, DType::F32)?;
            l.cross_attn.forward_residual(gpu, rope, &h, &keys, memory, rope_keys, &x)?;
            let h = l.norm3.forward_to(gpu, &x, DType::F32)?;
            l.linear2.forward_residual(gpu, &l.linear1.forward(gpu, &h, Act::Relu)?, &x)?;
        }
        self.norm.forward_to(gpu, &x, DType::F32)
    }
}

/// Input channels from which a downsampler conv runs as im2col and one GEMM: the two
/// deeper layers have 16 and 64 input channels (144 and 576 products an output), while
/// the first two have 9 and 36
const DOWN_GEMM_CIN: usize = 16;

/// A stride-2 3x3 conv of the mask downsampler
enum DownConv {
    /// PyTorch's `[cout, cin, 3, 3]` and the bias, through the direct kernel
    Direct(Tensor, Tensor),
    /// The weights as `[cout, cin * 9]`, against the im2col
    Gemm(Linear),
}

/// One stride-2 conv of the mask downsampler, then LayerNorm2d and GELU
struct DownLayer {
    conv: DownConv,
    norm_w: Tensor,
    norm_b: Tensor,
    cin: usize,
    cout: usize,
}

/// `LayerNorm2d` epsilon of the mask downsampler
const DOWN_EPS: f32 = 1e-6;

/// A ConvNeXt block of the memory fuser, with its layer scale folded into `fc2`
struct CxBlock {
    /// `[256, 1, 7, 7]`
    dw_w: Tensor,
    dw_b: Tensor,
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// One antialiased bilinear resize (`_upsample_bilinear2d_aa`) of single planes from one
/// square size to another, with the taps uploaded once
struct AaResize {
    rows: (Tensor, Tensor, Tensor, usize),
    cols: (Tensor, Tensor, Tensor, usize),
    input: usize,
    output: usize,
}

impl AaResize {
    fn new(gpu: &Gpu, input: usize, output: usize) -> Result<Self> {
        let upload = |t: Taps| -> Result<(Tensor, Tensor, Tensor, usize)> {
            let n = t.starts.len();
            let max = t.weights.len() / n;
            Ok((
                gpu.upload(&t.starts, &[n], DType::I32)?,
                gpu.upload(&t.counts, &[n], DType::I32)?,
                gpu.upload_f32(&t.weights, &[t.weights.len()])?,
                max,
            ))
        };
        // square planes: the same taps along both axes
        Ok(Self { rows: upload(aa_taps(input, output))?, cols: upload(aa_taps(input, output))?, input, output })
    }

    /// `x [input * input]` -> `[output * output]`, columns first as the image transform
    fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        let (i, o) = (self.input, self.output);
        ensure!(x.len() == i * i, "resize of {:?} from {i}x{i}", x.shape());
        let wide = gpu.empty(&[i, o], DType::F32)?;
        let (s, c, w, max) = &self.cols;
        let args = [wide.arg(), x.arg(), s.arg(), c.arg(), w.arg(), i.into(), o.into(), (*max).into()];
        gpu.dispatch_1d("resample_cols", wide.len(), &args)?;
        let out = gpu.empty(&[o, o], DType::F32)?;
        let (s, c, w, max) = &self.rows;
        let args = [out.arg(), wide.arg(), s.arg(), c.arg(), w.arg(), o.into(), i.into(), o.into(), (*max).into()];
        gpu.dispatch_1d("resample_rows", out.len(), &args)?;
        Ok(out)
    }
}

/// The memory encoder (`SimpleMaskEncoder`): the mask, downsampled to the 72x72 grid,
/// added to projected frame features, two ConvNeXt blocks, then a projection to 64
pub struct MemoryEncoder {
    resize: AaResize,
    down: Vec<DownLayer>,
    final_conv: Linear,
    pix_proj: Linear,
    fuser: Vec<CxBlock>,
    out_proj: Linear,
}

impl MemoryEncoder {
    fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "tracker_model.memory_encoder";
        let lin = |name: &str| Linear::load(gpu, wts, name, DType::F32);
        let mut down = Vec::new();
        for i in 0..4 {
            let l = format!("{p}.mask_downsampler.layers.{i}");
            let w = wts.get(gpu, &format!("{l}.conv.weight"))?;
            let (cout, cin) = (w.shape()[0], w.shape()[1]);
            let b = wts.get(gpu, &format!("{l}.conv.bias"))?;
            let conv = if cin >= DOWN_GEMM_CIN {
                DownConv::Gemm(Linear::new(w.reshape(&[cout, cin * 9]), Some(b)))
            } else {
                DownConv::Direct(w, b)
            };
            down.push(DownLayer {
                conv,
                norm_w: wts.get(gpu, &format!("{l}.layer_norm.weight"))?,
                norm_b: wts.get(gpu, &format!("{l}.layer_norm.bias"))?,
                cin,
                cout,
            });
        }
        let mut fuser = Vec::new();
        for i in 0..2 {
            let l = format!("{p}.memory_fuser.layers.{i}");
            // gamma * (x W2^T + b2) as (gamma W2) and gamma b2: one rounding apart
            let (gamma, _) = wts.host(&format!("{l}.scale"))?;
            let (mut w2, shape) = wts.host(&format!("{l}.pointwise_conv2.weight"))?;
            let (mut b2, _) = wts.host(&format!("{l}.pointwise_conv2.bias"))?;
            ensure!(gamma.len() == shape[0], "{l}: {} scales for {shape:?}", gamma.len());
            for (o, g) in gamma.iter().enumerate() {
                w2[o * shape[1]..][..shape[1]].iter_mut().for_each(|v| *v *= g);
                b2[o] *= g;
            }
            fuser.push(CxBlock {
                dw_w: wts.get(gpu, &format!("{l}.depthwise_conv.weight"))?,
                dw_b: wts.get(gpu, &format!("{l}.depthwise_conv.bias"))?,
                norm: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm"), 1e-6)?,
                fc1: lin(&format!("{l}.pointwise_conv1"))?,
                fc2: Linear::new(gpu.upload_f32(&w2, &shape)?, Some(gpu.upload_f32(&b2, &[shape[0]])?)),
            });
        }
        Ok(Self {
            resize: AaResize::new(gpu, IMAGE_SIZE, MASK_SIDE)?,
            down,
            final_conv: lin(&format!("{p}.mask_downsampler.final_conv"))?,
            pix_proj: lin(&format!("{p}.feature_projection"))?,
            fuser,
            out_proj: lin(&format!("{p}.projection"))?,
        })
    }

    /// Frame features `pix_feat [5184, 256]` and the mask input `mask [1008 * 1008]`
    /// (already through [`mask_for_mem`]) -> memory `[5184, 64]`
    pub fn forward(&self, gpu: &Gpu, pix_feat: &Tensor, mask: &Tensor) -> Result<Tensor> {
        ensure!(mask.len() == IMAGE_SIZE * IMAGE_SIZE, "memory encoder mask {:?}", mask.shape());
        ensure!(pix_feat.shape() == [TOKENS, D], "memory encoder features {:?}", pix_feat.shape());
        let mut x = self.resize.forward(gpu, mask)?.reshape(&[MASK_SIDE * MASK_SIDE, 1]);
        let mut side = MASK_SIDE;
        for l in &self.down {
            let (ih, iw) = (side, side);
            side /= 2;
            let y = match &l.conv {
                DownConv::Direct(w, b) => {
                    let y = gpu.empty(&[side * side, l.cout], DType::F32)?;
                    let args = [y.arg(), x.arg(), w.arg(), b.arg(), ih.into(), iw.into(), l.cin.into(), l.cout.into()];
                    gpu.dispatch_1d("conv3x3_s2", y.len(), &args)?;
                    y
                }
                DownConv::Gemm(conv) => {
                    let cols = gpu.empty(&[side * side, l.cin * 9], DType::F32)?;
                    gpu.dispatch_1d(
                        "im2col3x3_s2",
                        cols.len(),
                        &[cols.arg(), x.arg(), ih.into(), iw.into(), l.cin.into()],
                    )?;
                    conv.forward(gpu, &cols, Act::None)?
                }
            };
            let args = [y.arg(), l.norm_w.arg(), l.norm_b.arg(), l.cout.into(), DOWN_EPS.into(), Act::Gelu.into()];
            gpu.dispatch_1d("layernorm_act_rows", side * side, &args)?;
            x = y;
        }
        ensure!(side == GRID, "the mask downsampler ends at {side}, not {GRID}");
        let x = self.final_conv.forward(gpu, &x, Act::None)?;
        self.pix_proj.forward_residual(gpu, pix_feat, &x)?;
        for b in &self.fuser {
            let y = gpu.empty(x.shape(), DType::F32)?;
            let c = x.cols();
            let args = [y.arg(), x.arg(), b.dw_w.arg(), b.dw_b.arg(), GRID.into(), GRID.into(), c.into()];
            gpu.dispatch_1d("dwconv7x7", y.len(), &args)?;
            b.norm.forward_inplace(gpu, &y)?;
            b.fc2.forward_residual(gpu, &b.fc1.forward(gpu, &y, Act::Gelu)?, &x)?;
        }
        self.out_proj.forward(gpu, &x, Act::None)
    }
}

/// `x [n] -> (binarize ? x > 0 : sigmoid(x)) * 20 - 10`, the memory encoder's mask input
pub fn mask_for_mem(gpu: &Gpu, x: &Tensor, binarize: bool) -> Result<Tensor> {
    let out = gpu.empty(x.shape(), DType::F32)?;
    gpu.dispatch_1d("mask_for_mem", out.len(), &[out.arg(), x.arg(), u32::from(binarize).into()])?;
    Ok(out)
}

/// The tracker's own layers and constants, on top of the image path's backbone, tracker
/// neck and point decoder
pub struct Tracker {
    pub mem_attn: MemoryAttention,
    pub mem_enc: MemoryEncoder,
    pub obj_ptr_proj: Mlp,
    /// `obj_ptr_tpos_proj`, 256 to 64
    tpos_proj: Linear,
    no_obj_ptr: Tensor,
    /// `no_obj_embed_spatial [64]`, added to an absent object's memory
    pub(crate) no_obj_spatial: Tensor,
    /// `maskmem_pos_enc + maskmem_tpos_enc[i]` per temporal index i, `[5184, 64]` each
    mem_pos: Vec<Tensor>,
}

impl Tracker {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "tracker_model";
        let (tpos, _) = wts.host(&format!("{p}.memory_temporal_positional_encoding"))?;
        let sine = sine_2d(GRID, MEM_D);
        let mem_pos = tpos
            .chunks(MEM_D)
            .map(|t| {
                let pos: Vec<f32> = sine.iter().enumerate().map(|(i, v)| v + t[i % MEM_D]).collect();
                gpu.upload_f32(&pos, &[TOKENS, MEM_D])
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(mem_pos.len() == NUM_MASKMEM, "{} temporal encodings, expected {NUM_MASKMEM}", mem_pos.len());
        let no_obj_spatial = wts.get(gpu, &format!("{p}.occlusion_spatial_embedding_parameter"))?;
        Ok(Self {
            mem_attn: MemoryAttention::load(gpu, wts)?,
            mem_enc: MemoryEncoder::load(gpu, wts)?,
            obj_ptr_proj: Mlp::load_in_out(gpu, wts, &format!("{p}.object_pointer_proj"), 3, Act::Relu)?,
            tpos_proj: Linear::load(
                gpu,
                wts,
                &format!("{p}.temporal_positional_encoding_projection_layer"),
                DType::F32,
            )?,
            no_obj_ptr: wts.get(gpu, &format!("{p}.no_object_pointer"))?,
            no_obj_spatial: no_obj_spatial.reshape(&[MEM_D]),
            mem_pos,
        })
    }

    /// Object pointer from the chosen mask token, or `no_obj_ptr` for an absent object
    pub fn pointer(&self, gpu: &Gpu, token: &Tensor, present: bool) -> Result<Tensor> {
        if present { self.obj_ptr_proj.forward(gpu, token) } else { clone(gpu, &self.no_obj_ptr) }
    }

    /// The memory of one object on one frame: the encoder's output, plus
    /// `no_obj_embed_spatial` for an absent object
    ///
    /// Upstream stores the bank in BF16; this keeps FP32, which the video parity checks
    /// hold to the FP32 reference without the rounding
    pub fn memory(&self, gpu: &Gpu, pix_feat: &Tensor, mask: &Tensor, present: bool) -> Result<Tensor> {
        let mem = self.mem_enc.forward(gpu, pix_feat, mask)?;
        if !present {
            add_tiled(gpu, &mem, &self.no_obj_spatial)?;
        }
        Ok(mem)
    }

    /// `obj_ptr_tpos_proj(get_1d_sine_pe(pos / (max_abs_pos - 1), 256))` per pointer
    fn pointer_pos(&self, gpu: &Gpu, positions: &[usize], max_abs_pos: usize) -> Result<Option<Tensor>> {
        if positions.is_empty() {
            return Ok(None);
        }
        let t_max = max_abs_pos.saturating_sub(1).max(1) as f32;
        let half = D / 2;
        let dims: Vec<f32> = (0..half).map(|i| 10000f32.powf((2 * (i / 2)) as f32 / half as f32)).collect();
        let mut pe = Vec::with_capacity(positions.len() * D);
        for &p in positions {
            let v = p as f32 / t_max;
            pe.extend(dims.iter().map(|d| ((v / d) as f64).sin() as f32));
            pe.extend(dims.iter().map(|d| ((v / d) as f64).cos() as f32));
        }
        Ok(Some(self.tpos_proj.forward(gpu, &gpu.upload_f32(&pe, &[positions.len(), D])?, Act::None)?))
    }
}

/// One frame's memory for every object, and its object pointers
struct MemFrame {
    frame: usize,
    /// `[5184, 64]` per object
    mem: Vec<Tensor>,
    /// `[1, 256]` per object
    ptr: Vec<Tensor>,
}

/// The memory frames and object pointers upstream's selection gives one frame
pub(crate) struct Selection<'a> {
    /// Memory frames with their temporal encoding index, in bank order
    frames: Vec<(&'a MemFrame, usize)>,
    /// Frames whose object pointers join the memory
    ptr_frames: Vec<&'a MemFrame>,
    /// `[ptr_frames, 64]`, each pointer's temporal position, when there are pointers
    ptr_pos: Option<Tensor>,
}

impl Selection<'_> {
    /// Memory tokens from object pointers, 4 a pointer
    pub(crate) fn ptr_tokens(&self) -> usize {
        self.ptr_frames.len() * (D / MEM_D)
    }
}

/// Memory-attention inputs of one frame, `(memory, memory_pos)` per object
pub(crate) struct MemoryInputs {
    pub(crate) objects: Vec<(Tensor, Tensor)>,
}

/// Follows clicked objects through a video one frame at a time, starting from the frame
/// they were clicked on (see [`Sam3::track`])
///
/// Holds the memory bank: the clicked frame's memory and the earlier frames upstream's
/// memory selection can still pick, which is a constant amount whatever the video's length
pub struct VideoTracker<'a> {
    model: &'a Sam3,
    width: usize,
    height: usize,
    num_frames: usize,
    next_frame: usize,
    objects: usize,
    cond: Vec<MemFrame>,
    /// The last frames (up to `min(num_frames, 16) - 1`) whose score passed
    /// [`MEMORY_SCORE_THRESHOLD`], oldest first
    passing: VecDeque<MemFrame>,
    /// The previous frame when its score didn't pass: upstream always keeps it
    prev: Option<MemFrame>,
}

/// Features of one video frame
pub(crate) struct FrameFeatures {
    pub(crate) point: PointImage,
    /// Tracker neck's `fpn.coarse` `[5184, 256]`, without the no-memory embedding
    pub(crate) coarse: Tensor,
}

impl Sam3 {
    /// Backbone and tracker neck of one video frame
    pub(crate) fn video_features(&self, pixels: &Tensor) -> Result<FrameFeatures> {
        let gpu = &self.gpu;
        let trunk = self.backbone.forward(gpu, pixels)?;
        let fpn = self.tracker_neck.forward(gpu, &Trunk::new(gpu, trunk)?)?;
        let point = self.point.prepare_image(gpu, &fpn)?;
        Ok(FrameFeatures { point, coarse: fpn.coarse })
    }

    /// Starts tracking objects through a video of `num_frames` frames: segments each
    /// object from its clicks on `first` (frame 0), whose masks come back with the tracker
    ///
    /// Each entry of `objects` is one object's clicks, in `first`'s pixels; past 16, only
    /// the first and last 8 reach the decoder, as upstream. Upstream scales the object
    /// pointers' temporal positions by the video's length, so `num_frames` must be the
    /// real frame count; [`VideoTracker::next`] stops after it
    pub fn track(
        &self,
        first: &RgbImage,
        objects: &[Vec<PointPrompt>],
        num_frames: usize,
    ) -> Result<(VideoTracker<'_>, TrackedFrame)> {
        let pixels = preprocess(&self.gpu, first)?;
        self.track_pixels(&pixels, first.width(), first.height(), objects, num_frames)
    }

    /// [`Self::track`] from preprocessed pixels `[3, 1008, 1008]` of a `width` x
    /// `height` frame
    pub(crate) fn track_pixels(
        &self,
        pixels: &Tensor,
        width: usize,
        height: usize,
        objects: &[Vec<PointPrompt>],
        num_frames: usize,
    ) -> Result<(VideoTracker<'_>, TrackedFrame)> {
        ensure!(!objects.is_empty(), "track needs at least one object");
        ensure!(objects.iter().all(|o| !o.is_empty()), "every object needs at least one click");
        ensure!(num_frames >= 1, "a video has at least one frame");
        let mut tracker = VideoTracker {
            model: self,
            width,
            height,
            num_frames,
            next_frame: 1,
            objects: objects.len(),
            cond: Vec::new(),
            passing: VecDeque::new(),
            prev: None,
        };
        let out = tracker.prompt_frame(pixels, objects)?;
        Ok((tracker, out))
    }
}

impl VideoTracker<'_> {
    /// Tracks the objects into the next frame, which must have the first frame's size
    pub fn next(&mut self, frame: &RgbImage) -> Result<TrackedFrame> {
        ensure!(
            (frame.width(), frame.height()) == (self.width, self.height),
            "frame is {}x{}, the video {}x{}",
            frame.width(),
            frame.height(),
            self.width,
            self.height
        );
        let pixels = preprocess(&self.model.gpu, frame)?;
        self.next_pixels(&pixels)
    }

    /// The clicked frame: each object decoded from its clicks without memory, then the
    /// memory encoder over all objects' masks with overlaps resolved, as upstream's
    /// preflight consolidation does
    fn prompt_frame(&mut self, pixels: &Tensor, objects: &[Vec<PointPrompt>]) -> Result<TrackedFrame> {
        let (m, gpu) = (self.model, &self.model.gpu);
        let feats = self.model.video_features(pixels)?;
        let side = LOW_RES * LOW_RES;
        let low_res = gpu.empty(&[objects.len(), side], DType::F32)?;
        let (mut scores, mut ptrs, mut picks) = (Vec::new(), Vec::new(), Vec::new());
        for (b, points) in objects.iter().enumerate() {
            let points = &cap_clicks(points);
            let dec =
                m.point.decode(gpu, &feats.point.embed, &feats.point, points, (self.width, self.height), None, 0)?;
            let score = gpu.download(&m.point.obj_score(gpu, &dec)?)?[0];
            let iou = gpu.download(&dec.iou)?;
            // one click (or none) gives three candidates; more give the single-mask token,
            // unless its mask is unstable (`_dynamic_multimask_via_stability`)
            let (mask, token) = if points.len() <= 1 {
                let best = 1 + argmax(&iou[1..]);
                (best, best)
            } else {
                let stable = stability(&gpu.download(&row(gpu, &dec.low_res, 0)?)?) >= 0.98;
                (if stable { 0 } else { 1 + argmax(&iou[1..]) }, 0)
            };
            self.select_mask(&dec, mask, score > 0.0, &low_res, b)?;
            let token = row(gpu, &dec.tokens, MASK_TOKEN_ROW + token)?;
            ptrs.push(m.tracker.pointer(gpu, &token, score > 0.0)?);
            scores.push(score);
            picks.push((iou, mask));
        }

        let high = upsample(gpu, &low_res, 0, objects.len(), IMAGE_SIZE, IMAGE_SIZE)?;
        let n = IMAGE_SIZE * IMAGE_SIZE;
        let high = if objects.len() > 1 {
            let out = gpu.empty(high.shape(), DType::F32)?;
            gpu.dispatch_1d("nonoverlap", out.len(), &[out.arg(), high.arg(), objects.len().into(), n.into()])?;
            out
        } else {
            high
        };
        let input = mask_for_mem(gpu, &high, true)?;
        let mut mem = Vec::new();
        for (b, &score) in scores.iter().enumerate() {
            let mask = gpu.empty(&[n], DType::F32)?;
            copy(gpu, &mask, 0, &input, b * n, n)?;
            mem.push(m.tracker.memory(gpu, &feats.coarse, &mask, score > 0.0)?);
        }
        self.cond.push(MemFrame { frame: 0, mem, ptr: ptrs });
        self.output(0, low_res, &scores, picks, 0, 0)
    }

    /// Writes mask `index` of a decoder pass into row `b` of `low_res`, or
    /// `NO_OBJ_SCORE` everywhere for an absent object
    fn select_mask(&self, dec: &Decoded, index: usize, present: bool, low_res: &Tensor, b: usize) -> Result<()> {
        let gpu = &self.model.gpu;
        let side = LOW_RES * LOW_RES;
        let first = 4 - dec.low_res.rows();
        ensure!(index >= first, "mask {index} of a pass that kept masks {first}..4");
        copy(gpu, low_res, b * side, &dec.low_res, (index - first) * side, side)?;
        if !present {
            let args = [low_res.at(b * side), NO_OBJ_SCORE.into(), NO_OBJ_SCORE.into()];
            gpu.dispatch_1d("clamp_f32", side, &args)?;
        }
        Ok(())
    }

    /// Frames upstream's `frame_filter` selects for frame `t`, oldest first
    fn selected(&self) -> Vec<&MemFrame> {
        let mut out: Vec<&MemFrame> = self.passing.iter().collect();
        out.extend(self.prev.as_ref());
        out
    }

    /// The bank's selection for the next frame, as upstream's
    /// `_prepare_memory_conditioned_features` makes it
    pub(crate) fn selection(&self) -> Result<Selection<'_>> {
        let (m, gpu) = (self.model, &self.model.gpu);
        let t = self.next_frame;
        // memories: up to 4 conditioning frames at temporal index 6, then the last 6
        // selected frames at indices 5 (oldest) to 0
        let selected = self.selected();
        let cond = &self.cond[self.cond.len().saturating_sub(MAX_COND_FRAMES)..];
        let mut frames: Vec<(&MemFrame, usize)> = cond.iter().map(|f| (f, NUM_MASKMEM - 1)).collect();
        for t_pos in 1..NUM_MASKMEM {
            let t_rel = NUM_MASKMEM - t_pos;
            if t_rel <= selected.len() {
                frames.push((selected[selected.len() - t_rel], NUM_MASKMEM - 1 - t_pos));
            }
        }
        // pointers: the conditioning frames' at their distance, then selected frames
        // -1 to -(L - 1) at positions 1 to L - 1 (upstream stops one short of the oldest)
        let max_abs_pos = self.num_frames.min(MAX_OBJ_PTRS);
        let mut ptr_frames: Vec<(&MemFrame, usize)> = cond.iter().map(|f| (f, t - f.frame)).collect();
        for t_diff in 1..max_abs_pos {
            if t_diff >= selected.len() {
                break;
            }
            ptr_frames.push((selected[selected.len() - t_diff], t_diff));
        }
        let positions: Vec<usize> = ptr_frames.iter().map(|(_, p)| *p).collect();
        let ptr_pos = m.tracker.pointer_pos(gpu, &positions, max_abs_pos)?;
        Ok(Selection { frames, ptr_frames: ptr_frames.into_iter().map(|(f, _)| f).collect(), ptr_pos })
    }

    /// Memory-attention inputs of selection `sel`: each object's memory `[L, 64]` and its
    /// positional encoding, as upstream's `_prepare_memory_conditioned_features` assembles
    /// them. Each 256-d object pointer is 4 tokens of 64, all at the pointer's position
    pub(crate) fn inputs_of(&self, sel: &Selection) -> Result<MemoryInputs> {
        let (m, gpu) = (self.model, &self.model.gpu);
        let ptr_tokens = sel.ptr_tokens();
        let mem_tokens = sel.frames.len() * TOKENS + ptr_tokens;
        ensure!(mem_tokens > 0, "no memory to attend to");
        let frame = TOKENS * MEM_D;
        let base = sel.frames.len() * frame;
        let split = D / MEM_D;

        let mut objects = Vec::with_capacity(self.objects);
        for b in 0..self.objects {
            let memory = gpu.empty(&[mem_tokens, MEM_D], DType::F32)?;
            let memory_pos = gpu.empty(&[mem_tokens, MEM_D], DType::F32)?;
            for (i, (f, tpos)) in sel.frames.iter().enumerate() {
                copy(gpu, &memory, i * frame, &f.mem[b], 0, frame)?;
                copy(gpu, &memory_pos, i * frame, &m.tracker.mem_pos[*tpos], 0, frame)?;
            }
            for (i, f) in sel.ptr_frames.iter().enumerate() {
                copy(gpu, &memory, base + i * D, &f.ptr[b], 0, D)?;
                let pos = sel.ptr_pos.as_ref().expect("positions of every pointer");
                for s in 0..split {
                    copy(gpu, &memory_pos, base + i * D + s * MEM_D, pos, i * MEM_D, MEM_D)?;
                }
            }
            objects.push((memory, memory_pos));
        }
        Ok(MemoryInputs { objects })
    }

    pub(crate) fn next_pixels(&mut self, pixels: &Tensor) -> Result<TrackedFrame> {
        ensure!(self.next_frame < self.num_frames, "the video has {} frames", self.num_frames);
        let (m, gpu) = (self.model, &self.model.gpu);
        let t = self.next_frame;
        let feats = self.model.video_features(pixels)?;
        let max_abs_pos = self.num_frames.min(MAX_OBJ_PTRS);
        let sel = self.selection()?;
        let ptr_tokens = sel.ptr_tokens();
        let mem_tokens = sel.frames.len() * TOKENS + ptr_tokens;
        let inputs = self.inputs_of(&sel)?;
        drop(sel);

        let side = LOW_RES * LOW_RES;
        let low_res = gpu.empty(&[self.objects, side], DType::F32)?;
        let (mut scores, mut ptrs, mut mems, mut picks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut norm_sum, mut iou_sum) = (0f32, 0f32);
        ensure!(inputs.objects.len() == self.objects, "{} memories for {} objects", inputs.objects.len(), self.objects);
        let prepared = m.tracker.mem_attn.prepare(gpu, &feats.coarse)?;
        for (b, (memory, memory_pos)) in inputs.objects.iter().enumerate() {
            let embed = m.tracker.mem_attn.forward(gpu, &prepared, memory, memory_pos, ptr_tokens)?;
            let dec = m.point.decode(gpu, &embed, &feats.point, &[], (self.width, self.height), None, 1)?;
            let score = gpu.download(&m.point.obj_score(gpu, &dec)?)?[0];
            let iou = gpu.download(&dec.iou)?;
            let best = 1 + argmax(&iou[1..]);
            let present = score > 0.0;
            self.select_mask(&dec, best, present, &low_res, b)?;
            ptrs.push(m.tracker.pointer(gpu, &row(gpu, &dec.tokens, MASK_TOKEN_ROW + best)?, present)?);

            let high = upsample(gpu, &low_res, b, 1, IMAGE_SIZE, IMAGE_SIZE)?;
            let input = mask_for_mem(gpu, &high, false)?;
            mems.push(m.tracker.memory(gpu, &feats.coarse, &input, present)?);
            // `cal_mem_score`: present objects' rescaled score and their best IoU
            norm_sum += if present { 2.0 / (1.0 + (-score).exp()) - 1.0 } else { 0.0 };
            iou_sum += iou[best];
            scores.push(score);
            picks.push((iou, best));
        }
        drop(inputs);
        // upstream multiplies scores shaped [B, 1] by IoUs shaped [B], so its mean over the
        // broadcast [B, B] product is mean(score) * mean(IoU), not the mean of per-object
        // products; with several objects the two land on opposite sides of the threshold
        let n = self.objects as f32;
        let eff = (norm_sum / n) * (iou_sum / n);

        let frame = MemFrame { frame: t, mem: mems, ptr: ptrs };
        self.prev = None;
        if eff > MEMORY_SCORE_THRESHOLD {
            self.passing.push_back(frame);
            // `frame_filter` stops at `min(num_frames, 16) - 1` passing frames
            if self.passing.len() >= max_abs_pos {
                self.passing.pop_front();
            }
        } else {
            self.prev = Some(frame);
        }
        self.next_frame += 1;
        self.output(t, low_res, &scores, picks, mem_tokens, ptr_tokens)
    }

    /// Masks at the video's size from `low_res [objects, 288^2]`
    fn output(
        &self,
        frame: usize,
        low_res: Tensor,
        scores: &[f32],
        picks: Vec<(Vec<f32>, usize)>,
        mem_tokens: usize,
        ptr_tokens: usize,
    ) -> Result<TrackedFrame> {
        let gpu = &self.model.gpu;
        let (w, h) = (self.width, self.height);
        let plane = w * h;
        let full = upsample_masks(gpu, &low_res, 0, self.objects, w, h)?;
        let mut bytes = vec![0u8; self.objects * plane];
        gpu.read(&full.buf, 0, &mut bytes)?;
        let logits = gpu.download(&low_res)?;
        let side = LOW_RES * LOW_RES;
        ensure!(
            scores.len() == self.objects && picks.len() == self.objects && logits.len() == self.objects * side,
            "{} scores, {} picks and {} logits for {} objects",
            scores.len(),
            picks.len(),
            logits.len(),
            self.objects
        );
        let objects = scores
            .iter()
            .zip(bytes.chunks_exact(plane))
            .zip(logits.chunks_exact(side))
            .zip(picks)
            .map(|(((&score, mask), low_res), (ious, chosen))| TrackedObject {
                score,
                mask: mask.to_vec(),
                low_res: low_res.to_vec(),
                ious,
                chosen,
            })
            .collect();
        Ok(TrackedFrame { frame, objects, memory_tokens: mem_tokens, pointer_tokens: ptr_tokens })
    }
}

/// Bilinear upsampling (`align_corners=False`) of logits `first..first + n` of
/// `low_res [*, 288^2]` to `[n, h * w]`
fn upsample(gpu: &Gpu, low_res: &Tensor, first: usize, n: usize, h: usize, w: usize) -> Result<Tensor> {
    let side = LOW_RES * LOW_RES;
    ensure!((first + n) * side <= low_res.len(), "logits {first}..{} of {:?}", first + n, low_res.shape());
    let out = gpu.empty(&[n, h * w], DType::F32)?;
    let args = [out.arg(), low_res.at(first * side), LOW_RES.into(), LOW_RES.into(), h.into(), w.into()];
    gpu.dispatch_1d("resize_bilinear", out.len(), &args)?;
    Ok(out)
}

/// Upstream's `_get_stability_scores` at delta 0.05: the area above +delta over the area
/// above -delta
fn stability(logits: &[f32]) -> f32 {
    let inter = logits.iter().filter(|&&v| v > 0.05).count();
    let union = logits.iter().filter(|&&v| v > -0.05).count();
    if union > 0 { inter as f32 / union as f32 } else { 1.0 }
}

/// All the clicks, or the first and last half of [`MAX_CLICKS`] when there are more
fn cap_clicks(points: &[PointPrompt]) -> Vec<PointPrompt> {
    if points.len() <= MAX_CLICKS {
        return points.to_vec();
    }
    let first = MAX_CLICKS / 2;
    points[..first].iter().chain(&points[points.len() - (MAX_CLICKS - first)..]).copied().collect()
}
