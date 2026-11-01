//! SAM 3 video tracking (the tracker-only "SAM 3 for the SAM 2 video task" path): objects
//! clicked on the first frame, followed forward through the video with the tracker's
//! memory bank
//!
//! Per frame: the image backbone and tracker neck, memory attention from the frame's
//! features to earlier frames' memories and object pointers, the point decoder on the
//! result, then the memory encoder on the chosen mask, whose output joins the bank. This
//! follows upstream's `Sam3TrackerPredictor` (`sam3_tracker_base.py` and
//! `sam3_tracking_predictor.py` at f1647e8): all objects share one state, so the memory
//! selection uses their mean score, and the bank is stored in BF16 in every mode

use std::collections::VecDeque;

use eyre::{Result, ensure, eyre};

use crate::gpu::{Arg, Buf, Gpu, Precision, Tensor, launch};
use crate::image::{AaResize, RgbImage, preprocess};
use crate::nvtx::range;
use crate::ops::{
    ACT_GELU, ACT_RELU, B256, LayerNorm, Linear, Mask, Mlp, add, attention_f16_d256, attention_f16_qk256_v64, ew, mha,
};
use crate::pipeline::Sam3;
use crate::point::{Decoded, LOW_RES, MASK_TOKEN_ROW, PointImage, PointPrompt, row};
use crate::posenc::sine_2d;
use crate::vision::{GRID, IMAGE_SIZE, Trunk};
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

/// `RoPEAttention` with one head of 256: projections, then axial RoPE on q and on the
/// first `rope_keys` keys
struct RopeAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// `o_proj(v_proj(.))` as one linear, `Wo Wv` and `Wo bv + bo`, for the memory
    /// cross-attention in BF16 mode (see [`RopeAttention::forward_residual_folded`])
    vo: Option<Linear>,
}

impl RopeAttention {
    fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        // BF16 GEMMs in BF16 mode, as the reference's autocast runs them
        let l = |n: &str| Linear::load(gpu, wts, &format!("{prefix}.{n}"))?.bf16(gpu);
        Ok(Self { q: l("q_proj")?, k: l("k_proj")?, v: l("v_proj")?, o: l("o_proj")?, vo: None })
    }

    /// [`RopeAttention::load`] for the cross-attention to the 64-channel memory, which in
    /// BF16 mode also folds `v_proj` into `o_proj`
    fn load_cross(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let mut out = Self::load(gpu, wts, prefix)?;
        if gpu.precision() == Precision::Bf16 {
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
            out.vo = Some(Linear::new(gpu.upload(w, &[D, MEM_D])?, Some(gpu.upload(b, &[D])?)).bf16(gpu)?);
        }
        Ok(out)
    }

    /// `residual += o_proj(attn(rope(q_in Wq), rope(k_in Wk), v_in Wv))`
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
        let (q, k, v) = (self.q.forward(gpu, q_in)?, self.k.forward(gpu, k_in)?, self.v.forward(gpu, v_in)?);
        let attn = if gpu.precision() == Precision::Bf16 {
            // FP16 operands for the fused kernel, rounded once here rather than by each of
            // the 81 query blocks that read every key
            let (q16, k16, v16) =
                (rope.apply_f16(gpu, &q, q.rows())?, rope.apply_f16(gpu, &k, rope_keys)?, rope.apply_f16(gpu, &v, 0)?);
            attention_f16_d256(gpu, &q16, &k16, &v16, q.rows(), k.rows())?
        } else {
            let (q, k) = (rope.apply(gpu, &q, q.rows())?, rope.apply(gpu, &k, rope_keys)?);
            mha(gpu, (q.ptr(), D), (k.ptr(), D), (v.ptr(), D), q.rows(), k.rows(), 1, D, Mask::default())?
        };
        self.o.forward_residual(gpu, &attn, residual)
    }

    /// `residual += o_proj(attn(rope(q_in Wq), rope(k_in Wk), memory Wv))` in BF16 mode,
    /// computed as `(P memory) (Wo Wv)^T + Wo bv + bo`: softmax rows sum to 1, so the
    /// value projection's bias passes through P unchanged and both projections fold into
    /// one 64 -> 256 linear after the attention, which then runs on 64-wide values.
    /// `memory16` is the memory `[L, 64]` in FP16
    #[allow(clippy::too_many_arguments)]
    fn forward_residual_folded(
        &self,
        gpu: &Gpu,
        rope: &Rope,
        q_in: &Tensor,
        k_in: &Tensor,
        memory16: &Buf,
        rope_keys: usize,
        residual: &Tensor,
    ) -> Result<()> {
        let k = self.k.forward(gpu, k_in)?;
        let k16 = rope.apply_f16(gpu, &k, rope_keys)?;
        self.attend_folded(gpu, &self.query16(gpu, rope, q_in)?, &k16, memory16, k.rows(), residual)
    }

    /// The rotated queries `rope(q_in Wq)` in FP16, `[5184, 256]`
    fn query16(&self, gpu: &Gpu, rope: &Rope, q_in: &Tensor) -> Result<Buf> {
        let q = self.q.forward(gpu, q_in)?;
        rope.apply_f16(gpu, &q, q.rows())
    }

    /// [`RopeAttention::forward_residual_folded`] from queries and keys already
    /// projected, rotated and in FP16: `q16 [5184, 256]`, `k16 [lk, 256]`
    fn attend_folded(
        &self,
        gpu: &Gpu,
        q16: &Buf,
        k16: &Buf,
        memory16: &Buf,
        lk: usize,
        residual: &Tensor,
    ) -> Result<()> {
        let vo = self.vo.as_ref().expect("folded projection loaded in BF16 mode");
        let attn = attention_f16_qk256_v64(gpu, q16, k16, memory16, TOKENS, lk)?;
        vo.forward_residual(gpu, &attn, residual)
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
        Ok(Self { cos: gpu.upload(cos, &[TOKENS, D / 2])?, sin: gpu.upload(sin, &[TOKENS, D / 2])? })
    }

    /// `x [n, 256]` with its first `rows` rows rotated at positions `row % 5184`
    fn apply(&self, gpu: &Gpu, x: &Tensor, rows: usize) -> Result<Tensor> {
        let out = gpu.empty(x.shape())?;
        launch!(gpu, "rope_rows", ew(out.len()), B256, 0; &out, x, &self.cos, &self.sin, D, rows)?;
        Ok(out)
    }

    /// [`Rope::apply`] into FP16; `rows = 0` only converts
    fn apply_f16(&self, gpu: &Gpu, x: &Tensor, rows: usize) -> Result<Buf> {
        let out = gpu.alloc_bytes(x.len() * 2)?;
        launch!(gpu, "rope_rows_f16", ew(x.len() / 2), B256, 0; out.slice::<u32>(), x, &self.cos, &self.sin, D, rows)?;
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

/// One memory frame's keys and values for one object in BF16 mode, made once when the
/// frame joins the bank and kept while memory selection can still pick it
pub(crate) struct FrameKeys {
    /// Per layer, `rope(Wk (m + pos) + bk)` in FP16 `[5184, 256]`: the key without the
    /// slot's temporal encoding, which [`MemoryAttention::forward_cached`] adds
    keys: Vec<Buf>,
    /// The memory in FP16 `[5184, 64]`
    values: Buf,
}

/// Layer 0 up to its cross-attention, which depends only on the frame and so is shared by
/// every object (BF16 mode): the residual stream after layer 0's self-attention, and the
/// cross-attention's rotated FP16 queries
pub(crate) struct FrameStart {
    x: Tensor,
    q16: Buf,
}

/// What each layer's cross-attention reads
enum Bank<'a> {
    /// The assembled memory and its position, projected by every layer each frame
    Assembled { keys: Tensor, memory: &'a Tensor, memory16: Option<Buf>, rope_keys: usize },
    /// Per layer FP16 keys `[lk, 256]` and the FP16 values `[lk, 64]` (BF16 mode)
    Cached { keys: Vec<Buf>, values: Buf, lk: usize },
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
    /// The memory's sine position `[5184, 64]`, the same for every frame
    mem_sine: Tensor,
    /// Per layer, `Wk tpos_j` for each temporal encoding j, `[7, 256]` (BF16 mode)
    tpos_keys: Vec<Tensor>,
}

impl MemoryAttention {
    fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "tracker_model.memory_attention";
        let layers = (0..4)
            .map(|i| {
                let l = format!("{p}.layers.{i}");
                let ln = |n: &str| LayerNorm::load(gpu, wts, &format!("{l}.{n}"), 1e-5);
                Ok(MemoryLayer {
                    norm1: ln("layer_norm1")?,
                    self_attn: RopeAttention::load(gpu, wts, &format!("{l}.self_attn"))?,
                    norm2: ln("layer_norm2")?,
                    cross_attn: RopeAttention::load_cross(gpu, wts, &format!("{l}.cross_attn_image"))?,
                    norm3: ln("layer_norm3")?,
                    linear1: Linear::load(gpu, wts, &format!("{l}.linear1"))?.bf16(gpu)?,
                    linear2: Linear::load(gpu, wts, &format!("{l}.linear2"))?.bf16(gpu)?,
                })
            })
            .collect::<Result<_>>()?;
        // f32 products, as `output + 0.1 * src_pos` rounds them
        let src_pos = sine_2d(GRID, D).into_iter().map(|v| 0.1 * v).collect();
        let mut tpos_keys = Vec::new();
        if gpu.precision() == Precision::Bf16 {
            let (tpos, _) = wts.host("tracker_model.memory_temporal_positional_encoding")?;
            ensure!(tpos.len() == NUM_MASKMEM * MEM_D, "{} temporal encoding values", tpos.len());
            for i in 0..4 {
                let (wk, shape) = wts.host(&format!("{p}.layers.{i}.cross_attn_image.k_proj.weight"))?;
                ensure!(shape == [D, MEM_D], "k_proj {shape:?}");
                // f64 sums, rounded once; no bias, which the cached keys carry
                let c: Vec<f32> = tpos
                    .chunks(MEM_D)
                    .flat_map(|t| {
                        let wk = &wk;
                        (0..D).map(move |o| {
                            wk[o * MEM_D..][..MEM_D].iter().zip(t).map(|(&w, &x)| w as f64 * x as f64).sum::<f64>()
                                as f32
                        })
                    })
                    .collect();
                tpos_keys.push(gpu.upload(c, &[NUM_MASKMEM, D])?);
            }
        }
        Ok(Self {
            layers,
            norm: LayerNorm::load(gpu, wts, &format!("{p}.layer_norm"), 1e-5)?,
            rope: Rope::new(gpu)?,
            src_pos: gpu.upload(src_pos, &[TOKENS, D])?,
            mem_sine: gpu.upload(sine_2d(GRID, MEM_D), &[TOKENS, MEM_D])?,
            tpos_keys,
        })
    }

    /// The keys and values of one object's memory frame `mem [5184, 64]`, made once for
    /// every frame that will attend to it (BF16 mode)
    pub(crate) fn frame_keys(&self, gpu: &Gpu, mem: &Tensor) -> Result<FrameKeys> {
        assert_eq!(mem.shape(), &[TOKENS, MEM_D]);
        let keys_in = add(gpu, mem, &self.mem_sine)?;
        let keys = self
            .layers
            .iter()
            .map(|l| self.rope.apply_f16(gpu, &l.cross_attn.k.forward(gpu, &keys_in)?, TOKENS))
            .collect::<Result<_>>()?;
        Ok(FrameKeys { keys, values: self.rope.apply_f16(gpu, mem, 0)? })
    }

    /// [`MemoryAttention::forward`] in BF16 mode from cached frames: `slots` are the
    /// selected memory frames with their temporal encoding index, oldest first, and
    /// `ptr [P, 64]` the object pointer tokens with their positions `ptr_pos`. Each
    /// layer's keys are the frames' cached keys plus the rotated `Wk tpos`, so only the
    /// pointer tokens are projected here
    pub(crate) fn forward_cached(
        &self,
        gpu: &Gpu,
        start: &FrameStart,
        slots: &[(&FrameKeys, usize)],
        ptr: &Tensor,
        ptr_pos: &Tensor,
    ) -> Result<Tensor> {
        assert_eq!(ptr.shape(), ptr_pos.shape());
        let p = ptr.rows();
        let lk = slots.len() * TOKENS + p;
        ensure!(lk > 0, "no memory to attend to");
        let frame_bytes = |w: usize| TOKENS * w * 2;
        let values = gpu.alloc_bytes(lk * MEM_D * 2)?;
        for (i, (f, _)) in slots.iter().enumerate() {
            gpu.copy_dtod(f.values.ptr(), values.ptr() + (i * frame_bytes(MEM_D)) as u64, frame_bytes(MEM_D))?;
        }
        let tail = |buf: &Buf, w: usize| buf.ptr() + (slots.len() * frame_bytes(w)) as u64;
        let ptr_keys_in = if p > 0 {
            let v16 = self.rope.apply_f16(gpu, ptr, 0)?;
            gpu.copy_dtod(v16.ptr(), tail(&values, MEM_D), p * MEM_D * 2)?;
            Some(add(gpu, ptr, ptr_pos)?)
        } else {
            None
        };

        let mut keys = Vec::with_capacity(self.layers.len());
        for (l, tpos_keys) in self.layers.iter().zip(&self.tpos_keys) {
            let k = gpu.alloc_bytes(lk * D * 2)?;
            for (i, (f, tpos)) in slots.iter().enumerate() {
                let out = Arg::slice(k.ptr() + (i * frame_bytes(D)) as u64, TOKENS * D / 2);
                let c = Arg::slice(tpos_keys.at(tpos * D), D);
                launch!(gpu, "slot_keys_f16", ew(TOKENS * D / 2), B256, 0;
                    out, f.keys[keys.len()].slice::<u32>(), c, &self.rope.cos, &self.rope.sin)?;
            }
            if let Some(ptr_keys_in) = &ptr_keys_in {
                // pointer tokens get no RoPE
                let k16 = self.rope.apply_f16(gpu, &l.cross_attn.k.forward(gpu, ptr_keys_in)?, 0)?;
                gpu.copy_dtod(k16.ptr(), tail(&k, D), p * D * 2)?;
            }
            keys.push(k);
        }
        // the residual stream changes in place, and other objects start from it too
        self.layers_from(gpu, gpu.clone_tensor(&start.x)?, Some(&start.q16), &Bank::Cached { keys, values, lk })
    }

    /// [`FrameStart`] for frame features `src [5184, 256]`
    pub(crate) fn frame_start(&self, gpu: &Gpu, src: &Tensor) -> Result<FrameStart> {
        assert_eq!(src.shape(), &[TOKENS, D]);
        let x = add(gpu, src, &self.src_pos)?;
        let l = &self.layers[0];
        let h = l.norm1.forward(gpu, &x)?;
        l.self_attn.forward_residual(gpu, &self.rope, &h, &h, &h, TOKENS, &x)?;
        let q16 = l.cross_attn.query16(gpu, &self.rope, &l.norm2.forward(gpu, &x)?)?;
        Ok(FrameStart { x, q16 })
    }

    /// Frame features `src [5184, 256]` conditioned on `memory [L, 64]` (its positional
    /// encoding `memory_pos` alongside), whose last `ptr_tokens` rows are object pointer
    /// tokens without RoPE
    pub fn forward(
        &self,
        gpu: &Gpu,
        src: &Tensor,
        memory: &Tensor,
        memory_pos: &Tensor,
        ptr_tokens: usize,
    ) -> Result<Tensor> {
        assert_eq!(memory.shape(), memory_pos.shape());
        // keys see the memory plus its position, values the memory alone
        let keys = add(gpu, memory, memory_pos)?;
        let rope_keys = memory.rows() - ptr_tokens;
        // the values every layer reads, FP16 once for all four: exact for the spatial
        // memory, which is BF16-valued and within FP16's normal range, and one rounding
        // for the pointer tokens
        let memory16 =
            if gpu.precision() == Precision::Bf16 { Some(self.rope.apply_f16(gpu, memory, 0)?) } else { None };
        self.run(gpu, src, &Bank::Assembled { keys, memory, memory16, rope_keys })
    }

    /// The four layers and the final norm over `src`, cross-attending to `bank`
    fn run(&self, gpu: &Gpu, src: &Tensor, bank: &Bank) -> Result<Tensor> {
        assert_eq!(src.shape(), &[TOKENS, D]);
        self.layers_from(gpu, add(gpu, src, &self.src_pos)?, None, bank)
    }

    /// The layers on the residual stream `x`, from layer 0's self-attention on, or with
    /// layer 0's cross-attention queries `q0` (and `x` past its self-attention) from that
    /// cross-attention on
    fn layers_from(&self, gpu: &Gpu, x: Tensor, q0: Option<&Buf>, bank: &Bank) -> Result<Tensor> {
        let bf16 = gpu.precision() == Precision::Bf16;
        let rope = &self.rope;
        for (i, l) in self.layers.iter().enumerate() {
            let q16 = if i == 0 { q0 } else { None };
            if q16.is_none() {
                let h = l.norm1.forward(gpu, &x)?;
                l.self_attn.forward_residual(gpu, rope, &h, &h, &h, TOKENS, &x)?;
            }
            match bank {
                Bank::Assembled { keys, memory16: Some(m16), rope_keys, .. } => {
                    assert!(q16.is_none(), "an assembled bank starts at layer 0");
                    let h = l.norm2.forward(gpu, &x)?;
                    l.cross_attn.forward_residual_folded(gpu, rope, &h, keys, m16, *rope_keys, &x)?
                }
                Bank::Assembled { keys, memory, memory16: None, rope_keys } => {
                    assert!(q16.is_none(), "an assembled bank starts at layer 0");
                    let h = l.norm2.forward(gpu, &x)?;
                    l.cross_attn.forward_residual(gpu, rope, &h, keys, memory, *rope_keys, &x)?
                }
                Bank::Cached { keys, values, lk } => {
                    let own;
                    let q16 = match q16 {
                        Some(q) => q,
                        None => {
                            own = l.cross_attn.query16(gpu, rope, &l.norm2.forward(gpu, &x)?)?;
                            &own
                        }
                    };
                    l.cross_attn.attend_folded(gpu, q16, &keys[i], values, *lk, &x)?
                }
            }
            if bf16 {
                // the 2048-wide hidden stays BF16 between the two GEMMs
                let hidden = l.linear1.forward16_act16(gpu, &l.norm3.forward16(gpu, &x)?, ACT_RELU)?;
                l.linear2.forward16_residual(gpu, &hidden, &x)?;
            } else {
                let h = l.norm3.forward(gpu, &x)?;
                l.linear2.forward_residual(gpu, &l.linear1.forward_act(gpu, &h, ACT_RELU)?, &x)?;
            }
        }
        self.norm.forward(gpu, &x)
    }
}

/// Input channels from which a downsampler conv runs as im2col and one GEMM: the two
/// deeper layers (16 and 64 channels, 144 and 576 products an output) were most of the
/// direct kernel's 2.7 ms a frame, while the first two have 9 and 36
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
    norm: LayerNorm,
    cin: usize,
    cout: usize,
}

/// A ConvNeXt block of the memory fuser, with its layer scale folded into `fc2`
struct CxBlock {
    /// `[256, 1, 7, 7]`
    dw_w: Tensor,
    dw_b: Tensor,
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
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
                norm: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm"), 1e-6)?,
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
            for (o, g) in gamma.iter().enumerate() {
                w2[o * shape[1]..][..shape[1]].iter_mut().for_each(|v| *v *= g);
                b2[o] *= g;
            }
            fuser.push(CxBlock {
                dw_w: wts.get(gpu, &format!("{l}.depthwise_conv.weight"))?,
                dw_b: wts.get(gpu, &format!("{l}.depthwise_conv.bias"))?,
                norm: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm"), 1e-6)?,
                fc1: Linear::load(gpu, wts, &format!("{l}.pointwise_conv1"))?,
                fc2: Linear::new(gpu.upload(w2, &shape)?, Some(gpu.upload(b2, &[shape[0]])?)),
            });
        }
        Ok(Self {
            resize: AaResize::new(gpu, (IMAGE_SIZE, IMAGE_SIZE), (MASK_SIDE, MASK_SIDE))?,
            down,
            final_conv: Linear::load(gpu, wts, &format!("{p}.mask_downsampler.final_conv"))?,
            pix_proj: Linear::load(gpu, wts, &format!("{p}.feature_projection"))?,
            fuser,
            out_proj: Linear::load(gpu, wts, &format!("{p}.projection"))?,
        })
    }

    /// Frame features `pix_feat [5184, 256]` and the mask input `mask [1008 * 1008]`
    /// (already through [`mask_for_mem`]) -> memory `[5184, 64]`
    pub fn forward(&self, gpu: &Gpu, pix_feat: &Tensor, mask: &Tensor) -> Result<Tensor> {
        assert_eq!(mask.len(), IMAGE_SIZE * IMAGE_SIZE);
        let mut x = self.resize.forward(gpu, mask)?.reshape(&[MASK_SIDE * MASK_SIDE, 1]);
        let mut side = MASK_SIDE;
        for l in &self.down {
            side /= 2;
            let y = match &l.conv {
                DownConv::Direct(w, b) => {
                    let y = gpu.empty(&[side * side, l.cout])?;
                    launch!(gpu, "conv3x3_s2", ew(y.len()), B256, 0; &y, &x, w, b, 2 * side, l.cin)?;
                    y
                }
                DownConv::Gemm(conv) => {
                    let cols = gpu.empty(&[side * side, l.cin * 9])?;
                    launch!(gpu, "im2col3x3_s2", ew(cols.len()), B256, 0; &cols, &x, 2 * side, l.cin)?;
                    conv.forward(gpu, &cols)?
                }
            };
            let n = &l.norm;
            launch!(gpu, "layernorm_act_rows", ew(side * side), B256, 0; &y, &n.w, &n.b, l.cout, n.eps, ACT_GELU)?;
            x = y;
        }
        assert_eq!(side, GRID);
        let x = self.final_conv.forward(gpu, &x)?;
        self.pix_proj.forward_residual(gpu, pix_feat, &x)?;
        for b in &self.fuser {
            let y = gpu.empty(x.shape())?;
            launch!(gpu, "dwconv7x7", ew(y.len()), B256, 0; &y, &x, &b.dw_w, &b.dw_b, GRID)?;
            b.norm.forward_inplace(gpu, &y)?;
            b.fc2.forward_residual(gpu, &b.fc1.forward_act(gpu, &y, ACT_GELU)?, &x)?;
        }
        self.out_proj.forward(gpu, &x)
    }
}

/// `x [n] -> (binarize ? x > 0 : sigmoid(x)) * 20 - 10`, the memory encoder's mask input
pub fn mask_for_mem(gpu: &Gpu, x: &Tensor, binarize: bool) -> Result<Tensor> {
    let out = gpu.empty(x.shape())?;
    launch!(gpu, "mask_for_mem", ew(out.len()), B256, 0; &out, x, binarize as i32)?;
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
                gpu.upload(pos, &[TOKENS, MEM_D])
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(mem_pos.len() == NUM_MASKMEM, "{} temporal encodings, expected {NUM_MASKMEM}", mem_pos.len());
        let no_obj_spatial = wts.get(gpu, &format!("{p}.occlusion_spatial_embedding_parameter"))?;
        Ok(Self {
            mem_attn: MemoryAttention::load(gpu, wts)?,
            mem_enc: MemoryEncoder::load(gpu, wts)?,
            obj_ptr_proj: Mlp::load_in_out(gpu, wts, &format!("{p}.object_pointer_proj"), 3, ACT_RELU)?,
            tpos_proj: Linear::load(gpu, wts, &format!("{p}.temporal_positional_encoding_projection_layer"))?,
            no_obj_ptr: wts.get(gpu, &format!("{p}.no_object_pointer"))?,
            no_obj_spatial: no_obj_spatial.reshape(&[MEM_D]),
            mem_pos,
        })
    }

    /// Object pointer from the chosen mask token, or `no_obj_ptr` for an absent object
    pub fn pointer(&self, gpu: &Gpu, token: &Tensor, present: bool) -> Result<Tensor> {
        if present { self.obj_ptr_proj.forward(gpu, token) } else { gpu.clone_tensor(&self.no_obj_ptr) }
    }

    /// The memory of one object on one frame: the encoder's output, plus
    /// `no_obj_embed_spatial` for an absent object, rounded to BF16 as upstream stores it
    pub fn memory(&self, gpu: &Gpu, pix_feat: &Tensor, mask: &Tensor, present: bool) -> Result<Tensor> {
        let mem = self.mem_enc.forward(gpu, pix_feat, mask)?;
        if !present {
            launch!(gpu, "add_row", ew(mem.len()), B256, 0; &mem, &self.no_obj_spatial)?;
        }
        launch!(gpu, "round_bf16", ew(mem.len()), B256, 0; &mem)?;
        Ok(mem)
    }

    /// `obj_ptr_tpos_proj(get_1d_sine_pe(pos / (max_abs_pos - 1), 256))` per pointer
    fn pointer_pos(&self, gpu: &Gpu, positions: &[usize], max_abs_pos: usize) -> Result<Tensor> {
        let t_max = max_abs_pos.saturating_sub(1).max(1) as f32;
        let half = D / 2;
        let dims: Vec<f32> = (0..half).map(|i| 10000f32.powf((2 * (i / 2)) as f32 / half as f32)).collect();
        let mut pe = Vec::with_capacity(positions.len() * D);
        for &p in positions {
            let v = p as f32 / t_max;
            pe.extend(dims.iter().map(|d| ((v / d) as f64).sin() as f32));
            pe.extend(dims.iter().map(|d| ((v / d) as f64).cos() as f32));
        }
        self.tpos_proj.forward(gpu, &gpu.upload(pe, &[positions.len(), D])?)
    }
}

/// One frame's memory for every object, and its object pointers
struct MemFrame {
    frame: usize,
    /// `[5184, 64]` per object
    mem: Vec<Tensor>,
    /// `[1, 256]` per object
    ptr: Vec<Tensor>,
    /// Per object, the memory attention's keys and values for this frame in BF16 mode;
    /// emptied once memory selection can no longer pick the frame
    keys: Vec<FrameKeys>,
}

/// The memory frames and object pointers upstream's selection gives one frame
pub(crate) struct Selection<'a> {
    /// Memory frames with their temporal encoding index, in bank order
    frames: Vec<(&'a MemFrame, usize)>,
    /// Frames whose object pointers join the memory
    ptr_frames: Vec<&'a MemFrame>,
    /// `[ptr_frames, 64]`, each pointer's temporal position
    ptr_pos: Tensor,
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
    /// the last frames (up to `min(num_frames, 16) - 1`) whose score passed
    /// [`MEMORY_SCORE_THRESHOLD`], oldest first
    passing: VecDeque<MemFrame>,
    /// the previous frame when its score didn't pass: upstream always keeps it
    prev: Option<MemFrame>,
}

/// Features of one video frame
pub(crate) struct FrameFeatures {
    pub(crate) point: PointImage,
    /// tracker neck 72x72 level `[5184, 256]`, without the no-memory embedding
    pub(crate) fpn2: Tensor,
}

impl Sam3 {
    /// Backbone and tracker neck of one video frame
    pub(crate) fn video_features(&self, pixels: &Tensor) -> Result<FrameFeatures> {
        let gpu = &self.gpu;
        let trunk = {
            let _r = range(c"backbone");
            self.backbone.forward(gpu, pixels)?
        };
        let _r = range(c"neck");
        let mut fpn = self.tracker_neck.forward(gpu, &Trunk::new(gpu, trunk)?)?;
        let point = self.point.prepare_image(gpu, &fpn)?;
        Ok(FrameFeatures { point, fpn2: fpn.pop().expect("three levels") })
    }

    /// Starts tracking objects through a video of `num_frames` frames: segments each
    /// object from its clicks on `first` (frame 0), whose masks come back with the tracker
    ///
    /// Each entry of `objects` is one object's clicks, in `first`'s pixels; past 16, only
    /// the first and last 8 reach the decoder, as upstream. Upstream
    /// scales the object pointers' temporal positions by the video's length, so
    /// `num_frames` must be the real frame count; [`VideoTracker::next`] stops after it
    pub fn track(
        &self,
        first: &RgbImage,
        objects: &[Vec<PointPrompt>],
        num_frames: usize,
    ) -> Result<(VideoTracker<'_>, TrackedFrame)> {
        self.gpu.bind()?;
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
        let gpu = &self.model.gpu;
        gpu.bind()?;
        let pixels = preprocess(gpu, frame)?;
        self.next_pixels(&pixels)
    }

    /// The clicked frame: each object decoded from its clicks without memory, then the
    /// memory encoder over all objects' masks with overlaps resolved, as upstream's
    /// preflight consolidation does
    fn prompt_frame(&mut self, pixels: &Tensor, objects: &[Vec<PointPrompt>]) -> Result<TrackedFrame> {
        let (m, gpu) = (self.model, &self.model.gpu);
        let feats = self.model.video_features(pixels)?;
        let side = LOW_RES * LOW_RES;
        let low_res = gpu.empty(&[objects.len(), side])?;
        let (mut scores, mut ptrs, mut picks) = (Vec::new(), Vec::new(), Vec::new());
        for (b, points) in objects.iter().enumerate() {
            let _r = range(c"decoder");
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

        let _r = range(c"memory_encoder");
        let high = upsample(gpu, (&low_res).into(), objects.len(), IMAGE_SIZE, IMAGE_SIZE)?;
        let high = if objects.len() > 1 {
            let out = gpu.empty(high.shape())?;
            launch!(gpu, "nonoverlap", ew(out.len()), B256, 0; &out, &high, objects.len())?;
            out
        } else {
            high
        };
        let input = mask_for_mem(gpu, &high, true)?;
        let n = IMAGE_SIZE * IMAGE_SIZE;
        let mut mem = Vec::new();
        for (b, &score) in scores.iter().enumerate() {
            let mask = gpu.empty(&[n])?;
            gpu.copy_dtod(input.at(b * n), mask.ptr(), n * 4)?;
            mem.push(m.tracker.memory(gpu, &feats.fpn2, &mask, score > 0.0)?);
        }
        let keys = self.frame_keys(&mem)?;
        self.cond.push(MemFrame { frame: 0, mem, ptr: ptrs, keys });
        self.output(0, low_res, &scores, picks, 0, 0)
    }

    /// Writes mask `index` of a decoder pass into row `b` of `low_res`, or
    /// `NO_OBJ_SCORE` everywhere for an absent object
    fn select_mask(&self, dec: &Decoded, index: usize, present: bool, low_res: &Tensor, b: usize) -> Result<()> {
        let gpu = &self.model.gpu;
        let side = LOW_RES * LOW_RES;
        let first = 4 - dec.low_res.rows();
        gpu.copy_dtod(dec.low_res.at((index - first) * side), low_res.at(b * side), side * 4)?;
        if !present {
            launch!(gpu, "clamp", ew(side), B256, 0; Arg::slice(low_res.at(b * side), side), NO_OBJ_SCORE, NO_OBJ_SCORE)?;
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

    /// Object `b`'s pointer tokens `[4 * pointers, 64]` and their positions: each 256-d
    /// pointer is 4 tokens of 64, all at the pointer's position
    fn pointer_tokens(&self, sel: &Selection, b: usize) -> Result<(Tensor, Tensor)> {
        let gpu = &self.model.gpu;
        let split = D / MEM_D;
        let rows = sel.ptr_frames.len() * split;
        let (ptr, pos) = (gpu.empty(&[rows, MEM_D])?, gpu.empty(&[rows, MEM_D])?);
        for (i, f) in sel.ptr_frames.iter().enumerate() {
            gpu.copy_dtod(f.ptr[b].ptr(), ptr.at(i * D), D * 4)?;
            for s in 0..split {
                gpu.copy_dtod(sel.ptr_pos.at(i * MEM_D), pos.at(i * D + s * MEM_D), MEM_D * 4)?;
            }
        }
        Ok((ptr, pos))
    }

    /// Memory-attention inputs for the next frame: each object's memory `[L, 64]` and its
    /// positional encoding, which select from the bank as upstream's
    /// `_prepare_memory_conditioned_features` does
    pub(crate) fn memory_inputs(&self) -> Result<MemoryInputs> {
        let (m, gpu) = (self.model, &self.model.gpu);
        let sel = self.selection()?;
        let ptr_tokens = sel.ptr_tokens();
        let mem_tokens = sel.frames.len() * TOKENS + ptr_tokens;

        let mut objects = Vec::with_capacity(self.objects);
        for b in 0..self.objects {
            let memory = gpu.empty(&[mem_tokens, MEM_D])?;
            let memory_pos = gpu.empty(&[mem_tokens, MEM_D])?;
            let bytes = TOKENS * MEM_D * 4;
            for (i, (f, tpos)) in sel.frames.iter().enumerate() {
                gpu.copy_dtod(f.mem[b].ptr(), memory.at(i * TOKENS * MEM_D), bytes)?;
                gpu.copy_dtod(m.tracker.mem_pos[*tpos].ptr(), memory_pos.at(i * TOKENS * MEM_D), bytes)?;
            }
            let base = sel.frames.len() * TOKENS * MEM_D;
            let (ptr, pos) = self.pointer_tokens(&sel, b)?;
            gpu.copy_dtod(ptr.ptr(), memory.at(base), ptr_tokens * MEM_D * 4)?;
            gpu.copy_dtod(pos.ptr(), memory_pos.at(base), ptr_tokens * MEM_D * 4)?;
            objects.push((memory, memory_pos));
        }
        Ok(MemoryInputs { objects })
    }

    /// Object `b`'s memory attention over the bank from the frames' cached keys (BF16
    /// mode)
    pub(crate) fn memory_attention_cached(&self, sel: &Selection, b: usize, start: &FrameStart) -> Result<Tensor> {
        let slots: Vec<(&FrameKeys, usize)> = sel
            .frames
            .iter()
            .map(|(f, tpos)| {
                let keys = f.keys.get(b).ok_or_else(|| eyre!("frame {} has no cached keys", f.frame))?;
                Ok((keys, *tpos))
            })
            .collect::<Result<_>>()?;
        let (ptr, pos) = self.pointer_tokens(sel, b)?;
        self.model.tracker.mem_attn.forward_cached(&self.model.gpu, start, &slots, &ptr, &pos)
    }

    /// Every object's [`FrameKeys`] for a new memory frame in BF16 mode, else none
    fn frame_keys(&self, mems: &[Tensor]) -> Result<Vec<FrameKeys>> {
        let gpu = &self.model.gpu;
        if gpu.precision() != Precision::Bf16 {
            return Ok(Vec::new());
        }
        mems.iter().map(|mem| self.model.tracker.mem_attn.frame_keys(gpu, mem)).collect()
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
        // BF16 reads each frame's cached keys; the other modes assemble the memory
        let mut inputs =
            if gpu.precision() == Precision::Bf16 { None } else { Some(self.memory_inputs()?.objects.into_iter()) };
        // layer 0 up to its cross-attention is the same for every object
        let start = if inputs.is_none() {
            let _r = range(c"memory_attention");
            Some(m.tracker.mem_attn.frame_start(gpu, &feats.fpn2)?)
        } else {
            None
        };

        let side = LOW_RES * LOW_RES;
        let low_res = gpu.empty(&[self.objects, side])?;
        let (mut scores, mut ptrs, mut mems, mut picks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut norm_sum, mut iou_sum) = (0f32, 0f32);
        for b in 0..self.objects {
            let embed = {
                let _r = range(c"memory_attention");
                match inputs.as_mut().and_then(Iterator::next) {
                    Some((memory, memory_pos)) => {
                        m.tracker.mem_attn.forward(gpu, &feats.fpn2, &memory, &memory_pos, ptr_tokens)?
                    }
                    None => self.memory_attention_cached(&sel, b, start.as_ref().expect("BF16 frame start"))?,
                }
            };
            let _r = range(c"decoder");
            let dec = m.point.decode(gpu, &embed, &feats.point, &[], (self.width, self.height), None, 1)?;
            let score = gpu.download(&m.point.obj_score(gpu, &dec)?)?[0];
            let iou = gpu.download(&dec.iou)?;
            let best = 1 + argmax(&iou[1..]);
            let present = score > 0.0;
            self.select_mask(&dec, best, present, &low_res, b)?;
            ptrs.push(m.tracker.pointer(gpu, &row(gpu, &dec.tokens, MASK_TOKEN_ROW + best)?, present)?);

            let _r = range(c"memory_encoder");
            let high = upsample(gpu, Arg::slice(low_res.at(b * side), side), 1, IMAGE_SIZE, IMAGE_SIZE)?;
            let input = mask_for_mem(gpu, &high, false)?;
            mems.push(m.tracker.memory(gpu, &feats.fpn2, &input, present)?);
            // `cal_mem_score`: present objects' rescaled score and their best IoU
            norm_sum += if present { 2.0 / (1.0 + (-score).exp()) - 1.0 } else { 0.0 };
            iou_sum += iou[best];
            scores.push(score);
            picks.push((iou, best));
        }
        // upstream multiplies scores shaped [B, 1] by IoUs shaped [B], so its mean over the
        // broadcast [B, B] product is mean(score) * mean(IoU), not the mean of per-object
        // products; with several objects the two land on opposite sides of the threshold
        let n = self.objects as f32;
        let eff = (norm_sum / n) * (iou_sum / n);

        drop(sel);
        let keys = self.frame_keys(&mems)?;
        let frame = MemFrame { frame: t, mem: mems, ptr: ptrs, keys };
        self.prev = None;
        if eff > MEMORY_SCORE_THRESHOLD {
            self.passing.push_back(frame);
            // `frame_filter` stops at `min(num_frames, 16) - 1` passing frames
            if self.passing.len() >= max_abs_pos {
                self.passing.pop_front();
            }
            // the memory takes the last 6 selected frames, and passing frames only leave
            // that window: older ones keep their pointers but drop their keys
            let window = self.passing.len().saturating_sub(NUM_MASKMEM - 1);
            for f in self.passing.iter_mut().take(window) {
                f.keys.clear();
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
        let _r = range(c"postprocess");
        let (w, h) = (self.width, self.height);
        let up = upsample(gpu, (&low_res).into(), self.objects, h, w)?;
        let mask = gpu.alloc_bytes(up.len())?;
        launch!(gpu, "threshold_u8", ew(up.len()), B256, 0; mask.slice::<u8>(), &up, 0.0f32)?;
        let mut all = vec![0u8; self.objects * w * h];
        gpu.download_into(&mask, &mut all)?;
        let masks: Vec<Vec<u8>> = all.chunks_exact(w * h).map(<[u8]>::to_vec).collect();
        let logits = gpu.download(&low_res)?;
        let objects = scores
            .iter()
            .zip(masks)
            .zip(logits.chunks(LOW_RES * LOW_RES))
            .zip(picks)
            .map(|(((&score, mask), low_res), (ious, chosen))| TrackedObject {
                score,
                mask,
                low_res: low_res.to_vec(),
                ious,
                chosen,
            })
            .collect();
        Ok(TrackedFrame { frame, objects, memory_tokens: mem_tokens, pointer_tokens: ptr_tokens })
    }
}

/// Bilinear upsampling (`align_corners=False`) of `[n, 288^2]` logits to `[n, h * w]`
fn upsample(gpu: &Gpu, low_res: Arg, n: usize, h: usize, w: usize) -> Result<Tensor> {
    let out = gpu.empty(&[n, h * w])?;
    launch!(gpu, "resize_bilinear", ew(out.len()), B256, 0; &out, low_res, LOW_RES, LOW_RES, h, w)?;
    Ok(out)
}

/// Upstream's `_get_stability_scores` at delta 0.05: the area above +delta over the area
/// above -delta
fn stability(logits: &[f32]) -> f32 {
    let inter = logits.iter().filter(|&&v| v > 0.05).count();
    let union = logits.iter().filter(|&&v| v > -0.05).count();
    if union > 0 { inter as f32 / union as f32 } else { 1.0 }
}

/// First index of the maximum, like `torch.argmax`
/// All the clicks, or the first and last half of [`MAX_CLICKS`] when there are more
fn cap_clicks(points: &[PointPrompt]) -> Vec<PointPrompt> {
    if points.len() <= MAX_CLICKS {
        return points.to_vec();
    }
    let first = MAX_CLICKS / 2;
    points[..first].iter().chain(&points[points.len() - (MAX_CLICKS - first)..]).copied().collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
