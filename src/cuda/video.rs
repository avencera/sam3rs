//! The video tracker's stages on CUDA: memory attention, memory encoder and the object
//! pointer heads, plus the per-frame steps [`crate::tracker`] runs them through. The bank
//! is stored in BF16 in every mode, and BF16 mode keeps each bank frame's memory-attention
//! keys while memory selection can still pick it

use eyre::{Result, ensure, eyre};

use crate::cuda::gpu::{Arg, Buf, Gpu, Tensor, launch};
use crate::cuda::image::AaResize;
use crate::cuda::ops::{
    Act, B256, LayerNorm, Linear, Mask, Mlp, add, attention_f16_d256, attention_f16_qk256_v64, ew, mha,
};
use crate::cuda::pipeline::Cuda;
use crate::cuda::point::PointImage;
use crate::cuda::vision::Trunk;
use crate::host::{fold_layer_scale, fold_value_out, memory_positions};
use crate::model::point::LOW_RES;
use crate::model::tracker::{D, MASK_SIDE, MEM_D, NUM_MASKMEM, TOKENS};
use crate::model::vision::{GRID, IMAGE_SIZE};
use crate::nvtx::range;
use crate::posenc::{memory_rope, memory_src_pos, pointer_sine, sine_2d};
use crate::precision::Precision;
use crate::tracker::Selection;
use crate::weights::Weights;

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
            let (w, b) = fold_value_out(wts, prefix)?;
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

/// Axial RoPE angles for the 72x72 grid at head dim 256 (see [`memory_rope`])
struct Rope {
    /// `[TOKENS, 128]`
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(gpu: &Gpu) -> Result<Self> {
        let (cos, sin) = memory_rope();
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
            src_pos: gpu.upload(memory_src_pos(), &[TOKENS, D])?,
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
                let hidden = l.linear1.forward16_act16(gpu, &l.norm3.forward16(gpu, &x)?, Act::Relu)?;
                l.linear2.forward16_residual(gpu, &hidden, &x)?;
            } else {
                let h = l.norm3.forward(gpu, &x)?;
                l.linear2.forward_residual(gpu, &l.linear1.forward_act(gpu, &h, Act::Relu)?, &x)?;
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
            let (w2, shape, b2) = fold_layer_scale(wts, &l)?;
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
            launch!(gpu, "layernorm_act_rows", ew(side * side), B256, 0; &y, &n.w, &n.b, l.cout, n.eps, Act::Gelu)?;
            x = y;
        }
        assert_eq!(side, GRID);
        let x = self.final_conv.forward(gpu, &x)?;
        self.pix_proj.forward_residual(gpu, pix_feat, &x)?;
        for b in &self.fuser {
            let y = gpu.empty(x.shape())?;
            launch!(gpu, "dwconv7x7", ew(y.len()), B256, 0; &y, &x, &b.dw_w, &b.dw_b, GRID)?;
            b.norm.forward_inplace(gpu, &y)?;
            b.fc2.forward_residual(gpu, &b.fc1.forward_act(gpu, &y, Act::Gelu)?, &x)?;
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
        let mem_pos =
            memory_positions(wts)?.into_iter().map(|pos| gpu.upload(pos, &[TOKENS, MEM_D])).collect::<Result<_>>()?;
        let no_obj_spatial = wts.get(gpu, &format!("{p}.occlusion_spatial_embedding_parameter"))?;
        Ok(Self {
            mem_attn: MemoryAttention::load(gpu, wts)?,
            mem_enc: MemoryEncoder::load(gpu, wts)?,
            obj_ptr_proj: Mlp::load_in_out(gpu, wts, &format!("{p}.object_pointer_proj"), 3, Act::Relu)?,
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
        let pe = pointer_sine(positions, max_abs_pos);
        self.tpos_proj.forward(gpu, &gpu.upload(pe, &[positions.len(), D])?)
    }
}

/// Features of one video frame
pub(crate) struct FrameFeatures {
    pub(crate) point: PointImage,
    /// Tracker neck's `fpn.coarse` `[5184, 256]`, without the no-memory embedding
    pub(crate) coarse: Tensor,
}

/// What the bank keeps of one frame for every object
pub(crate) struct FrameMemory {
    /// `[5184, 64]` per object
    pub(crate) mem: Vec<Tensor>,
    /// `[1, 256]` per object
    pub(crate) ptr: Vec<Tensor>,
    /// Per object, the memory attention's keys and values for this frame in BF16 mode;
    /// emptied once memory selection can no longer pick the frame as a memory
    pub(crate) keys: Vec<FrameKeys>,
}

/// The memory attention's state for one frame, which every object's attention reads
pub(crate) enum FrameAttention {
    /// Each object's assembled memory and its position, in object order (FP32 and TF32)
    Assembled(std::vec::IntoIter<(Tensor, Tensor)>),
    /// Layer 0's shared start and the pointers' positions, for attention over the bank
    /// frames' cached keys (BF16)
    Cached { start: FrameStart, ptr_pos: Tensor },
}

type Sel<'a> = Selection<'a, FrameMemory>;

impl Cuda {
    /// Backbone and tracker neck of one video frame
    pub(crate) fn video_features(&self, pixels: &Tensor) -> Result<FrameFeatures> {
        let gpu = &self.gpu;
        let trunk = {
            let _r = range(c"backbone");
            self.backbone.forward(gpu, pixels)?
        };
        let _r = range(c"neck");
        let fpn = self.tracker_neck.forward(gpu, &Trunk::new(gpu, trunk)?)?;
        let point = self.point.prepare_image(gpu, &fpn)?;
        Ok(FrameFeatures { point, coarse: fpn.coarse })
    }

    /// `[pointers, 64]`, each selected pointer's temporal position
    pub(crate) fn pointer_pos(&self, sel: &Sel) -> Result<Tensor> {
        self.tracker.pointer_pos(&self.gpu, &sel.positions, sel.max_abs_pos)
    }

    /// Object `b`'s pointer tokens `[4 * pointers, 64]` and their positions: each 256-d
    /// pointer is 4 tokens of 64, all at the pointer's position
    fn pointer_tokens(&self, sel: &Sel, ptr_pos: &Tensor, b: usize) -> Result<(Tensor, Tensor)> {
        let gpu = &self.gpu;
        let split = D / MEM_D;
        let rows = sel.pointers.len() * split;
        let (ptr, pos) = (gpu.empty(&[rows, MEM_D])?, gpu.empty(&[rows, MEM_D])?);
        for (i, f) in sel.pointers.iter().enumerate() {
            gpu.copy_dtod(f.memory.ptr[b].ptr(), ptr.at(i * D), D * 4)?;
            for s in 0..split {
                gpu.copy_dtod(ptr_pos.at(i * MEM_D), pos.at(i * D + s * MEM_D), MEM_D * 4)?;
            }
        }
        Ok((ptr, pos))
    }

    /// Memory-attention inputs for a frame: each object's memory `[L, 64]` and its
    /// positional encoding, from the bank's selection `sel`
    pub(crate) fn memory_inputs(&self, sel: &Sel, ptr_pos: &Tensor) -> Result<Vec<(Tensor, Tensor)>> {
        let gpu = &self.gpu;
        let (ptr_tokens, mem_tokens) = (sel.ptr_tokens(), sel.mem_tokens());
        let objects = sel.frames.first().map_or(0, |(f, _)| f.memory.mem.len());
        let mut inputs = Vec::with_capacity(objects);
        for b in 0..objects {
            let memory = gpu.empty(&[mem_tokens, MEM_D])?;
            let memory_pos = gpu.empty(&[mem_tokens, MEM_D])?;
            let bytes = TOKENS * MEM_D * 4;
            for (i, (f, tpos)) in sel.frames.iter().enumerate() {
                gpu.copy_dtod(f.memory.mem[b].ptr(), memory.at(i * TOKENS * MEM_D), bytes)?;
                gpu.copy_dtod(self.tracker.mem_pos[*tpos].ptr(), memory_pos.at(i * TOKENS * MEM_D), bytes)?;
            }
            let base = sel.frames.len() * TOKENS * MEM_D;
            let (ptr, pos) = self.pointer_tokens(sel, ptr_pos, b)?;
            gpu.copy_dtod(ptr.ptr(), memory.at(base), ptr_tokens * MEM_D * 4)?;
            gpu.copy_dtod(pos.ptr(), memory_pos.at(base), ptr_tokens * MEM_D * 4)?;
            inputs.push((memory, memory_pos));
        }
        Ok(inputs)
    }

    /// Object `b`'s memory attention over the bank from the frames' cached keys (BF16
    /// mode)
    pub(crate) fn memory_attention_cached(
        &self,
        sel: &Sel,
        ptr_pos: &Tensor,
        b: usize,
        start: &FrameStart,
    ) -> Result<Tensor> {
        let slots: Vec<(&FrameKeys, usize)> = sel
            .frames
            .iter()
            .map(|(f, tpos)| {
                let keys = f.memory.keys.get(b).ok_or_else(|| eyre!("frame {} has no cached keys", f.frame))?;
                Ok((keys, *tpos))
            })
            .collect::<Result<_>>()?;
        let (ptr, pos) = self.pointer_tokens(sel, ptr_pos, b)?;
        self.tracker.mem_attn.forward_cached(&self.gpu, start, &slots, &ptr, &pos)
    }
}

/// Bilinear upsampling (`align_corners=False`) of `[n, 288^2]` logits to `[n, h * w]`
pub(crate) fn upsample(gpu: &Gpu, low_res: Arg, n: usize, h: usize, w: usize) -> Result<Tensor> {
    let out = gpu.empty(&[n, h * w])?;
    launch!(gpu, "resize_bilinear", ew(out.len()), B256, 0; &out, low_res, LOW_RES, LOW_RES, h, w)?;
    Ok(out)
}
