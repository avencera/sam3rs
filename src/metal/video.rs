//! The video tracker's stages on Metal: memory attention, memory encoder and the object
//! pointer heads, plus the per-frame steps [`crate::tracker`] runs them through
//!
//! Everything after the tracker neck runs in FP32 in both precisions, as on the rest of
//! the Metal backend, so the memory attention assembles each frame's memory and projects
//! it in every layer rather than caching reduced-precision keys per frame

use eyre::{Result, ensure};

use crate::host::{fold_layer_scale, fold_value_out, memory_positions};
use crate::image::{Taps, aa_taps};
use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::ops::{Act, LayerNorm, Linear, Mask, Mlp, View, attention};
use crate::metal::pipeline::Metal;
use crate::metal::point::{PointImage, add, add_tiled, clone};
use crate::metal::vision::Trunk;
use crate::model::point::LOW_RES;
use crate::model::tracker::{D, MASK_SIDE, MEM_D, TOKENS};
use crate::model::vision::{GRID, IMAGE_SIZE};
use crate::posenc::{memory_rope, memory_src_pos, pointer_sine};
use crate::tracker::Selection;
use crate::weights::Weights;

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
        let (w, b) = fold_value_out(wts, prefix)?;
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

/// Axial RoPE angles for the 72x72 grid at head dim 256 (see [`memory_rope`])
struct Rope {
    /// `[TOKENS, 128]`
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(gpu: &Gpu) -> Result<Self> {
        let (cos, sin) = memory_rope();
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
        Ok(Self {
            layers,
            norm: LayerNorm::load(gpu, wts, &format!("{p}.layer_norm"), 1e-5)?,
            rope: Rope::new(gpu)?,
            src_pos: gpu.upload_f32(&memory_src_pos(), &[TOKENS, D])?,
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
            let (w2, shape, b2) = fold_layer_scale(wts, &l)?;
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
        let mem_pos =
            memory_positions(wts)?.iter().map(|pos| gpu.upload_f32(pos, &[TOKENS, MEM_D])).collect::<Result<_>>()?;
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
    pub(crate) fn pointer_pos(&self, gpu: &Gpu, positions: &[usize], max_abs_pos: usize) -> Result<Option<Tensor>> {
        if positions.is_empty() {
            return Ok(None);
        }
        let pe = pointer_sine(positions, max_abs_pos);
        Ok(Some(self.tpos_proj.forward(gpu, &gpu.upload_f32(&pe, &[positions.len(), D])?, Act::None)?))
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
}

/// The memory attention's state for one frame: each object's assembled memory and its
/// position, in object order, and the frame's features through the first layer's
/// self-attention, which every object shares
pub(crate) struct FrameAttention {
    pub(crate) inputs: std::vec::IntoIter<(Tensor, Tensor)>,
    pub(crate) prepared: Tensor,
}

type Sel<'a> = Selection<'a, FrameMemory>;

impl Metal {
    /// Backbone and tracker neck of one video frame
    pub(crate) fn video_features(&self, pixels: &Tensor) -> Result<FrameFeatures> {
        let gpu = &self.gpu;
        let trunk = self.backbone.forward(gpu, pixels)?;
        let fpn = self.tracker_neck.forward(gpu, &Trunk::new(gpu, trunk)?)?;
        let point = self.point.prepare_image(gpu, &fpn)?;
        Ok(FrameFeatures { point, coarse: fpn.coarse })
    }

    /// Memory-attention inputs of selection `sel`: each object's memory `[L, 64]` and its
    /// positional encoding, as upstream's `_prepare_memory_conditioned_features` assembles
    /// them. Each 256-d object pointer is 4 tokens of 64, all at the pointer's position
    pub(crate) fn memory_inputs(&self, sel: &Sel) -> Result<Vec<(Tensor, Tensor)>> {
        let gpu = &self.gpu;
        let ptr_pos = self.tracker.pointer_pos(gpu, &sel.positions, sel.max_abs_pos)?;
        let mem_tokens = sel.mem_tokens();
        ensure!(mem_tokens > 0, "no memory to attend to");
        let frame = TOKENS * MEM_D;
        let base = sel.frames.len() * frame;
        let split = D / MEM_D;
        let objects = sel.frames.first().map_or(0, |(f, _)| f.memory.mem.len());

        let mut inputs = Vec::with_capacity(objects);
        for b in 0..objects {
            let memory = gpu.empty(&[mem_tokens, MEM_D], DType::F32)?;
            let memory_pos = gpu.empty(&[mem_tokens, MEM_D], DType::F32)?;
            for (i, (f, tpos)) in sel.frames.iter().enumerate() {
                copy(gpu, &memory, i * frame, &f.memory.mem[b], 0, frame)?;
                copy(gpu, &memory_pos, i * frame, &self.tracker.mem_pos[*tpos], 0, frame)?;
            }
            for (i, f) in sel.pointers.iter().enumerate() {
                copy(gpu, &memory, base + i * D, &f.memory.ptr[b], 0, D)?;
                let pos = ptr_pos.as_ref().expect("positions of every pointer");
                for s in 0..split {
                    copy(gpu, &memory_pos, base + i * D + s * MEM_D, pos, i * MEM_D, MEM_D)?;
                }
            }
            inputs.push((memory, memory_pos));
        }
        Ok(inputs)
    }
}

/// `n` FP32 elements of `src` from element `from` into `dst` from element `to`
pub(crate) fn copy(gpu: &Gpu, dst: &Tensor, to: usize, src: &Tensor, from: usize, n: usize) -> Result<()> {
    ensure!(from + n <= src.len() && to + n <= dst.len(), "copy of {n} from {:?} into {:?}", src.shape(), dst.shape());
    gpu.dispatch_1d("copy_f32", n, &[dst.at(to), src.at(from)])
}

/// Bilinear upsampling (`align_corners=False`) of logits `first..first + n` of
/// `low_res [*, 288^2]` to `[n, h * w]`
pub(crate) fn upsample(gpu: &Gpu, low_res: &Tensor, first: usize, n: usize, h: usize, w: usize) -> Result<Tensor> {
    let side = LOW_RES * LOW_RES;
    ensure!((first + n) * side <= low_res.len(), "logits {first}..{} of {:?}", first + n, low_res.shape());
    let out = gpu.empty(&[n, h * w], DType::F32)?;
    let args = [out.arg(), low_res.at(first * side), LOW_RES.into(), LOW_RES.into(), h.into(), w.into()];
    gpu.dispatch_1d("resize_bilinear", out.len(), &args)?;
    Ok(out)
}
