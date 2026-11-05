//! SAM 2 style interactive decoder for point prompts on Metal: prompt encoder, two-way
//! transformer, high-resolution upscaling and the multimask heads, all in FP32

use eyre::{Result, ensure};

use crate::host::PointEncoding;
use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::ops::{Act, Attention, ConvT2x2, LayerNorm, Linear, Mask, Mlp, add_assign, bias_act};
use crate::metal::vision::Fpn;
use crate::model::point::{D, HEADS, LOW_RES, MASK_TOKEN_ROW};
use crate::model::vision::GRID;
use crate::prompt::PointPrompt;
use crate::weights::Weights;

/// Image features the point decoder reads, computed once per image
pub struct PointImage {
    /// Tracker `fpn.coarse` + no-memory embedding `[72^2, 256]`
    pub embed: Tensor,
    /// conv_s0(tracker `fpn.fine`) `[288^2, 32]`
    pub feat_s0: Tensor,
    /// conv_s1(tracker `fpn.mid`) `[144^2, 64]`
    pub feat_s1: Tensor,
}

/// One decoder pass over `K` prompts with every output still on the device
pub struct Decoded {
    /// `[K * (4 - first), 288 * 288]` logits of mask tokens `first..4`, prompt by prompt.
    /// Upstream clamps them to +-32 before reuse as a mask input, but upsamples the
    /// unclamped logits for the final masks
    pub low_res: Tensor,
    /// `[K, 4]` predicted IoU of all four mask tokens, after the sigmoid
    pub iou: Tensor,
    /// `[K * (6 + prompts), 256]` output tokens of each prompt: object score, IoU, then
    /// mask tokens 0 to 3
    pub tokens: Tensor,
}

struct TwoWayLayer {
    self_attn: Attention,
    norm1: LayerNorm,
    token_to_image: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    norm3: LayerNorm,
    image_to_token: Attention,
    norm4: LayerNorm,
    /// The image side's projections as one GEMM over the keys (see [`image_proj`])
    image: Linear,
}

/// Columns of a layer's fused image projection: token-to-image k and v, then the
/// image-to-token q
const IMG_K: usize = 0;
const IMG_V: usize = 128;
const IMG_Q: usize = 256;

/// Every projection that reads a pass's image keys, as one GEMM: token-to-image k (of
/// keys + dense PE) and v (of keys), then image-to-token q (of keys + dense PE) when the
/// layer has one. The dense PE is a constant, so its projections fold into the bias and
/// a pass never forms keys + PE
fn image_proj(gpu: &Gpu, t2i: &Attention, i2t: Option<&Attention>, dense_pe: &Tensor) -> Result<Linear> {
    let mut parts = vec![(&t2i.k, true), (&t2i.v, false)];
    parts.extend(i2t.map(|a| (&a.q, true)));
    ensure!(
        parts.iter().all(|(l, _)| l.out_features() == IMG_V) && t2i.q.out_features() == IMG_V,
        "the point decoder's cross-attentions run at width {IMG_V}"
    );
    Linear::fuse_shifted(gpu, &parts, dense_pe)
}

/// The tracker's prompt encoder and mask decoder, which answer point prompts on an image
/// and decode every tracked frame of a video
pub struct PointDecoder {
    prompts: PointEncoding,
    no_mask: Tensor,
    mask_conv1: Linear,
    mask_ln1: LayerNorm,
    mask_conv2: Linear,
    mask_ln2: LayerNorm,
    mask_conv3: Linear,
    /// Object score, IoU and 4 mask tokens `[6, 256]`
    output_tokens: Vec<f32>,
    layers: Vec<TwoWayLayer>,
    final_attn: Attention,
    /// The final attention's k and v projections as one GEMM (see [`image_proj`])
    final_image: Linear,
    final_norm: LayerNorm,
    upscale1: ConvT2x2,
    upscale_ln: LayerNorm,
    upscale2: ConvT2x2,
    heads: Heads,
    obj_score_head: Mlp,
    no_mem: Tensor,
    conv_s0: Linear,
    conv_s1: Linear,
}

impl PointDecoder {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let pe = "tracker_model.prompt_encoder";
        let md = "tracker_model.mask_decoder";
        let f32_linear = |name: &str| Linear::load(gpu, wts, name, DType::F32);
        let prompts = PointEncoding::load(wts)?;
        let dense = prompts.dense_pe();

        let mut output_tokens = wts.host(&format!("{md}.obj_score_token.weight"))?.0;
        output_tokens.extend(wts.host(&format!("{md}.iou_token.weight"))?.0);
        output_tokens.extend(wts.host(&format!("{md}.mask_tokens.weight"))?.0);

        let dense_pe = gpu.upload_f32(&dense, &[GRID * GRID, D])?;
        let t = format!("{md}.transformer");
        let mut layers = Vec::new();
        for i in 0..2 {
            let l = format!("{t}.layers.{i}");
            let ln = |n: &str| LayerNorm::load(gpu, wts, &format!("{l}.{n}"), 1e-5);
            let at = |n: &str| Attention::load(gpu, wts, &format!("{l}.{n}"), HEADS);
            let (token_to_image, image_to_token) = (at("cross_attn_token_to_image")?, at("cross_attn_image_to_token")?);
            layers.push(TwoWayLayer {
                self_attn: at("self_attn")?,
                norm1: ln("layer_norm1")?,
                image: image_proj(gpu, &token_to_image, Some(&image_to_token), &dense_pe)?,
                token_to_image,
                norm2: ln("layer_norm2")?,
                mlp: Mlp::load(gpu, wts, &[format!("{l}.mlp.proj_in"), format!("{l}.mlp.proj_out")], Act::Relu)?,
                norm3: ln("layer_norm3")?,
                image_to_token,
                norm4: ln("layer_norm4")?,
            });
        }
        let final_attn = Attention::load(gpu, wts, &format!("{t}.final_attn_token_to_image"), HEADS)?;

        Ok(Self {
            prompts,
            no_mask: wts.get(gpu, &format!("{pe}.no_mask_embed.weight"))?,
            mask_conv1: f32_linear(&format!("{pe}.mask_embed.conv1"))?,
            mask_ln1: LayerNorm::load(gpu, wts, &format!("{pe}.mask_embed.layer_norm1"), 1e-6)?,
            mask_conv2: f32_linear(&format!("{pe}.mask_embed.conv2"))?,
            mask_ln2: LayerNorm::load(gpu, wts, &format!("{pe}.mask_embed.layer_norm2"), 1e-6)?,
            mask_conv3: f32_linear(&format!("{pe}.mask_embed.conv3"))?,
            final_image: image_proj(gpu, &final_attn, None, &dense_pe)?,
            output_tokens,
            layers,
            final_attn,
            final_norm: LayerNorm::load(gpu, wts, &format!("{t}.layer_norm_final_attn"), 1e-5)?,
            upscale1: ConvT2x2::load(gpu, wts, &format!("{md}.upscale_conv1"), DType::F32)?,
            upscale_ln: LayerNorm::load(gpu, wts, &format!("{md}.upscale_layer_norm"), 1e-6)?,
            upscale2: ConvT2x2::load(gpu, wts, &format!("{md}.upscale_conv2"), DType::F32)?,
            heads: Heads::load(gpu, wts, md)?,
            obj_score_head: Mlp::load_in_out(gpu, wts, &format!("{md}.pred_obj_score_head"), 3, Act::Relu)?,
            no_mem: wts.get(gpu, "tracker_model.no_memory_embedding")?,
            conv_s0: f32_linear(&format!("{md}.conv_s0"))?,
            conv_s1: f32_linear(&format!("{md}.conv_s1"))?,
        })
    }

    /// Tracker neck outputs -> decoder image features
    pub fn prepare_image(&self, gpu: &Gpu, fpn: &Fpn) -> Result<PointImage> {
        let embed = clone(gpu, &fpn.coarse)?;
        add_tiled(gpu, &embed, &self.no_mem)?;
        Ok(PointImage {
            embed,
            feat_s0: self.conv_s0.forward(gpu, &fpn.fine, Act::None)?,
            feat_s1: self.conv_s1.forward(gpu, &fpn.mid, Act::None)?,
        })
    }

    /// Dense prompt embeddings `[k * 72^2, 256]` of `k` prompts, from their low-res masks
    /// `[k, 288^2]` or the no-mask embedding
    fn dense(&self, gpu: &Gpu, mask: Option<&Tensor>, k: usize) -> Result<Tensor> {
        let Some(mask) = mask else {
            let out = gpu.empty(&[k * GRID * GRID, D], DType::F32)?;
            gpu.dispatch_1d("tile_rows", out.len(), &[out.arg(), self.no_mask.arg(), self.no_mask.len().into()])?;
            return Ok(out);
        };
        ensure!(mask.len() == k * LOW_RES * LOW_RES, "mask inputs {:?} for {k} prompts of {LOW_RES}^2", mask.shape());
        // the k masks stacked as one image k times as tall: the 2x2 patches and the
        // rowwise convs and norms never cross from one mask into the next
        // `c` channels per pixel: the masks are `[k, 288^2]`, one channel
        let stage = |x: &Tensor, side: usize, c: usize, conv: &Linear, ln: &LayerNorm| -> Result<Tensor> {
            let patches = gpu.empty(&[x.len() / (4 * c), c * 4], DType::F32)?;
            gpu.dispatch_1d("patchify2x2", patches.len(), &[patches.arg(), x.arg(), side.into(), c.into()])?;
            let y = conv.forward(gpu, &patches, Act::None)?;
            ln.forward_inplace(gpu, &y)?;
            bias_act(gpu, &y, None, Act::Gelu)?;
            Ok(y)
        };
        let x = stage(mask, LOW_RES, 1, &self.mask_conv1, &self.mask_ln1)?;
        let x = stage(&x, LOW_RES / 2, x.cols(), &self.mask_conv2, &self.mask_ln2)?;
        self.mask_conv3.forward(gpu, &x, Act::None)
    }

    /// The decoder over image embedding `embed` (the memory-conditioned features on a
    /// tracked frame, `img.embed` otherwise) and the high-resolution features of `img`,
    /// with masks for mask tokens `first..4` only
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &self,
        gpu: &Gpu,
        embed: &Tensor,
        img: &PointImage,
        points: &[PointPrompt],
        size: (usize, usize),
        mask_input: Option<&Tensor>,
        first: usize,
    ) -> Result<Decoded> {
        self.decode_batch(gpu, embed, img, &[points], size, mask_input, first)
    }

    /// [`Self::decode`] for `K` independent prompts with the same number of points, which
    /// share `embed` and `img`, and their mask inputs `[K, 288^2]`
    ///
    /// The prompts are stacked along the rows of every activation, so each GEMM, norm and
    /// conv runs once for all of them. Equal point counts give every prompt the same token
    /// count, so each prompt's attention is the one a single decode would run: no padding
    /// tokens and no masks
    #[allow(clippy::too_many_arguments)]
    pub fn decode_batch(
        &self,
        gpu: &Gpu,
        embed: &Tensor,
        img: &PointImage,
        prompts: &[&[PointPrompt]],
        (width, height): (usize, usize),
        mask_input: Option<&Tensor>,
        first: usize,
    ) -> Result<Decoded> {
        ensure!(first < 4, "mask tokens run 0 to 3, asked for {first}..4");
        let k = prompts.len();
        ensure!(k > 0, "a decoder batch needs at least one prompt");
        let mut tokens = Vec::new();
        let mut per_prompt = None;
        for points in prompts {
            let start = tokens.len();
            tokens.extend_from_slice(&self.output_tokens);
            tokens.extend(self.prompts.sparse_tokens(points, width, height));
            // compared per prompt, since unequal counts can still sum to a multiple of k
            let len = tokens.len() - start;
            ensure!(*per_prompt.get_or_insert(len) == len, "the prompts of one decoder batch need equal point counts");
        }
        let n = tokens.len() / (k * D);
        let prompt_pe = gpu.upload_f32(&tokens, &[k * n, D])?;
        let mut queries = clone(gpu, &prompt_pe)?;

        let keys = self.dense(gpu, mask_input, k)?;
        add_tiled(gpu, &keys, embed)?;
        // token-to-image attention of the queries (plus prompt PE) over the image side's
        // fused projections `proj`
        let token_to_image = |attn: &Attention, proj: &Tensor, queries: &Tensor| -> Result<()> {
            let q = attn.q.forward(gpu, &add(gpu, queries, &prompt_pe)?, Act::None)?;
            let out = attn.attend_projected(gpu, (&q, 0), (proj, IMG_K), (proj, IMG_V), k, Mask::default())?;
            attn.o.forward_residual(gpu, &out, queries)
        };

        for (i, l) in self.layers.iter().enumerate() {
            if i == 0 {
                // the first layer replaces the queries with self-attention (no residual, no pe)
                queries = l.self_attn.forward_batch(gpu, &queries, &queries, &queries, k)?;
            } else {
                let q = add(gpu, &queries, &prompt_pe)?;
                l.self_attn.forward_residual_batch(gpu, &q, &q, &queries, k, &queries)?;
            }
            l.norm1.forward_inplace(gpu, &queries)?;

            // the keys don't change until image-to-token, so one GEMM serves both
            let proj = l.image.forward(gpu, &keys, Act::None)?;
            token_to_image(&l.token_to_image, &proj, &queries)?;
            l.norm2.forward_inplace(gpu, &queries)?;

            let m = l.mlp.forward(gpu, &queries)?;
            add_assign(gpu, &queries, &m)?;
            l.norm3.forward_inplace(gpu, &queries)?;

            let i2t = &l.image_to_token;
            let tk = i2t.k.forward(gpu, &add(gpu, &queries, &prompt_pe)?, Act::None)?;
            let tv = i2t.v.forward(gpu, &queries, Act::None)?;
            let out = i2t.attend_projected(gpu, (&proj, IMG_Q), (&tk, 0), (&tv, 0), k, Mask::default())?;
            i2t.o.forward_residual(gpu, &out, &keys)?;
            l.norm4.forward_inplace(gpu, &keys)?;
        }
        let proj = self.final_image.forward(gpu, &keys, Act::None)?;
        token_to_image(&self.final_attn, &proj, &queries)?;
        self.final_norm.forward_inplace(gpu, &queries)?;

        // the k image grids stacked as one grid k times as tall, which the transposed
        // convs' pixel shuffle keeps prompt by prompt
        // the skip features repeat per stacked image, so the convs add them in their
        // epilogues; the norm runs a SIMD group a row over the 64-channel map
        let up = self.upscale1.forward_add(gpu, &keys, k * GRID, GRID, Some(&img.feat_s1), Act::None)?;
        self.upscale_ln.forward_act_narrow(gpu, &up, Act::Gelu)?;
        let up = self.upscale2.forward_add(gpu, &up, 2 * k * GRID, 2 * GRID, Some(&img.feat_s0), Act::Gelu)?;

        // mask token 0 is the single-mask output, 1 to 3 the multimask ones
        let side = LOW_RES * LOW_RES;
        let masks = 4 - first;
        let c = up.cols();
        ensure!(c == HYPER_OUT, "hypernetwork width {HYPER_OUT} vs upscaled channels {c}");
        let hyper = gpu.empty(&[k, masks * c], DType::F32)?;
        let iou = gpu.empty(&[k, IOU_OUT], DType::F32)?;
        self.heads.forward(gpu, &queries, n, first, &hyper, &iou)?;

        // prompt b's mask j is its hypernetwork row times its own upscaled map
        let low_res = gpu.empty(&[k * masks, side], DType::F32)?;
        let args = [low_res.arg(), hyper.arg(), up.arg(), side.into(), c.into(), masks.into(), (masks * c).into()];
        gpu.dispatch_1d("mask_dots", k * side, &args)?;
        Ok(Decoded { low_res, iou, tokens: queries })
    }

    /// `[1, 1]` object score logit of a decoder pass; the object is present when it is
    /// above 0. The image path has no use for it, so [`Self::decode`] leaves it out
    pub fn obj_score(&self, gpu: &Gpu, out: &Decoded) -> Result<Tensor> {
        self.obj_score_head.forward(gpu, &row(gpu, &out.tokens, 0)?)
    }
}

/// Hypernetwork output width: the upscaled map's channels
const HYPER_OUT: usize = 32;
/// IoU head outputs, one per mask token
const IOU_OUT: usize = 4;
/// Floats in one packed head: 3 square layers' weights and biases (see `decoder_heads`)
const HEAD_SLOT: usize = 3 * D * D + 3 * D;
/// The IoU head's slot, after the 4 hypernetworks
const IOU_SLOT: usize = 4;

/// The 4 mask tokens' hypernetwork MLPs and the IoU head, packed so one `decoder_heads`
/// launch runs every head of every prompt instead of a gather and 3 GEMMs per head
struct Heads {
    /// `[5, HEAD_SLOT]`: hypernetworks 0 to 3, then the IoU head
    weights: Tensor,
    /// `HeadMeta` rows for decodes from mask token 0 and from 1
    meta: [Tensor; 2],
}

impl Heads {
    fn load(gpu: &Gpu, wts: &Weights, md: &str) -> Result<Self> {
        let mut packed = Vec::with_capacity(5 * HEAD_SLOT);
        let prefixes =
            (0..4).map(|i| format!("{md}.output_hypernetworks_mlps.{i}")).chain([format!("{md}.iou_prediction_head")]);
        for p in prefixes {
            for (i, layer) in ["proj_in", "layers.0", "proj_out"].iter().enumerate() {
                let (w, ws) = wts.host(&format!("{p}.{layer}.weight"))?;
                let (b, _) = wts.host(&format!("{p}.{layer}.bias"))?;
                let out = if i == 2 { ws[0] } else { D };
                ensure!(ws == [out, D] && b.len() == out && out <= D, "{p}.{layer} of shape {ws:?}");
                // the last layer pads to D rows, of which the kernel reads `out`
                packed.extend_from_slice(&w);
                packed.resize(packed.len() + (D - out) * D, 0.0);
                packed.extend_from_slice(&b);
                packed.resize(packed.len() + D - out, 0.0);
            }
        }
        ensure!(packed.len() == 5 * HEAD_SLOT, "packed decoder heads of {} floats", packed.len());

        let meta = |first: usize| {
            let masks = 4 - first;
            // slot, token row, outputs, activation, destination, column
            let mut m: Vec<u32> = (first..4)
                .enumerate()
                .flat_map(|(j, t)| [t, MASK_TOKEN_ROW + t, HYPER_OUT, Act::None as usize, 0, j * HYPER_OUT])
                .map(|v| v as u32)
                .collect();
            m.extend([IOU_SLOT, 1, IOU_OUT, Act::Sigmoid as usize, 1, 0].map(|v| v as u32));
            gpu.upload(&m, &[masks + 1, 6], DType::I32)
        };
        Ok(Self { weights: gpu.upload_f32(&packed, &[5, HEAD_SLOT])?, meta: [meta(0)?, meta(1)?] })
    }

    /// Mask tokens `first..4` of each prompt's `n` tokens in `queries` -> `hyper` `[k,
    /// masks * 32]`, and its IoU token -> sigmoid IoU `[k, 4]`
    fn forward(&self, gpu: &Gpu, queries: &Tensor, n: usize, first: usize, hyper: &Tensor, iou: &Tensor) -> Result<()> {
        let meta = &self.meta[first.min(1)];
        ensure!(first <= 1, "decoder heads from mask token {first}");
        let args = [
            hyper.arg(),
            iou.arg(),
            queries.arg(),
            self.weights.arg(),
            meta.arg(),
            n.into(),
            hyper.cols().into(),
            iou.cols().into(),
        ];
        gpu.dispatch("decoder_heads", [meta.rows(), iou.rows(), 1], [D, 1, 1], &args)
    }
}

/// A copy of FP32 `x`
pub fn clone(gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
    ensure!(x.dtype() == DType::F32, "clone of a {:?} tensor", x.dtype());
    let out = gpu.empty(x.shape(), DType::F32)?;
    gpu.dispatch_1d("copy_f32", out.len(), &[out.arg(), x.arg()])?;
    Ok(out)
}

/// `a + b` as a new FP32 tensor
pub fn add(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure!(a.shape() == b.shape(), "add of {:?} and {:?}", a.shape(), b.shape());
    let out = gpu.empty(a.shape(), DType::F32)?;
    gpu.dispatch_1d("add", out.len(), &[out.arg(), a.arg(), b.arg()])?;
    Ok(out)
}

/// Row `r` of `x` as a `[1, cols]` tensor
pub fn row(gpu: &Gpu, x: &Tensor, r: usize) -> Result<Tensor> {
    let d = x.cols();
    ensure!(r < x.rows(), "row {r} of {:?}", x.shape());
    let out = gpu.empty(&[1, d], DType::F32)?;
    gpu.dispatch_1d("copy_f32", d, &[out.arg(), x.at(r * d)])?;
    Ok(out)
}

/// `x += b`, with `b` repeated over `x` when `x` stacks several copies of it
pub fn add_tiled(gpu: &Gpu, x: &Tensor, b: &Tensor) -> Result<()> {
    ensure!(x.len().is_multiple_of(b.len()), "{:?} is not a stack of {:?}", x.shape(), b.shape());
    if x.len() == b.len() {
        return add_assign(gpu, x, b);
    }
    gpu.dispatch_1d("add_tiled", x.len(), &[x.arg(), b.arg(), b.len().into()])
}

/// Each prompt's best multimask candidate from a pass with masks for tokens 1 to 3,
/// `low_res [K * 3, 288^2]` and `iou [K, 4]`, picked on the GPU as `[K, 288^2]`: the
/// first argmax of the prompt's candidate IoUs, as [`crate::host::argmax`] picks on the host. With
/// `clamp`, clamped to +-32 as the next pass's mask input
pub fn best_masks(gpu: &Gpu, low_res: &Tensor, iou: &Tensor, clamp: bool) -> Result<Tensor> {
    let k = iou.rows();
    let side = LOW_RES * LOW_RES;
    ensure!(
        iou.shape() == [k, 4] && low_res.shape() == [k * 3, side],
        "best masks of {:?} by IoUs {:?}",
        low_res.shape(),
        iou.shape()
    );
    let out = gpu.empty(&[k, side], DType::F32)?;
    let args =
        [out.arg(), low_res.arg(), iou.arg(), side.into(), u32::from(clamp).into(), (-32.0f32).into(), 32.0f32.into()];
    gpu.dispatch_1d("best_mask", out.len(), &args)?;
    Ok(out)
}

/// Upstream's `postprocess_masks` hole filling, in place on the GPU: in each low-res
/// mask of `logits [n, 288^2]`, background regions (logit <= 0), 8-connected, of at most
/// `max_area` pixels become foreground (+10). Every mask goes through the same four
/// launches, so a batch costs what one mask does until the GPU fills up
pub fn fill_holes(gpu: &Gpu, logits: &Tensor, max_area: usize) -> Result<()> {
    let n = logits.len();
    ensure!(
        n.is_multiple_of(LOW_RES * LOW_RES) && logits.dtype() == DType::F32,
        "{:?} {:?} is not a stack of FP32 low-res masks",
        logits.shape(),
        logits.dtype()
    );
    let labels = gpu.alloc(n * 4)?;
    let area = gpu.alloc(n * 4)?;
    gpu.dispatch_1d("holes_init", n, &[labels.arg(), area.arg(), logits.arg(), LOW_RES.into()])?;
    gpu.dispatch_1d("holes_merge", n, &[labels.arg(), logits.arg(), LOW_RES.into()])?;
    gpu.dispatch_1d("holes_count", n, &[labels.arg(), area.arg()])?;
    gpu.dispatch_1d("holes_fill", n, &[logits.arg(), labels.arg(), area.arg(), max_area.into()])
}
