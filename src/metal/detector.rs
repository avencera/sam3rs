//! SAM 3 detector on Metal: geometry (box) encoder, fusion encoder, DETR decoder with
//! presence token, dot-product scoring and the segmentation head
//!
//! Every layer runs FP32 in both precisions: Apple GPUs do FP32 math at the BF16 rate,
//! and the image features it reads (the detector neck's levels) are FP32 either way

use eyre::{Result, ensure, eyre};

use crate::metal::gpu::{DType, Gpu, Tensor, groups};
use crate::metal::ops::{
    Act, Attention, BoxRpb, Conv3x3, LayerNorm, Linear, Mask, Mlp, add_assign, bias_act, concat_rows, gather_rows,
};
use crate::metal::text::TextFeatures;
use crate::metal::vision::{Fpn, GRID};
use crate::posenc::{box_encoding, sine_2d};
use crate::prompt::BoxPrompt;
use crate::weights::Weights;

const D: usize = 256;
const HEADS: usize = 8;
/// Object queries of the DETR decoder, one score, box and mask each
pub const QUERIES: usize = 200;
const ROI: usize = 7;
/// Groups of the pixel decoder's GroupNorm
const NORM_GROUPS: usize = 8;
/// Pixels per partial-sum threadgroup of the group norm statistics
const NORM_CHUNK: usize = 256;
/// Threads of the one-threadgroup reductions
const THREADS: usize = 256;

/// Image features the detector reads, computed once per image
pub struct DetectorImage {
    pub fpn: Fpn,
    /// LayerNorm(fpn.coarse) for box ROI pooling
    pub coarse_normed: Tensor,
}

/// Raw detector outputs for one prompt, all on the device
pub struct DetectorOutput {
    /// `[200]` logits before presence
    pub logits: Tensor,
    /// `[1]`
    pub presence: Tensor,
    /// `[200, 4]` normalized cxcywh
    pub boxes: Tensor,
    /// `[200, 288 * 288]`
    pub masks: Tensor,
}

/// `a + b` as a new FP32 tensor
fn add(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure!(a.len() == b.len(), "add of {:?} and {:?}", a.shape(), b.shape());
    let out = gpu.empty(a.shape(), DType::F32)?;
    gpu.dispatch_1d("add", out.len(), &[out.arg(), a.arg(), b.arg()])?;
    Ok(out)
}

/// Rows `first..first + n` of FP32 `x` as a new tensor
fn rows(gpu: &Gpu, x: &Tensor, first: usize, n: usize) -> Result<Tensor> {
    ensure!(first + n <= x.rows(), "rows {first}..{} of {:?}", first + n, x.shape());
    let out = gpu.empty(&[n, x.cols()], DType::F32)?;
    gpu.dispatch_1d("copy_f32", out.len(), &[out.arg(), x.at(first * x.cols())])?;
    Ok(out)
}

/// A copy of FP32 `x`
fn copy(gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
    Ok(rows(gpu, x, 0, x.rows())?.reshape(x.shape()))
}

/// GroupNorm over channels-last `[hw, c]` with a fused activation, in two halves: this
/// computes the per-group statistics, and the returned norm applies them later, once
/// the next level's conv is about to read the tensor
fn group_norm_stats<'a>(
    gpu: &Gpu,
    x: &Tensor,
    (w, b): &'a (Tensor, Tensor),
    eps: f32,
    act: Act,
) -> Result<PendingNorm<'a>> {
    let (hw, c) = (x.rows(), x.cols());
    ensure!(c.is_multiple_of(NORM_GROUPS) && w.len() == c && b.len() == c, "group norm of {:?}", x.shape());
    let chunks = groups(hw, NORM_CHUNK);
    let part = gpu.empty(&[NORM_GROUPS * chunks, 2], DType::F32)?;
    let args = [part.arg(), x.arg(), hw.into(), c.into(), NORM_GROUPS.into(), NORM_CHUNK.into()];
    gpu.dispatch("group_stats", [chunks, NORM_GROUPS, 1], [THREADS, 1, 1], &args)?;
    let stats = gpu.empty(&[NORM_GROUPS, 2], DType::F32)?;
    gpu.dispatch(
        "group_stats_reduce",
        [NORM_GROUPS, 1, 1],
        [THREADS, 1, 1],
        &[stats.arg(), part.arg(), chunks.into()],
    )?;
    Ok(PendingNorm { stats, w, b, hw, eps, act })
}

/// A group norm whose per-group sums are computed but not yet applied
struct PendingNorm<'a> {
    /// `[groups, 2]`: sum and sum of squares
    stats: Tensor,
    w: &'a Tensor,
    b: &'a Tensor,
    hw: usize,
    eps: f32,
    act: Act,
}

impl PendingNorm<'_> {
    /// Normalizes `x`, the tensor the statistics came from, in place
    fn apply(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        ensure!(x.rows() == self.hw, "group norm statistics of {} pixels applied to {:?}", self.hw, x.shape());
        let args = [
            x.arg(),
            self.stats.arg(),
            self.w.arg(),
            self.b.arg(),
            self.hw.into(),
            x.cols().into(),
            NORM_GROUPS.into(),
            self.eps.into(),
            self.act.into(),
        ];
        gpu.dispatch_1d("group_norm_apply", x.len(), &args)
    }
}

/// The conv of `fine + up2(coarse)` over a fine map of `h` x `w` pixels, where `up2` is
/// nearest 2x upsampling and `norm`, if any, is still pending on `coarse` and is applied
/// to it in place first
fn conv_up(
    gpu: &Gpu,
    conv: &Conv3x3,
    fine: &Tensor,
    coarse: &Tensor,
    norm: Option<&PendingNorm>,
    (h, w): (usize, usize),
) -> Result<Tensor> {
    let c = fine.cols();
    ensure!(fine.shape() == [h * w, c], "fine level {:?} for {h}x{w}", fine.shape());
    ensure!(coarse.shape() == [h * w / 4, c], "coarse level {:?} for {h}x{w}", coarse.shape());
    if let Some(norm) = norm {
        norm.apply(gpu, coarse)?;
    }
    let up = gpu.empty(&[h * w, c], DType::F32)?;
    gpu.dispatch_1d("upsample2_add", up.len(), &[up.arg(), fine.arg(), coarse.arg(), (w / 2).into(), c.into()])?;
    conv.forward(gpu, &up, h, w)
}

/// Pre-norm encoder layer shared by the geometry and fusion encoders
struct EncoderLayer {
    norm1: LayerNorm,
    self_attn: Attention,
    norm2: LayerNorm,
    cross_attn: Attention,
    norm3: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderLayer {
    fn load(gpu: &Gpu, wts: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::load(gpu, wts, &format!("{p}.layer_norm1"), 1e-5)?,
            self_attn: Attention::load(gpu, wts, &format!("{p}.self_attn"), HEADS)?,
            norm2: LayerNorm::load(gpu, wts, &format!("{p}.layer_norm2"), 1e-5)?,
            cross_attn: Attention::load(gpu, wts, &format!("{p}.cross_attn"), HEADS)?,
            norm3: LayerNorm::load(gpu, wts, &format!("{p}.layer_norm3"), 1e-5)?,
            fc1: Linear::load(gpu, wts, &format!("{p}.mlp.fc1"), DType::F32)?,
            fc2: Linear::load(gpu, wts, &format!("{p}.mlp.fc2"), DType::F32)?,
        })
    }

    /// `x += fc2(relu(fc1(norm3(x))))`
    fn ffn(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        let h = self.norm3.forward_to(gpu, x, DType::F32)?;
        let hidden = self.fc1.forward(gpu, &h, Act::Relu)?;
        self.fc2.forward_residual(gpu, &hidden, x)
    }
}

struct GeometryEncoder {
    direct: Linear,
    pool: Linear,
    pos: Linear,
    label: Tensor,
    cls: Tensor,
    final_proj: Linear,
    prompt_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    out_norm: LayerNorm,
}

struct DecoderLayer {
    self_attn: Attention,
    self_norm: LayerNorm,
    text_attn: Attention,
    text_norm: LayerNorm,
    image_attn: Attention,
    image_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    mlp_norm: LayerNorm,
}

/// The detector's weights
pub struct Detector {
    vision_norm: LayerNorm,
    geometry: GeometryEncoder,
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
    query_embed: Tensor,
    reference_points: Tensor,
    presence_token: Tensor,
    ref_point_head: Mlp,
    rpb_x: RpbMlp,
    rpb_y: RpbMlp,
    box_head: Mlp,
    presence_head: Mlp,
    presence_norm: LayerNorm,
    out_norm: LayerNorm,
    text_mlp: Mlp,
    text_mlp_norm: LayerNorm,
    text_proj: Linear,
    query_proj: Linear,
    seg_attn: Attention,
    seg_attn_norm: LayerNorm,
    pixel_convs: Vec<Conv3x3>,
    pixel_norms: Vec<(Tensor, Tensor)>,
    instance_proj: Linear,
    mask_embed: Mlp,
    /// Sine position encoding of the 72x72 grid `[5184, 256]`
    pos: Tensor,
    /// `[1, 256]` zeros, the presence token's query position
    zero_row: Tensor,
}

impl Detector {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "detector_model";
        let linear = |name: &str| Linear::load(gpu, wts, name, DType::F32);
        let g = format!("{p}.geometry_encoder");
        let geometry = GeometryEncoder {
            direct: linear(&format!("{g}.boxes_direct_project"))?,
            pool: linear(&format!("{g}.boxes_pool_project"))?,
            pos: linear(&format!("{g}.boxes_pos_enc_project"))?,
            label: wts.get(gpu, &format!("{g}.label_embed.weight"))?,
            cls: wts.get(gpu, &format!("{g}.cls_embed.weight"))?,
            final_proj: linear(&format!("{g}.final_proj"))?,
            prompt_norm: LayerNorm::load(gpu, wts, &format!("{g}.prompt_layer_norm"), 1e-5)?,
            layers: (0..3).map(|i| EncoderLayer::load(gpu, wts, &format!("{g}.layers.{i}"))).collect::<Result<_>>()?,
            out_norm: LayerNorm::load(gpu, wts, &format!("{g}.output_layer_norm"), 1e-5)?,
        };
        ensure!(geometry.cls.shape() == [1, D], "geometry CLS token {:?}", geometry.cls.shape());
        ensure!(
            geometry.pool.in_features() == D * ROI * ROI,
            "box pooling projection of {}",
            geometry.pool.in_features()
        );
        let encoder = (0..6)
            .map(|i| EncoderLayer::load(gpu, wts, &format!("{p}.detr_encoder.layers.{i}")))
            .collect::<Result<_>>()?;
        let dd = format!("{p}.detr_decoder");
        let mut decoder = Vec::new();
        for i in 0..6 {
            let l = format!("{dd}.layers.{i}");
            let ln = |n: &str| LayerNorm::load(gpu, wts, &format!("{l}.{n}"), 1e-5);
            let at = |n: &str| Attention::load(gpu, wts, &format!("{l}.{n}"), HEADS);
            decoder.push(DecoderLayer {
                self_attn: at("self_attn")?,
                self_norm: ln("self_attn_layer_norm")?,
                text_attn: at("text_cross_attn")?,
                text_norm: ln("text_cross_attn_layer_norm")?,
                image_attn: at("vision_cross_attn")?,
                image_norm: ln("vision_cross_attn_layer_norm")?,
                fc1: linear(&format!("{l}.mlp.fc1"))?,
                fc2: linear(&format!("{l}.mlp.fc2"))?,
                mlp_norm: ln("mlp_layer_norm")?,
            });
        }
        let s = format!("{p}.dot_product_scoring");
        let m = format!("{p}.mask_decoder");
        let detector = Self {
            vision_norm: LayerNorm::load(gpu, wts, &format!("{g}.vision_layer_norm"), 1e-5)?,
            geometry,
            encoder,
            decoder,
            query_embed: wts.get(gpu, &format!("{dd}.query_embed.weight"))?,
            reference_points: wts.get(gpu, &format!("{dd}.reference_points.weight"))?,
            presence_token: wts.get(gpu, &format!("{dd}.presence_token.weight"))?,
            ref_point_head: Mlp::load_numbered(gpu, wts, &format!("{dd}.ref_point_head"), 2, Act::Relu)?,
            rpb_x: RpbMlp::load(gpu, wts, &format!("{dd}.box_rpb_embed_x"))?,
            rpb_y: RpbMlp::load(gpu, wts, &format!("{dd}.box_rpb_embed_y"))?,
            box_head: Mlp::load_numbered(gpu, wts, &format!("{dd}.box_head"), 3, Act::Relu)?,
            presence_head: Mlp::load_numbered(gpu, wts, &format!("{dd}.presence_head"), 3, Act::Relu)?,
            presence_norm: LayerNorm::load(gpu, wts, &format!("{dd}.presence_layer_norm"), 1e-5)?,
            out_norm: LayerNorm::load(gpu, wts, &format!("{dd}.output_layer_norm"), 1e-5)?,
            text_mlp: Mlp::load_numbered(gpu, wts, &format!("{s}.text_mlp"), 2, Act::Relu)?,
            text_mlp_norm: LayerNorm::load(gpu, wts, &format!("{s}.text_mlp_out_norm"), 1e-5)?,
            text_proj: linear(&format!("{s}.text_proj"))?,
            query_proj: linear(&format!("{s}.query_proj"))?,
            seg_attn: Attention::load(gpu, wts, &format!("{m}.prompt_cross_attn"), HEADS)?,
            seg_attn_norm: LayerNorm::load(gpu, wts, &format!("{m}.prompt_cross_attn_norm"), 1e-5)?,
            // the pixel decoder has three convs but only two upsampling stages run
            pixel_convs: (0..2)
                .map(|i| Conv3x3::load(gpu, wts, &format!("{m}.pixel_decoder.conv_layers.{i}"), DType::F32))
                .collect::<Result<_>>()?,
            pixel_norms: (0..2)
                .map(|i| {
                    let n = format!("{m}.pixel_decoder.norms.{i}");
                    Ok((wts.get(gpu, &format!("{n}.weight"))?, wts.get(gpu, &format!("{n}.bias"))?))
                })
                .collect::<Result<_>>()?,
            instance_proj: linear(&format!("{m}.instance_projection"))?,
            mask_embed: Mlp::load_indexed(gpu, wts, &format!("{m}.mask_embedder"), 3, Act::Relu)?,
            pos: gpu.upload_f32(&sine_2d(GRID, D), &[GRID * GRID, D])?,
            zero_row: gpu.upload_f32(&[0.0; D], &[1, D])?,
        };
        ensure!(
            detector.query_embed.shape() == [QUERIES, D] && detector.reference_points.shape() == [QUERIES, 4],
            "decoder queries {:?} and reference points {:?}",
            detector.query_embed.shape(),
            detector.reference_points.shape()
        );
        Ok(detector)
    }

    pub fn prepare_image(&self, gpu: &Gpu, fpn: Fpn) -> Result<DetectorImage> {
        ensure!(fpn.coarse.shape() == [GRID * GRID, D], "detector coarse level {:?}", fpn.coarse.shape());
        let coarse_normed = self.vision_norm.forward_to(gpu, &fpn.coarse, DType::F32)?;
        Ok(DetectorImage { fpn, coarse_normed })
    }

    /// Geometry tokens: one per box, then the CLS token, all valid
    fn encode_geometry(&self, gpu: &Gpu, img: &DetectorImage, boxes: &[BoxPrompt]) -> Result<Tensor> {
        let g = &self.geometry;
        let n = boxes.len();
        let embeds = if n == 0 {
            copy(gpu, &g.cls)?
        } else {
            let raw: Vec<f32> = boxes.iter().flat_map(BoxPrompt::cxcywh).collect();
            let raw = gpu.upload_f32(&raw, &[n, 4])?;
            let x = g.direct.forward(gpu, &raw, Act::None)?;

            let side = GRID as f32;
            let xyxy: Vec<f32> = boxes
                .iter()
                .flat_map(|b| {
                    let [cx, cy, w, h] = b.cxcywh();
                    [(cx - 0.5 * w) * side, (cy - 0.5 * h) * side, (cx + 0.5 * w) * side, (cy + 0.5 * h) * side]
                })
                .collect();
            let xyxy = gpu.upload_f32(&xyxy, &[n, 4])?;
            let pooled = gpu.empty(&[n, D * ROI * ROI], DType::F32)?;
            let args =
                [pooled.arg(), img.coarse_normed.arg(), xyxy.arg(), GRID.into(), GRID.into(), D.into(), ROI.into()];
            gpu.dispatch_1d("roi_align", pooled.len(), &args)?;
            g.pool.forward_residual(gpu, &pooled, &x)?;

            let enc: Vec<f32> = boxes.iter().flat_map(|b| box_encoding(b.cxcywh())).collect();
            let enc = gpu.upload_f32(&enc, &[n, 258])?;
            g.pos.forward_residual(gpu, &enc, &x)?;

            let labels: Vec<i32> = boxes.iter().map(|b| i32::from(b.positive())).collect();
            let label = gather_rows(gpu, &g.label, &labels)?;
            add_assign(gpu, &x, &label)?;
            concat_rows(gpu, &x, &g.cls)?
        };

        let x = g.final_proj.forward(gpu, &embeds, Act::None)?;
        g.prompt_norm.forward_inplace(gpu, &x)?;
        let mem_pos = add(gpu, &img.fpn.coarse, &self.pos)?;
        for l in &g.layers {
            let h = l.norm1.forward_to(gpu, &x, DType::F32)?;
            l.self_attn.forward_residual(gpu, &h, &h, &h, Mask::default(), &x)?;
            let h = l.norm2.forward_to(gpu, &x, DType::F32)?;
            l.cross_attn.forward_residual(gpu, &h, &mem_pos, &img.fpn.coarse, Mask::default(), &x)?;
            l.ffn(gpu, &x)?;
        }
        g.out_norm.forward_inplace(gpu, &x)?;
        Ok(x)
    }

    /// The detector over one prompt: text features plus any boxes
    pub fn forward(
        &self,
        gpu: &Gpu,
        img: &DetectorImage,
        text: &TextFeatures,
        boxes: &[BoxPrompt],
    ) -> Result<DetectorOutput> {
        let geo = self.encode_geometry(gpu, img, boxes)?;
        let prompt = concat_rows(gpu, &text.feats, &geo)?;
        let mut pad = text.pad.to_vec();
        pad.resize(prompt.rows(), 0);
        let pad = gpu.upload(&pad, &[prompt.rows()], DType::U8)?;
        let prompt_mask = Mask { key_pad: Some(&pad), ..Mask::default() };

        let memory = self.fusion_encoder(gpu, img, &prompt, prompt_mask)?;
        let (hs, boxes, presence) = self.decode(gpu, &memory, &prompt, prompt_mask)?;
        let logits = self.score(gpu, &hs, &prompt, &pad)?;
        let masks = self.segment(gpu, img, &memory, &hs, &prompt, prompt_mask)?;
        Ok(DetectorOutput { logits, presence, boxes, masks })
    }

    fn fusion_encoder(&self, gpu: &Gpu, img: &DetectorImage, prompt: &Tensor, prompt_mask: Mask) -> Result<Tensor> {
        let x = copy(gpu, &img.fpn.coarse)?;
        for l in &self.encoder {
            let h = l.norm1.forward_to(gpu, &x, DType::F32)?;
            let qk = add(gpu, &h, &self.pos)?;
            l.self_attn.forward_residual(gpu, &qk, &qk, &h, Mask::default(), &x)?;
            drop((h, qk));
            let h = l.norm2.forward_to(gpu, &x, DType::F32)?;
            l.cross_attn.forward_residual(gpu, &h, prompt, prompt, prompt_mask, &x)?;
            l.ffn(gpu, &x)?;
        }
        Ok(x)
    }

    /// Returns normed final query states `[200, 256]`, boxes `[200, 4]` and the presence
    /// logit `[1]`
    fn decode(
        &self,
        gpu: &Gpu,
        memory: &Tensor,
        prompt: &Tensor,
        prompt_mask: Mask,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // row 0 is the presence token, which has a zero query position
        let x = concat_rows(gpu, &self.presence_token, &self.query_embed)?;
        let mut boxes = copy(gpu, &self.reference_points)?;
        bias_act(gpu, &boxes, None, Act::Sigmoid)?;
        let mem_pos = add(gpu, memory, &self.pos)?;
        let mut hs = None;

        for (i, l) in self.decoder.iter().enumerate() {
            let sine = gpu.empty(&[QUERIES, 2 * D], DType::F32)?;
            gpu.dispatch_1d("box_sine_embed", sine.len(), &[sine.arg(), boxes.arg()])?;
            let pos = self.ref_point_head.forward(gpu, &sine)?;
            let qpos = concat_rows(gpu, &self.zero_row, &pos)?;
            let [dx, dy] = self.box_rpb(gpu, &boxes)?;

            let qk = add(gpu, &x, &qpos)?;
            l.self_attn.forward_residual(gpu, &qk, &qk, &x, Mask::default(), &x)?;
            l.self_norm.forward_inplace(gpu, &x)?;

            let q = add(gpu, &x, &qpos)?;
            l.text_attn.forward_residual(gpu, &q, prompt, prompt, prompt_mask, &x)?;
            l.text_norm.forward_inplace(gpu, &x)?;

            let q = add(gpu, &x, &qpos)?;
            let rpb = Mask { rpb: Some(BoxRpb { dx: &dx, dy: &dy, side: GRID }), ..Mask::default() };
            l.image_attn.forward_residual(gpu, &q, &mem_pos, memory, rpb, &x)?;
            l.image_norm.forward_inplace(gpu, &x)?;

            let hidden = l.fc1.forward(gpu, &x, Act::Relu)?;
            l.fc2.forward_residual(gpu, &hidden, &x)?;
            l.mlp_norm.forward_inplace(gpu, &x)?;

            let queries = rows(gpu, &x, 1, QUERIES)?;
            let normed = self.out_norm.forward_to(gpu, &queries, DType::F32)?;
            let delta = self.box_head.forward(gpu, &normed)?;
            let refined = gpu.empty(&[QUERIES, 4], DType::F32)?;
            gpu.dispatch_1d("box_refine", refined.len(), &[refined.arg(), boxes.arg(), delta.arg()])?;
            boxes = refined;

            // only the last layer's queries are kept, so earlier ones free before the next layer
            if i + 1 == self.decoder.len() {
                hs = Some(normed);
            }
        }
        let hs = hs.ok_or_else(|| eyre!("the detector has no decoder layers"))?;

        // the presence token is row 0 of the last layer's output
        let p = self.presence_norm.forward_to(gpu, &rows(gpu, &x, 0, 1)?, DType::F32)?;
        let presence = self.presence_head.forward(gpu, &p)?;
        Ok((hs, boxes, presence))
    }

    /// Box relative position bias for the image cross-attention in factored form, `dx`
    /// and `dy` as `[8, 200, 72]` (see [`BoxRpb`])
    fn box_rpb(&self, gpu: &Gpu, boxes: &Tensor) -> Result<[Tensor; 2]> {
        Ok([self.rpb_x.forward(gpu, boxes, 0)?, self.rpb_y.forward(gpu, boxes, 1)?])
    }

    fn score(&self, gpu: &Gpu, hs: &Tensor, prompt: &Tensor, pad: &Tensor) -> Result<Tensor> {
        let p = self.text_mlp.forward(gpu, prompt)?;
        add_assign(gpu, &p, prompt)?;
        self.text_mlp_norm.forward_inplace(gpu, &p)?;
        ensure!(p.cols() == D && pad.len() == p.rows(), "masked mean of {:?} with {} mask bytes", p.shape(), pad.len());
        let pooled = gpu.empty(&[1, D], DType::F32)?;
        gpu.dispatch_1d("masked_mean", D, &[pooled.arg(), p.arg(), pad.arg(), D.into(), p.rows().into()])?;
        let pp = self.text_proj.forward(gpu, &pooled, Act::None)?;
        let ph = self.query_proj.forward(gpu, hs, Act::None)?;
        ensure!(ph.shape() == [QUERIES, D] && pp.shape() == [1, D], "scores of {:?} by {:?}", ph.shape(), pp.shape());
        let scores = gpu.empty(&[QUERIES], DType::F32)?;
        let scale = 1.0 / (D as f32).sqrt();
        let args = [scores.arg(), ph.arg(), pp.arg(), QUERIES.into(), D.into(), scale.into(), 12.0f32.into()];
        // eight SIMD groups of 32, one query each
        gpu.dispatch("dot_scores", [groups(QUERIES, THREADS / 32), 1, 1], [THREADS, 1, 1], &args)?;
        Ok(scores)
    }

    fn segment(
        &self,
        gpu: &Gpu,
        img: &DetectorImage,
        memory: &Tensor,
        hs: &Tensor,
        prompt: &Tensor,
        prompt_mask: Mask,
    ) -> Result<Tensor> {
        let h = self.seg_attn_norm.forward_to(gpu, memory, DType::F32)?;
        let enc = copy(gpu, memory)?;
        self.seg_attn.forward_residual(gpu, &h, prompt, prompt, prompt_mask, &enc)?;
        drop(h);

        // each level's group norm is applied as the next level's conv reads it
        let mut prev = enc;
        let mut norm = None;
        let mut side = GRID;
        for (i, fine) in [&img.fpn.mid, &img.fpn.fine].into_iter().enumerate() {
            side *= 2;
            prev = conv_up(gpu, &self.pixel_convs[i], fine, &prev, norm.as_ref(), (side, side))?;
            norm = Some(group_norm_stats(gpu, &prev, &self.pixel_norms[i], 1e-5, Act::Relu)?);
        }
        norm.ok_or_else(|| eyre!("the pixel decoder has no levels"))?.apply(gpu, &prev)?;
        let instance = self.instance_proj.forward(gpu, &prev, Act::None)?;
        drop(prev);
        let embed = self.mask_embed.forward(gpu, hs)?;
        // masks [200, 288^2] = embed [200, 256] . instance^T, the instance map as a
        // linear's weights
        Linear::new(instance, None).forward(gpu, &embed, Act::None)
    }
}

/// One axis of the box relative position bias MLP, `Linear(2, hidden)`, ReLU and
/// `Linear(hidden, 8)`, run by one fused kernel together with its input deltas
struct RpbMlp {
    /// `[hidden, 2]`
    w1: Tensor,
    b1: Tensor,
    /// `[8, hidden]`
    w2: Tensor,
    b2: Tensor,
}

impl RpbMlp {
    fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let get = |name: &str| wts.get(gpu, &format!("{prefix}.{name}"));
        let mlp = Self {
            w1: get("layer1.weight")?,
            b1: get("layer1.bias")?,
            w2: get("layer2.weight")?,
            b2: get("layer2.bias")?,
        };
        let hidden = mlp.w1.shape()[0];
        ensure!(mlp.w1.shape() == [hidden, 2], "{prefix}.layer1 is {:?}", mlp.w1.shape());
        ensure!(
            mlp.w2.shape() == [HEADS, hidden],
            "{prefix}.layer2 is {:?}, the kernel assumes {HEADS} heads",
            mlp.w2.shape()
        );
        Ok(mlp)
    }

    /// Biases along `axis` (0 for x, 1 for y) of every query box and grid position,
    /// head-major `[8, 200 * 72]`
    fn forward(&self, gpu: &Gpu, boxes: &Tensor, axis: usize) -> Result<Tensor> {
        let rows = QUERIES * GRID;
        let out = gpu.empty(&[HEADS, rows], DType::F32)?;
        let args = [
            out.arg(),
            boxes.arg(),
            self.w1.arg(),
            self.b1.arg(),
            self.w2.arg(),
            self.b2.arg(),
            GRID.into(),
            axis.into(),
            self.w1.shape()[0].into(),
            rows.into(),
        ];
        gpu.dispatch_1d("rpb_mlp", rows, &args)?;
        Ok(out)
    }
}
