//! SAM 3 detector: geometry (box) encoder, fusion encoder, DETR decoder with presence
//! token, dot-product scoring and the segmentation head

use eyre::{Result, WrapErr, ensure};

use crate::gpu::{Gpu, Tensor, launch};
use crate::nvtx::range;
use crate::ops::{
    ACT_RELU, Attention, B256, Conv3x3, LayerNorm, Linear, Mask, Mlp, add, concat_rows, ew, group_norm, upload_i32,
};
use crate::posenc::box_encoding;
use crate::text::TextFeatures;
use crate::vision::GRID;
use crate::weights::Weights;

const D: usize = 256;
const HEADS: usize = 8;
pub const QUERIES: usize = 200;
const ROI: usize = 7;

/// A box prompt in normalized cxcywh with its label (true = positive)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxPrompt {
    cxcywh: [f32; 4],
    positive: bool,
}

impl BoxPrompt {
    /// A box from pixel corners `[x0, y0, x1, y1]` in a `width` x `height` image
    ///
    /// The box may extend past the image by up to one image size on each side
    pub fn from_xyxy(xyxy: [f32; 4], width: usize, height: usize, positive: bool) -> Result<Self> {
        ensure!(width > 0 && height > 0, "image size {width}x{height} is empty");
        ensure!(xyxy.iter().all(|v| v.is_finite()), "box {xyxy:?} has a non-finite coordinate");
        let (w, h) = (width as f32, height as f32);
        let [x0, y0, x1, y1] = xyxy;
        Self::from_cxcywh([(x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h, (x1 - x0) / w, (y1 - y0) / h], positive)
            .wrap_err_with(|| format!("box {xyxy:?} in a {width}x{height} image"))
    }

    /// A box from its center and size as fractions of the image width and height
    ///
    /// Width and height must be positive, and the corners within `[-1, 2]`: no more than
    /// one image size past each edge. That bounds the ROI pooling work per box
    pub fn from_cxcywh(cxcywh: [f32; 4], positive: bool) -> Result<Self> {
        let [cx, cy, w, h] = cxcywh;
        ensure!(cxcywh.iter().all(|v| v.is_finite()), "box {cxcywh:?} has a non-finite value");
        ensure!(w > 0.0 && h > 0.0, "box {cxcywh:?} has no area");
        let corners = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
        ensure!(
            corners.iter().all(|c| (-1.0..=2.0).contains(c)),
            "box {cxcywh:?} reaches more than one image size past the image"
        );
        Ok(Self { cxcywh, positive })
    }

    /// Center x, center y, width and height as fractions of the image size
    pub fn cxcywh(&self) -> [f32; 4] {
        self.cxcywh
    }

    /// Whether the box is a positive exemplar rather than a negative one
    pub fn positive(&self) -> bool {
        self.positive
    }
}

/// Image features the detector reads, computed once per image
pub struct DetectorImage {
    /// fpn levels `[288^2, 256]`, `[144^2, 256]`, `[72^2, 256]`
    pub fpn: Vec<Tensor>,
    /// LayerNorm(fpn2) for box ROI pooling
    pub fpn2_normed: Tensor,
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
            fc1: Linear::load(gpu, wts, &format!("{p}.mlp.fc1"))?,
            fc2: Linear::load(gpu, wts, &format!("{p}.mlp.fc2"))?,
        })
    }

    fn ffn(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        let h = self.norm3.forward(gpu, x)?;
        let hidden = self.fc1.forward_act(gpu, &h, ACT_RELU)?;
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

pub struct Detector {
    vision_norm: LayerNorm,
    geometry: GeometryEncoder,
    encoder: Vec<EncoderLayer>,
    decoder: Vec<DecoderLayer>,
    query_embed: Tensor,
    reference_points: Tensor,
    presence_token: Tensor,
    ref_point_head: Mlp,
    rpb_x: Mlp,
    rpb_y: Mlp,
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
    /// sine position encoding of the 72x72 grid `[5184, 256]`
    pos: Tensor,
}

impl Detector {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "detector_model";
        let g = format!("{p}.geometry_encoder");
        let geometry = GeometryEncoder {
            direct: Linear::load(gpu, wts, &format!("{g}.boxes_direct_project"))?,
            pool: Linear::load(gpu, wts, &format!("{g}.boxes_pool_project"))?,
            pos: Linear::load(gpu, wts, &format!("{g}.boxes_pos_enc_project"))?,
            label: wts.get(gpu, &format!("{g}.label_embed.weight"))?,
            cls: wts.get(gpu, &format!("{g}.cls_embed.weight"))?,
            final_proj: Linear::load(gpu, wts, &format!("{g}.final_proj"))?,
            prompt_norm: LayerNorm::load(gpu, wts, &format!("{g}.prompt_layer_norm"), 1e-5)?,
            layers: (0..3).map(|i| EncoderLayer::load(gpu, wts, &format!("{g}.layers.{i}"))).collect::<Result<_>>()?,
            out_norm: LayerNorm::load(gpu, wts, &format!("{g}.output_layer_norm"), 1e-5)?,
        };
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
                fc1: Linear::load(gpu, wts, &format!("{l}.mlp.fc1"))?,
                fc2: Linear::load(gpu, wts, &format!("{l}.mlp.fc2"))?,
                mlp_norm: ln("mlp_layer_norm")?,
            });
        }
        let s = format!("{p}.dot_product_scoring");
        let m = format!("{p}.mask_decoder");
        Ok(Self {
            vision_norm: LayerNorm::load(gpu, wts, &format!("{g}.vision_layer_norm"), 1e-5)?,
            geometry,
            encoder,
            decoder,
            query_embed: wts.get(gpu, &format!("{dd}.query_embed.weight"))?,
            reference_points: wts.get(gpu, &format!("{dd}.reference_points.weight"))?,
            presence_token: wts.get(gpu, &format!("{dd}.presence_token.weight"))?,
            ref_point_head: Mlp::load_numbered(gpu, wts, &format!("{dd}.ref_point_head"), 2, ACT_RELU)?,
            rpb_x: Mlp::load_numbered(gpu, wts, &format!("{dd}.box_rpb_embed_x"), 2, ACT_RELU)?,
            rpb_y: Mlp::load_numbered(gpu, wts, &format!("{dd}.box_rpb_embed_y"), 2, ACT_RELU)?,
            box_head: Mlp::load_numbered(gpu, wts, &format!("{dd}.box_head"), 3, ACT_RELU)?,
            presence_head: Mlp::load_numbered(gpu, wts, &format!("{dd}.presence_head"), 3, ACT_RELU)?,
            presence_norm: LayerNorm::load(gpu, wts, &format!("{dd}.presence_layer_norm"), 1e-5)?,
            out_norm: LayerNorm::load(gpu, wts, &format!("{dd}.output_layer_norm"), 1e-5)?,
            text_mlp: Mlp::load_numbered(gpu, wts, &format!("{s}.text_mlp"), 2, ACT_RELU)?,
            text_mlp_norm: LayerNorm::load(gpu, wts, &format!("{s}.text_mlp_out_norm"), 1e-5)?,
            text_proj: Linear::load(gpu, wts, &format!("{s}.text_proj"))?,
            query_proj: Linear::load(gpu, wts, &format!("{s}.query_proj"))?,
            seg_attn: Attention::load(gpu, wts, &format!("{m}.prompt_cross_attn"), HEADS)?,
            seg_attn_norm: LayerNorm::load(gpu, wts, &format!("{m}.prompt_cross_attn_norm"), 1e-5)?,
            // the pixel decoder has three convs but only two upsampling stages run
            pixel_convs: (0..2)
                .map(|i| Conv3x3::load(gpu, wts, &format!("{m}.pixel_decoder.conv_layers.{i}")))
                .collect::<Result<_>>()?,
            pixel_norms: (0..2)
                .map(|i| {
                    let n = format!("{m}.pixel_decoder.norms.{i}");
                    Ok((wts.get(gpu, &format!("{n}.weight"))?, wts.get(gpu, &format!("{n}.bias"))?))
                })
                .collect::<Result<_>>()?,
            instance_proj: Linear::load(gpu, wts, &format!("{m}.instance_projection"))?,
            mask_embed: Mlp::load_indexed(gpu, wts, &format!("{m}.mask_embedder"), 3, ACT_RELU)?,
            pos: gpu.upload(crate::posenc::sine_2d(GRID), &[GRID * GRID, D])?,
        })
    }

    pub fn prepare_image(&self, gpu: &Gpu, fpn: Vec<Tensor>) -> Result<DetectorImage> {
        let fpn2_normed = self.vision_norm.forward(gpu, &fpn[2])?;
        Ok(DetectorImage { fpn, fpn2_normed })
    }

    /// Geometry tokens: one per box, then the CLS token, all valid
    fn encode_geometry(&self, gpu: &Gpu, img: &DetectorImage, boxes: &[BoxPrompt]) -> Result<Tensor> {
        let g = &self.geometry;
        let n = boxes.len();
        let embeds = if n == 0 {
            gpu.clone_tensor(&g.cls)?
        } else {
            let raw: Vec<f32> = boxes.iter().flat_map(|b| b.cxcywh).collect();
            let raw = gpu.upload(raw, &[n, 4])?;
            let x = g.direct.forward(gpu, &raw)?;

            let side = GRID as f32;
            let xyxy: Vec<f32> = boxes
                .iter()
                .flat_map(|b| {
                    let [cx, cy, w, h] = b.cxcywh;
                    [(cx - 0.5 * w) * side, (cy - 0.5 * h) * side, (cx + 0.5 * w) * side, (cy + 0.5 * h) * side]
                })
                .collect();
            let xyxy = gpu.upload(xyxy, &[n, 4])?;
            let pooled = gpu.empty(&[n, D * ROI * ROI])?;
            launch!(gpu, "roi_align", ew(pooled.len()), B256, 0;
                &pooled, &img.fpn2_normed, &xyxy, n, GRID, GRID, D, ROI)?;
            g.pool.forward_residual(gpu, &pooled, &x)?;

            let enc: Vec<f32> = boxes.iter().flat_map(|b| box_encoding(b.cxcywh)).collect();
            let enc = gpu.upload(enc, &[n, 258])?;
            g.pos.forward_residual(gpu, &enc, &x)?;

            let labels: Vec<i32> = boxes.iter().map(|b| b.positive as i32).collect();
            let labels = upload_i32(gpu, labels)?;
            let label = gpu.empty(&[n, D])?;
            launch!(gpu, "gather_rows", ew(n * D), B256, 0; &label, &g.label, labels.ptr(), n, D)?;
            crate::ops::add_assign(gpu, &x, &label)?;
            concat_rows(gpu, &x, &g.cls)?
        };

        let x = g.final_proj.forward(gpu, &embeds)?;
        g.prompt_norm.forward_inplace(gpu, &x)?;
        let mem_pos = add(gpu, &img.fpn[2], &self.pos)?;
        for l in &g.layers {
            let h = l.norm1.forward(gpu, &x)?;
            l.self_attn.forward_residual(gpu, &h, &h, &h, Mask::default(), &x)?;
            let h = l.norm2.forward(gpu, &x)?;
            l.cross_attn.forward_residual(gpu, &h, &mem_pos, &img.fpn[2], Mask::default(), &x)?;
            l.ffn(gpu, &x)?;
        }
        g.out_norm.forward_inplace(gpu, &x)?;
        Ok(x)
    }

    pub fn forward(
        &self,
        gpu: &Gpu,
        img: &DetectorImage,
        text: &TextFeatures,
        boxes: &[BoxPrompt],
    ) -> Result<DetectorOutput> {
        let geo = {
            let _r = range(c"geometry");
            self.encode_geometry(gpu, img, boxes)?
        };
        let prompt = concat_rows(gpu, &text.feats, &geo)?;
        let mut pad = text.pad.to_vec();
        pad.resize(prompt.rows(), 0);
        let pad = gpu.upload_buf(pad)?;
        let prompt_mask = Mask { key_pad: pad.ptr(), ..Mask::default() };

        let memory = {
            let _r = range(c"fusion_encoder");
            self.fusion_encoder(gpu, img, &prompt, prompt_mask)?
        };
        let (hs, boxes, presence) = {
            let _r = range(c"decoder");
            self.decode(gpu, &memory, &prompt, prompt_mask)?
        };
        let logits = self.score(gpu, &hs, &prompt, &pad, prompt.rows())?;
        let masks = {
            let _r = range(c"seg_head");
            self.segment(gpu, img, &memory, &hs, &prompt, prompt_mask)?
        };
        Ok(DetectorOutput { logits, presence, boxes, masks })
    }

    fn fusion_encoder(&self, gpu: &Gpu, img: &DetectorImage, prompt: &Tensor, prompt_mask: Mask) -> Result<Tensor> {
        let x = gpu.clone_tensor(&img.fpn[2])?;
        for l in &self.encoder {
            let h = l.norm1.forward(gpu, &x)?;
            let qk = add(gpu, &h, &self.pos)?;
            l.self_attn.forward_residual(gpu, &qk, &qk, &h, Mask::default(), &x)?;
            drop((h, qk));
            let h = l.norm2.forward(gpu, &x)?;
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
        let n = QUERIES + 1;
        // row 0 is the presence token, which has a zero query position
        let x = concat_rows(gpu, &self.presence_token, &self.query_embed)?;
        let mut boxes = gpu.clone_tensor(&self.reference_points)?;
        crate::ops::bias_act(gpu, &boxes, None, crate::ops::ACT_SIGMOID)?;
        let mem_pos = add(gpu, memory, &self.pos)?;
        let qpos = gpu.zeros(&[n, D])?;
        let bias = gpu.empty(&[HEADS, n, GRID * GRID])?;
        let mut hs = None;
        let mut presence = None;

        for (i, l) in self.decoder.iter().enumerate() {
            let sine = gpu.empty(&[QUERIES, 2 * D])?;
            launch!(gpu, "box_sine_embed", ew(sine.len()), B256, 0; &sine, &boxes, QUERIES)?;
            let pos = self.ref_point_head.forward(gpu, &sine)?;
            gpu.copy_dtod(pos.ptr(), qpos.at(D), pos.len() * 4)?;
            self.box_rpb(gpu, &boxes, &bias)?;

            let qk = add(gpu, &x, &qpos)?;
            l.self_attn.forward_residual(gpu, &qk, &qk, &x, Mask::default(), &x)?;
            l.self_norm.forward_inplace(gpu, &x)?;

            let q = add(gpu, &x, &qpos)?;
            l.text_attn.forward_residual(gpu, &q, prompt, prompt, prompt_mask, &x)?;
            l.text_norm.forward_inplace(gpu, &x)?;

            let q = add(gpu, &x, &qpos)?;
            let rpb = Mask { bias: bias.ptr(), bias_div: 1, ..Mask::default() };
            l.image_attn.forward_residual(gpu, &q, &mem_pos, memory, rpb, &x)?;
            l.image_norm.forward_inplace(gpu, &x)?;

            let hidden = l.fc1.forward_act(gpu, &x, ACT_RELU)?;
            l.fc2.forward_residual(gpu, &hidden, &x)?;
            l.mlp_norm.forward_inplace(gpu, &x)?;

            let queries = gpu.empty(&[QUERIES, D])?;
            gpu.copy_dtod(x.at(D), queries.ptr(), QUERIES * D * 4)?;
            let normed = self.out_norm.forward(gpu, &queries)?;
            let delta = self.box_head.forward(gpu, &normed)?;
            let refined = gpu.empty(&[QUERIES, 4])?;
            launch!(gpu, "box_refine", ew(QUERIES * 4), B256, 0; &refined, &boxes, &delta, QUERIES)?;
            boxes = refined;

            if i == self.decoder.len() - 1 {
                let p = gpu.empty(&[1, D])?;
                gpu.copy_dtod(x.ptr(), p.ptr(), D * 4)?;
                let p = self.presence_norm.forward(gpu, &p)?;
                presence = Some(self.presence_head.forward(gpu, &p)?);
                hs = Some(normed);
            }
        }
        Ok((hs.unwrap(), boxes, presence.unwrap()))
    }

    /// Box relative position bias for the image cross-attention, `[8, 201, 5184]`
    fn box_rpb(&self, gpu: &Gpu, boxes: &Tensor, bias: &Tensor) -> Result<()> {
        let mut deltas = Vec::new();
        for (axis, mlp) in [(0i32, &self.rpb_x), (1, &self.rpb_y)] {
            let d = gpu.empty(&[QUERIES * GRID, 2])?;
            launch!(gpu, "rpb_deltas", ew(d.len()), B256, 0; &d, boxes, QUERIES, GRID, axis)?;
            deltas.push(mlp.forward(gpu, &d)?);
        }
        launch!(gpu, "rpb_combine", ew(bias.len()), B256, 0; bias, &deltas[0], &deltas[1], QUERIES, GRID, HEADS)
    }

    fn score(&self, gpu: &Gpu, hs: &Tensor, prompt: &Tensor, pad: &crate::gpu::Buf, len: usize) -> Result<Tensor> {
        let p = self.text_mlp.forward(gpu, prompt)?;
        crate::ops::add_assign(gpu, &p, prompt)?;
        self.text_mlp_norm.forward_inplace(gpu, &p)?;
        let pooled = gpu.empty(&[1, D])?;
        launch!(gpu, "masked_mean", (1, 1, 1), B256, 0; &pooled, &p, pad.ptr(), len, D)?;
        let pp = self.text_proj.forward(gpu, &pooled)?;
        let ph = self.query_proj.forward(gpu, hs)?;
        let scores = gpu.empty(&[QUERIES])?;
        let scale = 1.0 / (D as f32).sqrt();
        launch!(gpu, "dot_scores", ((QUERIES as u32).div_ceil(8), 1, 1), B256, 0; &scores, &ph, &pp, QUERIES, D, scale, 12.0f32)?;
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
        let h = self.seg_attn_norm.forward(gpu, memory)?;
        let enc = gpu.clone_tensor(memory)?;
        self.seg_attn.forward_residual(gpu, &h, prompt, prompt, prompt_mask, &enc)?;
        drop(h);

        let mut prev = enc;
        let mut side = GRID;
        for (i, fine) in [&img.fpn[1], &img.fpn[0]].into_iter().enumerate() {
            let up = gpu.empty(&[4 * side * side, D])?;
            launch!(gpu, "upsample2_add", ew(up.len()), B256, 0; &up, fine, &prev, side, side, D)?;
            side *= 2;
            prev = self.pixel_convs[i].forward(gpu, &up, side, side)?;
            let (w, b) = &self.pixel_norms[i];
            group_norm(gpu, &prev, 8, w, b, 1e-5, ACT_RELU)?;
        }
        let instance = self.instance_proj.forward(gpu, &prev)?;
        drop(prev);
        let embed = self.mask_embed.forward(gpu, hs)?;
        // masks [200, 288^2] = embed [200, 256] @ instance^T
        let masks = gpu.empty(&[QUERIES, side * side])?;
        let lin = Linear::new(instance, None);
        lin.forward_raw(gpu, embed.ptr(), D, QUERIES, masks.ptr(), 0.0)?;
        Ok(masks)
    }
}
