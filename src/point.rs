//! SAM 2 style interactive decoder for point prompts: prompt encoder, two-way
//! transformer, high-resolution upscaling and the multimask heads

use std::f64::consts::PI;

use eyre::{Result, ensure};

use crate::gpu::{Gpu, Tensor, launch};
use crate::ops::{Act, Attention, B256, ConvT2x2, LayerNorm, Linear, Mask, Mlp, add, bias_act, ew};
use crate::vision::{Fpn, GRID, IMAGE_SIZE};
use crate::weights::Weights;

const D: usize = 256;
const HEADS: usize = 8;
/// side of the low-resolution mask logits
pub const LOW_RES: usize = 4 * GRID;

/// A click in original image pixels; positive = foreground
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointPrompt {
    x: f32,
    y: f32,
    positive: bool,
}

impl PointPrompt {
    /// A click at `(x, y)` in original image pixels; fails on a non-finite coordinate
    pub fn new(x: f32, y: f32, positive: bool) -> Result<Self> {
        ensure!(x.is_finite() && y.is_finite(), "point ({x}, {y}) has a non-finite coordinate");
        Ok(Self { x, y, positive })
    }

    /// Horizontal position in pixels
    pub fn x(&self) -> f32 {
        self.x
    }

    /// Vertical position in pixels
    pub fn y(&self) -> f32 {
        self.y
    }

    /// Whether the click marks foreground rather than background
    pub fn positive(&self) -> bool {
        self.positive
    }
}

/// Image features the point decoder reads, computed once per image
pub struct PointImage {
    /// tracker `fpn.coarse` + no-memory embedding `[72^2, 256]`
    pub embed: Tensor,
    /// conv_s0(tracker `fpn.fine`) `[288^2, 32]`
    pub feat_s0: Tensor,
    /// conv_s1(tracker `fpn.mid`) `[144^2, 64]`
    pub feat_s1: Tensor,
}

/// One decoder pass with every output still on the device, for the video tracker
pub struct Decoded {
    /// `[4 - first, 288 * 288]` logits of mask tokens `first..4`
    pub low_res: Tensor,
    /// `[1, 4]` predicted IoU of all four mask tokens, after the sigmoid
    pub iou: Tensor,
    /// `[6 + prompts, 256]` output tokens: object score, IoU, then mask tokens 0 to 3
    pub tokens: Tensor,
}

/// Row of mask token `m` in [`Decoded::tokens`]
pub const MASK_TOKEN_ROW: usize = 2;

/// Three candidate masks from one decoder pass
pub struct PointOutput {
    /// `[3, 288 * 288]` logits. Upstream clamps them to +-32 before reuse as a mask
    /// input, but upsamples the unclamped logits for the final masks
    pub low_res: Tensor,
    pub iou: [f32; 3],
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
}

pub struct PointDecoder {
    gaussian: Vec<f32>,
    point_embed: Vec<f32>,
    not_a_point: Vec<f32>,
    no_mask: Tensor,
    mask_conv1: Linear,
    mask_ln1: LayerNorm,
    mask_conv2: Linear,
    mask_ln2: LayerNorm,
    mask_conv3: Linear,
    dense_pe: Tensor,
    /// obj score, iou and 4 mask tokens `[6, 256]`
    output_tokens: Vec<f32>,
    layers: Vec<TwoWayLayer>,
    final_attn: Attention,
    final_norm: LayerNorm,
    upscale1: ConvT2x2,
    upscale_ln: LayerNorm,
    upscale2: ConvT2x2,
    hyper: Vec<Mlp>,
    iou_head: Mlp,
    obj_score_head: Mlp,
    no_mem: Tensor,
    conv_s0: Linear,
    conv_s1: Linear,
}

/// Random Fourier features of normalized (x, y) coords in [0, 1]
fn fourier(gaussian: &[f32], x: f32, y: f32, out: &mut [f32]) {
    let (cx, cy) = (2.0 * x - 1.0, 2.0 * y - 1.0);
    let n = gaussian.len() / 2;
    for i in 0..n {
        let v = (cx * gaussian[i] + cy * gaussian[n + i]) * (2.0 * PI) as f32;
        out[i] = (v as f64).sin() as f32;
        out[n + i] = (v as f64).cos() as f32;
    }
}

impl PointDecoder {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let pe = "tracker_model.prompt_encoder";
        let md = "tracker_model.mask_decoder";
        let gaussian = wts.host(&format!("{pe}.shared_embedding.positional_embedding"))?.0;

        let mut dense = vec![0f32; GRID * GRID * D];
        for y in 0..GRID {
            for x in 0..GRID {
                let (fx, fy) = ((x as f32 + 0.5) / GRID as f32, (y as f32 + 0.5) / GRID as f32);
                fourier(&gaussian, fx, fy, &mut dense[(y * GRID + x) * D..][..D]);
            }
        }

        let mut output_tokens = wts.host(&format!("{md}.obj_score_token.weight"))?.0;
        output_tokens.extend(wts.host(&format!("{md}.iou_token.weight"))?.0);
        output_tokens.extend(wts.host(&format!("{md}.mask_tokens.weight"))?.0);

        let t = format!("{md}.transformer");
        let mut layers = Vec::new();
        for i in 0..2 {
            let l = format!("{t}.layers.{i}");
            let ln = |n: &str| LayerNorm::load(gpu, wts, &format!("{l}.{n}"), 1e-5);
            let at = |n: &str| Attention::load(gpu, wts, &format!("{l}.{n}"), HEADS);
            layers.push(TwoWayLayer {
                self_attn: at("self_attn")?,
                norm1: ln("layer_norm1")?,
                token_to_image: at("cross_attn_token_to_image")?,
                norm2: ln("layer_norm2")?,
                mlp: Mlp::load(gpu, wts, &[format!("{l}.mlp.proj_in"), format!("{l}.mlp.proj_out")], Act::Relu)?,
                norm3: ln("layer_norm3")?,
                image_to_token: at("cross_attn_image_to_token")?,
                norm4: ln("layer_norm4")?,
            });
        }

        Ok(Self {
            gaussian,
            point_embed: wts.host(&format!("{pe}.point_embed.weight"))?.0,
            not_a_point: wts.host(&format!("{pe}.not_a_point_embed.weight"))?.0,
            no_mask: wts.get(gpu, &format!("{pe}.no_mask_embed.weight"))?,
            mask_conv1: Linear::load(gpu, wts, &format!("{pe}.mask_embed.conv1"))?,
            mask_ln1: LayerNorm::load(gpu, wts, &format!("{pe}.mask_embed.layer_norm1"), 1e-6)?,
            mask_conv2: Linear::load(gpu, wts, &format!("{pe}.mask_embed.conv2"))?,
            mask_ln2: LayerNorm::load(gpu, wts, &format!("{pe}.mask_embed.layer_norm2"), 1e-6)?,
            mask_conv3: Linear::load(gpu, wts, &format!("{pe}.mask_embed.conv3"))?,
            dense_pe: gpu.upload(dense, &[GRID * GRID, D])?,
            output_tokens,
            layers,
            final_attn: Attention::load(gpu, wts, &format!("{t}.final_attn_token_to_image"), HEADS)?,
            final_norm: LayerNorm::load(gpu, wts, &format!("{t}.layer_norm_final_attn"), 1e-5)?,
            upscale1: ConvT2x2::load(gpu, wts, &format!("{md}.upscale_conv1"))?,
            upscale_ln: LayerNorm::load(gpu, wts, &format!("{md}.upscale_layer_norm"), 1e-6)?,
            upscale2: ConvT2x2::load(gpu, wts, &format!("{md}.upscale_conv2"))?,
            hyper: (0..4)
                .map(|i| Mlp::load_in_out(gpu, wts, &format!("{md}.output_hypernetworks_mlps.{i}"), 3, Act::Relu))
                .collect::<Result<_>>()?,
            iou_head: Mlp::load_in_out(gpu, wts, &format!("{md}.iou_prediction_head"), 3, Act::Relu)?,
            obj_score_head: Mlp::load_in_out(gpu, wts, &format!("{md}.pred_obj_score_head"), 3, Act::Relu)?,
            no_mem: wts.get(gpu, "tracker_model.no_memory_embedding")?,
            conv_s0: Linear::load(gpu, wts, &format!("{md}.conv_s0"))?,
            conv_s1: Linear::load(gpu, wts, &format!("{md}.conv_s1"))?,
        })
    }

    /// Tracker neck outputs -> decoder image features
    pub fn prepare_image(&self, gpu: &Gpu, fpn: &Fpn) -> Result<PointImage> {
        let embed = gpu.clone_tensor(&fpn.coarse)?;
        launch!(gpu, "add_row", ew(embed.len()), B256, 0; &embed, &self.no_mem)?;
        Ok(PointImage {
            embed,
            feat_s0: self.conv_s0.forward(gpu, &fpn.fine)?,
            feat_s1: self.conv_s1.forward(gpu, &fpn.mid)?,
        })
    }

    /// Sparse prompt tokens: one per point plus the padding point. Without points (a
    /// tracked frame) upstream still passes one point labelled -1 before the padding, so
    /// there are two padding tokens
    fn sparse_tokens(&self, points: &[PointPrompt], width: usize, height: usize) -> Vec<f32> {
        if points.is_empty() {
            return [self.not_a_point.as_slice(), self.not_a_point.as_slice()].concat();
        }
        let n = points.len() + 1;
        let mut out = vec![0f32; n * D];
        let size = IMAGE_SIZE as f32;
        for (i, p) in points.iter().enumerate() {
            // pixels -> 1008 input frame, shifted to the pixel center, normalized back
            let x = (p.x / width as f32 * size + 0.5) / size;
            let y = (p.y / height as f32 * size + 0.5) / size;
            let row = &mut out[i * D..][..D];
            fourier(&self.gaussian, x, y, row);
            let label = p.positive as usize;
            for (v, e) in row.iter_mut().zip(&self.point_embed[label * D..][..D]) {
                *v += e;
            }
        }
        out[(n - 1) * D..].copy_from_slice(&self.not_a_point);
        out
    }

    /// Dense prompt embedding `[72^2, 256]` from a low-res mask `[288^2]` or the
    /// no-mask embedding
    fn dense(&self, gpu: &Gpu, mask: Option<&Tensor>) -> Result<Tensor> {
        let Some(mask) = mask else {
            let out = gpu.zeros(&[GRID * GRID, D])?;
            launch!(gpu, "add_row", ew(out.len()), B256, 0; &out, &self.no_mask)?;
            return Ok(out);
        };
        let mut side = LOW_RES;
        let mut x = gpu.clone_tensor(mask)?.reshape(&[side * side, 1]);
        for (conv, ln) in [(&self.mask_conv1, &self.mask_ln1), (&self.mask_conv2, &self.mask_ln2)] {
            let c = x.cols();
            let patches = gpu.empty(&[(side / 2) * (side / 2), c * 4])?;
            launch!(gpu, "patchify2x2", ew(patches.len()), B256, 0; &patches, &x, side, c)?;
            side /= 2;
            x = conv.forward(gpu, &patches)?;
            ln.forward_inplace(gpu, &x)?;
            bias_act(gpu, &x, None, Act::Gelu)?;
        }
        self.mask_conv3.forward(gpu, &x)
    }

    pub fn forward(
        &self,
        gpu: &Gpu,
        img: &PointImage,
        points: &[PointPrompt],
        width: usize,
        height: usize,
        mask_input: Option<&Tensor>,
    ) -> Result<PointOutput> {
        let out = self.decode(gpu, &img.embed, img, points, (width, height), mask_input, 1)?;
        let iou = gpu.download(&out.iou)?;
        Ok(PointOutput { low_res: out.low_res, iou: [iou[1], iou[2], iou[3]] })
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
        (width, height): (usize, usize),
        mask_input: Option<&Tensor>,
        first: usize,
    ) -> Result<Decoded> {
        assert!(first < 4, "mask tokens 0 to 3");
        let mut tokens = self.output_tokens.clone();
        tokens.extend(self.sparse_tokens(points, width, height));
        let n = tokens.len() / D;
        let prompt_pe = gpu.upload(tokens, &[n, D])?;
        let queries = gpu.clone_tensor(&prompt_pe)?;

        let keys = self.dense(gpu, mask_input)?;
        crate::ops::add_assign(gpu, &keys, embed)?;
        let none = Mask::default();

        let mut queries = queries;
        for (i, l) in self.layers.iter().enumerate() {
            if i == 0 {
                // the first layer replaces the queries with self-attention (no residual, no pe)
                queries = l.self_attn.forward(gpu, &queries, &queries, &queries, none)?;
            } else {
                let q = add(gpu, &queries, &prompt_pe)?;
                l.self_attn.forward_residual(gpu, &q, &q, &queries, none, &queries)?;
            }
            l.norm1.forward_inplace(gpu, &queries)?;

            let q = add(gpu, &queries, &prompt_pe)?;
            let k = add(gpu, &keys, &self.dense_pe)?;
            l.token_to_image.forward_residual(gpu, &q, &k, &keys, none, &queries)?;
            l.norm2.forward_inplace(gpu, &queries)?;

            let m = l.mlp.forward(gpu, &queries)?;
            crate::ops::add_assign(gpu, &queries, &m)?;
            l.norm3.forward_inplace(gpu, &queries)?;

            let q = add(gpu, &queries, &prompt_pe)?;
            l.image_to_token.forward_residual(gpu, &k, &q, &queries, none, &keys)?;
            l.norm4.forward_inplace(gpu, &keys)?;
        }
        let q = add(gpu, &queries, &prompt_pe)?;
        let k = add(gpu, &keys, &self.dense_pe)?;
        self.final_attn.forward_residual(gpu, &q, &k, &keys, none, &queries)?;
        self.final_norm.forward_inplace(gpu, &queries)?;

        let up = self.upscale1.forward(gpu, &keys, GRID, GRID)?;
        crate::ops::add_assign(gpu, &up, &img.feat_s1)?;
        self.upscale_ln.forward_inplace(gpu, &up)?;
        bias_act(gpu, &up, None, Act::Gelu)?;
        let up = self.upscale2.forward(gpu, &up, 2 * GRID, 2 * GRID)?;
        crate::ops::add_assign(gpu, &up, &img.feat_s0)?;
        bias_act(gpu, &up, None, Act::Gelu)?;

        // mask token 0 is the single-mask output, 1 to 3 the multimask ones
        let side = LOW_RES * LOW_RES;
        let low_res = gpu.empty(&[4 - first, side])?;
        let proj = Linear::new(up, None);
        for (j, m) in (first..4).enumerate() {
            let hyper = self.hyper[m].forward(gpu, &row(gpu, &queries, MASK_TOKEN_ROW + m)?)?;
            proj.forward_raw(gpu, hyper.ptr(), hyper.cols(), 1, low_res.at(j * side), 0.0)?;
        }

        let iou = self.iou_head.forward(gpu, &row(gpu, &queries, 1)?)?;
        bias_act(gpu, &iou, None, Act::Sigmoid)?;
        Ok(Decoded { low_res, iou, tokens: queries })
    }

    /// `[1, 1]` object score logit of a decoder pass; the object is present when it is
    /// above 0. The image path has no use for it, so [`Self::decode`] leaves it out
    pub fn obj_score(&self, gpu: &Gpu, out: &Decoded) -> Result<Tensor> {
        self.obj_score_head.forward(gpu, &row(gpu, &out.tokens, 0)?)
    }
}

/// Row `r` of `x` as a `[1, cols]` tensor
pub fn row(gpu: &Gpu, x: &Tensor, r: usize) -> Result<Tensor> {
    let d = x.cols();
    let out = gpu.empty(&[1, d])?;
    gpu.copy_dtod(x.at(r * d), out.ptr(), d * 4)?;
    Ok(out)
}

/// One low-res mask `[288^2]` clamped to +-32, as the next pass's mask input
pub fn clamped_mask(gpu: &Gpu, low_res: &Tensor, index: usize) -> Result<Tensor> {
    let n = LOW_RES * LOW_RES;
    let out = gpu.empty(&[n])?;
    gpu.copy_dtod(low_res.at(index * n), out.ptr(), n * 4)?;
    launch!(gpu, "clamp", ew(n), B256, 0; &out, -32.0f32, 32.0f32)?;
    Ok(out)
}

/// Upstream's `postprocess_masks` hole filling: background regions (logit <= 0), 8-connected,
/// of at most `max_area` pixels become foreground (+10)
pub fn fill_holes(logits: &mut [f32], side: usize, max_area: usize) {
    let mut seen = vec![false; logits.len()];
    let mut stack = Vec::new();
    let mut component = Vec::new();
    for start in 0..logits.len() {
        if seen[start] || logits[start] > 0.0 {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        component.clear();
        while let Some(p) = stack.pop() {
            component.push(p);
            let (y, x) = ((p / side) as isize, (p % side) as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || nx < 0 || ny >= side as isize || nx >= side as isize {
                        continue;
                    }
                    let q = ny as usize * side + nx as usize;
                    if !seen[q] && logits[q] <= 0.0 {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        if component.len() <= max_area {
            for &p in &component {
                logits[p] = 10.0;
            }
        }
    }
}

/// Hole-fills `n` low-res masks `[n, 288^2]` on the host, upsamples them bilinearly to
/// the original size and thresholds at 0, returning `[n, height * width]` 0/1 bytes
/// into `out`, which must hold `n * width * height` bytes
pub fn full_res_masks(
    gpu: &Gpu,
    low_res: &Tensor,
    n: usize,
    width: usize,
    height: usize,
    out: &mut [u8],
) -> Result<()> {
    let side = LOW_RES;
    assert_eq!(low_res.len(), n * side * side);
    let mut logits = gpu.download(low_res)?;
    for m in logits.chunks_mut(side * side) {
        fill_holes(m, side, 256);
    }
    let filled = gpu.upload(logits, &[n, side * side])?;
    let up = gpu.empty(&[n, height * width])?;
    launch!(gpu, "resize_bilinear", ew(up.len()), B256, 0; &up, &filled, side, side, height, width)?;
    let mask = gpu.alloc_bytes(up.len())?;
    launch!(gpu, "threshold_u8", ew(up.len()), B256, 0; mask.slice::<u8>(), &up, 0.0f32)?;
    ensure!(out.len() == up.len(), "mask buffer of {} bytes for {} pixels", out.len(), up.len());
    gpu.download_into(&mask, out)
}
