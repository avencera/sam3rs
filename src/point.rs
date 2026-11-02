//! SAM 2 style interactive decoder for point prompts: prompt encoder, two-way
//! transformer, high-resolution upscaling and the multimask heads

use std::f64::consts::PI;
use std::ops::Range;

use eyre::{Result, ensure};

use crate::gpu::{Arg, Buf, Gpu, Mat, Tensor, launch};
use crate::ops::{Act, Attention, B256, ConvT2x2, LayerNorm, Linear, Mlp, add, add_assign, bias_act, ew};
use crate::vision::{Fpn, GRID, IMAGE_SIZE};
use crate::weights::Weights;

const D: usize = 256;
const HEADS: usize = 8;
/// Side of the low-resolution mask logits
pub const LOW_RES: usize = 4 * GRID;
/// Most mask pixels [`MaskResize`] resizes per launch: 1 GiB of FP32 staging
const MASK_GROUP_ELEMENTS: usize = 1 << 28;

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

/// Row of mask token `m` in [`Decoded::tokens`]
pub const MASK_TOKEN_ROW: usize = 2;

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
    /// Object score, IoU and 4 mask tokens `[6, 256]`
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

    /// Dense prompt embeddings `[k * 72^2, 256]` of `k` prompts, from their low-res masks
    /// `[k, 288^2]` or the no-mask embedding
    fn dense(&self, gpu: &Gpu, mask: Option<&Tensor>, k: usize) -> Result<Tensor> {
        let Some(mask) = mask else {
            let out = gpu.zeros(&[k * GRID * GRID, D])?;
            launch!(gpu, "add_row", ew(out.len()), B256, 0; &out, &self.no_mask)?;
            return Ok(out);
        };
        let mut side = LOW_RES;
        assert_eq!(mask.len(), k * side * side, "one low-res mask per prompt");
        // the k masks stacked as one image k times as tall: the 2x2 patches and the
        // rowwise convs and norms never cross from one mask into the next
        let mut x = gpu.clone_tensor(mask)?.reshape(&[k * side * side, 1]);
        for (conv, ln) in [(&self.mask_conv1, &self.mask_ln1), (&self.mask_conv2, &self.mask_ln2)] {
            let c = x.cols();
            let patches = gpu.empty(&[k * (side / 2) * (side / 2), c * 4])?;
            launch!(gpu, "patchify2x2", ew(patches.len()), B256, 0; &patches, &x, side, c)?;
            side /= 2;
            x = conv.forward(gpu, &patches)?;
            ln.forward_inplace(gpu, &x)?;
            bias_act(gpu, &x, None, Act::Gelu)?;
        }
        self.mask_conv3.forward(gpu, &x)
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
    /// tokens and no masks. One prompt issues exactly the work [`Self::decode`] always did
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
        assert!(first < 4, "mask tokens 0 to 3");
        let k = prompts.len();
        ensure!(k > 0, "a decoder batch needs at least one prompt");
        let mut tokens = Vec::new();
        for points in prompts {
            tokens.extend_from_slice(&self.output_tokens);
            tokens.extend(self.sparse_tokens(points, width, height));
        }
        ensure!(tokens.len().is_multiple_of(k * D), "the prompts of one decoder batch need equal point counts");
        let n = tokens.len() / (k * D);
        ensure!(tokens.len() == k * n * D, "the prompts of one decoder batch need equal point counts");
        let prompt_pe = gpu.upload(tokens, &[k * n, D])?;
        let queries = gpu.clone_tensor(&prompt_pe)?;

        let keys = self.dense(gpu, mask_input, k)?;
        add_tiled(gpu, &keys, embed)?;
        let dense_pe = |keys: &Tensor| -> Result<Tensor> {
            let out = gpu.clone_tensor(keys)?;
            add_tiled(gpu, &out, &self.dense_pe)?;
            Ok(out)
        };

        let mut queries = queries;
        for (i, l) in self.layers.iter().enumerate() {
            if i == 0 {
                // the first layer replaces the queries with self-attention (no residual, no pe)
                queries = l.self_attn.forward_batch(gpu, &queries, &queries, &queries, k)?;
            } else {
                let q = add(gpu, &queries, &prompt_pe)?;
                l.self_attn.forward_residual_batch(gpu, &q, &q, &queries, k, &queries)?;
            }
            l.norm1.forward_inplace(gpu, &queries)?;

            let q = add(gpu, &queries, &prompt_pe)?;
            let kp = dense_pe(&keys)?;
            l.token_to_image.forward_residual_batch(gpu, &q, &kp, &keys, k, &queries)?;
            l.norm2.forward_inplace(gpu, &queries)?;

            let m = l.mlp.forward(gpu, &queries)?;
            add_assign(gpu, &queries, &m)?;
            l.norm3.forward_inplace(gpu, &queries)?;

            let q = add(gpu, &queries, &prompt_pe)?;
            l.image_to_token.forward_residual_batch(gpu, &kp, &q, &queries, k, &keys)?;
            l.norm4.forward_inplace(gpu, &keys)?;
        }
        let q = add(gpu, &queries, &prompt_pe)?;
        let kp = dense_pe(&keys)?;
        self.final_attn.forward_residual_batch(gpu, &q, &kp, &keys, k, &queries)?;
        self.final_norm.forward_inplace(gpu, &queries)?;

        // the k image grids stacked as one grid k times as tall, which the transposed
        // convs' pixel shuffle keeps prompt by prompt
        let up = self.upscale1.forward(gpu, &keys, k * GRID, GRID)?;
        add_tiled(gpu, &up, &img.feat_s1)?;
        self.upscale_ln.forward_inplace(gpu, &up)?;
        bias_act(gpu, &up, None, Act::Gelu)?;
        let up = self.upscale2.forward(gpu, &up, 2 * k * GRID, 2 * GRID)?;
        add_tiled(gpu, &up, &img.feat_s0)?;
        bias_act(gpu, &up, None, Act::Gelu)?;

        // mask token 0 is the single-mask output, 1 to 3 the multimask ones
        let side = LOW_RES * LOW_RES;
        let masks = 4 - first;
        let low_res = gpu.empty(&[k * masks, side])?;
        let c = up.cols();
        for (j, m) in (first..4).enumerate() {
            let tokens = rows(gpu, &queries, (0..k).map(|b| b * n + MASK_TOKEN_ROW + m))?;
            let hyper = self.hyper[m].forward(gpu, &tokens)?;
            // prompt b's mask j is its hypernetwork row times its own upscaled map
            let x = Mat::new(hyper.ptr(), hyper.cols()).stride(hyper.cols());
            let maps = Mat::new(up.ptr(), c).t().stride(side * c);
            let out = Mat::new(low_res.at(j * side), side).stride(masks * side);
            gpu.gemm(1, side, c, x, maps, out, k, 1.0, 0.0)?;
        }

        let iou = self.iou_head.forward(gpu, &rows(gpu, &queries, (0..k).map(|b| b * n + 1))?)?;
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

/// Rows `idx` of `x`, in that order, as one `[idx.len(), cols]` tensor
pub fn rows(gpu: &Gpu, x: &Tensor, idx: impl IntoIterator<Item = usize>) -> Result<Tensor> {
    let idx: Vec<i32> = idx.into_iter().map(|r| r as i32).collect();
    let d = x.cols();
    assert!(idx.iter().all(|&r| (r as usize) < x.rows()), "rows {idx:?} of {} rows", x.rows());
    if let [r] = idx[..] {
        return row(gpu, x, r as usize);
    }
    let out = gpu.empty(&[idx.len(), d])?;
    let idx = gpu.upload_buf(idx)?;
    launch!(gpu, "gather_rows", ew(out.len()), B256, 0; &out, x, idx.slice::<i32>(), d)?;
    Ok(out)
}

/// `x += b`, with `b` repeated over `x` when `x` stacks several prompts' copies of it
fn add_tiled(gpu: &Gpu, x: &Tensor, b: &Tensor) -> Result<()> {
    assert!(x.len().is_multiple_of(b.len()), "{:?} is not a stack of {:?}", x.shape(), b.shape());
    if x.len() == b.len() {
        return add_assign(gpu, x, b);
    }
    launch!(gpu, "add_row", ew(x.len()), B256, 0; x, b)
}

/// First index of the maximum, like `torch.argmax`
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// The best candidate of each prompt from a pass's `low_res [K * candidates, 288^2]`:
/// row `best[b]` of prompt `b`'s candidates, as `[K, 288^2]`
pub fn pick_masks(gpu: &Gpu, low_res: &Tensor, best: &[usize]) -> Result<Tensor> {
    let candidates = low_res.rows() / best.len();
    assert_eq!(low_res.rows(), best.len() * candidates, "{} candidates of {} prompts", low_res.rows(), best.len());
    rows(gpu, low_res, best.iter().enumerate().map(|(b, &i)| b * candidates + i))
}

/// [`pick_masks`] clamped to +-32, as the next pass's mask inputs
pub fn clamped_masks(gpu: &Gpu, low_res: &Tensor, best: &[usize]) -> Result<Tensor> {
    let out = pick_masks(gpu, low_res, best)?;
    launch!(gpu, "clamp", ew(out.len()), B256, 0; &out, -32.0f32, 32.0f32)?;
    Ok(out)
}

/// Upstream's `postprocess_masks` hole filling, in place on the GPU: in each low-res
/// mask of `logits [n, 288^2]`, background regions (logit <= 0), 8-connected, of at most
/// `max_area` pixels become foreground (+10). Every mask goes through the same four
/// launches, so a batch costs what one mask does until the GPU fills up
pub fn fill_holes(gpu: &Gpu, logits: &Tensor, max_area: usize) -> Result<()> {
    let n = logits.len();
    assert!(n.is_multiple_of(LOW_RES * LOW_RES), "{:?} is not a stack of low-res masks", logits.shape());
    let labels = gpu.alloc_bytes(n * 4)?;
    let area = gpu.alloc_bytes(n * 4)?;
    gpu.memset_zero(area.ptr(), n * 4)?;
    let (labels, area) = (labels.slice::<u32>(), area.slice::<u32>());
    launch!(gpu, "holes_init", ew(n), B256, 0; labels, logits, LOW_RES)?;
    launch!(gpu, "holes_merge", ew(n), B256, 0; labels, logits, LOW_RES)?;
    launch!(gpu, "holes_count", ew(n), B256, 0; labels, area)?;
    launch!(gpu, "holes_fill", ew(n), B256, 0; logits, labels, area, max_area)
}

/// Low-res mask logits `[n, 288^2]` to full-size 0/1 masks: bilinear upsampling to
/// `height` x `width`, then a threshold at 0
///
/// Masks go through in groups whose FP32 staging stays bounded and whose element count
/// fits the kernels' 32-bit thread index; one mask at the largest image size is 2^28
/// elements, so a group always holds at least one. Callers choose where each group's
/// bytes go, so they can keep every mask on the GPU or download group by group
pub struct MaskResize {
    n: usize,
    width: usize,
    height: usize,
    /// Masks per group
    group: usize,
    /// FP32 upsampled logits of one group
    up: Tensor,
}

impl MaskResize {
    pub fn new(gpu: &Gpu, n: usize, width: usize, height: usize) -> Result<Self> {
        let plane = width * height;
        let group = (MASK_GROUP_ELEMENTS / plane).min(n).max(1);
        Ok(Self { n, width, height, group, up: gpu.empty(&[group, plane])? })
    }

    /// Most masks in one group
    pub fn group(&self) -> usize {
        self.group
    }

    /// The mask indices of each group, in order
    pub fn groups(&self) -> impl Iterator<Item = Range<usize>> + use<> {
        let (n, group) = (self.n, self.group);
        (0..n).step_by(group).map(move |start| start..n.min(start + group))
    }

    /// Every mask of `low_res` into one device buffer of `n * height * width` bytes
    pub fn all(&self, gpu: &Gpu, low_res: &Tensor) -> Result<Buf> {
        let plane = self.width * self.height;
        let mask = gpu.alloc_bytes(self.n * plane)?;
        for range in self.groups() {
            let dst = Arg::slice(mask.ptr() + (range.start * plane) as u64, range.len() * plane);
            self.run(gpu, low_res, range, dst)?;
        }
        Ok(mask)
    }

    /// Masks `range` of `low_res` into `dst`, `range.len() * height * width` bytes
    pub fn run(&self, gpu: &Gpu, low_res: &Tensor, range: Range<usize>, dst: Arg) -> Result<()> {
        let side = LOW_RES * LOW_RES;
        assert!(range.len() <= self.group && range.end * side <= low_res.len());
        let n = range.len() * self.width * self.height;
        let input = Arg::slice(low_res.at(range.start * side), range.len() * side);
        let up = Arg::slice(self.up.ptr(), n);
        launch!(gpu, "resize_bilinear", ew(n), B256, 0; up, input, LOW_RES, LOW_RES, self.height, self.width)?;
        // sigmoid(x) > 0.5 is x > 0
        launch!(gpu, "threshold_u8", ew(n), B256, 0; dst, up, 0.0f32)
    }
}

#[cfg(test)]
mod tests {
    use eyre::Result;

    use super::{LOW_RES, fill_holes};
    use crate::gpu::{Gpu, Precision};

    /// The host flood fill the GPU fill replaced: background regions (logit <= 0),
    /// 8-connected, of at most `max_area` pixels become foreground (+10). NaN is
    /// foreground, as upstream's `mask <= mask_threshold`; the old fill let a NaN pixel
    /// start a region, so its regions depended on scan order
    fn fill_holes_host(logits: &mut [f32], side: usize, max_area: usize) {
        let mut seen = vec![false; logits.len()];
        let mut stack = Vec::new();
        let mut component = Vec::new();
        for start in 0..logits.len() {
            if seen[start] || logits[start] > 0.0 || logits[start].is_nan() {
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

    /// Logits with regions of many sizes and shapes: box-blurred noise of growing radius,
    /// with NaN pixels, then an all-background and an all-foreground mask
    fn logits(masks: usize) -> Vec<f32> {
        let side = LOW_RES;
        let mut state = 0x9e37_79b9_u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 - 0.5
        };
        let mut out = Vec::new();
        for m in 0..masks {
            let raw: Vec<f32> = (0..side * side).map(|_| noise()).collect();
            let r = 1 + m as isize;
            let last = side as isize - 1;
            let mut mask = vec![0f32; side * side];
            for y in 0..side as isize {
                for x in 0..side as isize {
                    let mut sum = 0.0;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let (yy, xx) = ((y + dy).clamp(0, last), (x + dx).clamp(0, last));
                            sum += raw[yy as usize * side + xx as usize];
                        }
                    }
                    mask[y as usize * side + x as usize] = sum + 0.05 * m as f32;
                }
            }
            for p in (m * 7919..side * side).step_by(9973) {
                mask[p] = f32::NAN;
            }
            out.extend(mask);
        }
        out.extend(vec![-1.0; side * side]);
        out.extend(vec![1.0; side * side]);
        out
    }

    #[test]
    #[ignore = "needs a CUDA GPU"]
    fn gpu_fill_matches_the_host_flood_fill() -> Result<()> {
        let gpu = Gpu::new(0, Precision::Bf16)?;
        let side = LOW_RES;
        let host = logits(6);
        let n = host.len() / (side * side);
        let mut want = host.clone();
        want.chunks_mut(side * side).for_each(|m| fill_holes_host(m, side, 256));
        let changed = want.iter().zip(&host).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert!(changed > 1000, "the fixture fills only {changed} pixels");

        let t = gpu.upload(host, &[n, side * side])?;
        fill_holes(&gpu, &t, 256)?;
        let got = gpu.download(&t)?;
        assert_eq!(got.len(), want.len());
        let wrong = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(wrong, 0, "{wrong} of {} pixels differ from the host fill", got.len());
        Ok(())
    }
}
