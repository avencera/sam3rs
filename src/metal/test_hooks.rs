//! The Metal backend's [`TestHooks`] for the shared stage and video suites
//!
//! Only the backbone and necks run in BF16 on Metal, so every later video stage is held
//! to the FP32 limits in both modes

use eyre::{Result, ensure};

use crate::backend::Backend;
use crate::metal::detector::{DetectorImage, DetectorOutput};
use crate::metal::gpu::{Gpu, Tensor};
use crate::metal::masks::upsample_masks;
use crate::metal::pipeline::Metal;
use crate::metal::point::{Decoded, PointImage, clone, fill_holes};
use crate::metal::text::TextFeatures;
use crate::metal::video::{FrameFeatures, FrameMemory};
use crate::metal::vision::{Fpn, Trunk};
use crate::model::point::LOW_RES;
use crate::model::vision::IMAGE_SIZE;
use crate::output::HOLE_AREA;
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::test_support::{DecodedOutputs, LONG_REL_L2, REDUCED_BACKBONE_REL_L2, STAGE_REL_L2, TestHooks, Tolerances};
use crate::tracker::{Selection, VideoTracker};

/// Metal adds no end-to-end checks of its own
#[derive(Default)]
pub(crate) struct NoChecks;

/// `[rows, cols]` -> `[cols, rows]` on the host
fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..cols).flat_map(|c| (0..rows).map(move |r| v[r * cols + c])).collect()
}

impl TestHooks for Metal {
    type Tensor = Tensor;
    type Trunk = Trunk;
    type Fpn = Fpn;
    type PointImage = PointImage;
    type DetectorImage = DetectorImage;
    type E2eChecks = NoChecks;

    fn fill_holes_on_gpu(logits: Vec<f32>) -> Result<Vec<f32>> {
        let gpu = Gpu::new(0, Precision::Fp32)?;
        let side = LOW_RES * LOW_RES;
        let t = gpu.upload_f32(&logits, &[logits.len() / side, side])?;
        fill_holes(&gpu, &t, HOLE_AREA)?;
        gpu.download(&t)
    }

    fn sync(&self) -> Result<()> {
        self.gpu.synchronize()
    }

    fn take_profile(&self) -> Vec<(&'static str, usize, f64)> {
        self.gpu.take_profile()
    }

    fn tolerances(&self) -> Tolerances {
        let backbone = match self.precision() {
            Precision::Fp32 => STAGE_REL_L2,
            Precision::Tf32 | Precision::Bf16 => REDUCED_BACKBONE_REL_L2,
        };
        Tolerances { stage: STAGE_REL_L2, long: LONG_REL_L2, backbone }
    }

    fn upload(&self, v: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        self.gpu.upload_f32(&v, shape)
    }

    fn download(&self, t: &Tensor) -> Result<Vec<f32>> {
        self.gpu.download(t)
    }

    fn chw(&self, t: &Tensor) -> Result<Vec<f32>> {
        Ok(transpose(&self.gpu.download(t)?, t.rows(), t.cols()))
    }

    fn upload_chw(&self, v: Vec<f32>, c: usize) -> Result<Tensor> {
        let hw = v.len() / c;
        self.gpu.upload_f32(&transpose(&v, c, hw), &[hw, c])
    }

    fn upload_pixels(&self, v: Vec<f32>) -> Result<Tensor> {
        self.gpu.upload_f32(&v, &[3, IMAGE_SIZE, IMAGE_SIZE])
    }

    fn download_pixels(&self, px: &Tensor) -> Result<Vec<f32>> {
        self.gpu.download(px)
    }

    fn trunk(&self, px: &Tensor) -> Result<Tensor> {
        self.backbone.forward(&self.gpu, px)
    }

    fn neck_input(&self, trunk: Tensor) -> Result<Trunk> {
        Trunk::new(&self.gpu, trunk)
    }

    fn neck(&self, trunk: &Trunk, tracker: bool) -> Result<Fpn> {
        let neck = if tracker { &self.tracker_neck } else { &self.neck };
        neck.forward(&self.gpu, trunk)
    }

    fn fpn_levels<'a>(&self, fpn: &'a Fpn) -> [&'a Tensor; 3] {
        [&fpn.fine, &fpn.mid, &fpn.coarse]
    }

    fn point_image(&self, fpn: &Fpn) -> Result<PointImage> {
        self.point.prepare_image(&self.gpu, fpn)
    }

    fn point_image_from(&self, embed: Tensor, feat_s0: Tensor, feat_s1: Tensor) -> PointImage {
        PointImage { embed, feat_s0, feat_s1 }
    }

    fn point_levels<'a>(&self, img: &'a PointImage) -> [&'a Tensor; 2] {
        [&img.feat_s0, &img.feat_s1]
    }

    fn point_decode(
        &self,
        img: &PointImage,
        points: &[PointPrompt],
        size: (usize, usize),
        mask_input: Option<&[f32]>,
    ) -> Result<Decoded> {
        let gpu = &self.gpu;
        let mask_input = mask_input.map(|m| gpu.upload_f32(m, &[1, m.len()])).transpose()?;
        self.point.decode(gpu, &img.embed, img, points, size, mask_input.as_ref(), 1)
    }

    fn decoded_outputs(&self, dec: &Decoded) -> Result<DecodedOutputs> {
        let gpu = &self.gpu;
        let score = gpu.download(&self.point.obj_score(gpu, dec)?)?;
        ensure!(score.len() == 1, "{} object scores for one prompt", score.len());
        Ok(DecodedOutputs {
            iou: gpu.download(&dec.iou)?,
            score: score[0],
            low_res: gpu.download(&dec.low_res)?,
            tokens: gpu.download(&dec.tokens)?,
        })
    }

    fn full_masks(&self, dec: &Decoded, n: usize, (w, h): (usize, usize)) -> Result<Vec<u8>> {
        let gpu = &self.gpu;
        ensure!(dec.low_res.len() == n * LOW_RES * LOW_RES, "{n} masks from {:?}", dec.low_res.shape());
        let filled = clone(gpu, &dec.low_res)?;
        fill_holes(gpu, &filled, HOLE_AREA)?;
        let mut out = vec![0u8; n * w * h];
        gpu.read(&upsample_masks(gpu, &filled, 0, n, w, h)?.buf, 0, &mut out)?;
        Ok(out)
    }

    fn detector_image(&self, [fine, mid, coarse]: [Tensor; 3]) -> Result<DetectorImage> {
        self.detector.prepare_image(&self.gpu, Fpn { fine, mid, coarse })
    }

    fn text_features(&self, text: &TextFeatures) -> Result<Vec<f32>> {
        self.gpu.download(&text.feats)
    }

    fn detect_from(&self, img: &DetectorImage, text: &TextFeatures, boxes: &[BoxPrompt]) -> Result<DetectorOutput> {
        self.detector.forward(&self.gpu, img, text, boxes)
    }

    fn mask_logits(&self, det: &DetectorOutput) -> Result<Vec<f32>> {
        self.gpu.download(&det.masks)
    }

    fn frame_levels<'a>(&self, frame: &'a FrameFeatures) -> [&'a Tensor; 3] {
        [&frame.coarse, &frame.point.feat_s0, &frame.point.feat_s1]
    }

    fn memory_src_pos(&self) -> Result<Vec<f32>> {
        self.gpu.download(&self.tracker.mem_attn.src_pos)
    }

    fn memory_attention(
        &self,
        src: &Tensor,
        memory: &Tensor,
        memory_pos: &Tensor,
        ptr_tokens: usize,
    ) -> Result<Tensor> {
        let attn = &self.tracker.mem_attn;
        let prepared = attn.prepare(&self.gpu, src)?;
        attn.forward(&self.gpu, &prepared, memory, memory_pos, ptr_tokens)
    }

    fn object_pointer(&self, input: &Tensor) -> Result<Tensor> {
        self.tracker.obj_ptr_proj.forward(&self.gpu, input)
    }

    fn memory_encoder(&self, pix_feat: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.tracker.mem_enc.forward(&self.gpu, pix_feat, mask)
    }

    fn no_obj_spatial(&self) -> Result<Vec<f32>> {
        self.gpu.download(&self.tracker.no_obj_spatial)
    }

    fn assembled_memory(&self, sel: &Selection<'_, FrameMemory>) -> Result<Vec<(Tensor, Tensor)>> {
        self.memory_inputs(sel)
    }

    fn e2e_check(
        &self,
        _checks: &mut NoChecks,
        _tracker: &VideoTracker<'_>,
        _pixels: &Tensor,
        _t: usize,
        _frames: usize,
        _failures: &mut Vec<String>,
    ) -> Result<()> {
        Ok(())
    }

    fn e2e_finish(&self, _checks: NoChecks) -> Result<()> {
        Ok(())
    }
}
