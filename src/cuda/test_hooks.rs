//! The CUDA backend's [`TestHooks`] for the shared stage and video suites, and its own
//! end-to-end check: in BF16 mode, the memory attention over cached keys against the FP32
//! model's

use eyre::{Result, ensure};

use crate::backend::Backend;
use crate::cuda::detector::{DetectorImage, DetectorOutput};
use crate::cuda::gpu::{Gpu, Tensor, launch};
use crate::cuda::ops::{B256, ew};
use crate::cuda::pipeline::Cuda;
use crate::cuda::point::{Decoded, MaskResize, PointImage, fill_holes};
use crate::cuda::text::TextFeatures;
use crate::cuda::video::{FrameFeatures, FrameMemory};
use crate::cuda::vision::{Fpn, Trunk};
use crate::model::point::LOW_RES;
use crate::model::vision::IMAGE_SIZE;
use crate::output::HOLE_AREA;
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::sam3::Sam3;
use crate::test_support::{
    DecodedOutputs, LONG_REL_L2, REDUCED_BACKBONE_REL_L2, STAGE_REL_L2, TestHooks, Tolerances, check, rel_l2,
};
use crate::tracker::{Selection, VideoTracker};

/// Stage tolerance in the reduced-precision modes, at BF16's own rounding: the point
/// decoder's BF16 linears alone put the IoU head at 3.2e-3 on FP32 inputs
const REDUCED_REL_L2: f32 = 5e-3;

/// `[rows, cols]` -> `[cols, rows]`
fn transpose(gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
    let out = gpu.empty(&[x.cols(), x.rows()])?;
    launch!(gpu, "transpose", ew(out.len()), B256, 0; &out, x, x.cols())?;
    Ok(out)
}

/// The BF16 cached-key checks of the end-to-end video test
#[derive(Default)]
pub(crate) struct CachedKeyChecks {
    /// The FP32 model whose memory attention is the truth, loaded on first use
    reference: Option<Sam3>,
    checks: usize,
}

impl TestHooks for Cuda {
    type Tensor = Tensor;
    type Trunk = Trunk;
    type Fpn = Fpn;
    type PointImage = PointImage;
    type DetectorImage = DetectorImage;
    type E2eChecks = CachedKeyChecks;

    fn fill_holes_on_gpu(logits: Vec<f32>) -> Result<Vec<f32>> {
        let gpu = Gpu::new(0, Precision::Bf16)?;
        let side = LOW_RES * LOW_RES;
        let n = logits.len() / side;
        let t = gpu.upload(logits, &[n, side])?;
        fill_holes(&gpu, &t, HOLE_AREA)?;
        gpu.download(&t)
    }

    fn sync(&self) -> Result<()> {
        self.gpu.synchronize()
    }

    fn take_profile(&self) -> Vec<(&'static str, usize, f64)> {
        Vec::new()
    }

    fn tolerances(&self) -> Tolerances {
        match self.precision() {
            Precision::Fp32 => Tolerances { stage: STAGE_REL_L2, long: LONG_REL_L2, backbone: STAGE_REL_L2 },
            // TF32 GEMMs and the BF16 blocks round every stage
            Precision::Tf32 | Precision::Bf16 => {
                Tolerances { stage: REDUCED_REL_L2, long: REDUCED_REL_L2, backbone: REDUCED_BACKBONE_REL_L2 }
            }
        }
    }

    fn upload(&self, v: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        self.gpu.upload(v, shape)
    }

    fn download(&self, t: &Tensor) -> Result<Vec<f32>> {
        self.gpu.download(t)
    }

    fn chw(&self, t: &Tensor) -> Result<Vec<f32>> {
        self.gpu.download(&transpose(&self.gpu, t)?)
    }

    fn upload_chw(&self, v: Vec<f32>, c: usize) -> Result<Tensor> {
        let hw = v.len() / c;
        transpose(&self.gpu, &self.gpu.upload(v, &[c, hw])?)
    }

    fn upload_pixels(&self, v: Vec<f32>) -> Result<Tensor> {
        self.gpu.upload(v, &[3, IMAGE_SIZE, IMAGE_SIZE])
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
        let mask_input = mask_input.map(|m| gpu.upload(m.to_vec(), &[1, m.len()])).transpose()?;
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
        let filled = gpu.clone_tensor(&dec.low_res)?;
        fill_holes(gpu, &filled, HOLE_AREA)?;
        let mask = MaskResize::new(gpu, n, w, h)?.all(gpu, &filled)?;
        let mut out = vec![0u8; n * w * h];
        ensure!(out.len() == mask.bytes(), "mask buffer of {} bytes for {} pixels", out.len(), mask.bytes());
        gpu.download_into(&mask, &mut out)?;
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
        self.tracker.mem_attn.forward(&self.gpu, src, memory, memory_pos, ptr_tokens)
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
        let ptr_pos = self.pointer_pos(sel)?;
        self.memory_inputs(sel, &ptr_pos)
    }

    fn e2e_check(
        &self,
        checks: &mut CachedKeyChecks,
        tracker: &VideoTracker<'_>,
        pixels: &Tensor,
        t: usize,
        frames: usize,
        failures: &mut Vec<String>,
    ) -> Result<()> {
        if self.precision() != Precision::Bf16 || ![1, 2, 8, 16, frames - 1].contains(&t) {
            return Ok(());
        }
        // BF16 attends from each frame's cached keys plus the slot's rotated temporal key,
        // from layer 0's shared start. The drifted memory has no fixture, so FP32 memory
        // attention on the same inputs is the truth; the cached path must meet the BF16
        // stage limit against it, as the assembled path does in video_stages (printed
        // alongside)
        if checks.reference.is_none() {
            checks.reference = Some(Sam3::load("models/sam3".as_ref(), 0, Precision::Fp32)?);
        }
        let fp32 = &checks.reference.as_ref().expect("loaded above").backend;
        let (gpu, rg) = (&self.gpu, &fp32.gpu);
        let to_ref = |x: &Tensor| -> Result<Tensor> { rg.upload(gpu.download(x)?, x.shape()) };
        let feats = self.video_features(pixels)?;
        let sel = tracker.selection();
        let ptr_pos = self.pointer_pos(&sel)?;
        let inputs = self.memory_inputs(&sel, &ptr_pos)?;
        let ptr_tokens = sel.ptr_tokens();
        let start = self.tracker.mem_attn.frame_start(gpu, &feats.coarse)?;
        let src32 = to_ref(&feats.coarse)?;
        for (b, (mem, mem_pos)) in inputs.iter().enumerate() {
            let truth = fp32.tracker.mem_attn.forward(rg, &src32, &to_ref(mem)?, &to_ref(mem_pos)?, ptr_tokens)?;
            let truth = rg.download(&truth)?;
            let cached = self.memory_attention_cached(&sel, &ptr_pos, b, &start)?;
            let assembled = self.tracker.mem_attn.forward(gpu, &feats.coarse, mem, mem_pos, ptr_tokens)?;
            rel_l2(&format!("f{t} assembled o{b}"), &gpu.download(&assembled)?, &truth);
            check(failures, &format!("f{t} cached keys o{b}"), &gpu.download(&cached)?, &truth, REDUCED_REL_L2);
            checks.checks += 1;
        }
        Ok(())
    }

    fn e2e_finish(&self, checks: CachedKeyChecks) -> Result<()> {
        if self.precision() == Precision::Bf16 {
            ensure!(checks.checks > 0, "BF16 cached-key memory attention never checked");
        }
        Ok(())
    }
}
