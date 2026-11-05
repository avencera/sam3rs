//! SAM 3 video tracking (the tracker-only "SAM 3 for the SAM 2 video task" path): objects
//! clicked on the first frame, followed forward through the video with the tracker's
//! memory bank
//!
//! Per frame: the image backbone and tracker neck, memory attention from the frame's
//! features to earlier frames' memories and object pointers, the point decoder on the
//! result, then the memory encoder on the chosen mask, whose output joins the bank. This
//! follows upstream's `Sam3TrackerPredictor` (`sam3_tracker_base.py` and
//! `sam3_tracking_predictor.py` at f1647e8): all objects share one state, so the memory
//! selection uses their mean score
//!
//! The flow, the bank and its memory selection live here; each backend runs the stages
//! (see [`Backend`]) and keeps what it likes per bank frame

use std::collections::VecDeque;

use eyre::{Result, ensure};

use crate::backend::{Active, Backend, ObjectPass};
use crate::host::argmax;
use crate::image::RgbImage;
use crate::model::point::LOW_RES;
use crate::model::tracker::{
    D, MAX_CLICKS, MAX_COND_FRAMES, MAX_OBJ_PTRS, MEM_D, MEMORY_SCORE_THRESHOLD, NUM_MASKMEM, TOKENS,
};
use crate::nvtx::range;
use crate::output::{TrackedFrame, TrackedObject};
use crate::prompt::PointPrompt;
use crate::sam3::Sam3;

type FrameMemory = <Active as Backend>::FrameMemory;
type Pixels = <Active as Backend>::Pixels;

/// One frame of the memory bank: what the backend keeps of it for every object
pub(crate) struct MemFrame<F> {
    /// Index of the frame in the video
    pub(crate) frame: usize,
    pub(crate) memory: F,
}

/// The memory frames and object pointers upstream's selection gives one frame
pub(crate) struct Selection<'a, F> {
    /// Memory frames with their temporal encoding index, in bank order
    pub(crate) frames: Vec<(&'a MemFrame<F>, usize)>,
    /// Frames whose object pointers join the memory
    pub(crate) pointers: Vec<&'a MemFrame<F>>,
    /// Each pointer's temporal position
    pub(crate) positions: Vec<usize>,
    /// `min(num_frames, 16)`, which scales the positions
    pub(crate) max_abs_pos: usize,
}

impl<F> Selection<'_, F> {
    /// Memory tokens from object pointers, 4 a pointer
    pub(crate) fn ptr_tokens(&self) -> usize {
        self.pointers.len() * (D / MEM_D)
    }

    /// Every memory token, object pointers included
    pub(crate) fn mem_tokens(&self) -> usize {
        self.frames.len() * TOKENS + self.ptr_tokens()
    }
}

/// The memory bank: the clicked frame's memory and the earlier frames upstream's memory
/// selection can still pick
struct Bank<F> {
    cond: Vec<MemFrame<F>>,
    /// The last frames (up to `min(num_frames, 16) - 1`) whose score passed
    /// [`MEMORY_SCORE_THRESHOLD`], oldest first
    passing: VecDeque<MemFrame<F>>,
    /// The previous frame when its score didn't pass: upstream always keeps it
    prev: Option<MemFrame<F>>,
}

impl<F> Bank<F> {
    fn new() -> Self {
        Self { cond: Vec::new(), passing: VecDeque::new(), prev: None }
    }

    /// Frames upstream's `frame_filter` selects, oldest first
    fn selected(&self) -> Vec<&MemFrame<F>> {
        let mut out: Vec<&MemFrame<F>> = self.passing.iter().collect();
        out.extend(self.prev.as_ref());
        out
    }

    /// The bank's selection for frame `t` of a `num_frames` video, as upstream's
    /// `_prepare_memory_conditioned_features` makes it
    fn selection(&self, t: usize, num_frames: usize) -> Selection<'_, F> {
        // memories: up to 4 conditioning frames at temporal index 6, then the last 6
        // selected frames at indices 5 (oldest) to 0
        let selected = self.selected();
        let cond = &self.cond[self.cond.len().saturating_sub(MAX_COND_FRAMES)..];
        let mut frames: Vec<(&MemFrame<F>, usize)> = cond.iter().map(|f| (f, NUM_MASKMEM - 1)).collect();
        for t_pos in 1..NUM_MASKMEM {
            let t_rel = NUM_MASKMEM - t_pos;
            if t_rel <= selected.len() {
                frames.push((selected[selected.len() - t_rel], NUM_MASKMEM - 1 - t_pos));
            }
        }
        // pointers: the conditioning frames' at their distance, then selected frames
        // -1 to -(L - 1) at positions 1 to L - 1 (upstream stops one short of the oldest)
        let max_abs_pos = num_frames.min(MAX_OBJ_PTRS);
        let mut pointers: Vec<(&MemFrame<F>, usize)> = cond.iter().map(|f| (f, t - f.frame)).collect();
        for t_diff in 1..max_abs_pos {
            if t_diff >= selected.len() {
                break;
            }
            pointers.push((selected[selected.len() - t_diff], t_diff));
        }
        let positions = pointers.iter().map(|(_, p)| *p).collect();
        Selection { frames, pointers: pointers.into_iter().map(|(f, _)| f).collect(), positions, max_abs_pos }
    }

    /// Adds a tracked frame with memory score `score`, calling `retire` on frames that
    /// leave the memory window but keep their pointers
    fn push(&mut self, frame: MemFrame<F>, score: f32, max_abs_pos: usize, retire: impl Fn(&mut F)) {
        self.prev = None;
        if score > MEMORY_SCORE_THRESHOLD {
            self.passing.push_back(frame);
            // `frame_filter` stops at `min(num_frames, 16) - 1` passing frames
            if self.passing.len() >= max_abs_pos {
                self.passing.pop_front();
            }
            // the memory takes the last 6 selected frames, and passing frames only leave
            // that window: older ones keep their pointers but nothing the memory
            // attention alone reads
            let window = self.passing.len().saturating_sub(NUM_MASKMEM - 1);
            for f in self.passing.iter_mut().take(window) {
                retire(&mut f.memory);
            }
        } else {
            self.prev = Some(frame);
        }
    }
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
    bank: Bank<FrameMemory>,
}

impl Sam3 {
    /// Starts tracking objects through a video of `num_frames` frames: segments each
    /// object from its clicks on `first` (frame 0), whose masks come back with the tracker
    ///
    /// Each entry of `objects` is one object's clicks, in `first`'s pixels; past 16, only
    /// the first and last 8 reach the decoder, as upstream. Upstream scales the object
    /// pointers' temporal positions by the video's length, so `num_frames` must be the
    /// real frame count; [`VideoTracker::next`] stops after it
    pub fn track(
        &self,
        first: &RgbImage,
        objects: &[Vec<PointPrompt>],
        num_frames: usize,
    ) -> Result<(VideoTracker<'_>, TrackedFrame)> {
        self.backend.bind()?;
        let pixels = self.backend.preprocess(first)?;
        self.track_pixels(&pixels, first.width(), first.height(), objects, num_frames)
    }

    /// [`Self::track`] from preprocessed pixels of a `width` x `height` frame
    pub(crate) fn track_pixels(
        &self,
        pixels: &Pixels,
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
            bank: Bank::new(),
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
        let b = &self.model.backend;
        b.bind()?;
        let pixels = b.preprocess(frame)?;
        self.next_pixels(&pixels)
    }

    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// The clicked frame: each object decoded from its clicks without memory, then the
    /// memory encoder over all objects' masks with overlaps resolved, as upstream's
    /// preflight consolidation does
    fn prompt_frame(&mut self, pixels: &Pixels, objects: &[Vec<PointPrompt>]) -> Result<TrackedFrame> {
        let b = &self.model.backend;
        let feats = b.video_frame(pixels)?;
        let low_res = b.frame_logits(objects.len())?;
        let (mut scores, mut ptrs, mut picks) = (Vec::new(), Vec::new(), Vec::new());
        for (o, points) in objects.iter().enumerate() {
            let _r = range(c"decoder");
            let points = &cap_clicks(points);
            let ObjectPass { decoded, score, ious } = b.decode_object(&feats, None, points, self.size(), 0)?;
            // one click (or none) gives three candidates; more give the single-mask token,
            // unless its mask is unstable (`_dynamic_multimask_via_stability`)
            let (mask, token) = if points.len() <= 1 {
                let best = 1 + argmax(&ious[1..]);
                (best, best)
            } else {
                let stable = stability(&b.first_mask(&decoded)?) >= 0.98;
                (if stable { 0 } else { 1 + argmax(&ious[1..]) }, 0)
            };
            b.keep_mask(&decoded, mask, score > 0.0, &low_res, o)?;
            ptrs.push(b.pointer(&decoded, token, score > 0.0)?);
            scores.push(score);
            picks.push((ious, mask));
        }

        let _r = range(c"memory_encoder");
        let present: Vec<bool> = scores.iter().map(|&s| s > 0.0).collect();
        let mems = b.prompt_memories(&feats, &low_res, &present)?;
        let memory = b.frame_memory(mems, ptrs)?;
        self.bank.cond.push(MemFrame { frame: 0, memory });
        self.output(0, low_res, &scores, picks, 0, 0)
    }

    /// The bank's selection for the next frame
    pub(crate) fn selection(&self) -> Selection<'_, FrameMemory> {
        self.bank.selection(self.next_frame, self.num_frames)
    }

    /// Tracks the objects into the next frame from its preprocessed pixels
    pub(crate) fn next_pixels(&mut self, pixels: &Pixels) -> Result<TrackedFrame> {
        ensure!(self.next_frame < self.num_frames, "the video has {} frames", self.num_frames);
        let b = &self.model.backend;
        let t = self.next_frame;
        let feats = b.video_frame(pixels)?;
        let sel = self.selection();
        let (max_abs_pos, ptr_tokens, mem_tokens) = (sel.max_abs_pos, sel.ptr_tokens(), sel.mem_tokens());
        let mut attention = b.attention(&feats, &sel)?;

        let low_res = b.frame_logits(self.objects)?;
        let (mut scores, mut ptrs, mut mems, mut picks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut norm_sum, mut iou_sum) = (0f32, 0f32);
        for o in 0..self.objects {
            let embed = {
                let _r = range(c"memory_attention");
                b.attend(&feats, &mut attention, &sel, o)?
            };
            let _r = range(c"decoder");
            let ObjectPass { decoded, score, ious } = b.decode_object(&feats, Some(&embed), &[], self.size(), 1)?;
            let best = 1 + argmax(&ious[1..]);
            let present = score > 0.0;
            b.keep_mask(&decoded, best, present, &low_res, o)?;
            ptrs.push(b.pointer(&decoded, best, present)?);

            let _r = range(c"memory_encoder");
            mems.push(b.object_memory(&feats, &low_res, o, present)?);
            // `cal_mem_score`: present objects' rescaled score and their best IoU
            norm_sum += if present { 2.0 / (1.0 + (-score).exp()) - 1.0 } else { 0.0 };
            iou_sum += ious[best];
            scores.push(score);
            picks.push((ious, best));
        }
        drop((attention, sel));
        // upstream multiplies scores shaped [B, 1] by IoUs shaped [B], so its mean over the
        // broadcast [B, B] product is mean(score) * mean(IoU), not the mean of per-object
        // products; with several objects the two land on opposite sides of the threshold
        let n = self.objects as f32;
        let eff = (norm_sum / n) * (iou_sum / n);

        let memory = b.frame_memory(mems, ptrs)?;
        self.bank.push(MemFrame { frame: t, memory }, eff, max_abs_pos, Active::retire);
        self.next_frame += 1;
        self.output(t, low_res, &scores, picks, mem_tokens, ptr_tokens)
    }

    /// Masks at the video's size from the kept low-res masks
    fn output(
        &self,
        frame: usize,
        low_res: <Active as Backend>::Logits,
        scores: &[f32],
        picks: Vec<(Vec<f32>, usize)>,
        mem_tokens: usize,
        ptr_tokens: usize,
    ) -> Result<TrackedFrame> {
        let b = &self.model.backend;
        let _r = range(c"postprocess");
        let plane = self.width * self.height;
        // one buffer per object, filled in place, so the frame's masks are never held twice
        let mut masks = vec![vec![0u8; plane]; self.objects];
        b.frame_masks(&low_res, self.size(), &mut masks)?;
        let logits = b.download_logits(&low_res)?;
        let side = LOW_RES * LOW_RES;
        ensure!(
            scores.len() == self.objects && picks.len() == self.objects && logits.len() == self.objects * side,
            "{} scores, {} picks and {} logits for {} objects",
            scores.len(),
            picks.len(),
            logits.len(),
            self.objects
        );
        let objects = scores
            .iter()
            .zip(masks)
            .zip(logits.chunks_exact(side))
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

/// Upstream's `_get_stability_scores` at delta 0.05: the area above +delta over the area
/// above -delta
fn stability(logits: &[f32]) -> f32 {
    let inter = logits.iter().filter(|&&v| v > 0.05).count();
    let union = logits.iter().filter(|&&v| v > -0.05).count();
    if union > 0 { inter as f32 / union as f32 } else { 1.0 }
}

/// All the clicks, or the first and last half of [`MAX_CLICKS`] when there are more
fn cap_clicks(points: &[PointPrompt]) -> Vec<PointPrompt> {
    if points.len() <= MAX_CLICKS {
        return points.to_vec();
    }
    let first = MAX_CLICKS / 2;
    points[..first].iter().chain(&points[points.len() - (MAX_CLICKS - first)..]).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::{Bank, MemFrame, cap_clicks};
    use crate::prompt::PointPrompt;

    #[test]
    fn long_click_lists_keep_the_first_and_last_eight() {
        let clicks: Vec<PointPrompt> = (0..20).map(|i| PointPrompt::new(i as f32, 0.0, true).unwrap()).collect();
        let kept: Vec<f32> = cap_clicks(&clicks).iter().map(PointPrompt::x).collect();
        let want: Vec<f32> = (0..8).chain(12..20).map(|i| i as f32).collect();
        assert_eq!(kept, want);
        assert_eq!(cap_clicks(&clicks[..16]).len(), 16);
    }

    /// Bank frame indices with their temporal slots, or with their positions
    type Picks = Vec<(usize, usize)>;

    /// The selected memory frames and pointers
    fn picked(bank: &Bank<()>, t: usize, num_frames: usize) -> (Picks, Picks) {
        let sel = bank.selection(t, num_frames);
        let frames = sel.frames.iter().map(|(f, tpos)| (f.frame, *tpos)).collect();
        let pointers = sel.pointers.iter().zip(&sel.positions).map(|(f, p)| (f.frame, *p)).collect();
        (frames, pointers)
    }

    /// Upstream's selection on a long video: the clicked frame at slot 6, then the last 6
    /// passing frames, and pointers from the last 15; a failing frame is kept only while
    /// it is the previous one
    #[test]
    fn selection_follows_the_last_passing_frames() {
        let mut bank = Bank::new();
        bank.cond.push(MemFrame { frame: 0, memory: () });
        for t in 1..20 {
            // frame 5 fails the memory score
            bank.push(MemFrame { frame: t, memory: () }, if t == 5 { 0.0 } else { 1.0 }, 16, |_| {});
            if t == 5 {
                let (frames, _) = picked(&bank, 6, 64);
                assert_eq!(frames, [(0, 6), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]);
            }
        }
        let (frames, pointers) = picked(&bank, 20, 64);
        assert_eq!(frames, [(0, 6), (14, 5), (15, 4), (16, 3), (17, 2), (18, 1), (19, 0)]);
        let want: Vec<(usize, usize)> = [(0, 20)].into_iter().chain((1..15).map(|d| (20 - d, d))).collect();
        assert_eq!(pointers, want);
    }
}
