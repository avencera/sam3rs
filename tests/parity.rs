//! End-to-end parity against the PyTorch FP32 fixtures from `scripts/reference.py dump`
//!
//! Runs every fixture image in models/parity through `Sam3` (preprocessing included) and
//! checks the gate: the same kept queries, scores within 0.01, boxes within 0.5% of the
//! image's longer side and mask IoU >= 0.99. The image comes from the fixture's decoded
//! RGB because the `image` crate's JPEG decoder differs from PIL's libjpeg-turbo on a few
//! percent of bytes. Needs a CUDA GPU and the weights from scripts/fetch-models.sh. Both
//! precision modes must pass the same gate
//!
//! The batched point API is checked against single calls on every image: the fixture's
//! point prompt mixed into more clicks than one decoder batch holds, with several click
//! counts interleaved, must give each object the score and mask its own call gives. FP32
//! runs the same math either way, so it must match to 1e-4. TF32 and BF16 must match at
//! the gate above, except on clicks whose single call already misses the FP32 result by
//! more than that: stacking prompts changes which GEMM kernels cuBLAS picks, and on a
//! background click with a near-zero first pass the second pass is ill-conditioned (one
//! such click scored 0.09 in FP32, 0.17 in TF32 and 0.60 in BF16)

use std::path::Path;

use eyre::{Result, ensure, eyre};
use safetensors::{Dtype, SafeTensors};
use sam3rs::{
    BoxPrompt, Embedding, POINT_BATCH, PointMask, PointPrompt, Precision, RgbImage, SCORE_THRESHOLD, Sam3,
    VISUAL_PROMPT,
};
use serde_json::Value;

const SCORE_TOL: f32 = 0.01;
const MIN_IOU: f64 = 0.99;
/// FP32 batched points against single calls
const FP32_SCORE_TOL: f32 = 1e-4;
const FP32_MIN_IOU: f64 = 0.999;
/// Box corner tolerance as a fraction of the image's longer side
const BOX_TOL: f32 = 0.005;

fn fixture(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|e| eyre!("{name}: {e}"))?;
    Ok(match t.dtype() {
        Dtype::F32 => bytemuck::pod_collect_to_vec(t.data()),
        Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
        d => return Err(eyre!("{name}: unsupported {d:?}")),
    })
}

fn as_f32(mask: &[u8]) -> Vec<f32> {
    mask.iter().map(|&b| b as f32).collect()
}

/// IoU of two masks of the same size; two empty masks match
fn iou(got: &[u8], want: &[f32]) -> Result<f64> {
    ensure!(got.len() == want.len(), "mask has {} pixels, the reference {}", got.len(), want.len());
    let (mut inter, mut union) = (0usize, 0usize);
    for (a, b) in got.iter().zip(want) {
        let (a, b) = (*a != 0, *b != 0.0);
        inter += (a && b) as usize;
        union += (a || b) as usize;
    }
    Ok(if union == 0 { 1.0 } else { inter as f64 / union as f64 })
}

/// The larger of two errors, NaN if either is: `f32::max` drops NaN, which would
/// let a NaN output score zero error and pass
fn worse(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}

/// Largest absolute difference between two coordinate lists, NaN if any side is NaN
fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, worse)
}

fn numbers(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn matches_pytorch() -> Result<()> {
    let model_dir = Path::new("models/sam3");
    if !model_dir.join("model.safetensors").exists() {
        return Err(eyre!("missing weights, run scripts/fetch-models.sh"));
    }
    let mut failures = Vec::new();
    // FP32 single-call results of the batched-points check, per image, which the reduced
    // precisions are judged against; FP32 has to run first
    let mut truth = Vec::new();
    for precision in [Precision::Fp32, Precision::Tf32, Precision::Bf16] {
        println!("== {precision:?}");
        let model = Sam3::load(model_dir, 0, precision)?;
        let worst = check(&model, &mut failures, precision, &mut truth)?;
        println!(
            "worst over {} checks: score delta {:.4}, box delta {:.2} px, mask iou {:.5}",
            worst.checks, worst.score, worst.box_px, worst.iou
        );
        println!(
            "batched points against single calls, worst over {} objects ({} excused): score delta {:.5}, mask iou {:.5}",
            worst.batch_checks, worst.batch_excused, worst.batch_score, worst.batch_iou
        );
    }
    assert!(failures.is_empty(), "parity failures:\n{}", failures.join("\n"));
    Ok(())
}

/// Worst values seen over every check, for the README's parity table
struct Worst {
    checks: usize,
    score: f32,
    box_px: f32,
    iou: f64,
    batch_checks: usize,
    batch_excused: usize,
    batch_score: f32,
    batch_iou: f64,
}

/// Runs every fixture prompt, pushing the checks that miss the gate onto `failures`
fn check(
    model: &Sam3,
    failures: &mut Vec<String>,
    precision: Precision,
    truth: &mut Vec<Vec<PointMask>>,
) -> Result<Worst> {
    let mut fixtures: Vec<_> =
        std::fs::read_dir("models/parity")?.map(|e| e.map(|e| e.path())).collect::<Result<_, _>>()?;
    fixtures.sort();
    assert!(!fixtures.is_empty(), "no fixtures, run scripts/reference.py dump");

    let mut worst = Worst {
        checks: 0,
        score: 0.0,
        box_px: 0.0,
        iou: 1.0,
        batch_checks: 0,
        batch_excused: 0,
        batch_score: 0.0,
        batch_iou: 1.0,
    };
    for (index, path) in fixtures.into_iter().enumerate() {
        let bytes = std::fs::read(&path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let (_, header) = SafeTensors::read_metadata(&bytes)?;
        let meta: Value = serde_json::from_str(&header.metadata().as_ref().unwrap()["meta"])?;
        let image = path.file_stem().unwrap().to_string_lossy().into_owned();

        let rgb = st.tensor("image_rgb")?;
        let (height, width) = (rgb.shape()[0], rgb.shape()[1]);
        let img = RgbImage::new(width, height, rgb.data().to_vec())?;
        let emb = model.encode(&img)?;
        let hw = width * height;
        let box_tol = BOX_TOL * width.max(height) as f32;
        let mut fixture_points = Vec::new();

        for prompt in meta["prompts"].as_array().unwrap() {
            let key = prompt["name"].as_str().unwrap().replace(' ', "_");
            let mut check = |ok: bool, what: String| {
                let line = format!("{image}/{key}: {what}");
                println!("{}{line}", if ok { "" } else { "FAIL " });
                if !ok {
                    failures.push(format!("{precision:?} {line}"));
                }
            };

            if let Some(points) = prompt.get("points") {
                let points: Vec<PointPrompt> = points
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| {
                        let v = numbers(p);
                        PointPrompt::new(v[0], v[1], v[2] != 0.0)
                    })
                    .collect::<Result<_>>()?;
                let got = model.segment_points(&emb, &points)?;
                fixture_points.push(points.clone());
                let want_iou = fixture(&st, &format!("{key}.iou"))?;
                let best = (0..3).fold(0, |b, i| if want_iou[i] > want_iou[b] { i } else { b });
                let want_mask = fixture(&st, &format!("{key}.masks"))?;
                let mask_iou = iou(&got.mask, &want_mask[best * hw..][..hw])?;
                let dscore = (got.score - want_iou[best]).abs();
                worst.checks += 1;
                worst.score = worse(worst.score, dscore);
                worst.iou = worst.iou.min(mask_iou);
                check(
                    mask_iou >= MIN_IOU && dscore <= SCORE_TOL,
                    format!("point score {:.4} vs {:.4}, mask iou {mask_iou:.5}", got.score, want_iou[best]),
                );
                continue;
            }

            let text = prompt.get("text").and_then(Value::as_str).unwrap_or(VISUAL_PROMPT);
            let boxes: Vec<BoxPrompt> = prompt
                .get("boxes")
                .map(|b| b.as_array().unwrap().iter().map(numbers).collect::<Vec<_>>())
                .unwrap_or_default()
                .iter()
                .map(|v| BoxPrompt::from_xyxy([v[0], v[1], v[2], v[3]], width, height, v[4] != 0.0))
                .collect::<Result<_>>()?;
            let got = model.detect(&emb, text, &boxes)?;
            let want_scores = fixture(&st, &format!("{key}.scores"))?;
            let want_boxes = fixture(&st, &format!("{key}.boxes_xyxy"))?;
            let want_masks = fixture(&st, &format!("{key}.masks"))?;
            let want_queries: Vec<usize> = fixture(&st, &format!("{key}.scores_all"))?
                .iter()
                .enumerate()
                .filter(|(_, p)| **p > SCORE_THRESHOLD)
                .map(|(i, _)| i)
                .collect();
            let got_queries: Vec<usize> = got.iter().map(|d| d.query).collect();
            worst.checks += 1;
            if got_queries != want_queries {
                check(false, format!("kept queries {got_queries:?} vs {want_queries:?}"));
                continue;
            }
            check(true, format!("kept queries {got_queries:?}"));
            // both sides keep detections in query order
            for (i, d) in got.iter().enumerate() {
                let mask_iou = iou(got.mask(i), &want_masks[i * hw..][..hw])?;
                let dscore = (d.score - want_scores[i]).abs();
                let dbox = max_abs_diff(&d.bbox, &want_boxes[i * 4..][..4]);
                worst.checks += 1;
                worst.score = worse(worst.score, dscore);
                worst.box_px = worse(worst.box_px, dbox);
                worst.iou = worst.iou.min(mask_iou);
                check(
                    mask_iou >= MIN_IOU && dscore <= SCORE_TOL && dbox <= box_tol,
                    format!(
                        "det {i} score {:.4} vs {:.4}, box delta {dbox:.2} px (tol {box_tol:.1}), mask iou {mask_iou:.5}",
                        d.score, want_scores[i]
                    ),
                );
            }
        }
        if precision == Precision::Fp32 {
            truth.push(Vec::new());
        }
        let truth = truth.get_mut(index).ok_or_else(|| eyre!("FP32 has to run first"))?;
        let batch = Batch { extra: &fixture_points, precision, truth };
        check_batch(model, &emb, batch, &mut worst, &mut |ok, what| {
            let line = format!("{image}/batched points: {what}");
            println!("{}{line}", if ok { "" } else { "FAIL " });
            if !ok {
                failures.push(format!("{precision:?} {line}"));
            }
        })?;
    }
    Ok(worst)
}

/// What [`check_batch`] runs on and against
struct Batch<'a> {
    /// The fixture's point prompts, mixed into the batch
    extra: &'a [Vec<PointPrompt>],
    precision: Precision,
    /// FP32 single-call results, written by the FP32 run and read by the others
    truth: &'a mut Vec<PointMask>,
}

/// [`Sam3::segment_points_batch`] against one [`Sam3::segment_points`] call per prompt.
/// The batch holds `extra` and more than [`POINT_BATCH`] grid clicks, plus prompts of two
/// and three clicks with background ones, interleaved, so it splits into several decoder
/// batches and its click-count groups are not runs of the input
fn check_batch(
    model: &Sam3,
    emb: &Embedding,
    Batch { extra, precision, truth }: Batch,
    worst: &mut Worst,
    check: &mut dyn FnMut(bool, String),
) -> Result<()> {
    let (w, h) = (emb.width() as f32, emb.height() as f32);
    let at = |fx: f32, fy: f32, positive: bool| PointPrompt::new(fx * w, fy * h, positive);
    let grid = POINT_BATCH + 4;
    let mut prompts: Vec<Vec<PointPrompt>> = Vec::new();
    for i in 0..grid {
        let (fx, fy) = ((i % 5) as f32 * 0.2 + 0.1, (i / 5) as f32 / (grid / 5 + 1) as f32 + 0.1);
        prompts.push(vec![at(fx, fy, true)?]);
        match i % 7 {
            2 => prompts.push(vec![at(fx, fy, true)?, at(fx + 0.05, fy + 0.05, false)?]),
            5 => prompts.push(vec![at(fx, fy, true)?, at(fx + 0.04, fy, true)?, at(fx, fy + 0.06, false)?]),
            _ => {}
        }
    }
    prompts.splice(3..3, extra.iter().cloned());

    // a batch of one is the single call itself
    let one = model.segment_points_batch(emb, &[&prompts[0]])?;
    let single = model.segment_points(emb, &prompts[0])?;
    check(
        one.len() == 1 && one.scores()[0].to_bits() == single.score.to_bits() && one.mask(0) == &single.mask[..],
        format!("batch of 1, score {:.5} vs {:.5}", one.scores()[0], single.score),
    );
    let empty = model.segment_points_batch(emb, &[])?;
    check(empty.is_empty() && empty.masks().is_empty(), format!("batch of 0 gave {} masks", empty.len()));

    let refs: Vec<&[PointPrompt]> = prompts.iter().map(Vec::as_slice).collect();
    let batch = model.segment_points_batch(emb, &refs)?;
    ensure!(batch.len() == prompts.len(), "{} masks for {} prompts", batch.len(), prompts.len());
    ensure!(batch.masks().len() == prompts.len() * emb.width() * emb.height(), "mask buffer size");
    let fp32 = precision == Precision::Fp32;
    let (score_tol, min_iou) = if fp32 { (FP32_SCORE_TOL, FP32_MIN_IOU) } else { (SCORE_TOL, MIN_IOU) };
    let mut excused = 0;
    for (i, points) in prompts.iter().enumerate() {
        let single = model.segment_points(emb, points)?;
        let mask_iou = iou(batch.mask(i), &as_f32(&single.mask))?;
        let dscore = (batch.scores()[i] - single.score).abs();
        let label = format!("prompt {i} ({} clicks)", points.len());
        let scores = format!("score {:.5} vs {:.5}, mask iou {mask_iou:.5}", batch.scores()[i], single.score);
        let ok = mask_iou >= min_iou && dscore <= score_tol;
        if fp32 {
            truth.push(single);
        } else if !ok {
            // ill-conditioned at this precision: the single call itself is off FP32
            let want = truth.get(i).ok_or_else(|| eyre!("no FP32 result for prompt {i}"))?;
            let (off_score, off_iou) = ((single.score - want.score).abs(), iou(&single.mask, &as_f32(&want.mask))?);
            if off_score.is_finite() && (off_score > SCORE_TOL || off_iou < MIN_IOU) {
                excused += 1;
                worst.batch_excused += 1;
                println!("{label}: excused, {scores}; the single call is off FP32 by {off_score:.4}, iou {off_iou:.5}");
                continue;
            }
        }
        worst.batch_checks += 1;
        worst.batch_score = worse(worst.batch_score, dscore);
        worst.batch_iou = worst.batch_iou.min(mask_iou);
        check(ok, format!("{label} {scores}"));
    }
    // a gate that excuses most prompts would pass anything
    check(excused * 4 <= prompts.len(), format!("{excused} of {} prompts excused", prompts.len()));
    Ok(())
}

#[test]
fn nan_errors_fail_the_gate() {
    let dbox = max_abs_diff(&[1.0, f32::NAN, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
    assert!(dbox.is_nan(), "a NaN box coordinate must not score zero error");
    assert!(worse(0.0, f32::NAN).is_nan() && worse(f32::NAN, 0.0).is_nan());
    assert_eq!(max_abs_diff(&[1.0, 2.5], &[1.0, 2.0]), 0.5);
}
