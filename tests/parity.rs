//! End-to-end parity against the PyTorch FP32 fixtures from `scripts/reference.py dump`
//!
//! Runs every fixture image in models/parity through `Sam3` (preprocessing included) and
//! checks the gate: the same kept queries, scores within 0.01, boxes within 0.5% of the
//! image's longer side and mask IoU >= 0.99. The image comes from the fixture's decoded
//! RGB because the `image` crate's JPEG decoder differs from PIL's libjpeg-turbo on a few
//! percent of bytes. Needs a CUDA GPU and the weights from scripts/fetch-models.sh. Both
//! precision modes must pass the same gate

use std::path::Path;

use eyre::{Result, ensure, eyre};
use safetensors::{Dtype, SafeTensors};
use sam3rs::{BoxPrompt, PointPrompt, Precision, RgbImage, SCORE_THRESHOLD, Sam3, VISUAL_PROMPT};
use serde_json::Value;

const SCORE_TOL: f32 = 0.01;
const MIN_IOU: f64 = 0.99;
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
    for precision in [Precision::Fp32, Precision::Bf16] {
        println!("== {precision:?}");
        let model = Sam3::load(model_dir, 0, precision)?;
        let worst = check(&model, &mut failures, precision)?;
        println!(
            "worst over {} checks: score delta {:.4}, box delta {:.2} px, mask iou {:.5}",
            worst.checks, worst.score, worst.box_px, worst.iou
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
}

/// Runs every fixture prompt, pushing the checks that miss the gate onto `failures`
fn check(model: &Sam3, failures: &mut Vec<String>, precision: Precision) -> Result<Worst> {
    let mut fixtures: Vec<_> =
        std::fs::read_dir("models/parity")?.map(|e| e.map(|e| e.path())).collect::<Result<_, _>>()?;
    fixtures.sort();
    assert!(!fixtures.is_empty(), "no fixtures, run scripts/reference.py dump");

    let mut worst = Worst { checks: 0, score: 0.0, box_px: 0.0, iou: 1.0 };
    for path in fixtures {
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
                let want_iou = fixture(&st, &format!("{key}.iou"))?;
                let best = (0..3).fold(0, |b, i| if want_iou[i] > want_iou[b] { i } else { b });
                let want_mask = fixture(&st, &format!("{key}.masks"))?;
                let mask_iou = iou(&got.mask, &want_mask[best * hw..][..hw])?;
                let dscore = (got.score - want_iou[best]).abs();
                worst.checks += 1;
                worst.score = worst.score.max(dscore);
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
                let mask_iou = iou(&d.mask, &want_masks[i * hw..][..hw])?;
                let dscore = (d.score - want_scores[i]).abs();
                let dbox =
                    d.bbox.iter().zip(&want_boxes[i * 4..][..4]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                worst.checks += 1;
                worst.score = worst.score.max(dscore);
                worst.box_px = worst.box_px.max(dbox);
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
    }
    Ok(worst)
}
