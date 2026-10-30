//! End-to-end parity against the PyTorch FP32 fixtures from `scripts/reference.py dump`
//!
//! Runs every fixture image in models/parity through `Sam3` (preprocessing included) and
//! checks the gate: same kept detections, scores within 0.01 and mask IoU >= 0.99. The
//! image comes from the fixture's decoded RGB because the `image` crate's JPEG decoder
//! differs from PIL's libjpeg-turbo on a few percent of bytes. Needs a CUDA GPU and the
//! weights from scripts/fetch-models.sh

use std::path::Path;

use eyre::{Result, eyre};
use safetensors::{Dtype, SafeTensors};
use sam3rs::detector::BoxPrompt;
use sam3rs::image::RgbImage;
use sam3rs::pipeline::{Sam3, VISUAL_PROMPT};
use sam3rs::point::PointPrompt;
use serde_json::Value;

const SCORE_TOL: f32 = 0.01;
const MIN_IOU: f64 = 0.99;

fn fixture(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|e| eyre!("{name}: {e}"))?;
    Ok(match t.dtype() {
        Dtype::F32 => t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
        d => return Err(eyre!("{name}: unsupported {d:?}")),
    })
}

fn iou(got: &[u8], want: &[f32]) -> f64 {
    let (mut inter, mut union) = (0usize, 0usize);
    for (a, b) in got.iter().zip(want) {
        let (a, b) = (*a != 0, *b != 0.0);
        inter += (a && b) as usize;
        union += (a || b) as usize;
    }
    if union == 0 { 1.0 } else { inter as f64 / union as f64 }
}

fn numbers(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn matches_pytorch_fp32() -> Result<()> {
    let model_dir = Path::new("models/sam3");
    if !model_dir.join("model.safetensors").exists() {
        return Err(eyre!("missing weights, run scripts/fetch-models.sh"));
    }
    let model = Sam3::load(model_dir, 0)?;
    let mut fixtures: Vec<_> = std::fs::read_dir("models/parity")?.map(|e| e.map(|e| e.path())).collect::<Result<_, _>>()?;
    fixtures.sort();
    assert!(!fixtures.is_empty(), "no fixtures, run scripts/reference.py dump");

    let mut failures = Vec::new();
    for path in fixtures {
        let bytes = std::fs::read(&path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let (_, header) = SafeTensors::read_metadata(&bytes)?;
        let meta: Value = serde_json::from_str(&header.metadata().as_ref().unwrap()["meta"])?;
        let image = path.file_stem().unwrap().to_string_lossy().into_owned();

        let rgb = st.tensor("image_rgb")?;
        let (height, width) = (rgb.shape()[0], rgb.shape()[1]);
        let img = RgbImage { width, height, data: rgb.data().to_vec() };
        let emb = model.encode(&img)?;
        let hw = width * height;

        for prompt in meta["prompts"].as_array().unwrap() {
            let key = prompt["name"].as_str().unwrap().replace(' ', "_");
            let mut check = |ok: bool, what: String| {
                let line = format!("{image}/{key}: {what}");
                println!("{}{line}", if ok { "" } else { "FAIL " });
                if !ok {
                    failures.push(line);
                }
            };

            if let Some(points) = prompt.get("points") {
                let points: Vec<PointPrompt> = points
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| {
                        let v = numbers(p);
                        PointPrompt { x: v[0], y: v[1], positive: v[2] != 0.0 }
                    })
                    .collect();
                let got = model.segment_points(&emb, &points)?;
                let want_iou = fixture(&st, &format!("{key}.iou"))?;
                let best = (0..3).fold(0, |b, i| if want_iou[i] > want_iou[b] { i } else { b });
                let want_mask = fixture(&st, &format!("{key}.masks"))?;
                let mask_iou = iou(&got.mask, &want_mask[best * hw..][..hw]);
                let dscore = (got.score - want_iou[best]).abs();
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
                .collect();
            let got = model.detect(&emb, text, &boxes)?;
            let want_scores = fixture(&st, &format!("{key}.scores"))?;
            let want_masks = fixture(&st, &format!("{key}.masks"))?;
            if got.len() != want_scores.len() {
                check(false, format!("{} detections vs {}", got.len(), want_scores.len()));
                continue;
            }
            // both sides keep detections in query order
            for (i, d) in got.iter().enumerate() {
                let mask_iou = iou(&d.mask, &want_masks[i * hw..][..hw]);
                let dscore = (d.score - want_scores[i]).abs();
                check(
                    mask_iou >= MIN_IOU && dscore <= SCORE_TOL,
                    format!("det {i} score {:.4} vs {:.4}, mask iou {mask_iou:.5}", d.score, want_scores[i]),
                );
            }
        }
    }
    assert!(failures.is_empty(), "parity failures:\n{}", failures.join("\n"));
    Ok(())
}
