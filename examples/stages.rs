//! Compares intermediate stages against the PyTorch FP32 fixtures in models/parity
//!
//!     cargo run --release --example stages -- models/parity/dogs.safetensors

use std::path::PathBuf;

use eyre::{Result, eyre};
use safetensors::{Dtype, SafeTensors};
use sam3rs::gpu::{Gpu, Tensor};
use sam3rs::image::{RgbImage, preprocess};
use sam3rs::detector::{BoxPrompt, Detector};
use sam3rs::ops::transpose;
use sam3rs::point::{LOW_RES, PointDecoder, PointPrompt, clamped_mask, full_res_masks};
use sam3rs::text::TextEncoder;
use sam3rs::tokenizer::Tokenizer;
use sam3rs::vision::{Backbone, Neck};
use sam3rs::weights::Weights;

fn fixture(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|e| eyre!("{name}: {e}"))?;
    Ok(match t.dtype() {
        Dtype::F32 => t.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
        d => return Err(eyre!("{name}: unsupported {d:?}")),
    })
}

fn compare(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut ref_sq = 0f64;
    for (g, w) in got.iter().zip(want) {
        max_abs = max_abs.max((g - w).abs());
        sum_sq += ((g - w) as f64).powi(2);
        ref_sq += (*w as f64).powi(2);
    }
    let rel = (sum_sq / ref_sq.max(1e-30)).sqrt();
    println!("{name:>12}: max_abs {max_abs:.3e}  rel_l2 {rel:.3e}");
}

/// channels-last device tensor -> CHW host vector
fn chw(gpu: &Gpu, t: &Tensor) -> Result<Vec<f32>> {
    gpu.download(&transpose(gpu, t)?)
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let path = PathBuf::from(std::env::args().nth(1).unwrap_or("models/parity/dogs.safetensors".into()));
    let bytes = std::fs::read(&path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let (_, meta) = SafeTensors::read_metadata(&bytes)?;
    let meta: serde_json::Value = serde_json::from_str(&meta.metadata().as_ref().unwrap()["meta"])?;
    let image_path = PathBuf::from(meta["image"].as_str().unwrap());

    let gpu = Gpu::new(0)?;
    let wts = Weights::open("models/sam3/model.safetensors".as_ref())?;

    let img = RgbImage::open(&image_path)?;
    let rgb: Vec<f32> = img.data.iter().map(|&b| b as f32).collect();
    let want_rgb = fixture(&st, "image_rgb")?;
    let diff = rgb.iter().zip(&want_rgb).filter(|(a, b)| a != b).count();
    println!("rgb bytes differing from PIL: {diff} of {}", rgb.len());
    let img = RgbImage { data: want_rgb.iter().map(|&v| v as u8).collect(), ..img };
    let pixels = preprocess(&gpu, &img)?;
    compare("pixels", &gpu.download(&pixels)?, &fixture(&st, "pixels")?);
    // isolate the backbone from resize differences
    let pixels = gpu.upload(&fixture(&st, "pixels")?, &[3, 1008, 1008])?;

    let backbone = Backbone::load(&gpu, &wts)?;
    let t0 = std::time::Instant::now();
    let trunk = backbone.forward(&gpu, &pixels)?;
    gpu.synchronize()?;
    println!("backbone {:?}", t0.elapsed());
    compare("trunk", &chw(&gpu, &trunk)?, &fixture(&st, "trunk")?);

    let trunk = gpu.upload(&fixture(&st, "trunk")?, &[1024, 72 * 72])?;
    let trunk = transpose(&gpu, &trunk)?;
    let neck = Neck::load(&gpu, &wts, "detector_model.vision_encoder.neck")?;
    let fpn = neck.forward(&gpu, &trunk)?;
    for (i, f) in fpn.iter().enumerate() {
        compare(&format!("fpn{i}"), &chw(&gpu, f)?, &fixture(&st, &format!("fpn{i}"))?);
    }
    drop((backbone, neck));

    let tracker_neck = Neck::load(&gpu, &wts, "tracker_neck")?;
    let sam2_fpn = tracker_neck.forward(&gpu, &trunk)?;
    let point_decoder = PointDecoder::load(&gpu, &wts)?;
    let point_img = point_decoder.prepare_image(&gpu, &sam2_fpn)?;
    compare("sam2_fpn0", &chw(&gpu, &point_img.feat_s0)?, &fixture(&st, "sam2_fpn0")?);
    compare("sam2_fpn1", &chw(&gpu, &point_img.feat_s1)?, &fixture(&st, "sam2_fpn1")?);
    compare("sam2_fpn2", &chw(&gpu, &sam2_fpn[2])?, &fixture(&st, "sam2_fpn2")?);
    drop((tracker_neck, sam2_fpn));
    for prompt in meta["prompts"].as_array().unwrap() {
        let Some(points) = prompt.get("points") else { continue };
        let key = prompt["name"].as_str().unwrap().replace(' ', "_");
        let points: Vec<PointPrompt> = points
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let v: Vec<f64> = p.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                PointPrompt { x: v[0] as f32, y: v[1] as f32, positive: v[2] != 0.0 }
            })
            .collect();
        println!("-- {key} ({} points)", points.len());
        let first = point_decoder.forward(&gpu, &point_img, &points, img.width, img.height, None)?;
        println!("  iou_first {:?} vs {:?}", first.iou, fixture(&st, &format!("{key}.iou_first"))?);
        let clamped: Vec<f32> = gpu.download(&first.low_res)?.iter().map(|v| v.clamp(-32.0, 32.0)).collect();
        compare("low_first", &clamped, &fixture(&st, &format!("{key}.low_res_first"))?);
        let best = (0..3).max_by(|a, b| first.iou[*a].total_cmp(&first.iou[*b])).unwrap();
        let mask_input = clamped_mask(&gpu, &first.low_res, best)?;
        let second = point_decoder.forward(&gpu, &point_img, &points, img.width, img.height, Some(&mask_input))?;
        println!("  iou {:?} vs {:?}", second.iou, fixture(&st, &format!("{key}.iou"))?);
        let clamped: Vec<f32> = gpu.download(&second.low_res)?.iter().map(|v| v.clamp(-32.0, 32.0)).collect();
        compare("low_res", &clamped, &fixture(&st, &format!("{key}.low_res"))?);
        let masks = full_res_masks(&gpu, &second.low_res, 3, img.width, img.height)?;
        let want = fixture(&st, &format!("{key}.masks"))?;
        let hw = img.width * img.height;
        for m in 0..3 {
            let (mut inter, mut union) = (0usize, 0usize);
            for (a, b) in masks[m * hw..][..hw].iter().zip(&want[m * hw..][..hw]) {
                let (a, b) = (*a != 0, *b != 0.0);
                inter += (a && b) as usize;
                union += (a || b) as usize;
            }
            println!("  mask {m} iou {:.5}", inter as f64 / union.max(1) as f64);
        }
        let _ = LOW_RES;
    }

    // detector stages from the reference fpn features
    let sides = [288usize, 144, 72];
    let fpn = (0..3)
        .map(|i| {
            let f = gpu.upload(&fixture(&st, &format!("fpn{i}"))?, &[256, sides[i] * sides[i]])?;
            transpose(&gpu, &f)
        })
        .collect::<Result<Vec<_>>>()?;
    let tokenizer = Tokenizer::open("models/sam3".as_ref())?;
    let text_encoder = TextEncoder::load(&gpu, &wts)?;
    let detector = Detector::load(&gpu, &wts)?;
    let img_state = detector.prepare_image(&gpu, fpn)?;
    let (w, h) = (img.width as f64, img.height as f64);
    for prompt in meta["prompts"].as_array().unwrap() {
        if prompt.get("points").is_some() {
            continue;
        }
        let key = prompt["name"].as_str().unwrap().replace(' ', "_");
        let text = prompt.get("text").and_then(|t| t.as_str()).unwrap_or("visual");
        let boxes: Vec<BoxPrompt> = prompt
            .get("boxes")
            .map(|b| b.as_array().unwrap().clone())
            .unwrap_or_default()
            .iter()
            .map(|b| {
                let v: Vec<f64> = b.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                BoxPrompt {
                    cxcywh: [
                        ((v[0] + v[2]) / 2.0 / w) as f32,
                        ((v[1] + v[3]) / 2.0 / h) as f32,
                        ((v[2] - v[0]) / w) as f32,
                        ((v[3] - v[1]) / h) as f32,
                    ],
                    positive: v[4] != 0.0,
                }
            })
            .collect();
        println!("-- {key} (text {text:?}, {} boxes)", boxes.len());
        let tokens = tokenizer.tokenize(text)?;
        let feats = text_encoder.forward(&gpu, &tokens)?;
        compare("lang_feats", &gpu.download(&feats.feats)?, &fixture(&st, &format!("{key}.language_features"))?);
        let out = detector.forward(&gpu, &img_state, &feats, &boxes)?;
        let presence = gpu.download(&out.presence)?[0];
        let logits = gpu.download(&out.logits)?;
        let pres_sig = 1.0 / (1.0 + (-presence).exp());
        let probs: Vec<f32> = logits.iter().map(|l| pres_sig / (1.0 + (-l).exp())).collect();
        println!("  presence {presence:.4} vs {:.4}", fixture(&st, &format!("{key}.presence_logit"))?[0]);
        compare("scores", &probs, &fixture(&st, &format!("{key}.scores_all"))?);
        compare("boxes", &gpu.download(&out.boxes)?, &fixture(&st, &format!("{key}.boxes_all"))?);
        compare("masks", &gpu.download(&out.masks)?, &fixture(&st, &format!("{key}.mask_logits_all"))?);
        let kept: Vec<String> = probs.iter().filter(|p| **p > 0.5).map(|p| format!("{p:.4}")).collect();
        println!("  kept {kept:?}");
    }
    Ok(())
}
