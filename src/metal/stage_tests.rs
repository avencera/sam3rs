//! Compares the Metal backend's intermediate stages against the PyTorch FP32 fixtures in
//! models/parity: pixels, trunk and FPN levels, then text features, scores, boxes and
//! mask logits per prompt. A diagnostic report rather than a gate (tests/parity.rs is
//! the gate), so it is ignored by default
//!
//!     SAM3RS_STAGES=models/parity/dogs.safetensors SAM3RS_PRECISION=bf16 \
//!         cargo test --release --lib stages -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use eyre::{Result, eyre};
use safetensors::{Dtype, SafeTensors};

use crate::image::RgbImage;
use crate::metal::detector::Detector;
use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::image::preprocess;
use crate::metal::masks::upsample_masks;
use crate::metal::point::{PointDecoder, best_masks, clone, fill_holes, row};
use crate::metal::text::TextEncoder;
use crate::metal::vision::{Backbone, Fpn, Neck, Trunk};
use crate::output::HOLE_AREA;
use crate::precision::Precision;
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::tokenizer::Tokenizer;
use crate::weights::Weights;

/// `[rows, cols]` -> `[cols, rows]` in FP32, for comparing channel-major references
fn transpose(gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
    let out = gpu.empty(&[x.cols(), x.rows()], DType::F32)?;
    gpu.dispatch_1d("transpose", out.len(), &[out.arg(), x.arg(), x.rows().into(), x.cols().into()])?;
    Ok(out)
}

fn fixture(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|e| eyre!("{name}: {e}"))?;
    Ok(match t.dtype() {
        Dtype::F32 => bytemuck::pod_collect_to_vec(t.data()),
        Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
        d => return Err(eyre!("{name}: unsupported {d:?}")),
    })
}

/// Prints the max and relative L2 error; NaNs count as an infinite error
fn compare(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut ref_sq = 0f64;
    for (g, w) in got.iter().zip(want) {
        let d = if g.is_finite() { (g - w).abs() } else { f32::INFINITY };
        max_abs = max_abs.max(d);
        sum_sq += (d as f64).powi(2);
        ref_sq += (*w as f64).powi(2);
    }
    let rel = (sum_sq / ref_sq.max(1e-30)).sqrt();
    println!("{name:>12}: max_abs {max_abs:.3e}  rel_l2 {rel:.3e}");
}

/// Channels-last device tensor -> CHW host vector
fn chw(gpu: &Gpu, t: &Tensor) -> Result<Vec<f32>> {
    gpu.download(&transpose(gpu, t)?)
}

/// Per-kernel GPU time when `SAM3RS_METAL_PROFILE` is set, summed over every dispatch since
/// the last report
fn print_profile(gpu: &Gpu) {
    let rows = gpu.take_profile();
    let total: f64 = rows.iter().map(|r| r.2).sum();
    for (name, count, t) in rows {
        println!("  {name:>24} {count:>6} {:>9.2} ms {:>5.1}%", t * 1e3, 100.0 * t / total);
    }
}

#[test]
#[ignore = "needs a Metal 4 GPU, the weights and the parity fixtures"]
fn stages() -> Result<()> {
    let path = std::env::var_os("SAM3RS_STAGES").map_or_else(|| "models/parity/dogs.safetensors".into(), PathBuf::from);
    let bytes = std::fs::read(&path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let (_, meta) = SafeTensors::read_metadata(&bytes)?;
    let meta: serde_json::Value = serde_json::from_str(&meta.metadata().as_ref().unwrap()["meta"])?;
    let image_path = PathBuf::from(meta["image"].as_str().unwrap());

    let precision: Precision = std::env::var("SAM3RS_PRECISION").as_deref().unwrap_or("fp32").parse()?;
    let t0 = Instant::now();
    let gpu = Gpu::new(0, precision)?;
    gpu.ready()?;
    println!("kernel build: {:?}", t0.elapsed());
    let wts = Weights::open("models/sam3/model.safetensors".as_ref())?;

    let img = RgbImage::open(&image_path)?;
    let want_rgb = fixture(&st, "image_rgb")?;
    let img = RgbImage::new(img.width(), img.height(), want_rgb.iter().map(|&v| v as u8).collect())?;
    let pixels = preprocess(&gpu, &img)?;
    compare("pixels", &gpu.download(&pixels)?, &fixture(&st, "pixels")?);
    // isolate the backbone from resize differences
    let pixels = gpu.upload_f32(&fixture(&st, "pixels")?, &[3, 1008, 1008])?;

    let backbone = Backbone::load(&gpu, &wts)?;
    gpu.synchronize()?;
    for run in 0..3 {
        let t0 = Instant::now();
        let trunk = backbone.forward(&gpu, &pixels)?;
        gpu.synchronize()?;
        println!("backbone run {run}: {:?}", t0.elapsed());
        if run == 0 {
            compare("trunk", &chw(&gpu, &trunk)?, &fixture(&st, "trunk")?);
        }
    }
    print_profile(&gpu);

    let trunk = gpu.upload_f32(&fixture(&st, "trunk")?, &[1024, 72 * 72])?;
    let trunk = Trunk::new(&gpu, transpose(&gpu, &trunk)?)?;
    for (prefix, key) in [("detector_model.vision_encoder.neck", "fpn"), ("tracker_neck", "sam2_fpn")] {
        let neck = Neck::load(&gpu, &wts, prefix)?;
        gpu.synchronize()?;
        let t0 = Instant::now();
        let fpn = neck.forward(&gpu, &trunk)?;
        gpu.synchronize()?;
        println!("{key} neck: {:?}", t0.elapsed());
        print_profile(&gpu);
        // the tracker fixtures hold the point decoder's inputs, whose first two levels
        // pass through its conv_s0 and conv_s1, so only the coarse level compares here
        let levels: &[usize] = if key == "fpn" { &[0, 1, 2] } else { &[2] };
        for &i in levels {
            let f = [&fpn.fine, &fpn.mid, &fpn.coarse][i];
            compare(&format!("{key}{i}"), &chw(&gpu, f)?, &fixture(&st, &format!("{key}{i}"))?);
        }
    }
    Ok(())
}

/// The text encoder and detector from the reference FPN levels, so their errors are
/// their own: text features, then scores, boxes and mask logits for every text and box
/// prompt of the fixture
#[test]
#[ignore = "needs a Metal 4 GPU, the weights and the parity fixtures"]
fn detector_stages() -> Result<()> {
    let path = std::env::var_os("SAM3RS_STAGES").map_or_else(|| "models/parity/dogs.safetensors".into(), PathBuf::from);
    let bytes = std::fs::read(&path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let (_, meta) = SafeTensors::read_metadata(&bytes)?;
    let meta: serde_json::Value = serde_json::from_str(&meta.metadata().as_ref().unwrap()["meta"])?;
    let rgb = st.tensor("image_rgb")?;
    let (w, h) = (rgb.shape()[1] as f64, rgb.shape()[0] as f64);

    let precision: Precision = std::env::var("SAM3RS_PRECISION").as_deref().unwrap_or("fp32").parse()?;
    let gpu = Gpu::new(0, precision)?;
    let wts = Weights::open("models/sam3/model.safetensors".as_ref())?;
    let level = |i: usize, side: usize| {
        let f = gpu.upload_f32(&fixture(&st, &format!("fpn{i}"))?, &[256, side * side])?;
        transpose(&gpu, &f)
    };
    let fpn = Fpn { fine: level(0, 288)?, mid: level(1, 144)?, coarse: level(2, 72)? };
    let tokenizer = Tokenizer::open("models/sam3".as_ref())?;
    let text_encoder = TextEncoder::load(&gpu, &wts)?;
    let detector = Detector::load(&gpu, &wts)?;
    let img_state = detector.prepare_image(&gpu, fpn)?;
    gpu.synchronize()?;

    for prompt in meta["prompts"].as_array().unwrap() {
        if prompt.get("points").is_some() {
            continue;
        }
        let key = prompt["name"].as_str().unwrap().replace(' ', "_");
        let text = prompt.get("text").and_then(|t| t.as_str()).unwrap_or(crate::VISUAL_PROMPT);
        let boxes: Vec<BoxPrompt> = prompt
            .get("boxes")
            .map(|b| b.as_array().unwrap().clone())
            .unwrap_or_default()
            .iter()
            .map(|b| {
                let v: Vec<f64> = b.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                let cxcywh = [(v[0] + v[2]) / 2.0 / w, (v[1] + v[3]) / 2.0 / h, (v[2] - v[0]) / w, (v[3] - v[1]) / h];
                BoxPrompt::from_cxcywh(cxcywh.map(|c| c as f32), v[4] != 0.0)
            })
            .collect::<Result<_>>()?;
        println!("-- {key} (text {text:?}, {} boxes)", boxes.len());
        let tokens = tokenizer.tokenize(text)?;
        for run in 0..3 {
            // the last run's dispatches alone, when SAM3RS_METAL_PROFILE is set
            if run == 2 {
                gpu.take_profile();
            }
            let t0 = Instant::now();
            let feats = text_encoder.forward(&gpu, &tokens)?;
            gpu.synchronize()?;
            let t_text = t0.elapsed();
            let t0 = Instant::now();
            let out = detector.forward(&gpu, &img_state, &feats, &boxes)?;
            gpu.synchronize()?;
            println!("  run {run}: text {t_text:?}, detector {:?}", t0.elapsed());
            if run == 2 {
                print_profile(&gpu);
            }
            if run > 0 {
                continue;
            }
            compare("lang_feats", &gpu.download(&feats.feats)?, &fixture(&st, &format!("{key}.language_features"))?);
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
    }
    Ok(())
}

/// The point decoder from the reference trunk: the tracker neck's conv_s0/conv_s1
/// features, then each point prompt's two passes (IoU and clamped low-res logits) and
/// its final hole-filled full-size masks
///
///     SAM3RS_STAGES=models/parity/dogs.safetensors cargo test --release --lib point_stages -- --ignored --nocapture
#[test]
#[ignore = "needs a Metal 4 GPU, the weights and the parity fixtures"]
fn point_stages() -> Result<()> {
    let path = std::env::var_os("SAM3RS_STAGES").map_or_else(|| "models/parity/dogs.safetensors".into(), PathBuf::from);
    let bytes = std::fs::read(&path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let (_, meta) = SafeTensors::read_metadata(&bytes)?;
    let meta: serde_json::Value = serde_json::from_str(&meta.metadata().as_ref().unwrap()["meta"])?;
    let image_path = PathBuf::from(meta["image"].as_str().unwrap());
    let (width, height) = {
        let img = RgbImage::open(&image_path)?;
        (img.width(), img.height())
    };

    let precision: Precision = std::env::var("SAM3RS_PRECISION").as_deref().unwrap_or("fp32").parse()?;
    let gpu = Gpu::new(0, precision)?;
    let wts = Weights::open("models/sam3/model.safetensors".as_ref())?;

    let trunk = gpu.upload_f32(&fixture(&st, "trunk")?, &[1024, 72 * 72])?;
    let trunk = Trunk::new(&gpu, transpose(&gpu, &trunk)?)?;
    let neck = Neck::load(&gpu, &wts, "tracker_neck")?;
    let fpn = neck.forward(&gpu, &trunk)?;
    let decoder = PointDecoder::load(&gpu, &wts)?;
    let img = decoder.prepare_image(&gpu, &fpn)?;
    compare("sam2_fpn0", &chw(&gpu, &img.feat_s0)?, &fixture(&st, "sam2_fpn0")?);
    compare("sam2_fpn1", &chw(&gpu, &img.feat_s1)?, &fixture(&st, "sam2_fpn1")?);
    compare("sam2_fpn2", &chw(&gpu, &fpn.coarse)?, &fixture(&st, "sam2_fpn2")?);
    drop((neck, fpn, trunk));

    let mut prompts = 0;
    for prompt in meta["prompts"].as_array().unwrap() {
        let Some(points) = prompt.get("points") else { continue };
        prompts += 1;
        let key = prompt["name"].as_str().unwrap().replace(' ', "_");
        let points: Vec<PointPrompt> = points
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let v: Vec<f64> = p.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                PointPrompt::new(v[0] as f32, v[1] as f32, v[2] != 0.0)
            })
            .collect::<Result<_>>()?;
        println!("-- {key} ({} points)", points.len());
        let size = (width, height);
        let decode = |mask_input: Option<&Tensor>| -> Result<(Tensor, Vec<f32>)> {
            let out = decoder.decode(&gpu, &img.embed, &img, &points, size, mask_input, 1)?;
            Ok((out.low_res, gpu.download(&out.iou)?[1..].to_vec()))
        };
        let t0 = Instant::now();
        let (low_first, iou_first) = decode(None)?;
        gpu.synchronize()?;
        println!("  first pass {:?}", t0.elapsed());
        let want_iou = fixture(&st, &format!("{key}.iou_first"))?;
        println!("  iou_first {iou_first:?} vs {want_iou:?}");
        compare("iou_first", &iou_first, &want_iou);
        let clamped: Vec<f32> = gpu.download(&low_first)?.iter().map(|v| v.clamp(-32.0, 32.0)).collect();
        compare("low_first", &clamped, &fixture(&st, &format!("{key}.low_res_first"))?);
        let best = (0..3).max_by(|a, b| iou_first[*a].total_cmp(&iou_first[*b])).unwrap();
        let mask_input = row(&gpu, &low_first, best)?;
        gpu.dispatch_1d("clamp_f32", mask_input.len(), &[mask_input.arg(), (-32.0f32).into(), 32.0f32.into()])?;
        let (low_res, iou) = decode(Some(&mask_input))?;
        let want_iou = fixture(&st, &format!("{key}.iou"))?;
        println!("  iou {iou:?} vs {want_iou:?}");
        compare("iou", &iou, &want_iou);
        let clamped: Vec<f32> = gpu.download(&low_res)?.iter().map(|v| v.clamp(-32.0, 32.0)).collect();
        compare("low_res", &clamped, &fixture(&st, &format!("{key}.low_res"))?);

        let filled = clone(&gpu, &low_res)?;
        fill_holes(&gpu, &filled, HOLE_AREA)?;
        let hw = width * height;
        let mut masks = vec![0u8; 3 * hw];
        gpu.read(&upsample_masks(&gpu, &filled, 0, 3, width, height)?.buf, 0, &mut masks)?;
        let want = fixture(&st, &format!("{key}.masks"))?;
        assert_eq!(masks.len(), 3 * hw, "{key}: mask bytes");
        assert_eq!(want.len(), 3 * hw, "{key}: reference mask pixels");
        for m in 0..3 {
            let (mut inter, mut union) = (0usize, 0usize);
            for (a, b) in masks[m * hw..][..hw].iter().zip(&want[m * hw..][..hw]) {
                let (a, b) = (*a != 0, *b != 0.0);
                inter += (a && b) as usize;
                union += (a || b) as usize;
            }
            println!("  mask {m} iou {:.5}", inter as f64 / union.max(1) as f64);
        }

        // the whole click as `segment_points` runs it, timed; the last run's dispatches
        // print when SAM3RS_METAL_PROFILE is set
        for run in 0..4 {
            if run == 3 {
                gpu.take_profile();
            }
            let t0 = Instant::now();
            let first = decoder.decode(&gpu, &img.embed, &img, &points, size, None, 1)?;
            let mask_input = best_masks(&gpu, &first.low_res, &first.iou, true)?;
            let second = decoder.decode(&gpu, &img.embed, &img, &points, size, Some(&mask_input), 1)?;
            let picked = best_masks(&gpu, &second.low_res, &second.iou, false)?;
            fill_holes(&gpu, &picked, HOLE_AREA)?;
            let full = upsample_masks(&gpu, &picked, 0, 1, width, height)?;
            gpu.download(&second.iou)?;
            let mut mask = vec![0u8; hw];
            gpu.read(&full.buf, 0, &mut mask)?;
            println!("  click run {run}: {:?}", t0.elapsed());
        }
        print_profile(&gpu);
    }
    assert!(prompts > 0, "{} has no point prompts", path.display());
    Ok(())
}
