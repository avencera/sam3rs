//! Compares the image path's intermediate stages against the PyTorch FP32 fixtures in
//! models/parity: pixels, trunk and FPN levels; the point decoder's two passes and masks
//! per point prompt; text features, scores, boxes and mask logits per text or box prompt.
//! Each stage starts from the reference's own inputs, so its error is its own. One suite
//! for every backend, through its [`TestHooks`]. A diagnostic report rather than a gate
//! (tests/parity.rs is the gate), so it is ignored by default
//!
//!     SAM3RS_STAGES=models/parity/dogs.safetensors SAM3RS_PRECISION=bf16 \
//!         cargo test --release --lib stage_tests:: -- --ignored --nocapture --test-threads 1
//!
//! Per-kernel GPU times print after the timed runs where the backend profiles kernels
//! (Metal with `SAM3RS_METAL_PROFILE` set)

use std::path::PathBuf;
use std::time::Instant;

use eyre::Result;

use crate::backend::{Active, Backend};
use crate::image::RgbImage;
use crate::model::point::LOW_RES;
use crate::model::vision::DIM;
use crate::output::{HOLE_AREA, VISUAL_PROMPT};
use crate::prompt::{BoxPrompt, PointPrompt};
use crate::sam3::Sam3;
use crate::test_support::{
    Fixture, TestHooks, compare, fill_holes_host, hole_fill_logits, load_model, stage_fixture_path,
};
use crate::tokenizer::Tokenizer;

/// Per-kernel GPU time since the last report, where the backend profiles kernels
fn print_profile(b: &Active) {
    let rows = b.take_profile();
    let total: f64 = rows.iter().map(|r| r.2).sum();
    for (name, count, t) in rows {
        println!("  {name:>24} {count:>6} {:>9.2} ms {:>5.1}%", t * 1e3, 100.0 * t / total);
    }
}

/// The fixture and the model, with the load time printed
fn open() -> Result<(Fixture, Sam3)> {
    let fx = Fixture::open(&stage_fixture_path())?;
    let t0 = Instant::now();
    let m = load_model()?;
    println!("model load: {:?}", t0.elapsed());
    Ok((fx, m))
}

/// The reference trunk, prepared for the necks
fn reference_trunk(b: &Active, fx: &Fixture) -> Result<<Active as TestHooks>::Trunk> {
    b.neck_input(b.upload_chw(fx.values("trunk")?, DIM)?)
}

/// Preprocessing, the backbone (timed over 3 runs) and both necks, each from the
/// reference's output of the stage before
#[test]
#[ignore = "needs a GPU, the weights and the parity fixtures"]
fn stages() -> Result<()> {
    let (fx, m) = open()?;
    let b = &m.backend;
    let image_path = PathBuf::from(fx.meta["image"].as_str().unwrap());

    let img = RgbImage::open(&image_path)?;
    let want_rgb = fx.values("image_rgb")?;
    let diff = img.data().iter().zip(&want_rgb).filter(|(a, b)| **a as f32 != **b).count();
    println!("rgb bytes differing from PIL: {diff} of {}", want_rgb.len());
    let img = RgbImage::new(img.width(), img.height(), want_rgb.iter().map(|&v| v as u8).collect())?;
    let pixels = b.preprocess(&img)?;
    compare("pixels", &b.download_pixels(&pixels)?, &fx.values("pixels")?);
    // isolate the backbone from resize differences
    let pixels = b.upload_pixels(fx.values("pixels")?)?;

    for run in 0..3 {
        let t0 = Instant::now();
        let trunk = b.trunk(&pixels)?;
        b.sync()?;
        println!("backbone run {run}: {:?}", t0.elapsed());
        if run == 0 {
            compare("trunk", &b.chw(&trunk)?, &fx.values("trunk")?);
        }
    }
    print_profile(b);

    let trunk = reference_trunk(b, &fx)?;
    for (tracker, key) in [(false, "fpn"), (true, "sam2_fpn")] {
        b.sync()?;
        let t0 = Instant::now();
        let fpn = b.neck(&trunk, tracker)?;
        b.sync()?;
        println!("{key} neck: {:?}", t0.elapsed());
        print_profile(b);
        // the tracker fixtures hold the point decoder's inputs, whose first two levels
        // pass through its conv_s0 and conv_s1 (see `point_stages`), so only the coarse
        // level compares here
        let levels: &[usize] = if tracker { &[2] } else { &[0, 1, 2] };
        for &i in levels {
            compare(&format!("{key}{i}"), &b.chw(b.fpn_levels(&fpn)[i])?, &fx.values(&format!("{key}{i}"))?);
        }
    }
    Ok(())
}

/// The point decoder from the reference trunk: the tracker neck's conv_s0/conv_s1
/// features, then each point prompt's two passes (IoU and clamped low-res logits), its
/// final hole-filled full-size masks, and the whole click as `segment_points` runs it,
/// timed
///
///     SAM3RS_STAGES=models/parity/dogs.safetensors cargo test --release --lib point_stages -- --ignored --nocapture
#[test]
#[ignore = "needs a GPU, the weights and the parity fixtures"]
fn point_stages() -> Result<()> {
    let (fx, m) = open()?;
    let b = &m.backend;
    let image = RgbImage::open(&PathBuf::from(fx.meta["image"].as_str().unwrap()))?;
    let size = (image.width(), image.height());

    let img = {
        let fpn = b.neck(&reference_trunk(b, &fx)?, true)?;
        let img = b.point_image(&fpn)?;
        let [s0, s1] = b.point_levels(&img);
        compare("sam2_fpn0", &b.chw(s0)?, &fx.values("sam2_fpn0")?);
        compare("sam2_fpn1", &b.chw(s1)?, &fx.values("sam2_fpn1")?);
        compare("sam2_fpn2", &b.chw(b.fpn_levels(&fpn)[2])?, &fx.values("sam2_fpn2")?);
        img
    };
    // the public path for the timed clicks, from the image itself
    let emb = m.encode(&image)?;

    let mut prompts = 0;
    for prompt in fx.meta["prompts"].as_array().unwrap() {
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
        let t0 = Instant::now();
        let first = b.decoded_outputs(&b.point_decode(&img, &points, size, None)?)?;
        println!("  first pass {:?}", t0.elapsed());
        let (iou_first, want_iou) = (&first.iou[1..], fx.values(&format!("{key}.iou_first"))?);
        println!("  iou_first {iou_first:?} vs {want_iou:?}");
        compare("iou_first", iou_first, &want_iou);
        let clamped: Vec<f32> = first.low_res.iter().map(|v| v.clamp(-32.0, 32.0)).collect();
        compare("low_first", &clamped, &fx.values(&format!("{key}.low_res_first"))?);

        // the best first-pass mask, clamped as upstream does before reusing it
        let best = (0..3).max_by(|a, b| iou_first[*a].total_cmp(&iou_first[*b])).unwrap();
        let side = LOW_RES * LOW_RES;
        let mask_input = &clamped[best * side..][..side];
        let second = b.point_decode(&img, &points, size, Some(mask_input))?;
        let out = b.decoded_outputs(&second)?;
        let (iou, want_iou) = (&out.iou[1..], fx.values(&format!("{key}.iou"))?);
        println!("  iou {iou:?} vs {want_iou:?}");
        compare("iou", iou, &want_iou);
        let clamped: Vec<f32> = out.low_res.iter().map(|v| v.clamp(-32.0, 32.0)).collect();
        compare("low_res", &clamped, &fx.values(&format!("{key}.low_res"))?);

        let masks = b.full_masks(&second, 3, size)?;
        let want = fx.values(&format!("{key}.masks"))?;
        let hw = size.0 * size.1;
        assert_eq!(masks.len(), 3 * hw, "{key}: mask bytes");
        assert_eq!(want.len(), 3 * hw, "{key}: reference mask pixels");
        for i in 0..3 {
            let (mut inter, mut union) = (0usize, 0usize);
            for (a, b) in masks[i * hw..][..hw].iter().zip(&want[i * hw..][..hw]) {
                let (a, b) = (*a != 0, *b != 0.0);
                inter += (a && b) as usize;
                union += (a || b) as usize;
            }
            println!("  mask {i} iou {:.5}", inter as f64 / union.max(1) as f64);
        }

        // the last run's kernels alone, where the backend profiles them
        for run in 0..4 {
            if run == 3 {
                b.take_profile();
            }
            let t0 = Instant::now();
            m.segment_points(&emb, &points)?;
            println!("  click run {run}: {:?}", t0.elapsed());
        }
        print_profile(b);
    }
    assert!(prompts > 0, "{} has no point prompts", stage_fixture_path().display());
    Ok(())
}

/// The text encoder and detector from the reference FPN levels, so their errors are
/// their own: text features, then scores, boxes and mask logits for every text and box
/// prompt of the fixture, each timed over 3 runs
#[test]
#[ignore = "needs a GPU, the weights and the parity fixtures"]
fn detector_stages() -> Result<()> {
    let (fx, m) = open()?;
    let b = &m.backend;
    let (_, rgb_shape) = fx.get("image_rgb")?;
    let (w, h) = (rgb_shape[1] as f64, rgb_shape[0] as f64);
    let level = |i: usize| b.upload_chw(fx.values(&format!("fpn{i}"))?, 256);
    let img = b.detector_image([level(0)?, level(1)?, level(2)?])?;
    let tokenizer = Tokenizer::open("models/sam3".as_ref())?;
    b.sync()?;

    for prompt in fx.meta["prompts"].as_array().unwrap() {
        if prompt.get("points").is_some() {
            continue;
        }
        let key = prompt["name"].as_str().unwrap().replace(' ', "_");
        let text = prompt.get("text").and_then(|t| t.as_str()).unwrap_or(VISUAL_PROMPT);
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
            // the last run's kernels alone, where the backend profiles them
            if run == 2 {
                b.take_profile();
            }
            let t0 = Instant::now();
            let feats = b.encode_text(&tokens)?;
            b.sync()?;
            let t_text = t0.elapsed();
            let t0 = Instant::now();
            let out = b.detect_from(&img, &feats, &boxes)?;
            b.sync()?;
            println!("  run {run}: text {t_text:?}, detector {:?}", t0.elapsed());
            if run == 2 {
                print_profile(b);
            }
            if run > 0 {
                continue;
            }
            compare("lang_feats", &b.text_features(&feats)?, &fx.values(&format!("{key}.language_features"))?);
            let scores = b.detection_scores(&out)?;
            let pres_sig = 1.0 / (1.0 + (-scores.presence).exp());
            let probs: Vec<f32> = scores.logits.iter().map(|l| pres_sig / (1.0 + (-l).exp())).collect();
            println!("  presence {:.4} vs {:.4}", scores.presence, fx.values(&format!("{key}.presence_logit"))?[0]);
            compare("scores", &probs, &fx.values(&format!("{key}.scores_all"))?);
            compare("boxes", &scores.boxes, &fx.values(&format!("{key}.boxes_all"))?);
            compare("masks", &b.mask_logits(&out)?, &fx.values(&format!("{key}.mask_logits_all"))?);
            let kept: Vec<String> = probs.iter().filter(|p| **p > 0.5).map(|p| format!("{p:.4}")).collect();
            println!("  kept {kept:?}");
        }
    }
    Ok(())
}

/// The GPU hole fill against the host flood fill it replaced, bit for bit
#[test]
#[ignore = "needs a GPU"]
fn gpu_fill_matches_the_host_flood_fill() -> Result<()> {
    let side = LOW_RES;
    let host = hole_fill_logits(6);
    let mut want = host.clone();
    want.chunks_mut(side * side).for_each(|m| fill_holes_host(m, side, HOLE_AREA));
    let changed = want.iter().zip(&host).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert!(changed > 1000, "the fixture fills only {changed} pixels");

    let got = Active::fill_holes_on_gpu(host)?;
    assert_eq!(got.len(), want.len());
    let wrong = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(wrong, 0, "{wrong} of {} pixels differ from the host fill", got.len());
    Ok(())
}
