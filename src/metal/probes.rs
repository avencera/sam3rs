//! Timing probes, ignored by default: where a click's wall time goes and per-kernel GPU
//! time of tracked video frames

use std::time::Instant;

use eyre::Result;

use crate::image::RgbImage;
use crate::precision::Precision;
use crate::prompt::PointPrompt;
use crate::sam3::Sam3;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2] * 1e3
}

/// Where a click's wall time goes: the whole click, one decoder pass with its wait,
/// passes queued back to back, and the host encoding of a pass alone
#[test]
#[ignore = "needs a Metal 4 GPU and the weights; timing only"]
fn click_breakdown() -> Result<()> {
    let m = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
    let img = RgbImage::open("models/images/dogs.jpg".as_ref())?;
    let emb = m.encode(&img)?;
    let pts = [PointPrompt::new(1439.0, 504.0, true)?];
    let gpu = &m.backend.gpu;
    let size = emb.size();
    let pass =
        || m.backend.point.decode_batch(gpu, &emb.image.point.embed, &emb.image.point, &[&pts[..]], size, None, 1);
    let (mut full, mut one, mut encode, mut queued) = (vec![], vec![], vec![], vec![]);
    for i in 0..23 {
        let t0 = Instant::now();
        m.segment_points(&emb, &pts)?;
        let t_full = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let d = pass()?;
        let t_enc = t0.elapsed().as_secs_f64();
        gpu.download(&d.iou)?;
        let t_one = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        for _ in 0..10 {
            pass()?;
        }
        gpu.synchronize()?;
        let t_q = t0.elapsed().as_secs_f64() / 10.0;
        if i >= 3 {
            full.push(t_full);
            one.push(t_one);
            encode.push(t_enc);
            queued.push(t_q);
        }
    }
    println!(
        "click {:.2} ms; one pass {:.2} ms (host encode {:.2} ms); queued passes {:.2} ms each",
        median(full),
        median(one),
        median(encode),
        median(queued)
    );
    Ok(())
}

/// Per-kernel GPU time of tracked horses frames (3 objects), after 8 warm frames;
/// set `SAM3RS_METAL_PROFILE`
#[test]
#[ignore = "needs a Metal 4 GPU, the weights and the horses frames; timing only"]
fn video_profile() -> Result<()> {
    let m = Sam3::load("models/sam3".as_ref(), 0, Precision::Bf16)?;
    let mut paths: Vec<_> = std::fs::read_dir("models/video-frames/horses-720p-64")?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.sort();
    let frames: Vec<RgbImage> = paths.iter().take(14).map(|p| RgbImage::open(p)).collect::<Result<_>>()?;
    let (w, h) = (frames[0].width() as f32, frames[0].height() as f32);
    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("tests/video_prompts.json")?)?;
    let clip = prompts.as_array().unwrap().iter().find(|c| c["clip"] == "horses").unwrap();
    let clicks = clip["objects"]
        .as_array()
        .unwrap()
        .iter()
        .take(3)
        .map(|o| {
            let (x, y) = (o["point"][0].as_f64().unwrap() as f32, o["point"][1].as_f64().unwrap() as f32);
            Ok(vec![PointPrompt::new(x * w, y * h, true)?])
        })
        .collect::<Result<Vec<_>>>()?;

    let (mut tracker, _) = m.track(&frames[0], &clicks, frames.len())?;
    for f in &frames[1..9] {
        tracker.next(f)?;
    }
    m.backend.gpu.take_profile();
    let measured = &frames[9..];
    let t0 = Instant::now();
    for f in measured {
        tracker.next(f)?;
    }
    let wall = t0.elapsed().as_secs_f64() / measured.len() as f64;
    let mut rows = m.backend.gpu.take_profile();
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    let total: f64 = rows.iter().map(|r| r.2).sum();
    let per = 1e3 / measured.len() as f64;
    println!("frame wall {:.1} ms (profiled), GPU {:.1} ms", wall * 1e3, total * per);
    for (name, count, t) in rows {
        println!(
            "  {name:>28} {:>7.1} {:>9.2} ms {:>5.1}%",
            count as f64 / measured.len() as f64,
            t * per,
            100.0 * t / total
        );
    }
    Ok(())
}
