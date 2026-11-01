//! Times the video tracker on a directory of frames with the PyTorch baseline's protocol
//! (`scripts/video_reference.py bench`): frames decoded before timing, clicks on frame
//! 0, then per-frame latency with masks at the video's size on the host. Steady state is
//! the median over frames 8 onward; the first run warms up and the rest are reported.
//! `--outputs DIR` writes the last run's results for an A/B gate: `outputs.json` (size,
//! objects, frames, scores) and `masks.bin`, each frame's masks in object order as one
//! byte a pixel
//!
//!     cargo run --release --example bench_video -- --frames models/video-frames/horses-720p-64 \
//!         --clip horses --objects 3

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use eyre::{Result, eyre};
use sam3rs::{PointPrompt, Precision, RgbImage, Sam3};

/// Frames before steady state, as in the PyTorch harness
const WARM_FRAMES: usize = 8;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/sam3")]
    model: PathBuf,
    /// Directory of frames, read in name order
    #[arg(long)]
    frames: PathBuf,
    /// Clip name in the prompts file
    #[arg(long)]
    clip: String,
    #[arg(long, default_value_t = 1)]
    objects: usize,
    #[arg(long, default_value = "tests/video_prompts.json")]
    prompts: PathBuf,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value = "bf16")]
    precision: Precision,
    /// Directory for the last run's scores and masks
    #[arg(long)]
    outputs: Option<PathBuf>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let mut paths: Vec<PathBuf> =
        std::fs::read_dir(&args.frames)?.map(|e| e.map(|e| e.path())).collect::<Result<_, _>>()?;
    paths.sort();
    let frames: Vec<RgbImage> = paths.iter().map(|p| RgbImage::open(p)).collect::<Result<_>>()?;
    let (w, h) = (frames[0].width() as f32, frames[0].height() as f32);

    let prompts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args.prompts)?)?;
    let clip = prompts
        .as_array()
        .and_then(|a| a.iter().find(|c| c["clip"] == args.clip.as_str()))
        .ok_or_else(|| eyre!("no clip {} in {}", args.clip, args.prompts.display()))?;
    let clicks = clip["objects"]
        .as_array()
        .ok_or_else(|| eyre!("clip {} has no objects", args.clip))?
        .iter()
        .take(args.objects)
        .map(|o| {
            let (x, y) = (o["point"][0].as_f64().unwrap_or(0.0) as f32, o["point"][1].as_f64().unwrap_or(0.0) as f32);
            Ok(vec![PointPrompt::new(x * w, y * h, true)?])
        })
        .collect::<Result<Vec<_>>>()?;

    let model = Sam3::load(&args.model, args.device, args.precision)?;
    let (mut scores, mut masks) = (Vec::new(), Vec::new());
    for run in 0..args.runs {
        let t0 = Instant::now();
        let (mut tracker, first) = model.track(&frames[0], &clicks, frames.len())?;
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut frame_ms = Vec::with_capacity(frames.len());
        let mut tracked = first.objects.iter().filter(|o| o.score > 0.0).count();
        let mut outputs = vec![first];
        for frame in &frames[1..] {
            let t = Instant::now();
            let out = tracker.next(frame)?;
            frame_ms.push(t.elapsed().as_secs_f64() * 1e3);
            tracked = out.objects.iter().filter(|o| o.score > 0.0).count();
            outputs.push(out);
        }
        // kept outside the timed calls; only the last run's are written
        scores = outputs.iter().map(|f| f.objects.iter().map(|o| o.score).collect::<Vec<_>>()).collect();
        masks = outputs.into_iter().flat_map(|f| f.objects.into_iter().flat_map(|o| o.mask)).collect();
        let steady = median(frame_ms[WARM_FRAMES.min(frame_ms.len() - 1)..].to_vec());
        let label = if run == 0 { "warm-up" } else { "run" };
        println!(
            "{label} {run}: {} {} objects, prompt {prompt_ms:.1} ms, steady {steady:.2} ms ({:.1} FPS), tracked {tracked} at the end",
            args.clip,
            clicks.len(),
            1e3 / steady
        );
    }
    if let Some(dir) = &args.outputs {
        std::fs::create_dir_all(dir)?;
        let meta = serde_json::json!({
            "width": frames[0].width(),
            "height": frames[0].height(),
            "objects": clicks.len(),
            "frames": frames.len(),
            "scores": scores,
        });
        std::fs::write(dir.join("outputs.json"), serde_json::to_string(&meta)?)?;
        std::fs::write(dir.join("masks.bin"), &masks)?;
    }
    Ok(())
}
