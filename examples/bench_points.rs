//! Times K independent single-click objects against one embedding: K
//! `segment_points` calls in a loop against one `segment_points_batch` call, for each K.
//! Model load, JPEG decode and the encode are excluded; each arm gets one warm-up, then
//! the median of `--repeat` runs. The clicks sit at the centers of a near-square grid
//! over the image, so the input is fixed for a given image and K. Every call ends with
//! the full-resolution masks on the host
//!
//!     cargo run --release --example bench_points -- --image models/images/dogs.jpg

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use eyre::Result;
use sam3rs::{Embedding, PointPrompt, Precision, RgbImage, Sam3};

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Arm {
    Loop,
    Batch,
    Both,
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/sam3")]
    model: PathBuf,
    #[arg(long, default_value = "models/images/dogs.jpg")]
    image: PathBuf,
    /// Object counts to time
    #[arg(long, value_delimiter = ',', default_value = "1,4,8,16,32")]
    k: Vec<usize>,
    #[arg(long, default_value_t = 11)]
    repeat: usize,
    #[arg(long, default_value_t = 0)]
    device: usize,
    #[arg(long, default_value = "bf16")]
    precision: Precision,
    #[arg(long, value_enum, default_value = "both")]
    arm: Arm,
}

fn median_ms(repeat: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    let mut times = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        f()?;
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    Ok(times[times.len() / 2])
}

/// `k` clicks at the cell centers of a grid with about as many columns as rows
fn grid(k: usize, width: usize, height: usize) -> Result<Vec<Vec<PointPrompt>>> {
    let cols = (k as f64).sqrt().ceil() as usize;
    let rows = k.div_ceil(cols);
    (0..k)
        .map(|i| {
            let (c, r) = (i % cols, i / cols);
            let x = (c as f32 + 0.5) * width as f32 / cols as f32;
            let y = (r as f32 + 0.5) * height as f32 / rows as f32;
            Ok(vec![PointPrompt::new(x, y, true)?])
        })
        .collect()
}

fn time_loop(model: &Sam3, emb: &Embedding, prompts: &[Vec<PointPrompt>], repeat: usize) -> Result<f64> {
    median_ms(repeat, || {
        for p in prompts {
            // dropped before the next call, as a caller that keeps only a reduced copy would
            drop(model.segment_points(emb, p)?);
        }
        Ok(())
    })
}

fn time_batch(model: &Sam3, emb: &Embedding, prompts: &[Vec<PointPrompt>], repeat: usize) -> Result<f64> {
    let prompts: Vec<&[PointPrompt]> = prompts.iter().map(Vec::as_slice).collect();
    median_ms(repeat, || model.segment_points_batch(emb, &prompts).map(drop))
}

/// Worst score difference and lowest mask IoU of the batched call against the loop
fn check(model: &Sam3, emb: &Embedding, prompts: &[Vec<PointPrompt>]) -> Result<(f32, f64)> {
    let refs: Vec<&[PointPrompt]> = prompts.iter().map(Vec::as_slice).collect();
    let batch = model.segment_points_batch(emb, &refs)?;
    eyre::ensure!(batch.len() == prompts.len(), "{} masks for {} prompts", batch.len(), prompts.len());
    let (mut score, mut iou) = (0f32, 1f64);
    for (i, p) in prompts.iter().enumerate() {
        let single = model.segment_points(emb, p)?;
        let got = batch.mask(i);
        eyre::ensure!(got.len() == single.mask.len() && batch.scores()[i].is_finite(), "prompt {i}");
        score = score.max((batch.scores()[i] - single.score).abs());
        let inter = got.iter().zip(single.mask.iter()).filter(|(a, b)| **a != 0 && **b != 0).count();
        let union = got.iter().zip(single.mask.iter()).filter(|(a, b)| **a != 0 || **b != 0).count();
        iou = iou.min(if union == 0 { 1.0 } else { inter as f64 / union as f64 });
    }
    Ok((score, iou))
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let model = Sam3::load(&args.model, args.device, args.precision)?;
    let img = RgbImage::open(&args.image)?;
    let emb = model.encode(&img)?;
    let name = args.image.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());

    println!(
        "| Image | Precision | K | Loop ms | Batch ms | Speedup | Loop ms/obj | Batch ms/obj | Worst score diff | Min IoU |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for &k in &args.k {
        let prompts = grid(k, img.width(), img.height())?;
        let looped = match args.arm {
            Arm::Batch => f64::NAN,
            _ => time_loop(&model, &emb, &prompts, args.repeat)?,
        };
        let batched = match args.arm {
            Arm::Loop => f64::NAN,
            _ => time_batch(&model, &emb, &prompts, args.repeat)?,
        };
        let kf = k as f64;
        let (score, iou) = match args.arm {
            Arm::Both => check(&model, &emb, &prompts)?,
            _ => (f32::NAN, f64::NAN),
        };
        println!(
            "| {name} {}x{} | {:?} | {k} | {looped:.2} | {batched:.2} | {:.2}x | {:.2} | {:.2} | {score:.5} | {iou:.5} |",
            img.width(),
            img.height(),
            args.precision,
            looped / batched,
            looped / kf,
            batched / kf
        );
    }
    Ok(())
}
