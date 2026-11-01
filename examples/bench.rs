//! Times sam3rs with the osam benchmark protocol: dogs.jpg, model load and JPEG decode
//! excluded, one warm-up then the median of 5 runs for the image encoder and for each
//! prompt (point, box, negative box, text). Every prompt call ends with full-resolution
//! masks on the host. The text prompt re-runs the text encoder on every call, as osam
//! does; box prompts reuse the cached features of the "visual" text, which osam and
//! PyTorch re-encode each time
//!
//!     cargo run --release --example bench -- --precision bf16

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use eyre::Result;
use sam3rs::{BoxPrompt, Embedding, PointPrompt, Precision, RgbImage, Sam3, VISUAL_PROMPT};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/sam3")]
    model: PathBuf,
    #[arg(long, default_value = "models/images/dogs.jpg")]
    image: PathBuf,
    #[arg(long, default_value_t = 5)]
    repeat: usize,
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// fp32 (strict reference path), tf32 (FP32 with TF32 tensor-core GEMMs) or bf16
    /// (BF16 backbone, fused attention)
    #[arg(long, default_value = "bf16")]
    precision: Precision,
    /// After timing, write the outputs of one more encode and one call per prompt to
    /// this file (scores, boxes, DETR queries and masks), so a timed build can be
    /// checked against the control by hashing it
    #[arg(long)]
    outputs: Option<PathBuf>,
}

fn median_ms(repeat: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let mut times = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        f()?;
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    Ok(times[times.len() / 2])
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let model = Sam3::load(&args.model, args.device, args.precision)?;
    let img = RgbImage::open(&args.image)?;
    let (w, h) = (img.width(), img.height());

    // the osam benchmark prompts for dogs.jpg
    let positive = BoxPrompt::from_xyxy([1233.0, 376.0, 1649.0, 691.0], w, h, true)?;
    let negative = BoxPrompt::from_xyxy([1400.0, 500.0, 1500.0, 600.0], w, h, false)?;
    let point = [PointPrompt::new(1439.0, 504.0, true)?];
    type Prompt<'a> = Box<dyn Fn(&Embedding) -> Result<()> + 'a>;
    let prompts: [(&str, Prompt); 4] = [
        ("point", Box::new(|e| model.segment_points(e, &point).map(drop))),
        ("box", Box::new(|e| model.detect(e, VISUAL_PROMPT, &[positive]).map(drop))),
        ("negative box", Box::new(|e| model.detect(e, VISUAL_PROMPT, &[positive, negative]).map(drop))),
        ("text", Box::new(|e| model.detect(e, "dog", &[]).map(drop))),
    ];

    let emb = model.encode(&img)?;
    let mut row = vec![median_ms(args.repeat, || model.encode(&img).map(drop))?];
    for (_, prompt) in &prompts {
        prompt(&emb)?;
        row.push(median_ms(args.repeat, || prompt(&emb))?);
    }

    if let Some(path) = &args.outputs {
        let emb = model.encode(&img)?;
        let mut bytes = Vec::new();
        let pm = model.segment_points(&emb, &point)?;
        bytes.extend(pm.score.to_le_bytes());
        bytes.extend(pm.mask.iter());
        let detections: [(&str, &[BoxPrompt]); 3] =
            [(VISUAL_PROMPT, &[positive]), (VISUAL_PROMPT, &[positive, negative]), ("dog", &[])];
        for (text, boxes) in detections {
            let found = model.detect(&emb, text, boxes)?;
            bytes.extend((found.len() as u64).to_le_bytes());
            for (i, d) in found.iter().enumerate() {
                bytes.extend(d.score.to_le_bytes());
                for v in d.bbox {
                    bytes.extend(v.to_le_bytes());
                }
                bytes.extend((d.query as u64).to_le_bytes());
                bytes.extend(found.mask(i));
            }
        }
        std::fs::write(path, bytes)?;
    }

    let names: Vec<&str> = prompts.iter().map(|(n, _)| *n).collect();
    println!("| Runtime | Encode | {} |", names.join(" | "));
    // two decimals: at one, a 7 ms prompt moves in 1.4% steps, coarser than the run-to-run noise
    let cells: Vec<String> = row.iter().map(|ms| format!("{ms:.2}")).collect();
    println!("| sam3rs {:?} | {} |", args.precision, cells.join(" | "));
    Ok(())
}
