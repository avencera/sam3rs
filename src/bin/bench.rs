//! Times sam3rs with the osam benchmark protocol: dogs.jpg, model load and JPEG decode
//! excluded, one warm-up then the median of 5 runs for the image encoder and for each
//! prompt (point, box, negative box, text). Every prompt call ends with full-resolution
//! masks on the host, and detection prompts re-run the text encoder, as osam does

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use eyre::Result;
use sam3rs::detector::BoxPrompt;
use sam3rs::image::RgbImage;
use sam3rs::pipeline::{Embedding, Sam3, VISUAL_PROMPT};
use sam3rs::point::PointPrompt;

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
    let model = Sam3::load(&args.model, args.device)?;
    let img = RgbImage::open(&args.image)?;
    let (w, h) = (img.width, img.height);

    // the osam benchmark prompts for dogs.jpg
    let positive = BoxPrompt::from_xyxy([1233.0, 376.0, 1649.0, 691.0], w, h, true);
    let negative = BoxPrompt::from_xyxy([1400.0, 500.0, 1500.0, 600.0], w, h, false);
    let point = [PointPrompt { x: 1439.0, y: 504.0, positive: true }];
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

    let names: Vec<&str> = prompts.iter().map(|(n, _)| *n).collect();
    println!("| Runtime | Encode | {} |", names.join(" | "));
    let cells: Vec<String> = row.iter().map(|ms| format!("{ms:.1}")).collect();
    println!("| sam3rs | {} |", cells.join(" | "));
    Ok(())
}
