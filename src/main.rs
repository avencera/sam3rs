//! Segments an image with a text, box or point prompt and writes a mask overlay
//!
//!     sam3rs image.jpg --text dog --out dogs.png
//!     sam3rs image.jpg --box 1233,376,1649,691 --neg-box 1400,500,1500,600
//!     sam3rs image.jpg --point 1439,504 --neg-point 1500,600
//!
//! Without `--model`, the weights come from the gated facebook/sam3 Hugging Face repo

use std::path::PathBuf;

use clap::Parser;
use eyre::{Result, bail, eyre};
use sam3rs::{BoxPrompt, PointPrompt, Precision, RgbImage, Sam3, VISUAL_PROMPT};
use serde_json::json;

#[derive(Parser)]
#[command(about = "SAM 3 image segmentation on CUDA")]
struct Args {
    image: PathBuf,
    /// noun phrase to detect, alone or with boxes
    #[arg(long)]
    text: Option<String>,
    /// positive box exemplar x0,y0,x1,y1 in pixels
    #[arg(long = "box", value_parser = parse_coords::<4>)]
    boxes: Vec<[f32; 4]>,
    /// negative box exemplar x0,y0,x1,y1
    #[arg(long = "neg-box", value_parser = parse_coords::<4>)]
    neg_boxes: Vec<[f32; 4]>,
    /// foreground click x,y; points can't be combined with text or boxes
    #[arg(long = "point", value_parser = parse_coords::<2>)]
    points: Vec<[f32; 2]>,
    /// background click x,y
    #[arg(long = "neg-point", value_parser = parse_coords::<2>)]
    neg_points: Vec<[f32; 2]>,
    /// PNG with the masks drawn over the image
    #[arg(long)]
    out: Option<PathBuf>,
    /// directory with model.safetensors, vocab.json and merges.txt [default: download
    /// facebook/sam3 from Hugging Face]
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// fp32 (strict reference path), tf32 (FP32 with TF32 tensor-core GEMMs) or bf16
    /// (BF16 backbone, fused attention)
    #[arg(long, default_value = "bf16")]
    precision: Precision,
}

fn parse_coords<const N: usize>(s: &str) -> Result<[f32; N]> {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>()?;
    v.try_into().map_err(|_| eyre!("expected {N} comma-separated numbers"))
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let has_points = !args.points.is_empty() || !args.neg_points.is_empty();
    let has_boxes = !args.boxes.is_empty() || !args.neg_boxes.is_empty();
    if has_points && (has_boxes || args.text.is_some()) {
        bail!("point prompts can't be combined with text or boxes");
    }
    if !has_points && !has_boxes && args.text.is_none() {
        bail!("give --text, --box or --point");
    }

    let img = RgbImage::open(&args.image)?;
    let model = match &args.model {
        Some(dir) => Sam3::load(dir, args.device, args.precision)?,
        None => load_pretrained(args.device, args.precision)?,
    };
    let emb = model.encode(&img)?;

    let (results, masks) = if has_points {
        let clicks = args.points.iter().map(|p| (p, true)).chain(args.neg_points.iter().map(|p| (p, false)));
        let points: Vec<PointPrompt> =
            clicks.map(|(p, positive)| PointPrompt::new(p[0], p[1], positive)).collect::<Result<_>>()?;
        let m = model.segment_points(&emb, &points)?;
        (vec![json!({ "score": m.score })], vec![m.mask])
    } else {
        let exemplars = args.boxes.iter().map(|b| (b, true)).chain(args.neg_boxes.iter().map(|b| (b, false)));
        let boxes: Vec<BoxPrompt> = exemplars
            .map(|(b, positive)| BoxPrompt::from_xyxy(*b, img.width(), img.height(), positive))
            .collect::<Result<_>>()?;
        let text = args.text.as_deref().unwrap_or(VISUAL_PROMPT);
        let dets = model.detect(&emb, text, &boxes)?;
        let results = dets.iter().map(|d| json!({ "score": d.score, "bbox_xyxy": d.bbox })).collect();
        (results, dets.into_iter().map(|d| d.mask).collect())
    };
    for r in &results {
        println!("{r}");
    }
    if let Some(out) = &args.out {
        let masks: Vec<&[u8]> = masks.iter().map(Vec::as_slice).collect();
        img.overlay(&masks)?.save(out)?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}

#[cfg(feature = "online")]
fn load_pretrained(device: usize, precision: Precision) -> Result<Sam3> {
    Sam3::from_pretrained(device, precision)
}

#[cfg(not(feature = "online"))]
fn load_pretrained(_device: usize, _precision: Precision) -> Result<Sam3> {
    bail!("this build has no `online` feature, so pass --model with a local model directory")
}
