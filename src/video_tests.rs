//! Compares the video tracker with the PyTorch fixtures `scripts/video_reference.py dump`
//! writes: each stage on the reference's own inputs at the fully dumped frames, then the
//! whole tracker over the clip from the reference's pixels
//!
//! Both run against an FP32 fixture. In FP32 mode the end-to-end check holds the masks to
//! it; in the other modes it allows PyTorch BF16's own drift, read from the clip's BF16
//! fixture next to it
//!
//!     SAM3RS_VIDEO=models/video-parity/horses-720p-64-o3-fp32.safetensors \
//!         cargo test --release --lib video_ -- --ignored --nocapture --test-threads 1

use std::path::{Path, PathBuf};

use eyre::{Result, ensure, eyre};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::gpu::{Buf, Gpu, Precision, Tensor, launch};
use crate::pipeline::Sam3;
use crate::point::{LOW_RES, MASK_TOKEN_ROW, PointImage, PointPrompt};
use crate::video::VideoTracker;
use crate::vision::{GRID, IMAGE_SIZE};

/// Stage tolerances in FP32 (rel L2 on the reference's own inputs): every stage is a
/// handful of FP32 GEMMs and reductions, which reorder sums but don't lose precision
const STAGE_REL_L2: f32 = 1e-4;
/// The memory attention over ~36k keys and the low-res mask logits sum over more terms
const LONG_REL_L2: f32 = 5e-4;
/// Stage tolerance in the reduced-precision modes, at BF16's own rounding: the point
/// decoder's BF16 linears alone put the IoU head at 3.2e-3 on FP32 inputs
const REDUCED_REL_L2: f32 = 5e-3;
/// Backbone and neck stages in the reduced-precision modes: they are the image path's BF16
/// encoder, gated end to end by `tests/parity.rs`, and drift 1.4e-2 to 2.6e-2 from FP32 on
/// the video fixtures
const REDUCED_BACKBONE_REL_L2: f32 = 4e-2;
/// Outputs whose every error is below this pass whatever their rel L2, which is noise on
/// values near zero (an absent object's IoU predictions are about 1e-15)
const NEGLIGIBLE_ABS: f32 = 1e-6;

struct Fixture {
    mmap: Mmap,
    meta: Value,
}

impl Fixture {
    fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| eyre!("{}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (_, header) = SafeTensors::read_metadata(&mmap)?;
        let meta = serde_json::from_str(&header.metadata().as_ref().ok_or_else(|| eyre!("no metadata"))?["meta"])?;
        Ok(Self { mmap, meta })
    }

    fn st(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }

    fn has(&self, name: &str) -> bool {
        self.st().is_ok_and(|st| st.tensor(name).is_ok())
    }

    /// The tensor as f32 values and its shape
    fn get(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let st = self.st()?;
        let t = st.tensor(name).map_err(|e| eyre!("fixture {name}: {e}"))?;
        let v = match t.dtype() {
            Dtype::F32 => bytemuck::pod_collect_to_vec(t.data()),
            Dtype::U8 => t.data().iter().map(|&b| b as f32).collect(),
            d => return Err(eyre!("{name}: unsupported {d:?}")),
        };
        Ok((v, t.shape().to_vec()))
    }

    fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let st = self.st()?;
        Ok(st.tensor(name).map_err(|e| eyre!("fixture {name}: {e}"))?.data().to_vec())
    }

    fn frame(&self, t: usize) -> &Value {
        &self.meta["per_frame"][t]
    }

    /// Key prefix `f{t}.{name}.{k}` of the `nth` call to `name` on frame `t`
    fn call(&self, t: usize, name: &str, nth: usize) -> Option<String> {
        let calls = self.frame(t)["calls"].as_array()?;
        let k = calls.iter().enumerate().filter(|(_, c)| c.as_str() == Some(name)).nth(nth)?.0;
        Some(format!("f{t}.{name}.{k}"))
    }

    fn frames(&self) -> usize {
        self.meta["frames"].as_u64().unwrap() as usize
    }

    /// Frames in the whole clip, which sets the tracker's temporal scale even when the
    /// fixture dumped fewer
    fn video_frames(&self) -> usize {
        self.meta["video"]["frames"].as_u64().unwrap() as usize
    }

    fn size(&self) -> (usize, usize) {
        let v = &self.meta["video"];
        (v["width"].as_u64().unwrap() as usize, v["height"].as_u64().unwrap() as usize)
    }

    /// One positive click per object, in the video's pixels
    fn clicks(&self) -> Result<Vec<Vec<PointPrompt>>> {
        let (w, h) = self.size();
        self.meta["objects"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| {
                let p = &o["point"];
                let (x, y) = (p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32);
                Ok(vec![PointPrompt::new(x * w as f32, y * h as f32, true)?])
            })
            .collect()
    }
}

/// rel L2 of `got` against `want`, printed with the max abs error; fails on a length
/// mismatch or a non-finite value
fn rel_l2(name: &str, got: &[f32], want: &[f32]) -> f32 {
    rel_l2_max(name, got, want).0
}

/// [`rel_l2`] and the max abs error
fn rel_l2_max(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    assert!(!got.is_empty(), "{name}: empty");
    assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite output");
    let (mut max_abs, mut err, mut norm) = (0f32, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        max_abs = max_abs.max((g - w).abs());
        err += ((g - w) as f64).powi(2);
        norm += (*w as f64).powi(2);
    }
    let rel = (err / norm.max(1e-30)).sqrt() as f32;
    println!("{name:>28}: max_abs {max_abs:.3e}  rel_l2 {rel:.3e}");
    (rel, max_abs)
}

fn check(failures: &mut Vec<String>, name: &str, got: &[f32], want: &[f32], tol: f32) {
    let (rel, max_abs) = rel_l2_max(name, got, want);
    // a NaN reference fails too
    if rel.is_nan() || (rel > tol && max_abs >= NEGLIGIBLE_ABS) {
        failures.push(format!("{name}: rel_l2 {rel:.3e} > {tol:.0e}"));
    }
}

/// Object `b` of a sequence-first `[n, objects, c]` tensor, as `[n, c]`
fn seq_object(v: &[f32], shape: &[usize], b: usize) -> Vec<f32> {
    let (n, objects, c) = (shape[0], shape[1], shape[2]);
    (0..n).flat_map(|i| v[(i * objects + b) * c..][..c].iter().copied()).collect()
}

/// Object `b` of a batch-first tensor `[objects, ...]`
fn batch_object(v: &[f32], shape: &[usize], b: usize) -> Vec<f32> {
    let n = v.len() / shape[0];
    v[b * n..][..n].to_vec()
}

/// `[c, hw]` -> `[hw, c]`
fn to_hwc(v: &[f32], c: usize) -> Vec<f32> {
    let hw = v.len() / c;
    (0..hw).flat_map(|p| (0..c).map(move |ch| v[ch * hw + p])).collect()
}

fn download_hwc_as_chw(gpu: &Gpu, t: &Tensor) -> Result<Vec<f32>> {
    let v = gpu.download(t)?;
    let (hw, c) = (t.rows(), t.cols());
    Ok((0..c).flat_map(|ch| (0..hw).map(|p| v[p * c + ch]).collect::<Vec<_>>()).collect())
}

fn fixture_path() -> PathBuf {
    std::env::var_os("SAM3RS_VIDEO")
        .map_or_else(|| "models/video-parity/horses-720p-64-o3-fp32.safetensors".into(), PathBuf::from)
}

fn load_model() -> Result<Sam3> {
    let precision: Precision = std::env::var("SAM3RS_PRECISION").as_deref().unwrap_or("fp32").parse()?;
    Sam3::load("models/sam3".as_ref(), 0, precision)
}

fn pixels(gpu: &Gpu, fx: &Fixture, t: usize) -> Result<Tensor> {
    let (v, shape) = fx.get(&format!("f{t}.input"))?;
    ensure!(shape == [3, IMAGE_SIZE, IMAGE_SIZE], "f{t}.input is {shape:?}");
    gpu.upload(v, &[3, IMAGE_SIZE, IMAGE_SIZE])
}

/// Each tracker stage on the reference's inputs at every fully dumped frame
#[test]
#[ignore = "needs a CUDA GPU, the weights and the video fixtures"]
fn video_stages() -> Result<()> {
    let fx = Fixture::open(&fixture_path())?;
    ensure!(fx.meta["precision"] == "fp32", "stage checks need an FP32 fixture");
    let m = load_model()?;
    let gpu = &m.gpu;
    let (w, h) = fx.size();
    let clicks = fx.clicks()?;
    let objects = clicks.len();
    let mut failures = Vec::new();
    let mut checked = 0;
    // frames (or calls) each optional stage ran on, so a fixture missing a hook fails
    let (mut mem_attn_runs, mut obj_ptr_runs, mut mem_enc_runs) = (0, 0, 0);
    let (stage_tol, long_tol, backbone_tol) = match gpu.precision() {
        Precision::Fp32 => (STAGE_REL_L2, LONG_REL_L2, STAGE_REL_L2),
        Precision::Tf32 | Precision::Bf16 => (REDUCED_REL_L2, REDUCED_REL_L2, REDUCED_BACKBONE_REL_L2),
    };

    for t in 0..fx.frames() {
        let Some(bb) = fx.call(t, "backbone", 0) else { continue };
        if !fx.has(&format!("{bb}.fpn2")) {
            continue;
        }
        println!("frame {t}");
        let feats = m.video_features(&pixels(gpu, &fx, t)?)?;
        let fpn = |l: usize| fx.get(&format!("{bb}.fpn{l}")).map(|(v, _)| v);
        check(&mut failures, &format!("f{t} fpn2"), &download_hwc_as_chw(gpu, &feats.fpn2)?, &fpn(2)?, backbone_tol);
        check(
            &mut failures,
            &format!("f{t} feat_s0"),
            &download_hwc_as_chw(gpu, &feats.point.feat_s0)?,
            &fpn(0)?,
            backbone_tol,
        );
        check(
            &mut failures,
            &format!("f{t} feat_s1"),
            &download_hwc_as_chw(gpu, &feats.point.feat_s1)?,
            &fpn(1)?,
            backbone_tol,
        );

        if let Some(k) = fx.call(t, "mem_attn", 0) {
            mem_attn_runs += 1;
            let ptr_tokens = fx.frame(t)["num_obj_ptr_tokens"].as_u64().unwrap() as usize;
            let (src, ss) = fx.get(&format!("{k}.src"))?;
            let (src_pos, _) = fx.get(&format!("{k}.src_pos"))?;
            let (prompt, ps) = fx.get(&format!("{k}.prompt"))?;
            let (prompt_pos, _) = fx.get(&format!("{k}.prompt_pos"))?;
            let (out, os) = fx.get(&format!("{k}.out"))?;
            let ours_pos: Vec<f32> = gpu.download(&m.tracker.mem_attn.src_pos)?.iter().map(|v| v * 10.0).collect();
            check(&mut failures, &format!("f{t} src_pos"), &ours_pos, &seq_object(&src_pos, &ss, 0), stage_tol);
            for b in 0..ss[1] {
                let up = |v: Vec<f32>, c: usize| gpu.upload(v, &[ps[0], c]);
                let src_b = gpu.upload(seq_object(&src, &ss, b), &[GRID * GRID, 256])?;
                let memory = up(seq_object(&prompt, &ps, b), ps[2])?;
                let memory_pos = up(seq_object(&prompt_pos, &ps, b), ps[2])?;
                let got = m.tracker.mem_attn.forward(gpu, &src_b, &memory, &memory_pos, ptr_tokens)?;
                check(
                    &mut failures,
                    &format!("f{t} mem_attn o{b}"),
                    &gpu.download(&got)?,
                    &seq_object(&out, &os, b),
                    long_tol,
                );
            }
        }

        // the prompt frame decodes each object alone; tracked frames batch them
        let decoder_calls = if t == 0 { objects } else { 1 };
        for call in 0..decoder_calls {
            let k = fx.call(t, "mask_dec", call).ok_or_else(|| eyre!("f{t}: no mask_dec call {call}"))?;
            let (emb, es) = fx.get(&format!("{k}.image_embeddings"))?;
            let (hr0, h0s) = fx.get(&format!("{k}.high_res0"))?;
            let (hr1, h1s) = fx.get(&format!("{k}.high_res1"))?;
            let (ious, is) = fx.get(&format!("{k}.ious"))?;
            let (score, _) = fx.get(&format!("{k}.obj_score"))?;
            let (low, ls) = fx.get(&format!("{k}.low_res_multimasks"))?;
            let (tokens, ts) = fx.get(&format!("{k}.tokens"))?;
            ensure!(is[1] == 3, "f{t}: expected the multimask outputs, got ious {is:?}");
            for b in 0..es[0] {
                let o = if t == 0 { call } else { b };
                let img = PointImage {
                    embed: gpu.upload(to_hwc(&batch_object(&emb, &es, b), 256), &[GRID * GRID, 256])?,
                    feat_s0: gpu.upload(to_hwc(&batch_object(&hr0, &h0s, b), h0s[1]), &[LOW_RES * LOW_RES, h0s[1]])?,
                    feat_s1: gpu
                        .upload(to_hwc(&batch_object(&hr1, &h1s, b), h1s[1]), &[LOW_RES * LOW_RES / 4, h1s[1]])?,
                };
                let points: &[PointPrompt] = if t == 0 { &clicks[o] } else { &[] };
                let dec = m.point.decode(gpu, &img.embed, &img, points, (w, h), None, 1)?;
                let iou = gpu.download(&dec.iou)?;
                check(&mut failures, &format!("f{t} iou o{o}"), &iou[1..], &batch_object(&ious, &is, b), stage_tol);
                let got_score = gpu.download(&m.point.obj_score(gpu, &dec)?)?;
                check(
                    &mut failures,
                    &format!("f{t} obj_score o{o}"),
                    &got_score,
                    &batch_object(&score, &[es[0]], b),
                    stage_tol,
                );
                check(
                    &mut failures,
                    &format!("f{t} low_res o{o}"),
                    &gpu.download(&dec.low_res)?,
                    &batch_object(&low, &ls, b),
                    long_tol,
                );
                let tok = gpu.download(&dec.tokens)?;
                let got_tokens = &tok[(MASK_TOKEN_ROW + 1) * 256..][..3 * 256];
                check(
                    &mut failures,
                    &format!("f{t} mask tokens o{o}"),
                    got_tokens,
                    &batch_object(&tokens, &ts, b),
                    stage_tol,
                );
            }
        }

        for call in 0.. {
            let Some(k) = fx.call(t, "obj_ptr", call) else { break };
            obj_ptr_runs += 1;
            let (input, is) = fx.get(&format!("{k}.in"))?;
            let (out, _) = fx.get(&format!("{k}.out"))?;
            let got = m.tracker.obj_ptr_proj.forward(gpu, &gpu.upload(input, &[is[0], is[1]])?)?;
            check(&mut failures, &format!("f{t} obj_ptr {call}"), &gpu.download(&got)?, &out, stage_tol);
        }

        if let Some(k) = fx.call(t, "mem_enc", 0) {
            mem_enc_runs += 1;
            let (pix, ps) = fx.get(&format!("{k}.pix_feat"))?;
            let (mask, ms) = fx.get(&format!("{k}.mask"))?;
            let (want, fs) = fx.get(&format!("{k}.features"))?;
            // upstream adds the no-object embedding to the encoder's output in place, after
            // the hook captured it, so the fixture holds it for absent objects
            let (scores, _) = fx.get(&format!("f{t}.scores"))?;
            let no_obj = gpu.download(&m.tracker.no_obj_spatial)?;
            for (b, &score) in scores.iter().enumerate().take(ps[0]) {
                let pix_b = gpu.upload(to_hwc(&batch_object(&pix, &ps, b), 256), &[GRID * GRID, 256])?;
                let mask_b = gpu.upload(batch_object(&mask, &ms, b), &[IMAGE_SIZE * IMAGE_SIZE])?;
                let mut got = download_hwc_as_chw(gpu, &m.tracker.mem_enc.forward(gpu, &pix_b, &mask_b)?)?;
                if score <= 0.0 {
                    let hw = GRID * GRID;
                    got.iter_mut().enumerate().for_each(|(i, v)| *v += no_obj[i / hw]);
                }
                check(&mut failures, &format!("f{t} mem_enc o{b}"), &got, &batch_object(&want, &fs, b), stage_tol);
            }
        }
        checked += 1;
    }
    ensure!(checked >= 3, "only {checked} fully dumped frames");
    ensure!(
        mem_attn_runs > 0 && obj_ptr_runs > 0 && mem_enc_runs > 0,
        "stages unchecked: mem_attn {mem_attn_runs}, obj_ptr {obj_ptr_runs}, mem_enc {mem_enc_runs}"
    );
    ensure!(failures.is_empty(), "{} stage checks failed:\n{}", failures.len(), failures.join("\n"));
    Ok(())
}

/// Area of `a & b` over `a | b`, 1 when both are empty
fn iou(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "mask sizes differ");
    assert!(!a.is_empty(), "empty mask");
    let (mut inter, mut union) = (0usize, 0usize);
    for (&x, &y) in a.iter().zip(b) {
        inter += (x != 0 && y != 0) as usize;
        union += (x != 0 || y != 0) as usize;
    }
    if union == 0 { 1.0 } else { inter as f32 / union as f32 }
}

/// Mask `b` of `np.packbits(masks [objects, h, w] > 0, axis=-1)`
fn unpack(packed: &[u8], b: usize, (w, h): (usize, usize)) -> Vec<u8> {
    let row = w.div_ceil(8);
    let base = b * h * row;
    (0..h * w).map(|i| (packed[base + (i / w) * row + (i % w) / 8] >> (7 - (i % w) % 8)) & 1).collect()
}

/// The whole tracker over the clip from the reference's pixels: memory sizes must match
/// the reference's exactly, and the masks closely. In BF16 mode it also checks the
/// memory attention over cached keys against the FP32 model's on the same inputs
#[test]
#[ignore = "needs a CUDA GPU, the weights and the video fixtures"]
fn video_e2e() -> Result<()> {
    let path = fixture_path();
    let fx = Fixture::open(&path)?;
    ensure!(fx.meta["precision"] == "fp32", "the end-to-end check runs against an FP32 fixture");
    let m = load_model()?;
    let gpu = &m.gpu;
    let (w, h) = fx.size();
    let frames = fx.frames();
    let clicks = fx.clicks()?;
    let objects = clicks.len();
    let mut failures = Vec::new();
    let mut ious = vec![Vec::new(); objects];
    // per object, whether each frame's choice was a near-tie (see NEAR_TIE)
    let mut ties = vec![Vec::new(); objects];

    // FP32 holds to the FP32 reference; BF16 to PyTorch BF16's own drift from it, from the
    // clip's BF16 fixture: mean IoU at most 0.01 lower and no frame below its worst
    let limits: Vec<(f32, f32)> = if m.precision() == Precision::Fp32 {
        vec![(0.99, 0.95); objects]
    } else {
        let name = path.file_name().unwrap().to_string_lossy().replace("-fp32", "-bf16");
        let bf = Fixture::open(&path.with_file_name(name))?;
        (0..objects)
            .map(|b| {
                let v = (0..frames)
                    .map(|t| {
                        let a = unpack(&fx.bytes(&format!("f{t}.masks"))?, b, (w, h));
                        Ok(iou(&a, &unpack(&bf.bytes(&format!("f{t}.masks"))?, b, (w, h))))
                    })
                    .collect::<Result<Vec<f32>>>()?;
                let (mean, min) = (v.iter().sum::<f32>() / v.len() as f32, v.iter().copied().fold(1.0, f32::min));
                println!("PyTorch BF16 object {b}: mean IoU {mean:.4}, min {min:.4}");
                Ok((mean - 0.01, min))
            })
            .collect::<Result<_>>()?
    };
    let full_check = m.precision() == Precision::Fp32;
    let mut cached_checks = 0;
    // FP32 model for the cached-key check, loaded on first use
    let mut reference: Option<Sam3> = None;

    let (mut tracker, mut out): (VideoTracker, _) =
        m.track_pixels(&pixels(gpu, &fx, 0)?, w, h, &clicks, fx.video_frames())?;
    for t in 0..frames {
        if t > 0 {
            if full_check
                && let Some(k) = fx.call(t, "mem_attn", 0)
                && fx.has(&format!("{k}.prompt_pos"))
            {
                // positions are exact up to the sine; the memory itself carries the
                // drift of every earlier frame
                let inputs = tracker.memory_inputs()?;
                let (pos, ps) = fx.get(&format!("{k}.prompt_pos"))?;
                let (prompt, _) = fx.get(&format!("{k}.prompt"))?;
                for (b, (mem, mem_pos)) in inputs.objects.iter().enumerate() {
                    check(
                        &mut failures,
                        &format!("f{t} prompt_pos o{b}"),
                        &gpu.download(mem_pos)?,
                        &seq_object(&pos, &ps, b),
                        STAGE_REL_L2,
                    );
                    rel_l2(&format!("f{t} prompt o{b} (drift)"), &gpu.download(mem)?, &seq_object(&prompt, &ps, b));
                }
            }
            if m.precision() == Precision::Bf16 && [1, 2, 8, 16, frames - 1].contains(&t) {
                // BF16 attends from each frame's cached keys plus the slot's rotated
                // temporal key, from layer 0's shared start. The drifted memory has no
                // fixture, so FP32 memory attention on the same inputs is the truth; the
                // cached path must meet the BF16 stage limit against it, as the assembled
                // path does in video_stages (printed alongside)
                if reference.is_none() {
                    reference = Some(Sam3::load("models/sam3".as_ref(), 0, Precision::Fp32)?);
                }
                let fp32 = reference.as_ref().expect("loaded above");
                let rg = &fp32.gpu;
                let to_ref = |x: &Tensor| -> Result<Tensor> { rg.upload(gpu.download(x)?, x.shape()) };
                let feats = m.video_features(&pixels(gpu, &fx, t)?)?;
                let (sel, inputs) = (tracker.selection()?, tracker.memory_inputs()?);
                let ptr_tokens = sel.ptr_tokens();
                let start = m.tracker.mem_attn.frame_start(gpu, &feats.fpn2)?;
                let src32 = to_ref(&feats.fpn2)?;
                for (b, (mem, mem_pos)) in inputs.objects.iter().enumerate() {
                    let truth =
                        fp32.tracker.mem_attn.forward(rg, &src32, &to_ref(mem)?, &to_ref(mem_pos)?, ptr_tokens)?;
                    let truth = rg.download(&truth)?;
                    let cached = tracker.memory_attention_cached(&sel, b, &start)?;
                    let assembled = m.tracker.mem_attn.forward(gpu, &feats.fpn2, mem, mem_pos, ptr_tokens)?;
                    rel_l2(&format!("f{t} assembled o{b}"), &gpu.download(&assembled)?, &truth);
                    check(
                        &mut failures,
                        &format!("f{t} cached keys o{b}"),
                        &gpu.download(&cached)?,
                        &truth,
                        REDUCED_REL_L2,
                    );
                    cached_checks += 1;
                }
            }
            out = tracker.next_pixels(&pixels(gpu, &fx, t)?)?;
        }
        ensure!(
            out.frame == t && out.objects.len() == objects,
            "frame {t}: got frame {} with {} objects",
            out.frame,
            out.objects.len()
        );

        let meta = fx.frame(t);
        let want_tokens = meta["mem_tokens"].as_u64().unwrap_or(0) as usize;
        let want_ptr = meta["num_obj_ptr_tokens"].as_u64().unwrap_or(0) as usize;
        if (out.memory_tokens, out.pointer_tokens) != (want_tokens, want_ptr) {
            failures.push(format!(
                "f{t}: memory {} tokens ({} pointer), reference {want_tokens} ({want_ptr})",
                out.memory_tokens, out.pointer_tokens
            ));
        }

        let (scores, _) = fx.get(&format!("f{t}.scores"))?;
        let (low, _) = fx.get(&format!("f{t}.low_res"))?;
        let packed = fx.bytes(&format!("f{t}.masks"))?;
        let got_scores: Vec<f32> = out.objects.iter().map(|o| o.score).collect();
        let side = LOW_RES * LOW_RES;
        ensure!(low.len() == objects * side, "f{t}.low_res has {} values", low.len());
        // per object: rel L2 of the logits (an absent object's -1024s would swamp a joint
        // one) and low-res pixels on the other side of 0, split at |reference logit| 1
        let mut detail = Vec::new();
        for (b, o) in out.objects.iter().enumerate() {
            let want = &low[b * side..][..side];
            assert_eq!(o.low_res.len(), side, "f{t} o{b}: low-res length");
            let (err, norm) = o
                .low_res
                .iter()
                .zip(want)
                .fold((0f64, 0f64), |(e, n), (g, w)| (e + ((g - w) as f64).powi(2), n + (*w as f64).powi(2)));
            let flips: Vec<f32> =
                o.low_res.iter().zip(want).filter(|(g, w)| (**g > 0.0) != (**w > 0.0)).map(|(_, w)| *w).collect();
            let far = flips.iter().filter(|w| w.abs() >= 1.0).count();
            let ious: Vec<String> = o.ious.iter().map(|v| format!("{v:.3}")).collect();
            detail.push(format!(
                "o{b} rel {:.1e} flips {}/{far} pick {} of [{}]",
                (err / norm.max(1e-30)).sqrt(),
                flips.len(),
                o.chosen,
                ious.join(" ")
            ));
        }
        for (b, o) in out.objects.iter().enumerate() {
            // thresholding turns NaN logits into background, and the logit error above is
            // only printed, so non-finite values fail here before the masks are compared
            let ref_low = &low[b * side..][..side];
            if !o.low_res.iter().chain(&o.ious).all(|v| v.is_finite()) {
                failures.push(format!("f{t} o{b}: non-finite low-res logits or candidate IoUs"));
            }
            if !ref_low.iter().all(|v| v.is_finite()) || !scores[b].is_finite() {
                failures.push(format!("f{t} o{b}: non-finite reference logits or score"));
            }
            let want = unpack(&packed, b, (w, h));
            let v = iou(&o.mask, &want);
            ious[b].push(v);
            let mut cands = o.ious[1..].to_vec();
            cands.sort_by(|a, b| b.total_cmp(a));
            ties[b].push(o.chosen != 0 && cands.len() > 1 && cands[0] - cands[1] < NEAR_TIE);
            if !o.score.is_finite() || (o.score > 0.0) != (scores[b] > 0.0) {
                failures.push(format!("f{t} o{b}: score {:.3}, reference {:.3}", o.score, scores[b]));
            }
        }
        println!(
            "f{t:>3}: scores {got_scores:.3?} ref {scores:.3?}  iou {:.4?}  {}",
            ious.iter().map(|v| v[t]).collect::<Vec<_>>(),
            detail.join(", ")
        );
    }

    for (b, v) in ious.iter().enumerate() {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let (mean_tol, min_tol) = limits[b];
        // in the reduced modes, frames under the worst-frame limit that were near-ties
        let excused: Vec<usize> =
            (0..v.len()).filter(|&t| !full_check && v[t] < min_tol && v[t] >= NEAR_TIE_MIN_IOU && ties[b][t]).collect();
        let min = (0..v.len()).filter(|t| !excused.contains(t)).map(|t| v[t]).fold(f32::INFINITY, f32::min);
        println!("object {b}: mean IoU {mean:.4}, min {min:.4}, near-tie frames under the limit {excused:?}");
        if mean < mean_tol || min < min_tol || excused.len() > NEAR_TIE_FRAMES {
            failures.push(format!(
                "object {b}: mean IoU {mean:.4} (needs {mean_tol}), min {min:.4} (needs {min_tol}), {} near-tie frames excused (at most {NEAR_TIE_FRAMES})",
                excused.len()
            ));
        }
    }
    if m.precision() == Precision::Bf16 {
        ensure!(cached_checks > 0, "BF16 cached-key memory attention never checked");
    }
    ensure!(failures.is_empty(), "{} checks failed:\n{}", failures.len(), failures.join("\n"));
    Ok(())
}

/// A tracked frame whose top two candidate masks have predicted IoUs closer than this is
/// an argmax near-tie: reduced precision can pick the other candidate (FP32 picks 0.768
/// over 0.767 on horses frame 35, BF16 the reverse), so such a frame may fall below the
/// worst-frame limit, though not below NEAR_TIE_MIN_IOU, on at most NEAR_TIE_FRAMES
/// frames an object
const NEAR_TIE: f32 = 0.005;
const NEAR_TIE_MIN_IOU: f32 = 0.8;
const NEAR_TIE_FRAMES: usize = 2;

/// Error bound of the fused d256 attention against exact math: FP16 probabilities and
/// per-tile FP16 sums in P @ V, on inputs FP16 holds exactly. The memory cross-attention
/// kernel (64-wide values, FP16 sums of 16 terms in both products) is held to the same
/// bound
const ATTENTION_D256_REL_L2: f32 = 1e-3;

/// The fused head-dim-256 attentions against an f64 softmax, on FP16-exact inputs with
/// the memory attention's logit spread: `attention_f16_d256` (values of 256) and
/// `attention_f16_qk256_v64` (values of 64), each at 200 queries over 36,348 keys (split
/// keys, a partial last tile and a partial query block), and 5,184 queries over 100 keys
/// (one unsplit launch, a partial last query block of the 128-row kernel)
#[test]
#[ignore = "needs a CUDA GPU"]
fn attention_d256() -> Result<()> {
    const D: usize = 256;
    let gpu = Gpu::new(0, Precision::Bf16)?;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    // integers in [-1023, 1023] over a power of two, which FP16 holds exactly
    let mut values = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) % 2047) as i32 as f32 - 1023.0
            })
            .map(|v| v / scale)
            .collect()
    };
    let to_f16 = |gpu: &Gpu, x: &[f32], w: usize| -> Result<Buf> {
        let t = gpu.upload(x.to_vec(), &[x.len() / w, w])?;
        let out = gpu.alloc_bytes(x.len() * 2)?;
        // rope_rows = 0 only converts, so the angle tables are never read
        launch!(gpu, "rope_rows_f16", crate::ops::ew(x.len() / 2), crate::ops::B256, 0;
            out.slice::<u32>(), &t, &t, &t, w, 0usize)?;
        Ok(out)
    };

    for dv in [D, 64] {
        for (lq, lk) in [(200, 36_348), (5184, 100)] {
            // logits q.k / 16 with a spread of about 2.7, as peaked as the tracker's
            let (q, k, v) = (values(lq * D, 512.0), values(lk * D, 256.0), values(lk * dv, 256.0));
            let (q16, k16, v16) = (to_f16(&gpu, &q, D)?, to_f16(&gpu, &k, D)?, to_f16(&gpu, &v, dv)?);
            let out = if dv == D {
                crate::ops::attention_f16_d256(&gpu, &q16, &k16, &v16, lq, lk)?
            } else {
                crate::ops::attention_f16_qk256_v64(&gpu, &q16, &k16, &v16, lq, lk)?
            };
            ensure!(out.shape() == [lq, dv], "output shape {:?}", out.shape());
            let got = gpu.download(&out)?;

            let mut want = vec![0f32; lq * dv];
            let mut p = vec![0f64; lk];
            for i in 0..lq {
                let qi = &q[i * D..][..D];
                for (j, pj) in p.iter_mut().enumerate() {
                    *pj = qi.iter().zip(&k[j * D..][..D]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / 16.0;
                }
                let mx = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total: f64 = p
                    .iter_mut()
                    .map(|x| {
                        *x = (*x - mx).exp();
                        *x
                    })
                    .sum();
                let mut row = vec![0f64; dv];
                for (j, pj) in p.iter().enumerate() {
                    for (r, vj) in row.iter_mut().zip(&v[j * dv..][..dv]) {
                        *r += pj * *vj as f64;
                    }
                }
                for (w, r) in want[i * dv..][..dv].iter_mut().zip(row) {
                    *w = (r / total) as f32;
                }
            }
            let rel = rel_l2(&format!("dv{dv} {lq}x{lk}"), &got, &want);
            ensure!(
                rel <= ATTENTION_D256_REL_L2,
                "values of {dv}, {lq} queries over {lk} keys: rel L2 {rel:.3e} > {ATTENTION_D256_REL_L2:.0e}"
            );
        }
    }
    Ok(())
}
