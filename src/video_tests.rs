//! Compares the video tracker with the PyTorch fixtures `scripts/video_reference.py dump`
//! writes: each stage on the reference's own inputs at the fully dumped frames, then the
//! whole tracker over the clip from the reference's pixels. One suite for every backend,
//! through its [`TestHooks`]
//!
//! Both run against an FP32 fixture. In FP32 mode the end-to-end check holds the masks to
//! it; in the other modes it allows PyTorch BF16's own drift, read from the clip's BF16
//! fixture next to it. Which stages run in reduced precision is the backend's: its
//! [`TestHooks::tolerances`] give each stage's limit
//!
//!     SAM3RS_VIDEO=models/video-parity/horses-720p-64-o3-fp32.safetensors \
//!         cargo test --release --lib video_ -- --ignored --nocapture --test-threads 1

use std::time::Instant;

use eyre::{Result, ensure, eyre};

use crate::backend::Backend;
use crate::model::point::{LOW_RES, MASK_TOKEN_ROW};
use crate::model::vision::{GRID, IMAGE_SIZE};
use crate::precision::Precision;
use crate::prompt::PointPrompt;
use crate::sam3::Sam3;
use crate::test_support::{
    Fixture, NEAR_TIE, NEAR_TIE_FRAMES, NEAR_TIE_MIN_IOU, STAGE_REL_L2, TestHooks, batch_object, check, iou,
    load_model, rel_l2, seq_object, unpack, video_fixture_path,
};
use crate::tracker::VideoTracker;

type Pixels = <crate::backend::Active as Backend>::Pixels;

/// Frame `t`'s preprocessed pixels from the fixture
fn pixels(m: &Sam3, fx: &Fixture, t: usize) -> Result<Pixels> {
    let (v, shape) = fx.get(&format!("f{t}.input"))?;
    ensure!(shape == [3, IMAGE_SIZE, IMAGE_SIZE], "f{t}.input is {shape:?}");
    m.backend.upload_pixels(v)
}

/// `[c, hw]` -> `[hw, c]`
fn to_hwc(v: &[f32], c: usize) -> Vec<f32> {
    let hw = v.len() / c;
    (0..hw).flat_map(|p| (0..c).map(move |ch| v[ch * hw + p])).collect()
}

/// Each tracker stage on the reference's inputs at every fully dumped frame
#[test]
#[ignore = "needs a GPU, the weights and the video fixtures"]
fn video_stages() -> Result<()> {
    let fx = Fixture::open(&video_fixture_path())?;
    ensure!(fx.meta["precision"] == "fp32", "stage checks need an FP32 fixture");
    let m = load_model()?;
    let b = &m.backend;
    let (w, h) = fx.size();
    let clicks = fx.clicks()?;
    let objects = clicks.len();
    let mut failures = Vec::new();
    let mut checked = 0;
    // frames (or calls) each optional stage ran on, so a fixture missing a hook fails
    let (mut mem_attn_runs, mut obj_ptr_runs, mut mem_enc_runs) = (0, 0, 0);
    let tol = b.tolerances();

    for t in 0..fx.frames() {
        let Some(bb) = fx.call(t, "backbone", 0) else { continue };
        if !fx.has(&format!("{bb}.fpn2")) {
            continue;
        }
        println!("frame {t}");
        let feats = b.video_frame(&pixels(&m, &fx, t)?)?;
        let fpn = |l: usize| fx.values(&format!("{bb}.fpn{l}"));
        let [coarse, s0, s1] = b.frame_levels(&feats);
        check(&mut failures, &format!("f{t} fpn2"), &b.chw(coarse)?, &fpn(2)?, tol.backbone);
        check(&mut failures, &format!("f{t} feat_s0"), &b.chw(s0)?, &fpn(0)?, tol.backbone);
        check(&mut failures, &format!("f{t} feat_s1"), &b.chw(s1)?, &fpn(1)?, tol.backbone);

        if let Some(k) = fx.call(t, "mem_attn", 0) {
            mem_attn_runs += 1;
            let ptr_tokens = fx.frame(t)["num_obj_ptr_tokens"].as_u64().unwrap() as usize;
            let (src, ss) = fx.get(&format!("{k}.src"))?;
            let src_pos = fx.values(&format!("{k}.src_pos"))?;
            let (prompt, ps) = fx.get(&format!("{k}.prompt"))?;
            let prompt_pos = fx.values(&format!("{k}.prompt_pos"))?;
            let (out, os) = fx.get(&format!("{k}.out"))?;
            let ours_pos: Vec<f32> = b.memory_src_pos()?.iter().map(|v| v * 10.0).collect();
            check(&mut failures, &format!("f{t} src_pos"), &ours_pos, &seq_object(&src_pos, &ss, 0), tol.stage);
            for o in 0..ss[1] {
                let up = |v: Vec<f32>| b.upload(v, &[ps[0], ps[2]]);
                let src_o = b.upload(seq_object(&src, &ss, o), &[GRID * GRID, 256])?;
                let memory = up(seq_object(&prompt, &ps, o))?;
                let memory_pos = up(seq_object(&prompt_pos, &ps, o))?;
                b.sync()?;
                let t0 = Instant::now();
                let got = b.memory_attention(&src_o, &memory, &memory_pos, ptr_tokens)?;
                b.sync()?;
                println!("f{t} mem_attn o{o} over {} keys: {:?}", ps[0], t0.elapsed());
                check(
                    &mut failures,
                    &format!("f{t} mem_attn o{o}"),
                    &b.download(&got)?,
                    &seq_object(&out, &os, o),
                    tol.long,
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
            let score = fx.values(&format!("{k}.obj_score"))?;
            let (low, ls) = fx.get(&format!("{k}.low_res_multimasks"))?;
            let (tokens, ts) = fx.get(&format!("{k}.tokens"))?;
            ensure!(is[1] == 3, "f{t}: expected the multimask outputs, got ious {is:?}");
            for i in 0..es[0] {
                let o = if t == 0 { call } else { i };
                let img = b.point_image_from(
                    b.upload(to_hwc(&batch_object(&emb, &es, i), 256), &[GRID * GRID, 256])?,
                    b.upload(to_hwc(&batch_object(&hr0, &h0s, i), h0s[1]), &[LOW_RES * LOW_RES, h0s[1]])?,
                    b.upload(to_hwc(&batch_object(&hr1, &h1s, i), h1s[1]), &[LOW_RES * LOW_RES / 4, h1s[1]])?,
                );
                let points: &[PointPrompt] = if t == 0 { &clicks[o] } else { &[] };
                let dec = b.decoded_outputs(&b.point_decode(&img, points, (w, h), None)?)?;
                check(&mut failures, &format!("f{t} iou o{o}"), &dec.iou[1..], &batch_object(&ious, &is, i), tol.stage);
                check(
                    &mut failures,
                    &format!("f{t} obj_score o{o}"),
                    &[dec.score],
                    &batch_object(&score, &[es[0]], i),
                    tol.stage,
                );
                check(
                    &mut failures,
                    &format!("f{t} low_res o{o}"),
                    &dec.low_res,
                    &batch_object(&low, &ls, i),
                    tol.long,
                );
                let got_tokens = &dec.tokens[(MASK_TOKEN_ROW + 1) * 256..][..3 * 256];
                check(
                    &mut failures,
                    &format!("f{t} mask tokens o{o}"),
                    got_tokens,
                    &batch_object(&tokens, &ts, i),
                    tol.stage,
                );
            }
        }

        for call in 0.. {
            let Some(k) = fx.call(t, "obj_ptr", call) else { break };
            obj_ptr_runs += 1;
            let (input, is) = fx.get(&format!("{k}.in"))?;
            let out = fx.values(&format!("{k}.out"))?;
            let got = b.object_pointer(&b.upload(input, &[is[0], is[1]])?)?;
            check(&mut failures, &format!("f{t} obj_ptr {call}"), &b.download(&got)?, &out, tol.stage);
        }

        if let Some(k) = fx.call(t, "mem_enc", 0) {
            mem_enc_runs += 1;
            let (pix, ps) = fx.get(&format!("{k}.pix_feat"))?;
            let (mask, ms) = fx.get(&format!("{k}.mask"))?;
            let (want, fs) = fx.get(&format!("{k}.features"))?;
            // upstream adds the no-object embedding to the encoder's output in place, after
            // the hook captured it, so the fixture holds it for absent objects
            let scores = fx.values(&format!("f{t}.scores"))?;
            let no_obj = b.no_obj_spatial()?;
            for (o, &score) in scores.iter().enumerate().take(ps[0]) {
                let pix_o = b.upload(to_hwc(&batch_object(&pix, &ps, o), 256), &[GRID * GRID, 256])?;
                let mask_o = b.upload(batch_object(&mask, &ms, o), &[IMAGE_SIZE * IMAGE_SIZE])?;
                let mut got = b.chw(&b.memory_encoder(&pix_o, &mask_o)?)?;
                if score <= 0.0 {
                    let hw = GRID * GRID;
                    got.iter_mut().enumerate().for_each(|(i, v)| *v += no_obj[i / hw]);
                }
                check(&mut failures, &format!("f{t} mem_enc o{o}"), &got, &batch_object(&want, &fs, o), tol.stage);
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

/// The whole tracker over the clip from the reference's pixels: memory sizes must match
/// the reference's exactly, and the masks closely. The backend adds its own checks
/// through [`TestHooks::e2e_check`], such as CUDA's memory attention over cached keys in
/// BF16 mode
#[test]
#[ignore = "needs a GPU, the weights and the video fixtures"]
fn video_e2e() -> Result<()> {
    let path = video_fixture_path();
    let fx = Fixture::open(&path)?;
    ensure!(fx.meta["precision"] == "fp32", "the end-to-end check runs against an FP32 fixture");
    let m = load_model()?;
    let b = &m.backend;
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
            .map(|o| {
                let v = (0..frames)
                    .map(|t| {
                        let a = unpack(&fx.bytes(&format!("f{t}.masks"))?, o, (w, h));
                        Ok(iou(&a, &unpack(&bf.bytes(&format!("f{t}.masks"))?, o, (w, h))))
                    })
                    .collect::<Result<Vec<f32>>>()?;
                let (mean, min) = (v.iter().sum::<f32>() / v.len() as f32, v.iter().copied().fold(1.0, f32::min));
                println!("PyTorch BF16 object {o}: mean IoU {mean:.4}, min {min:.4}");
                Ok((mean - 0.01, min))
            })
            .collect::<Result<_>>()?
    };
    let full_check = m.precision() == Precision::Fp32;
    let mut own_checks = Default::default();

    let (mut tracker, mut out): (VideoTracker, _) =
        m.track_pixels(&pixels(&m, &fx, 0)?, w, h, &clicks, fx.video_frames())?;
    for t in 0..frames {
        if t > 0 {
            if full_check
                && let Some(k) = fx.call(t, "mem_attn", 0)
                && fx.has(&format!("{k}.prompt_pos"))
            {
                // positions are exact up to the sine; the memory itself carries the
                // drift of every earlier frame
                let inputs = b.assembled_memory(&tracker.selection())?;
                let (pos, ps) = fx.get(&format!("{k}.prompt_pos"))?;
                let prompt = fx.values(&format!("{k}.prompt"))?;
                for (o, (mem, mem_pos)) in inputs.iter().enumerate() {
                    check(
                        &mut failures,
                        &format!("f{t} prompt_pos o{o}"),
                        &b.download(mem_pos)?,
                        &seq_object(&pos, &ps, o),
                        STAGE_REL_L2,
                    );
                    rel_l2(&format!("f{t} prompt o{o} (drift)"), &b.download(mem)?, &seq_object(&prompt, &ps, o));
                }
            }
            let px = pixels(&m, &fx, t)?;
            b.e2e_check(&mut own_checks, &tracker, &px, t, frames, &mut failures)?;
            b.sync()?;
            let t0 = Instant::now();
            out = tracker.next_pixels(&px)?;
            println!("f{t} tracked in {:?}", t0.elapsed());
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

        let scores = fx.values(&format!("f{t}.scores"))?;
        let low = fx.values(&format!("f{t}.low_res"))?;
        let packed = fx.bytes(&format!("f{t}.masks"))?;
        let got_scores: Vec<f32> = out.objects.iter().map(|o| o.score).collect();
        let side = LOW_RES * LOW_RES;
        ensure!(low.len() == objects * side, "f{t}.low_res has {} values", low.len());
        // per object: rel L2 of the logits (an absent object's -1024s would swamp a joint
        // one) and low-res pixels on the other side of 0, split at |reference logit| 1
        let mut detail = Vec::new();
        for (i, o) in out.objects.iter().enumerate() {
            let want = &low[i * side..][..side];
            assert_eq!(o.low_res.len(), side, "f{t} o{i}: low-res length");
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
                "o{i} rel {:.1e} flips {}/{far} pick {} of [{}]",
                (err / norm.max(1e-30)).sqrt(),
                flips.len(),
                o.chosen,
                ious.join(" ")
            ));
        }
        for (i, o) in out.objects.iter().enumerate() {
            // thresholding turns NaN logits into background, and the logit error above is
            // only printed, so non-finite values fail here before the masks are compared
            let ref_low = &low[i * side..][..side];
            if !o.low_res.iter().chain(&o.ious).all(|v| v.is_finite()) {
                failures.push(format!("f{t} o{i}: non-finite low-res logits or candidate IoUs"));
            }
            if !ref_low.iter().all(|v| v.is_finite()) || !scores[i].is_finite() {
                failures.push(format!("f{t} o{i}: non-finite reference logits or score"));
            }
            let want = unpack(&packed, i, (w, h));
            let v = iou(&o.mask, &want);
            ious[i].push(v);
            let mut cands = o.ious[1..].to_vec();
            cands.sort_by(|a, b| b.total_cmp(a));
            ties[i].push(o.chosen != 0 && cands.len() > 1 && cands[0] - cands[1] < NEAR_TIE);
            if !o.score.is_finite() || (o.score > 0.0) != (scores[i] > 0.0) {
                failures.push(format!("f{t} o{i}: score {:.3}, reference {:.3}", o.score, scores[i]));
            }
        }
        println!(
            "f{t:>3}: scores {got_scores:.3?} ref {scores:.3?}  iou {:.4?}  {}",
            ious.iter().map(|v| v[t]).collect::<Vec<_>>(),
            detail.join(", ")
        );
    }

    for (o, v) in ious.iter().enumerate() {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let (mean_tol, min_tol) = limits[o];
        // in the reduced modes, frames under the worst-frame limit that were near-ties
        let excused: Vec<usize> =
            (0..v.len()).filter(|&t| !full_check && v[t] < min_tol && v[t] >= NEAR_TIE_MIN_IOU && ties[o][t]).collect();
        let min = (0..v.len()).filter(|t| !excused.contains(t)).map(|t| v[t]).fold(f32::INFINITY, f32::min);
        println!("object {o}: mean IoU {mean:.4}, min {min:.4}, near-tie frames under the limit {excused:?}");
        if mean < mean_tol || min < min_tol || excused.len() > NEAR_TIE_FRAMES {
            failures.push(format!(
                "object {o}: mean IoU {mean:.4} (needs {mean_tol}), min {min:.4} (needs {min_tol}), {} near-tie frames excused (at most {NEAR_TIE_FRAMES})",
                excused.len()
            ));
        }
    }
    b.e2e_finish(own_checks)?;
    ensure!(failures.is_empty(), "{} checks failed:\n{}", failures.len(), failures.join("\n"));
    Ok(())
}
