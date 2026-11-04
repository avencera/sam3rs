//! Kernels against f64 CPU references on sizes the model doesn't exercise in the stage
//! test: ragged tiles, strided views and every attention mask. Needs a Metal 4 GPU
//!
//!     cargo test --release --lib metal::kernel_tests -- --ignored

use eyre::Result;

use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::ops::{Act, BoxRpb, Linear, Mask, View, attention};
use crate::precision::Precision;

/// Deterministic values in [-1, 1)
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Fails on a length mismatch, any NaN, or an error past `tol` relative to the largest
/// reference value
fn assert_close(name: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(1e-30f64, |m, w| m.max(w.abs()));
    let mut worst = 0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{name}: non-finite {g} at {i}");
        worst = worst.max((*g as f64 - w).abs() / scale);
    }
    assert!(worst <= tol, "{name}: worst error {worst:.3e} of the largest value, over {tol:.0e}");
}

fn gpu() -> Gpu {
    Gpu::new(0, Precision::Fp32).expect("a Metal 4 GPU")
}

/// Reference attention of one batch entry, rows of q/k/v/out read through views
#[allow(clippy::too_many_arguments)]
fn attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (lq, lk, hd, dv): (usize, usize, usize, usize),
    qv: (usize, usize),
    kv: (usize, usize),
    vv: (usize, usize),
    bias: impl Fn(usize, usize) -> Option<f64>,
) -> Vec<f64> {
    let mut out = vec![0f64; lq * dv];
    for i in 0..lq {
        let scores: Vec<Option<f64>> = (0..lk)
            .map(|j| {
                let dot: f64 = (0..hd).map(|c| q[qv.0 + i * qv.1 + c] as f64 * k[kv.0 + j * kv.1 + c] as f64).sum();
                bias(i, j).map(|b| dot / (hd as f64).sqrt() + b)
            })
            .collect();
        let max = scores.iter().flatten().fold(f64::NEG_INFINITY, |m, &s| m.max(s));
        if max == f64::NEG_INFINITY {
            continue;
        }
        let p: Vec<f64> = scores.iter().map(|s| s.map_or(0.0, |s| (s - max).exp())).collect();
        let total: f64 = p.iter().sum();
        for c in 0..dv {
            out[i * dv + c] = (0..lk).map(|j| p[j] * v[vv.0 + j * vv.1 + c] as f64).sum::<f64>() / total;
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum Masking {
    None,
    KeyPad,
    Causal,
    Rpb,
}

/// q, k and v are column blocks of one fused `[rows, 3 * heads * hd]` projection, heads
/// as the batch, like the model's attention inputs
fn check_attention(gpu: &Gpu, (hd, dv): (usize, usize), masking: Masking) -> Result<()> {
    let heads = 3;
    let side = 7;
    let (lq, lk) = match masking {
        // long enough that few query blocks split their keys (see `kv_splits`), with a
        // causal row's later chunks fully masked
        Masking::Causal => (600, 600),
        Masking::Rpb => (19, side * side),
        // unmasked runs the tensor-op kernels: several threadgroups and a ragged last key
        // tile, with keys enough that head dim 256 over 64-wide values splits them in two
        Masking::None => (150, 2300),
        Masking::KeyPad => (37, 1000),
    };
    let width = heads * (2 * hd + dv);
    let rows = lq.max(lk);
    let fused = noise(rows * width, (hd * 31 + dv) as u64);
    // scaled up so the softmax is peaked and a wrong key shows
    let fused: Vec<f32> = fused.iter().map(|x| x * 2.0).collect();
    let t = gpu.upload_f32(&fused, &[rows, width])?;
    let pad: Vec<u8> = (0..lk).map(|j| u8::from(j % 3 == 1)).collect();
    let pad_t = gpu.upload(&pad, &[lk], DType::U8)?;
    let dx = noise(heads * (lq - 1) * side, 7);
    let dy = noise(heads * (lq - 1) * side, 8);
    let (dx_t, dy_t) = (gpu.upload_f32(&dx, &[dx.len()])?, gpu.upload_f32(&dy, &[dy.len()])?);
    let mask = match masking {
        Masking::None => Mask::default(),
        Masking::KeyPad => Mask { key_pad: Some(&pad_t), ..Mask::default() },
        Masking::Causal => Mask { causal: true, ..Mask::default() },
        Masking::Rpb => Mask { rpb: Some(BoxRpb { dx: &dx_t, dy: &dy_t, side }), ..Mask::default() },
    };
    let out = gpu.empty(&[lq, heads * dv], DType::F32)?;
    let (kc, vc) = (heads * hd, 2 * heads * hd);
    let view = |col: usize, stride: usize| View { t: &t, offset: col, ld: width, stride };
    let ov = View { t: &out, offset: 0, ld: heads * dv, stride: dv };
    attention(gpu, view(0, hd), view(kc, hd), view(vc, dv), ov, heads, lq, lk, (hd, dv), mask)?;
    let got = gpu.download(&out)?;

    let mut want = vec![0f64; lq * heads * dv];
    for h in 0..heads {
        let bias = |i: usize, j: usize| match masking {
            Masking::KeyPad if pad[j] != 0 => None,
            Masking::Causal if j > i => None,
            Masking::Rpb if i > 0 => {
                let base = (h * (lq - 1) + i - 1) * side;
                Some((dy[base + j / side] + dx[base + j % side]) as f64)
            }
            _ => Some(0.0),
        };
        let r = attention_ref(
            &fused,
            &fused,
            &fused,
            (lq, lk, hd, dv),
            (h * hd, width),
            (kc + h * hd, width),
            (vc + h * dv, width),
            bias,
        );
        for i in 0..lq {
            want[i * heads * dv + h * dv..][..dv].copy_from_slice(&r[i * dv..][..dv]);
        }
    }
    assert_close(&format!("attention d{hd} v{dv} {masking:?}"), &got, &want, 2e-5);
    Ok(())
}

#[test]
#[ignore = "needs a Metal 4 GPU"]
fn attention_matches_reference() -> Result<()> {
    let gpu = gpu();
    for dims in [(16, 16), (32, 32), (64, 64), (256, 256), (256, 64)] {
        for masking in [Masking::None, Masking::KeyPad, Masking::Causal, Masking::Rpb] {
            check_attention(&gpu, dims, masking)?;
        }
    }
    Ok(())
}

/// Ragged M, N and K, with the bias, an activation and the accumulating epilogue, in
/// FP32 and BF16: one tile grid on the plain GEMM (K too short to split), split-K into a
/// full and a ragged chunk, and a 16-way split as the text encoder's 32-token linears take
#[test]
#[ignore = "needs a Metal 4 GPU"]
fn linear_matches_reference() -> Result<()> {
    let gpu = gpu();
    for (m, n, k) in [(37, 70, 300), (37, 70, 588), (32, 64, 4096)] {
        check_linear(&gpu, m, n, k)?;
    }
    Ok(())
}

fn check_linear(gpu: &Gpu, m: usize, n: usize, k: usize) -> Result<()> {
    let x = noise(m * k, 1);
    let w = noise(n * k, 2);
    let b = noise(n, 3);
    let y0 = noise(m * n, 4);
    let round = |v: &[f32], dtype: DType| -> Vec<f32> {
        match dtype {
            DType::Bf16 => v.iter().map(|&x| half::bf16::from_f32(x).to_f32()).collect(),
            _ => v.to_vec(),
        }
    };
    for dtype in [DType::F32, DType::Bf16] {
        let (xr, wr) = (round(&x, dtype), round(&w, dtype));
        let dot = |i: usize, j: usize| (0..k).map(|c| xr[i * k + c] as f64 * wr[j * k + c] as f64).sum::<f64>();
        let lin = Linear::new(gpu.upload_as(&w, &[n, k], dtype)?, Some(gpu.upload_f32(&b, &[n])?));
        let xt = gpu.upload_as(&x, &[m, k], dtype)?;

        let got = gpu.download(&lin.forward_to(gpu, &xt, Act::Relu, DType::F32)?)?;
        let want: Vec<f64> = (0..m * n).map(|i| (dot(i / n, i % n) + b[i % n] as f64).max(0.0)).collect();
        assert_close(&format!("linear relu {dtype:?} m{m} n{n} k{k}"), &got, &want, 1e-5);

        if dtype == DType::F32 {
            let y: Tensor = gpu.upload_f32(&y0, &[m, n])?;
            lin.forward_residual(gpu, &xt, &y)?;
            let want: Vec<f64> = (0..m * n).map(|i| y0[i] as f64 + dot(i / n, i % n) + b[i % n] as f64).collect();
            assert_close(&format!("linear residual m{m} n{n} k{k}"), &gpu.download(&y)?, &want, 1e-5);
        }
    }
    Ok(())
}

/// Attention rates on the model's long shapes, batch by batch (a timing probe, not a check)
#[test]
#[ignore = "needs a Metal 4 GPU; timing only"]
fn attention_rates() -> Result<()> {
    let gpu = gpu();
    let time = |hd: usize, dv: usize, batch: usize, lq: usize, lk: usize| -> Result<()> {
        let q = gpu.upload_f32(&noise(batch * lq * hd, 1), &[batch * lq, hd])?;
        let k = gpu.upload_f32(&noise(batch * lk * hd, 2), &[batch * lk, hd])?;
        let v = gpu.upload_f32(&noise(batch * lk * dv, 3), &[batch * lk, dv])?;
        let out = gpu.empty(&[batch * lq, dv], DType::F32)?;
        fn view(t: &Tensor, rows: usize, w: usize) -> View<'_> {
            View { t, offset: 0, ld: w, stride: rows * w }
        }
        let mut times = Vec::new();
        for i in 0..13 {
            let t0 = std::time::Instant::now();
            attention(
                &gpu,
                view(&q, lq, hd),
                view(&k, lk, hd),
                view(&v, lk, dv),
                view(&out, lq, dv),
                batch,
                lq,
                lk,
                (hd, dv),
                Mask::default(),
            )?;
            gpu.synchronize()?;
            if i >= 3 {
                times.push(t0.elapsed().as_secs_f64());
            }
        }
        times.sort_by(f64::total_cmp);
        let t = times[times.len() / 2];
        let flops = 2.0 * (batch * lq * lk * (hd + dv)) as f64;
        println!(
            "d{hd} v{dv} batch {batch} lq {lq} lk {lk}: {:.2} ms, {:.2} ms per entry, {:.2} TFLOPS",
            t * 1e3,
            t * 1e3 / batch as f64,
            flops / t / 1e12
        );
        Ok(())
    };
    if std::env::var_os("SAM3RS_PROBE_QUICK").is_some() {
        time(256, 256, 1, 5184, 5184)?;
        time(256, 64, 1, 5184, 7 * 5184 + 64)?;
        // the same work over 4x the threadgroups, as a 4-way key split would give
        return Ok(());
    }
    for batch in [1, 2, 3, 4, 8] {
        time(256, 256, batch, 5184, 5184)?;
    }
    for batch in [1, 3] {
        time(256, 64, batch, 5184, 7 * 5184 + 64)?;
        time(256, 64, batch, 5184, 5184 + 16)?;
    }
    time(32, 32, 8, 5184, 5184)?;
    time(32, 32, 1, 200, 5184)?;
    Ok(())
}
