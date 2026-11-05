//! CUDA kernels against exact host math

use eyre::{Result, ensure};

use crate::cuda::gpu::{Buf, Gpu, launch};
use crate::cuda::ops::{B256, attention_f16_d256, attention_f16_qk256_v64, ew};
use crate::precision::Precision;
use crate::test_support::rel_l2;

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
        launch!(gpu, "rope_rows_f16", ew(x.len() / 2), B256, 0;
            out.slice::<u32>(), &t, &t, &t, w, 0usize)?;
        Ok(out)
    };

    for dv in [D, 64] {
        for (lq, lk) in [(200, 36_348), (5184, 100)] {
            // logits q.k / 16 with a spread of about 2.7, as peaked as the tracker's
            let (q, k, v) = (values(lq * D, 512.0), values(lk * D, 256.0), values(lk * dv, 256.0));
            let (q16, k16, v16) = (to_f16(&gpu, &q, D)?, to_f16(&gpu, &k, D)?, to_f16(&gpu, &v, dv)?);
            let out = if dv == D {
                attention_f16_d256(&gpu, &q16, &k16, &v16, lq, lk)?
            } else {
                attention_f16_qk256_v64(&gpu, &q16, &k16, &v16, lq, lk)?
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
