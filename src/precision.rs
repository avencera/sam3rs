//! The numeric mode a model runs in

use eyre::{Result, eyre};

/// Numeric mode for the whole pipeline
///
/// `Fp32` is the parity reference on every backend. The reduced modes spend lower
/// precision where the backend's GPU gains from it, so the same mode runs different
/// stages in BF16 on different backends; each variant says what it runs on CUDA and on
/// Metal (Apple silicon)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Strict FP32, no TF32 or BF16 anywhere: the parity reference path
    ///
    /// On CUDA, cuBLAS GEMMs run in pedantic math and cuBLASLt linears with FP32 compute,
    /// with bias and activations fused into their epilogues. The vision backbone's
    /// attention runs a fused flash-attention kernel on FP32 FMAs with an online softmax,
    /// so it never writes the score matrix; the detector's attention stays materialized.
    /// On Metal, every stage runs in FP32
    Fp32,
    /// The FP32 path with TF32 tensor-core GEMMs (FP32 accumulation), the way ONNX
    /// Runtime's CUDA provider runs FP32 models by default. CUDA only: loading a model in
    /// this mode on Metal fails
    ///
    /// Weights and activations stay FP32, with bias and activations fused into cuBLASLt
    /// epilogues. Attention without a causal mask or additive bias, the detector's
    /// box-biased cross-attention included, runs a fused flash-attention kernel on TF32
    /// tensor cores with an FP32 softmax; the rest stays materialized
    ///
    /// The first call of each GEMM shape times cuBLASLt's top candidate algorithms and
    /// switches from its first choice only to one more than 5% faster. A candidate near
    /// that margin can win in one run and lose in the next, so outputs can differ
    /// slightly between runs (about 0.00001 in score)
    Tf32,
    /// BF16 weights and activations in the stages where the backend gains from them, with
    /// FP32 accumulation
    ///
    /// On CUDA, GEMMs run on tensor cores. The vision backbone, necks, text encoder MLP
    /// and fusion encoder FFN keep BF16 weights and activations; the rest of the text
    /// encoder and fusion encoder, the detector and point decoder keep FP32 ones, and
    /// cuBLAS may round their GEMM inputs to BF16 (`CUBLAS_COMPUTE_32F_FAST_16BF`), though
    /// it picks TF32 kernels for most of them. Unmasked attention with head dim 32 or 64,
    /// and the detector's box-biased cross-attention, run the fused kernel. It computes
    /// scores from BF16 q and k as they are, or rounds FP32 q and k to FP16 and sums
    /// their scores in FP16; either way it rounds v to FP16 and sums each 64-key P @ V
    /// tile in FP16 before adding it to FP32 totals
    ///
    /// On Metal, only the vision backbone and both necks run in BF16 (FP32 accumulation
    /// and residual stream), and every later stage runs in FP32, since Apple GPUs run FP32
    /// matrix math within about 20% of BF16
    Bf16,
}

impl Precision {
    const ALL: [Self; 3] = [Self::Fp32, Self::Tf32, Self::Bf16];

    /// The mode's name, which `parse` reads back: `"fp32"`, `"tf32"` or `"bf16"`
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Tf32 => "tf32",
            Self::Bf16 => "bf16",
        }
    }
}

/// Writes [`Precision::as_str`]
impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(self.as_str())
    }
}

impl std::str::FromStr for Precision {
    type Err = eyre::Report;

    fn from_str(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.as_str() == s)
            .ok_or_else(|| eyre!("unknown precision {s:?}, expected fp32, tf32 or bf16"))
    }
}
