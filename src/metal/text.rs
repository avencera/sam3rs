//! CLIP-style causal text transformer (24 layers, width 1024) and its 1024 -> 256 resizer,
//! FP32 in both precisions

use eyre::{Result, ensure};

use crate::metal::gpu::{DType, Gpu, Tensor};
use crate::metal::ops::{Act, LayerNorm, Linear, Mask, View, mha};
use crate::tokenizer::CONTEXT;
use crate::weights::Weights;

const WIDTH: usize = 1024;
const HEADS: usize = 16;

struct Layer {
    ln1: LayerNorm,
    qkv: Linear,
    out: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// The text encoder of the detector
pub struct TextEncoder {
    tok: Tensor,
    pos: Tensor,
    layers: Vec<Layer>,
    final_ln: LayerNorm,
    resizer: Linear,
}

/// Encoded text prompt: features `[32, 256]` and the padding mask (1 = padding)
pub struct TextFeatures {
    pub feats: Tensor,
    pub pad: [u8; CONTEXT],
}

impl TextEncoder {
    pub fn load(gpu: &Gpu, wts: &Weights) -> Result<Self> {
        let p = "detector_model.text_encoder.text_model";
        let linear = |name: &str| Linear::load(gpu, wts, name, DType::F32);
        let mut layers = Vec::new();
        for i in 0..24 {
            let l = format!("{p}.encoder.layers.{i}");
            let a = format!("{l}.self_attn");
            let qkv = [format!("{a}.q_proj"), format!("{a}.k_proj"), format!("{a}.v_proj")];
            layers.push(Layer {
                ln1: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm1"), 1e-5)?,
                qkv: Linear::load_concat(gpu, wts, &qkv, DType::F32)?,
                out: linear(&format!("{a}.out_proj"))?,
                ln2: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm2"), 1e-5)?,
                fc1: linear(&format!("{l}.mlp.fc1"))?,
                fc2: linear(&format!("{l}.mlp.fc2"))?,
            });
        }
        let pos = wts.get(gpu, &format!("{p}.embeddings.position_embedding.weight"))?;
        ensure!(pos.rows() >= CONTEXT && pos.cols() == WIDTH, "text position table {:?}", pos.shape());
        Ok(Self {
            tok: wts.get(gpu, &format!("{p}.embeddings.token_embedding.weight"))?,
            pos,
            layers,
            final_ln: LayerNorm::load(gpu, wts, &format!("{p}.final_layer_norm"), 1e-5)?,
            resizer: linear("detector_model.text_projection")?,
        })
    }

    pub fn forward(&self, gpu: &Gpu, tokens: &[u32; CONTEXT]) -> Result<TextFeatures> {
        let vocab = self.tok.rows();
        ensure!(tokens.iter().all(|&t| (t as usize) < vocab), "token ids {tokens:?} past a vocabulary of {vocab}");
        let ids = gpu.upload(&tokens.map(|t| t as i32), &[CONTEXT], DType::I32)?;
        let x = gpu.empty(&[CONTEXT, WIDTH], DType::F32)?;
        gpu.dispatch_1d("text_embed", x.len(), &[x.arg(), self.tok.arg(), self.pos.arg(), ids.arg(), WIDTH.into()])?;

        // causal attention only: padding positions come after EOT, so causal masking
        // already keeps real tokens from seeing them
        let mask = Mask { causal: true, ..Mask::default() };
        let hd = WIDTH / HEADS;
        for l in &self.layers {
            let h = l.ln1.forward_to(gpu, &x, DType::F32)?;
            let qkv = l.qkv.forward(gpu, &h, Act::None)?;
            let (q, k, v) = (View::heads(&qkv, 0, hd), View::heads(&qkv, WIDTH, hd), View::heads(&qkv, 2 * WIDTH, hd));
            let o = mha(gpu, q, k, v, (CONTEXT, CONTEXT), HEADS, mask)?;
            l.out.forward_residual(gpu, &o, &x)?;
            let h = l.ln2.forward_to(gpu, &x, DType::F32)?;
            let hidden = l.fc1.forward(gpu, &h, Act::Gelu)?;
            l.fc2.forward_residual(gpu, &hidden, &x)?;
        }
        self.final_ln.forward_inplace(gpu, &x)?;
        let feats = self.resizer.forward(gpu, &x, Act::None)?;
        Ok(TextFeatures { feats, pad: tokens.map(|t| u8::from(t == 0)) })
    }
}
