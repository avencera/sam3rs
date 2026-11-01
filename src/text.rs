//! CLIP-style causal text transformer (24 layers, width 1024) and its 1024 -> 256 resizer

use eyre::Result;

use crate::gpu::{Gpu, Tensor, launch};
use crate::ops::{Act, B256, LayerNorm, Linear, Mask, ew, mha};
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
        let mut layers = Vec::new();
        for i in 0..24 {
            let l = format!("{p}.encoder.layers.{i}");
            let a = format!("{l}.self_attn");
            layers.push(Layer {
                ln1: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm1"), 1e-5)?,
                qkv: Linear::load_concat(
                    gpu,
                    wts,
                    &[format!("{a}.q_proj"), format!("{a}.k_proj"), format!("{a}.v_proj")],
                )?,
                out: Linear::load(gpu, wts, &format!("{a}.out_proj"))?,
                ln2: LayerNorm::load(gpu, wts, &format!("{l}.layer_norm2"), 1e-5)?,
                // two thirds of the encoder's weights, read once per prompt for 32 tokens;
                // BF16 attention projections cost more score accuracy for less time
                fc1: Linear::load(gpu, wts, &format!("{l}.mlp.fc1"))?.bf16(gpu)?,
                fc2: Linear::load(gpu, wts, &format!("{l}.mlp.fc2"))?.bf16(gpu)?,
            });
        }
        Ok(Self {
            tok: wts.get(gpu, &format!("{p}.embeddings.token_embedding.weight"))?,
            pos: wts.get(gpu, &format!("{p}.embeddings.position_embedding.weight"))?,
            layers,
            final_ln: LayerNorm::load(gpu, wts, &format!("{p}.final_layer_norm"), 1e-5)?,
            resizer: Linear::load(gpu, wts, "detector_model.text_projection")?,
        })
    }

    pub fn forward(&self, gpu: &Gpu, tokens: &[u32; CONTEXT]) -> Result<TextFeatures> {
        let ids = gpu.upload_buf(tokens.map(|t| t as i32).to_vec())?;
        let x = gpu.empty(&[CONTEXT, WIDTH])?;
        launch!(gpu, "text_embed", ew(x.len()), B256, 0; &x, &self.tok, &self.pos, ids.slice::<i32>())?;

        // causal attention only: padding positions come after EOT, so causal masking
        // already keeps real tokens from seeing them
        let mask = Mask { causal: true, ..Mask::default() };
        let hd = WIDTH / HEADS;
        for l in &self.layers {
            let h = l.ln1.forward(gpu, &x)?;
            let qkv = l.qkv.forward(gpu, &h)?;
            let ld = 3 * WIDTH;
            let o = mha(
                gpu,
                (qkv.ptr(), ld),
                (qkv.at(WIDTH), ld),
                (qkv.at(2 * WIDTH), ld),
                CONTEXT,
                CONTEXT,
                HEADS,
                hd,
                mask,
            )?;
            l.out.forward_residual(gpu, &o, &x)?;
            let h = l.ln2.forward(gpu, &x)?;
            let hidden = l.fc1.forward_act(gpu, &h, Act::Gelu)?;
            l.fc2.forward_residual(gpu, &hidden, &x)?;
        }
        self.final_ln.forward_inplace(gpu, &x)?;
        let feats = self.resizer.forward(gpu, &x)?;
        Ok(TextFeatures { feats, pad: tokens.map(|t| (t == 0) as u8) })
    }
}
