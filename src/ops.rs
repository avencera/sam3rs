//! Layers shared by every stage: linears, norms, attention and convolutions, all on
//! channels-last `[rows, features]` tensors

use eyre::Result;

use crate::gpu::{Gpu, Mat, Tensor, blocks};
use crate::launch;
use crate::weights::Weights;

pub const ACT_NONE: i32 = 0;
pub const ACT_RELU: i32 = 1;
pub const ACT_GELU: i32 = 2;
pub const ACT_GELU_TANH: i32 = 3;
pub const ACT_SIGMOID: i32 = 4;

/// Grid for grid-stride elementwise kernels
pub fn ew(n: usize) -> (u32, u32, u32) {
    (blocks(n, 256).min(65535 * 8), 1, 1)
}

pub const B256: (u32, u32, u32) = (256, 1, 1);

pub fn bias_act(gpu: &Gpu, x: &Tensor, bias: Option<&Tensor>, act: i32) -> Result<()> {
    let b = bias.map_or(0, Tensor::ptr);
    launch!(gpu, "bias_act", ew(x.len()), B256, 0; x, b, x.rows() as u64, x.cols(), act)
}

pub fn add(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    assert_eq!(a.len(), b.len());
    let out = gpu.empty(a.shape())?;
    launch!(gpu, "add", ew(a.len()), B256, 0; &out, a, b, a.len() as u64)?;
    Ok(out)
}

/// `a += b` in place
pub fn add_assign(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.len(), b.len());
    launch!(gpu, "add", ew(a.len()), B256, 0; a, a, b, a.len() as u64)
}

/// A dense layer `y = x @ W^T + b` with W stored `[out, in]` like PyTorch
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let w = wts.get(gpu, &format!("{prefix}.weight"))?;
        let w = match w.shape().len() {
            2 => w,
            // 1x1 conv weights [O, C, 1, 1]
            _ => {
                let (o, c) = (w.shape()[0], w.len() / w.shape()[0]);
                w.reshape(&[o, c])
            }
        };
        let bias = format!("{prefix}.bias");
        let b = if wts.contains(&bias) { Some(wts.get(gpu, &bias)?) } else { None };
        Ok(Self { w, b })
    }

    /// Fuses separate projections that read the same input, such as q/k/v
    pub fn load_concat(gpu: &Gpu, wts: &Weights, prefixes: &[String]) -> Result<Self> {
        let names = |s: &str| prefixes.iter().map(|p| format!("{p}.{s}")).collect::<Vec<_>>();
        let w = wts.concat(gpu, &names("weight"))?;
        let b = Some(wts.concat(gpu, &names("bias"))?);
        Ok(Self { w, b })
    }

    pub fn out_features(&self) -> usize {
        self.w.shape()[0]
    }

    pub fn in_features(&self) -> usize {
        self.w.shape()[1]
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        self.forward_act(gpu, x, ACT_NONE)
    }

    pub fn forward_act(&self, gpu: &Gpu, x: &Tensor, act: i32) -> Result<Tensor> {
        assert_eq!(x.cols(), self.in_features(), "linear input {:?} vs weight {:?}", x.shape(), self.w.shape());
        let rows = x.rows();
        let out = gpu.empty(&[rows, self.out_features()])?;
        self.forward_raw(gpu, x.ptr(), x.cols(), rows, out.ptr(), 0.0)?;
        if self.b.is_some() || act != ACT_NONE {
            bias_act(gpu, &out, self.b.as_ref(), act)?;
        }
        Ok(out)
    }

    /// `residual += x @ W^T + b`
    pub fn forward_residual(&self, gpu: &Gpu, x: &Tensor, residual: &Tensor) -> Result<()> {
        assert_eq!(residual.cols(), self.out_features());
        assert_eq!(x.rows(), residual.rows());
        self.forward_raw(gpu, x.ptr(), x.cols(), x.rows(), residual.ptr(), 1.0)?;
        if self.b.is_some() {
            bias_act(gpu, residual, self.b.as_ref(), ACT_NONE)?;
        }
        Ok(())
    }

    /// GEMM only, no bias: `out = x @ W^T + beta * out` over raw strided rows
    pub fn forward_raw(&self, gpu: &Gpu, x: u64, x_ld: usize, rows: usize, out: u64, beta: f32) -> Result<()> {
        let (n, k) = (self.out_features(), self.in_features());
        gpu.gemm(rows, n, k, Mat::new(x, x_ld), Mat::new(self.w.ptr(), k).t(), Mat::new(out, n), 1, 1.0, beta)
    }
}

/// A stack of linears with an activation between them (not after the last)
pub struct Mlp {
    pub layers: Vec<Linear>,
    pub act: i32,
}

impl Mlp {
    pub fn load(gpu: &Gpu, wts: &Weights, prefixes: &[String], act: i32) -> Result<Self> {
        let layers = prefixes.iter().map(|p| Linear::load(gpu, wts, p)).collect::<Result<_>>()?;
        Ok(Self { layers, act })
    }

    /// `prefix.layer1 .. prefix.layerN`
    pub fn load_numbered(gpu: &Gpu, wts: &Weights, prefix: &str, n: usize, act: i32) -> Result<Self> {
        let prefixes: Vec<String> = (1..=n).map(|i| format!("{prefix}.layer{i}")).collect();
        Self::load(gpu, wts, &prefixes, act)
    }

    /// `prefix.layers.0 .. prefix.layers.{N-1}`
    pub fn load_indexed(gpu: &Gpu, wts: &Weights, prefix: &str, n: usize, act: i32) -> Result<Self> {
        let prefixes: Vec<String> = (0..n).map(|i| format!("{prefix}.layers.{i}")).collect();
        Self::load(gpu, wts, &prefixes, act)
    }

    /// `prefix.proj_in`, `prefix.layers.*`, `prefix.proj_out`
    pub fn load_in_out(gpu: &Gpu, wts: &Weights, prefix: &str, n: usize, act: i32) -> Result<Self> {
        let mut prefixes = vec![format!("{prefix}.proj_in")];
        prefixes.extend((0..n - 2).map(|i| format!("{prefix}.layers.{i}")));
        prefixes.push(format!("{prefix}.proj_out"));
        Self::load(gpu, wts, &prefixes, act)
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        let last = self.layers.len() - 1;
        let mut h = self.layers[0].forward_act(gpu, x, if last == 0 { ACT_NONE } else { self.act })?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            h = layer.forward_act(gpu, &h, if i == last { ACT_NONE } else { self.act })?;
        }
        Ok(h)
    }
}

pub struct LayerNorm {
    pub w: Tensor,
    pub b: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self { w: wts.get(gpu, &format!("{prefix}.weight"))?, b: wts.get(gpu, &format!("{prefix}.bias"))?, eps })
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        let out = gpu.empty(x.shape())?;
        self.forward_into(gpu, x, &out)?;
        Ok(out)
    }

    pub fn forward_inplace(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        self.forward_into(gpu, x, x)
    }

    fn forward_into(&self, gpu: &Gpu, x: &Tensor, out: &Tensor) -> Result<()> {
        assert_eq!(x.cols(), self.w.len());
        launch!(gpu, "layernorm", (x.rows() as u32, 1, 1), B256, 0; out, x, &self.w, &self.b, x.cols(), self.eps)
    }
}

/// Softmax masking options for [`attention`]
#[derive(Clone, Copy, Default)]
pub struct Mask {
    /// additive bias `[batch / bias_div, lq, lk]`
    pub bias: u64,
    pub bias_div: usize,
    /// u8 per key, nonzero means the key is padding
    pub key_pad: u64,
    pub causal: bool,
}

/// Scaled dot-product attention over `batch` independent (q, k, v) slices described by
/// strided row-major views. Writes `out` rows of `hd` values per batch entry
#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    q: Mat,
    k: Mat,
    v: Mat,
    out: Mat,
    batch: usize,
    lq: usize,
    lk: usize,
    hd: usize,
    mask: Mask,
) -> Result<()> {
    let scores = gpu.empty(&[batch, lq, lk])?;
    let scale = 1.0 / (hd as f32).sqrt();
    let s = Mat::new(scores.ptr(), lk).stride(lq * lk);
    gpu.gemm(lq, lk, hd, q, k.t(), s, batch, scale, 0.0)?;
    launch!(gpu, "softmax", ((batch * lq) as u32, 1, 1), B256, 0;
        &scores, lk, lq, mask.bias, mask.bias_div.max(1), mask.key_pad, mask.causal as i32)?;
    gpu.gemm(lq, hd, lk, s, v, out, batch, 1.0, 0.0)
}

/// Multi-head attention where q, k, v are `[tokens, heads * hd]` row-major (possibly
/// columns of a wider fused projection, hence the explicit leading dims)
#[allow(clippy::too_many_arguments)]
pub fn mha(
    gpu: &Gpu,
    q: (u64, usize),
    k: (u64, usize),
    v: (u64, usize),
    lq: usize,
    lk: usize,
    heads: usize,
    hd: usize,
    mask: Mask,
) -> Result<Tensor> {
    let out = gpu.empty(&[lq, heads * hd])?;
    attention(
        gpu,
        Mat::new(q.0, q.1).stride(hd),
        Mat::new(k.0, k.1).stride(hd),
        Mat::new(v.0, v.1).stride(hd),
        Mat::new(out.ptr(), heads * hd).stride(hd),
        heads,
        lq,
        lk,
        hd,
        Mask { bias_div: mask.bias_div.max(1), ..mask },
    )?;
    Ok(out)
}

/// 3x3 conv, stride 1, padding 1, as nine shifted GEMMs over a padded input
pub struct Conv3x3 {
    /// `[9, O, C]`
    pub w: Tensor,
    pub b: Tensor,
}

impl Conv3x3 {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let (w, shape) = wts.host(&format!("{prefix}.weight"))?;
        let (o, c) = (shape[0], shape[1]);
        assert_eq!(&shape[2..], &[3, 3]);
        let mut taps = vec![0f32; w.len()];
        for oc in 0..o {
            for ic in 0..c {
                for t in 0..9 {
                    taps[(t * o + oc) * c + ic] = w[(oc * c + ic) * 9 + t];
                }
            }
        }
        Ok(Self { w: gpu.upload(&taps, &[9, o, c])?, b: wts.get(gpu, &format!("{prefix}.bias"))? })
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (o, c) = (self.w.shape()[1], self.w.shape()[2]);
        assert_eq!(x.shape(), &[h * w, c]);
        let wp = w + 2;
        let xp = gpu.zeros(&[(h + 2) * wp + 2, c])?;
        launch!(gpu, "pad1", ew(x.len()), B256, 0; &xp, x, h, w, c)?;
        let yp = gpu.empty(&[h * wp, o])?;
        for t in 0..9 {
            let (dy, dx) = (t / 3, t % 3);
            gpu.gemm(
                h * wp,
                o,
                c,
                Mat::new(xp.at((dy * wp + dx) * c), c),
                Mat::new(self.w.at(t * o * c), c).t(),
                Mat::new(yp.ptr(), o),
                1,
                1.0,
                if t == 0 { 0.0 } else { 1.0 },
            )?;
        }
        let y = gpu.empty(&[h * w, o])?;
        launch!(gpu, "crop_bias", ew(y.len()), B256, 0; &y, &yp, &self.b, h, w, o)?;
        Ok(y)
    }
}

/// ConvTranspose2d with kernel 2 and stride 2: a GEMM then a pixel-shuffle scatter
pub struct ConvT2x2 {
    /// `[C, O * 4]`, PyTorch's `[C, O, 2, 2]` as is
    pub w: Tensor,
    pub b: Tensor,
}

impl ConvT2x2 {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let w = wts.get(gpu, &format!("{prefix}.weight"))?;
        let c = w.shape()[0];
        let n = w.len() / c;
        Ok(Self { w: w.reshape(&[c, n]), b: wts.get(gpu, &format!("{prefix}.bias"))? })
    }

    pub fn out_channels(&self) -> usize {
        self.w.shape()[1] / 4
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (c, n) = (self.w.shape()[0], self.w.shape()[1]);
        assert_eq!(x.shape(), &[h * w, c]);
        let g = gpu.empty(&[h * w, n])?;
        gpu.gemm(h * w, n, c, Mat::new(x.ptr(), c), Mat::new(self.w.ptr(), n), Mat::new(g.ptr(), n), 1, 1.0, 0.0)?;
        let o = n / 4;
        let y = gpu.empty(&[4 * h * w, o])?;
        launch!(gpu, "convt2x2_scatter", ew(y.len()), B256, 0; &y, &g, &self.b, h, w, o)?;
        Ok(y)
    }
}

/// `[rows, cols]` -> `[cols, rows]`
pub fn transpose(gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
    let out = gpu.empty(&[x.cols(), x.rows()])?;
    launch!(gpu, "transpose", ew(x.len()), B256, 0; &out, x, x.rows() as u64, x.cols())?;
    Ok(out)
}

/// Multi-head attention with separate q/k/v/o projections (`prefix.{q,k,v,o}_proj`). The
/// projections may narrow to an internal width, as in SAM's downsampled attention
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub heads: usize,
}

impl Attention {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str, heads: usize) -> Result<Self> {
        let l = |n: &str| Linear::load(gpu, wts, &format!("{prefix}.{n}"));
        Ok(Self { q: l("q_proj")?, k: l("k_proj")?, v: l("v_proj")?, o: l("o_proj")?, heads })
    }

    /// `o_proj(attn(q_in Wq, k_in Wk, v_in Wv))`
    pub fn forward(&self, gpu: &Gpu, q_in: &Tensor, k_in: &Tensor, v_in: &Tensor, mask: Mask) -> Result<Tensor> {
        let attn = self.attend(gpu, q_in, k_in, v_in, mask)?;
        self.o.forward(gpu, &attn)
    }

    /// `residual += o_proj(attn(...))`
    pub fn forward_residual(
        &self,
        gpu: &Gpu,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        mask: Mask,
        residual: &Tensor,
    ) -> Result<()> {
        let attn = self.attend(gpu, q_in, k_in, v_in, mask)?;
        self.o.forward_residual(gpu, &attn, residual)
    }

    fn attend(&self, gpu: &Gpu, q_in: &Tensor, k_in: &Tensor, v_in: &Tensor, mask: Mask) -> Result<Tensor> {
        let q = self.q.forward(gpu, q_in)?;
        let k = self.k.forward(gpu, k_in)?;
        let v = self.v.forward(gpu, v_in)?;
        let dim = q.cols();
        mha(gpu, (q.ptr(), dim), (k.ptr(), dim), (v.ptr(), dim), q.rows(), k.rows(), self.heads, dim / self.heads, mask)
    }
}

/// GroupNorm over channels-last `[hw, c]` in place, with a fused activation
pub fn group_norm(gpu: &Gpu, x: &Tensor, groups: usize, w: &Tensor, b: &Tensor, eps: f32, act: i32) -> Result<()> {
    let (hw, c) = (x.rows(), x.cols());
    let stats = gpu.zeros(&[groups, 2])?;
    let chunk = 256usize;
    launch!(gpu, "group_stats", (hw.div_ceil(chunk) as u32, groups as u32, 1), (32, 1, 1), 0;
        &stats, x, hw as u64, c, groups, chunk)?;
    launch!(gpu, "group_norm_apply", ew(x.len()), B256, 0; x, &stats, w, b, hw as u64, c, groups, eps, act)
}

/// `[n, d]` rows of `a` followed by rows of `b`
pub fn concat_rows(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    assert_eq!(a.cols(), b.cols());
    let out = gpu.empty(&[a.rows() + b.rows(), a.cols()])?;
    gpu.copy_dtod(a.ptr(), out.ptr(), a.len() * 4)?;
    gpu.copy_dtod(b.ptr(), out.at(a.len()), b.len() * 4)?;
    Ok(out)
}

pub fn upload_i32(gpu: &Gpu, v: &[i32]) -> Result<crate::gpu::Buf> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.upload_bytes(&bytes)
}
