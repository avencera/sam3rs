//! Layers shared by every stage: linears, norms, attention and convolutions, all on
//! channels-last `[rows, features]` tensors

use eyre::{Result, bail, eyre};

use crate::blaslt::Epilogue;
use crate::gpu::{Arg, Buf, Gpu, Mat, Precision, Tensor, blocks, launch};
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

/// A row-major BF16 activation matrix
pub struct Bf16 {
    pub buf: Buf,
    pub rows: usize,
    pub cols: usize,
}

impl Bf16 {
    pub fn empty(gpu: &Gpu, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self { buf: gpu.alloc_bytes(rows * cols * 2)?, rows, cols })
    }

    pub fn from_f32(gpu: &Gpu, x: &Tensor) -> Result<Self> {
        Ok(Self { buf: to_bf16(gpu, x.ptr(), x.cols(), x.rows(), x.cols())?, rows: x.rows(), cols: x.cols() })
    }
}

/// A dense layer `y = x @ W^T + b` with W stored `[out, in]` like PyTorch
pub struct Linear {
    /// FP32 `[out, in]`, dropped once [`Self::bf16`] replaces it
    w: Option<Tensor>,
    /// `[out, in]`
    shape: [usize; 2],
    pub b: Option<Tensor>,
    /// BF16 copies of `w` and `b` in `Precision::Bf16`, for the cuBLASLt path; a BF16
    /// output needs a BF16 bias
    w16: Option<Buf>,
    b16: Option<Buf>,
}

impl Linear {
    /// A linear over existing device weights, run in FP32 whatever the precision
    pub fn new(w: Tensor, b: Option<Tensor>) -> Self {
        let shape = [w.rows(), w.cols()];
        Self { w: Some(w), shape, b, w16: None, b16: None }
    }

    /// In BF16 mode, keeps only a BF16 copy of the weights and runs this linear through
    /// BF16 cuBLASLt. The vision backbone and the fusion encoder's FFN opt in for their
    /// FLOPs and the text encoder's MLP for its weight reads (32 tokens over 800 MB of
    /// FP32 weights); the other detector layers keep FP32 activations and weights
    pub fn bf16(mut self, gpu: &Gpu) -> Result<Self> {
        if gpu.precision() == Precision::Bf16
            && let Some(w) = self.w.take()
        {
            self.w16 = Some(to_bf16(gpu, w.ptr(), w.cols(), w.rows(), w.cols())?);
            if let Some(b) = &self.b {
                self.b16 = Some(to_bf16(gpu, b.ptr(), b.len(), 1, b.len())?);
            }
        }
        Ok(self)
    }

    /// FP32 weights; a linear converted by [`Self::bf16`] has none
    fn w(&self) -> &Tensor {
        self.w.as_ref().expect("FP32 path of a linear that keeps only BF16 weights")
    }

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
        Ok(Self::new(w, b))
    }

    /// Fuses separate projections that read the same input, such as q/k/v
    pub fn load_concat(gpu: &Gpu, wts: &Weights, prefixes: &[String]) -> Result<Self> {
        let names = |s: &str| prefixes.iter().map(|p| format!("{p}.{s}")).collect::<Vec<_>>();
        let w = wts.concat(gpu, &names("weight"))?;
        let b = Some(wts.concat(gpu, &names("bias"))?);
        Ok(Self::new(w, b))
    }

    pub fn out_features(&self) -> usize {
        self.shape[0]
    }

    pub fn in_features(&self) -> usize {
        self.shape[1]
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        self.forward_act(gpu, x, ACT_NONE)
    }

    pub fn forward_act(&self, gpu: &Gpu, x: &Tensor, act: i32) -> Result<Tensor> {
        assert_eq!(x.cols(), self.in_features(), "linear input {:?} vs weight {:?}", x.shape(), self.shape);
        let rows = x.rows();
        let out = gpu.empty(&[rows, self.out_features()])?;
        if self.w16.is_some() {
            if !self.forward_bf16(gpu, x, out.ptr(), 0.0, act)? {
                bias_act(gpu, &out, None, act)?;
            }
            return Ok(out);
        }
        let (epilogue, fused) = self.epilogue(act);
        let bias = self.b.as_ref().map_or(0, Tensor::ptr);
        let mnk = (rows, self.out_features(), self.in_features());
        if gpu.linear_f32(mnk, x.ptr(), x.cols(), self.w().ptr(), out.ptr(), 0.0, bias, epilogue)? {
            if !fused {
                bias_act(gpu, &out, None, act)?;
            }
            return Ok(out);
        }
        self.forward_raw(gpu, x.ptr(), x.cols(), rows, out.ptr(), 0.0)?;
        if self.b.is_some() || act != ACT_NONE {
            bias_act(gpu, &out, self.b.as_ref(), act)?;
        }
        Ok(out)
    }

    /// The cuBLASLt epilogue for this layer's bias and `act`, and whether it applies
    /// `act`; if not, only the bias is fused
    fn epilogue(&self, act: i32) -> (Epilogue, bool) {
        let has_bias = self.b.is_some();
        match act {
            ACT_NONE => (if has_bias { Epilogue::Bias } else { Epilogue::None }, true),
            ACT_RELU => (if has_bias { Epilogue::ReluBias } else { Epilogue::Relu }, true),
            ACT_GELU_TANH => (if has_bias { Epilogue::GeluBias } else { Epilogue::Gelu }, true),
            _ => (if has_bias { Epilogue::Bias } else { Epilogue::None }, false),
        }
    }

    /// `residual += x @ W^T + b`
    pub fn forward_residual(&self, gpu: &Gpu, x: &Tensor, residual: &Tensor) -> Result<()> {
        assert_eq!(residual.cols(), self.out_features());
        assert_eq!(x.rows(), residual.rows());
        if self.w16.is_some() {
            self.forward_bf16(gpu, x, residual.ptr(), 1.0, ACT_NONE)?;
            return Ok(());
        }
        let (epilogue, _) = self.epilogue(ACT_NONE);
        let bias = self.b.as_ref().map_or(0, Tensor::ptr);
        let mnk = (x.rows(), self.out_features(), self.in_features());
        if gpu.linear_f32(mnk, x.ptr(), x.cols(), self.w().ptr(), residual.ptr(), 1.0, bias, epilogue)? {
            return Ok(());
        }
        self.forward_raw(gpu, x.ptr(), x.cols(), x.rows(), residual.ptr(), 1.0)?;
        if self.b.is_some() {
            bias_act(gpu, residual, self.b.as_ref(), ACT_NONE)?;
        }
        Ok(())
    }

    /// FP32 input: converts to BF16, then [`Self::forward_lt`]
    fn forward_bf16(&self, gpu: &Gpu, x: &Tensor, out: u64, beta: f32, act: i32) -> Result<bool> {
        let x16 = Bf16::from_f32(gpu, x)?;
        self.forward_lt(gpu, &x16, (out, false), beta, act)
    }

    /// BF16 `act(x16 @ W^T + b)`, for a following BF16 linear; `act` must be one the
    /// epilogue fuses
    pub fn forward16_act16(&self, gpu: &Gpu, x16: &Bf16, act: i32) -> Result<Bf16> {
        let out = Bf16::empty(gpu, x16.rows, self.out_features())?;
        if !self.forward_lt(gpu, x16, (out.buf.ptr(), true), 0.0, act)? {
            bail!("activation {act} has no cuBLASLt epilogue");
        }
        Ok(out)
    }

    /// `residual += x16 @ W^T + b`
    pub fn forward16_residual(&self, gpu: &Gpu, x16: &Bf16, residual: &Tensor) -> Result<()> {
        assert_eq!((residual.rows(), residual.cols()), (x16.rows, self.out_features()));
        self.forward_lt(gpu, x16, (residual.ptr(), false), 1.0, ACT_NONE)?;
        Ok(())
    }

    /// `out = act(x16 @ W^T + b) + beta * out` through cuBLASLt. Returns whether the
    /// epilogue applied `act`; if not, only the bias was added
    fn forward_lt(&self, gpu: &Gpu, x16: &Bf16, out: (u64, bool), beta: f32, act: i32) -> Result<bool> {
        let w16 = self.w16.as_ref().ok_or_else(|| eyre!("linear has no BF16 weights"))?;
        assert_eq!(x16.cols, self.in_features());
        let (m, n, k) = (x16.rows, self.out_features(), self.in_features());
        let bias = if out.1 { self.b16.as_ref().map_or(0, Buf::ptr) } else { self.b.as_ref().map_or(0, Tensor::ptr) };
        let (epilogue, fused) = self.epilogue(act);
        gpu.linear_bf16((m, n, k), x16.buf.ptr(), k, w16.ptr(), out, beta, bias, epilogue)?;
        Ok(fused)
    }

    /// GEMM only, no bias: `out = x @ W^T + beta * out` over raw strided rows
    pub fn forward_raw(&self, gpu: &Gpu, x: u64, x_ld: usize, rows: usize, out: u64, beta: f32) -> Result<()> {
        let (n, k) = (self.out_features(), self.in_features());
        gpu.gemm(rows, n, k, Mat::new(x, x_ld), Mat::new(self.w().ptr(), k).t(), Mat::new(out, n), 1, 1.0, beta)
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

    /// Normalized rows in BF16, as the input of a BF16 linear
    pub fn forward16(&self, gpu: &Gpu, x: &Tensor) -> Result<Bf16> {
        assert_eq!(x.cols(), self.w.len());
        let out = Bf16::empty(gpu, x.rows(), x.cols())?;
        launch!(gpu, "layernorm_bf16", (x.rows() as u32, 1, 1), B256, 0;
            out.buf.ptr(), x, &self.w, &self.b, x.cols(), self.eps)?;
        Ok(out)
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
    pub rpb: Option<BoxRpb>,
}

/// Key splits for the fused box-RPB attention
const RPB_SPLITS: usize = 16;

/// Other head-dim-32 attentions split their keys when they would launch fewer blocks than
/// this, aiming for about `SPLIT_BLOCKS` blocks. Below `SPLIT_MIN_KEYS` keys the extra
/// combine launch costs more than the idle SMs
const SPLIT_UNDER_BLOCKS: usize = 64;
const SPLIT_BLOCKS: usize = 256;
const SPLIT_MIN_KEYS: usize = 1024;

/// FP32 head-dim-32 attentions with at least this many queries and keys round K and V to
/// 16 bits once up front and run in 128-row blocks: each block re-reads all of K and V,
/// which made the fusion encoder's self-attention L2-bound. The BF16 encoder's global
/// attention measured slower in 128-row blocks
const TALL_MIN: usize = 512;

/// Box relative position bias in factored form: for head `h`, query row `1 + q` and key
/// `y * side + x`, the bias is `dy[h, q, y] + dx[h, q, x]`, and row 0 gets none. The
/// fused kernel adds it per score instead of reading a `[heads, lq, side^2]` tensor
#[derive(Clone, Copy)]
pub struct BoxRpb {
    /// `[heads, lq - 1, side]`, head-major so each query row's biases are contiguous
    pub dx: u64,
    /// `[heads, lq - 1, side]`
    pub dy: u64,
    /// even, so the fused kernel reads each thread's key pair from one grid row
    pub side: usize,
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
    if flash_ok(gpu, &[q, k, v, out], hd, &mask) {
        let kernel = match (q.bf16, hd, mask.rpb) {
            (true, _, _) => "flash_attn_bf16_d64",
            (false, 32, Some(_)) => "flash_attn_rpb_d32",
            (false, 32, None) => "flash_attn_d32",
            _ => "flash_attn_d64",
        };
        let scale_log2 = std::f32::consts::LOG2_E / (hd as f32).sqrt();
        let view = |m: Mat| [Arg::from(m.ptr), Arg::from(m.ld as u64), Arg::from(m.stride as u64)];
        if !q.bf16 && hd == 32 && mask.rpb.is_none() && lq >= TALL_MIN && lk >= TALL_MIN {
            // K as BF16 and V as FP16 [lk, batch * hd], exactly as the kernel would round
            // them while staging
            let kd = gpu.alloc_bytes(lk * batch * hd * 2)?;
            let vd = gpu.alloc_bytes(lk * batch * hd * 2)?;
            launch!(gpu, "kv_to_16", ew(lk * batch * hd / 8), B256, 0;
                kd.ptr(), vd.ptr(), k.ptr, k.ld as u64, k.stride as u64, v.ptr, v.ld as u64, v.stride as u64,
                lk, batch, hd)?;
            let packed = |ptr: u64| Mat::new(ptr, batch * hd).stride(hd);
            let tail = [Arg::from(lq as i32), Arg::from(lk as i32), Arg::from(scale_log2), Arg::from(mask.key_pad)];
            let args = [out, q, packed(kd.ptr()), packed(vd.ptr())].into_iter().flat_map(view).chain(tail).collect();
            let grid = (lq.div_ceil(128) as u32, batch as u32, 1);
            return gpu.launch("flash_attn_kv16_d32", grid, (256, 1, 1), 0, args);
        }

        let grid = (lq.div_ceil(64) as u32, batch as u32, 1);
        let mut args = Vec::with_capacity(16);
        for m in [out, q, k, v] {
            args.extend(view(m));
        }
        args.extend([Arg::from(lq as i32), Arg::from(lk as i32), Arg::from(scale_log2)]);

        // the decoder has 201 queries over 5184 keys and the geometry encoder a few: one
        // block per 64 queries and head would leave most SMs idle, so split the keys and
        // merge the partial softmaxes
        let tiles = (grid.0 * grid.1) as usize;
        let splits = match mask.rpb {
            Some(_) => RPB_SPLITS,
            None if hd == 32 && tiles < SPLIT_UNDER_BLOCKS && lk >= SPLIT_MIN_KEYS => SPLIT_BLOCKS.div_ceil(tiles),
            None => 1,
        };
        let chunk = lk.div_ceil(splits).next_multiple_of(64);
        let splits = lk.div_ceil(chunk);
        let part = if splits > 1 { Some(gpu.empty(&[splits, batch, lq, hd + 2])?) } else { None };
        match mask.rpb {
            Some(r) => {
                assert!(r.side.is_multiple_of(2), "the fused box RPB needs an even side");
                args.extend([Arg::from(r.dx), Arg::from(r.dy), Arg::from(r.side)]);
            }
            None => args.push(Arg::from(mask.key_pad)),
        }
        args.extend([Arg::from(part.as_ref().map_or(0, Tensor::ptr)), Arg::from(chunk)]);
        gpu.launch(kernel, (grid.0, grid.1, splits as u32), (128, 1, 1), 0, args)?;
        let Some(part) = part else { return Ok(()) };
        let warps = batch * lq;
        return launch!(gpu, "flash_combine_d32", (blocks(warps * 32, 256), 1, 1), B256, 0;
            out.ptr, out.ld as u64, out.stride as u64, &part, splits, batch, lq);
    }

    let mut mask = mask;
    let _bias = match mask.rpb {
        Some(r) => {
            assert!(mask.bias == 0 && lk == r.side * r.side, "box RPB covers a side x side key grid");
            let bias = gpu.empty(&[batch, lq, lk])?;
            launch!(gpu, "rpb_combine", ew(bias.len()), B256, 0;
                &bias, r.dx, r.dy, lq - 1, r.side, batch)?;
            mask.bias = bias.ptr();
            mask.bias_div = 1;
            Some(bias)
        }
        None => None,
    };
    let scores = gpu.empty(&[batch, lq, lk])?;
    let scale = 1.0 / (hd as f32).sqrt();
    let s = Mat::new(scores.ptr(), lk).stride(lq * lk);
    gpu.gemm(lq, lk, hd, q, k.t(), s, batch, scale, 0.0)?;
    launch!(gpu, "softmax", ((batch * lq) as u32, 1, 1), B256, 0;
        &scores, lk, lq, mask.bias, mask.bias_div.max(1), mask.key_pad, mask.causal as i32)?;
    gpu.gemm(lq, hd, lk, s, v, out, batch, 1.0, 0.0)
}

/// The fused kernel covers BF16 mode for FP32 q/k/v with head dims 32 and 64, which it
/// rounds to BF16 (q, k) and FP16 (v) for `mma.sync`, and BF16 q/k/v with head dim 64,
/// without additive bias or causal masking, except the box RPB, which it fuses for FP32
/// q/k/v with head dim 32 and no key padding. It loads q/k/v in 16-byte chunks and stores
/// float2. BF16 q/k/v must use it, since the materialized path reads FP32
fn flash_ok(gpu: &Gpu, [q, k, v, out]: &[Mat; 4], hd: usize, mask: &Mask) -> bool {
    let aligned = |m: &Mat, elems: usize, size: usize| {
        !m.trans
            && m.ptr.is_multiple_of((elems * size) as u64)
            && m.ld.is_multiple_of(elems)
            && m.stride.is_multiple_of(elems)
    };
    let bf16 = q.bf16;
    assert!(k.bf16 == bf16 && v.bf16 == bf16 && !out.bf16, "q/k/v share a type, out is FP32");
    let inputs = if bf16 { (8, 2) } else { (4, 4) };
    let ok = gpu.precision() == Precision::Bf16
        && (hd == 64 || (hd == 32 && !bf16))
        && (mask.rpb.is_none() || (hd == 32 && !bf16 && mask.key_pad == 0))
        && mask.bias == 0
        && !mask.causal
        && aligned(out, 2, 4)
        && [q, k, v].iter().all(|m| aligned(m, inputs.0, inputs.1));
    assert!(ok || !bf16, "BF16 attention inputs need the fused kernel");
    ok
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

/// Output pixels per im2col chunk in BF16 mode: 16Ki rows of 9 x 256 BF16 channels is a
/// 72 MiB buffer, where the whole 288x288 map at once would hold 382 MiB
const IM2COL_ROWS: usize = 16 << 10;

/// 3x3 conv, stride 1, padding 1: nine shifted GEMMs over a padded input in FP32 modes,
/// and in BF16 mode one cuBLASLt GEMM with K = 9C over BF16 im2col chunks, with the bias
/// in its epilogue
pub struct Conv3x3 {
    /// `[9, O, C]`
    pub w: Tensor,
    pub b: Tensor,
    /// BF16 `[O, 9 * C]` (tap-major within each row) in BF16 mode
    w16: Option<Buf>,
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
        let w16 = match gpu.precision() {
            Precision::Fp32 | Precision::Tf32 => None,
            Precision::Bf16 => {
                let mut rows = vec![0f32; w.len()];
                for oc in 0..o {
                    for t in 0..9 {
                        for ic in 0..c {
                            rows[(oc * 9 + t) * c + ic] = w[(oc * c + ic) * 9 + t];
                        }
                    }
                }
                let rows = gpu.upload(rows, &[o, 9 * c])?;
                Some(to_bf16(gpu, rows.ptr(), rows.len(), 1, rows.len())?)
            }
        };
        let w = gpu.upload(taps, &[9, o, c])?;
        Ok(Self { w, b: wts.get(gpu, &format!("{prefix}.bias"))?, w16 })
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (o, c) = (self.w.shape()[1], self.w.shape()[2]);
        assert_eq!(x.shape(), &[h * w, c]);
        if let Some(w16) = &self.w16 {
            return self.forward_im2col(gpu, x, w16, h, w);
        }

        let wp = w + 2;
        let rows = (h + 2) * wp + 2;
        let xp = gpu.alloc_bytes(rows * c * 4)?;
        gpu.memset_zero(xp.ptr(), xp.bytes())?;
        launch!(gpu, "pad1", ew(x.len()), B256, 0; xp.ptr(), x, h, w, c)?;
        let yp = gpu.empty(&[h * wp, o])?;
        for t in 0..9 {
            let (dy, dx) = (t / 3, t % 3);
            let input = Mat::new(xp.ptr() + ((dy * wp + dx) * c * 4) as u64, c);
            let weight = Mat::new(self.w.at(t * o * c), c).t();
            gpu.gemm(h * wp, o, c, input, weight, Mat::new(yp.ptr(), o), 1, 1.0, if t == 0 { 0.0 } else { 1.0 })?;
        }
        let y = gpu.empty(&[h * w, o])?;
        launch!(gpu, "crop_bias", ew(y.len()), B256, 0; &y, &yp, &self.b, h, w, o)?;
        Ok(y)
    }

    /// The conv of `fine + up2(coarse)` over a fine map of `h` x `w` pixels, where `up2` is
    /// nearest 2x upsampling and `norm`, if any, is still pending on `coarse`. BF16 mode
    /// computes that input inside the im2col; the other modes apply `norm` to `coarse` in
    /// place and materialize the sum
    pub fn forward_up(
        &self,
        gpu: &Gpu,
        fine: &Tensor,
        coarse: &Tensor,
        norm: Option<&PendingNorm>,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let c = self.w.shape()[2];
        assert_eq!(fine.shape(), &[h * w, c]);
        assert_eq!(coarse.shape(), &[h * w / 4, c]);
        let Some(w16) = &self.w16 else {
            if let Some(norm) = norm {
                norm.apply(gpu, coarse)?;
            }
            let up = gpu.empty(&[h * w, c])?;
            launch!(gpu, "upsample2_add", ew(up.len()), B256, 0; &up, fine, coarse, h / 2, w / 2, c)?;
            return self.forward(gpu, &up, h, w);
        };

        assert!(norm.is_none_or(|n| (c / n.groups).is_multiple_of(8)), "im2col normalizes 8 channels of one group");
        let (stats, gw, gb) = norm.map_or((0, 0, 0), |n| (n.stats.ptr(), n.w.ptr(), n.b.ptr()));
        let (groups, eps, act) = norm.map_or((1, 0.0, ACT_NONE), |n| (n.groups, n.eps, n.act));
        self.im2col_gemm(gpu, w16, h, w, |cols, p0, n| {
            launch!(gpu, "im2col3x3_up_bf16", ew(n * 9 * c / 8), B256, 0;
                cols, fine, coarse, stats, gw, gb, groups, eps, act, h, w, c, p0 as u64, n as u64)
        })
    }

    fn forward_im2col(&self, gpu: &Gpu, x: &Tensor, w16: &Buf, h: usize, w: usize) -> Result<Tensor> {
        let c = self.w.shape()[2];
        self.im2col_gemm(gpu, w16, h, w, |cols, p0, n| {
            launch!(gpu, "im2col3x3_bf16", ew(n * 9 * c / 8), B256, 0; cols, x, h, w, c, p0 as u64, n as u64)
        })
    }

    /// One K = 9C GEMM per chunk of output pixels, instead of nine GEMMs that each read
    /// and rewrite the whole FP32 output. `im2col(cols, p0, n)` writes the BF16 rows of
    /// output pixels `[p0, p0 + n)`
    fn im2col_gemm(
        &self,
        gpu: &Gpu,
        w16: &Buf,
        h: usize,
        w: usize,
        im2col: impl Fn(u64, usize, usize) -> Result<()>,
    ) -> Result<Tensor> {
        let (o, c) = (self.w.shape()[1], self.w.shape()[2]);
        assert!(c.is_multiple_of(8), "im2col copies 8 channels at a time");
        let (pixels, k) = (h * w, 9 * c);
        let y = gpu.empty(&[pixels, o])?;
        let cols = gpu.alloc_bytes(pixels.min(IM2COL_ROWS) * k * 2)?;
        for p0 in (0..pixels).step_by(IM2COL_ROWS) {
            let n = IM2COL_ROWS.min(pixels - p0);
            im2col(cols.ptr(), p0, n)?;
            let out = (y.at(p0 * o), false);
            gpu.linear_bf16((n, o, k), cols.ptr(), k, w16.ptr(), out, 0.0, self.b.ptr(), Epilogue::Bias)?;
        }
        Ok(y)
    }
}

/// ConvTranspose2d with kernel 2 and stride 2: a GEMM then a pixel-shuffle scatter
pub struct ConvT2x2 {
    /// `[C, O * 4]`, PyTorch's `[C, O, 2, 2]` as is
    pub w: Tensor,
    pub b: Tensor,
    /// BF16 copy of `w` in BF16 mode
    w16: Option<Buf>,
}

impl ConvT2x2 {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str) -> Result<Self> {
        let w = wts.get(gpu, &format!("{prefix}.weight"))?;
        let c = w.shape()[0];
        let n = w.len() / c;
        let w = w.reshape(&[c, n]);
        let w16 = bf16_weights(gpu, &w)?;
        Ok(Self { w, b: wts.get(gpu, &format!("{prefix}.bias"))?, w16 })
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (c, n) = (self.w.shape()[0], self.w.shape()[1]);
        assert_eq!(x.shape(), &[h * w, c]);
        let g = gpu.empty(&[h * w, n])?;
        match &self.w16 {
            Some(w16) => {
                let x16 = Bf16::from_f32(gpu, x)?;
                let (input, weight) = (Mat::new(x16.buf.ptr(), c).bf16(), Mat::new(w16.ptr(), n).bf16());
                gpu.gemm(h * w, n, c, input, weight, Mat::new(g.ptr(), n), 1, 1.0, 0.0)?;
            }
            None => gpu.gemm(
                h * w,
                n,
                c,
                Mat::new(x.ptr(), c),
                Mat::new(self.w.ptr(), n),
                Mat::new(g.ptr(), n),
                1,
                1.0,
                0.0,
            )?,
        }
        let o = n / 4;
        let y = gpu.empty(&[4 * h * w, o])?;
        launch!(gpu, "convt2x2_scatter", ew(y.len()), B256, 0; &y, &g, &self.b, h, w, o)?;
        Ok(y)
    }
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

/// GroupNorm over channels-last `[hw, c]` with a fused activation, in two halves: this
/// computes the per-group statistics, and the returned norm applies them later, in place
/// or as [`Conv3x3::forward_up`] reads its input
pub fn group_norm_stats<'a>(
    gpu: &Gpu,
    x: &Tensor,
    groups: usize,
    w: &'a Tensor,
    b: &'a Tensor,
    eps: f32,
    act: i32,
) -> Result<PendingNorm<'a>> {
    let (hw, c) = (x.rows(), x.cols());
    let chunk = 256usize;
    let blocks = hw.div_ceil(chunk);
    let part = gpu.empty(&[groups * blocks, 2])?;
    launch!(gpu, "group_stats", (blocks as u32, groups as u32, 1), B256, 0;
        &part, x, hw as u64, c, groups, chunk)?;
    let stats = gpu.empty(&[groups, 2])?;
    launch!(gpu, "group_stats_reduce", (groups as u32, 1, 1), B256, 0; &stats, &part, blocks)?;
    Ok(PendingNorm { stats, w, b, groups, eps, act })
}

/// A group norm whose per-group sums are computed but not yet applied
pub struct PendingNorm<'a> {
    /// `[groups, 2]`: sum and sum of squares
    stats: Tensor,
    w: &'a Tensor,
    b: &'a Tensor,
    groups: usize,
    eps: f32,
    act: i32,
}

impl PendingNorm<'_> {
    /// Normalizes `x`, the tensor the statistics came from, in place
    pub fn apply(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        let (hw, c) = (x.rows(), x.cols());
        launch!(gpu, "group_norm_apply", ew(x.len()), B256, 0;
            x, &self.stats, self.w, self.b, hw as u64, c, self.groups, self.eps, self.act)
    }
}

/// `[n, d]` rows of `a` followed by rows of `b`
pub fn concat_rows(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    assert_eq!(a.cols(), b.cols());
    let out = gpu.empty(&[a.rows() + b.rows(), a.cols()])?;
    gpu.copy_dtod(a.ptr(), out.ptr(), a.len() * 4)?;
    gpu.copy_dtod(b.ptr(), out.at(a.len()), b.len() * 4)?;
    Ok(out)
}

/// BF16 copy of a weight tensor in BF16 mode, for the conv layers in the necks
fn bf16_weights(gpu: &Gpu, w: &Tensor) -> Result<Option<Buf>> {
    Ok(match gpu.precision() {
        Precision::Fp32 | Precision::Tf32 => None,
        Precision::Bf16 => Some(to_bf16(gpu, w.ptr(), w.len(), 1, w.len())?),
    })
}

/// `rows` rows of `cols` floats (row stride `ld`) -> contiguous BF16 `[rows, cols]`
pub fn to_bf16(gpu: &Gpu, src: u64, cols: usize, rows: usize, ld: usize) -> Result<Buf> {
    let out = gpu.alloc_bytes(rows * cols * 2)?;
    launch!(gpu, "f32_to_bf16", ew(rows * cols), B256, 0; out.ptr(), src, rows as u64, cols, ld as u64)?;
    Ok(out)
}

pub fn upload_i32(gpu: &Gpu, v: Vec<i32>) -> Result<Buf> {
    gpu.upload_buf(v)
}
