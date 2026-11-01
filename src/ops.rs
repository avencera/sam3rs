//! Layers shared by every stage: linears, norms, attention and convolutions, all on
//! channels-last `[rows, features]` tensors

use eyre::{Result, bail, eyre};

use crate::blaslt::Epilogue;
use crate::gpu::{Arg, Buf, Gpu, Mat, Precision, Tensor, blocks, launch};
use crate::weights::Weights;

/// An activation a kernel or a cuBLASLt epilogue applies; the discriminants are the
/// kernels' `ACT_*` numbers (crates/sam3rs-kernels/src/common.rs)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Act {
    None = 0,
    Relu = 1,
    /// Exact erf GELU
    Gelu = 2,
    /// GELU's tanh approximation
    GeluTanh = 3,
    Sigmoid = 4,
}

impl Act {
    /// The cuBLASLt epilogue that adds the bias, if any, then applies this activation, or
    /// `None` when cuBLASLt has no epilogue for it
    fn epilogue(self, has_bias: bool) -> Option<Epilogue> {
        Some(match (self, has_bias) {
            (Self::None, false) => Epilogue::None,
            (Self::None, true) => Epilogue::Bias,
            (Self::Relu, false) => Epilogue::Relu,
            (Self::Relu, true) => Epilogue::ReluBias,
            (Self::GeluTanh, false) => Epilogue::Gelu,
            (Self::GeluTanh, true) => Epilogue::GeluBias,
            (Self::Gelu | Self::Sigmoid, _) => return None,
        })
    }
}

impl From<Act> for Arg {
    fn from(act: Act) -> Self {
        Arg::I32(act as i32)
    }
}

/// Grid for an elementwise kernel over `n` outputs: one thread each, in blocks of 256.
/// The kernels have no grid-stride loop, so the grid can't be capped
pub fn ew(n: usize) -> (u32, u32, u32) {
    // their thread index is 32-bit, which the rounded-up grid must not overflow
    assert!(n.next_multiple_of(256) <= 1 << 32, "{n} elements exceed the 2^32 threads of a 32-bit thread index");
    (blocks(n, 256), 1, 1)
}

pub const B256: (u32, u32, u32) = (256, 1, 1);

pub fn bias_act(gpu: &Gpu, x: &Tensor, bias: Option<&Tensor>, act: Act) -> Result<()> {
    // the kernel takes the bias length as the row width
    assert!(
        bias.is_none_or(|b| b.len() == x.cols()),
        "bias of {} for rows of {}",
        bias.map_or(0, Tensor::len),
        x.cols()
    );
    launch!(gpu, "bias_act", ew(x.len()), B256, 0; x, bias.map_or(Arg::EMPTY, Arg::from), act)
}

pub fn add(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    assert_eq!(a.len(), b.len());
    let out = gpu.empty(a.shape())?;
    launch!(gpu, "add", ew(a.len()), B256, 0; &out, a, b)?;
    Ok(out)
}

/// `a += b` in place
pub fn add_assign(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.len(), b.len());
    launch!(gpu, "add_assign", ew(a.len()), B256, 0; a, b)
}

/// `a += b16` in place
pub fn add_assign_bf16(gpu: &Gpu, a: &Tensor, b16: &Bf16) -> Result<()> {
    assert_eq!(a.len(), b16.rows * b16.cols);
    launch!(gpu, "add_assign_bf16", ew(a.len()), B256, 0; a, b16)
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
        Ok(Self { buf: to_bf16(gpu, x, x.cols(), x.rows(), x.cols())?, rows: x.rows(), cols: x.cols() })
    }
}

/// The whole matrix as a BF16 slice
impl From<&Bf16> for Arg {
    fn from(x: &Bf16) -> Self {
        x.buf.slice::<u16>()
    }
}

/// A dense layer `y = x @ W^T + b` with W stored `[out, in]` like PyTorch
pub struct Linear {
    w: LinearWeights,
    /// `[out, in]`
    shape: [usize; 2],
    pub b: Option<Tensor>,
}

/// The weights a [`Linear`] multiplies by
enum LinearWeights {
    /// FP32 `[out, in]`
    Fp32(Tensor),
    /// BF16 copies of the weights and bias for the cuBLASLt path, which replace the FP32
    /// weights in `Precision::Bf16`; a BF16 output needs a BF16 bias
    Bf16 { w: Buf, b: Option<Buf> },
}

impl Linear {
    /// A linear over existing device weights, run in FP32 whatever the precision
    pub fn new(w: Tensor, b: Option<Tensor>) -> Self {
        let shape = [w.rows(), w.cols()];
        Self { w: LinearWeights::Fp32(w), shape, b }
    }

    /// In BF16 mode, keeps only a BF16 copy of the weights and runs this linear through
    /// BF16 cuBLASLt. The vision backbone and the fusion encoder's FFN opt in for their
    /// FLOPs, the text encoder's MLP for its weight reads (32 tokens over 800 MB of FP32
    /// weights), and the video tracker's memory attention projections and FFN for its 36k
    /// memory keys; the other detector layers keep FP32 activations and weights
    pub fn bf16(mut self, gpu: &Gpu) -> Result<Self> {
        if gpu.precision() == Precision::Bf16
            && let LinearWeights::Fp32(w) = &self.w
        {
            let w16 = to_bf16(gpu, w, w.cols(), w.rows(), w.cols())?;
            let b16 = self.b.as_ref().map(|b| to_bf16(gpu, b, b.len(), 1, b.len())).transpose()?;
            self.w = LinearWeights::Bf16 { w: w16, b: b16 };
        }
        Ok(self)
    }

    /// Whether [`Self::bf16`] moved this linear onto BF16 weights
    fn is_bf16(&self) -> bool {
        matches!(self.w, LinearWeights::Bf16 { .. })
    }

    /// FP32 weights; a linear converted by [`Self::bf16`] has none
    fn w(&self) -> &Tensor {
        match &self.w {
            LinearWeights::Fp32(w) => w,
            LinearWeights::Bf16 { .. } => panic!("FP32 path of a linear that keeps only BF16 weights"),
        }
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
        self.forward_act(gpu, x, Act::None)
    }

    pub fn forward_act(&self, gpu: &Gpu, x: &Tensor, act: Act) -> Result<Tensor> {
        assert_eq!(x.cols(), self.in_features(), "linear input {:?} vs weight {:?}", x.shape(), self.shape);
        let rows = x.rows();
        let out = gpu.empty(&[rows, self.out_features()])?;
        if self.is_bf16() {
            if !self.forward_bf16(gpu, x, out.ptr(), 0.0, act)? {
                bias_act(gpu, &out, None, act)?;
            }
            return Ok(out);
        }
        let (epilogue, fused) = self.epilogue(act);
        let bias = self.b.as_ref().map_or(0, Tensor::ptr);
        let mnk = (rows, self.out_features(), self.in_features());
        gpu.linear_f32(mnk, x.ptr(), x.cols(), self.w().ptr(), out.ptr(), 0.0, bias, epilogue)?;
        if !fused {
            bias_act(gpu, &out, None, act)?;
        }
        Ok(out)
    }

    /// The cuBLASLt epilogue for this layer's bias and `act`, and whether it applies
    /// `act`; if not, only the bias is fused
    fn epilogue(&self, act: Act) -> (Epilogue, bool) {
        let has_bias = self.b.is_some();
        match act.epilogue(has_bias) {
            Some(epilogue) => (epilogue, true),
            None => (if has_bias { Epilogue::Bias } else { Epilogue::None }, false),
        }
    }

    /// `residual += x @ W^T + b`
    pub fn forward_residual(&self, gpu: &Gpu, x: &Tensor, residual: &Tensor) -> Result<()> {
        assert_eq!(residual.cols(), self.out_features());
        assert_eq!(x.rows(), residual.rows());
        if self.is_bf16() {
            self.forward_bf16(gpu, x, residual.ptr(), 1.0, Act::None)?;
            return Ok(());
        }
        let (epilogue, _) = self.epilogue(Act::None);
        let bias = self.b.as_ref().map_or(0, Tensor::ptr);
        let mnk = (x.rows(), self.out_features(), self.in_features());
        gpu.linear_f32(mnk, x.ptr(), x.cols(), self.w().ptr(), residual.ptr(), 1.0, bias, epilogue)
    }

    /// FP32 input: converts to BF16, then [`Self::forward_lt`]
    fn forward_bf16(&self, gpu: &Gpu, x: &Tensor, out: u64, beta: f32, act: Act) -> Result<bool> {
        let x16 = Bf16::from_f32(gpu, x)?;
        self.forward_lt(gpu, &x16, (out, false), beta, act)
    }

    /// BF16 `act(x16 @ W^T + b)`, for a following BF16 linear; `act` must be one the
    /// epilogue fuses
    pub fn forward16_act16(&self, gpu: &Gpu, x16: &Bf16, act: Act) -> Result<Bf16> {
        let out = Bf16::empty(gpu, x16.rows, self.out_features())?;
        if !self.forward_lt(gpu, x16, (out.buf.ptr(), true), 0.0, act)? {
            bail!("activation {act:?} has no cuBLASLt epilogue");
        }
        Ok(out)
    }

    /// BF16 `x16 @ W^T + b` to FP32: what [`Self::forward`] computes in BF16 mode once
    /// it has converted its FP32 input
    pub fn forward_from16(&self, gpu: &Gpu, x16: &Bf16) -> Result<Tensor> {
        let out = gpu.empty(&[x16.rows, self.out_features()])?;
        self.forward_lt(gpu, x16, (out.ptr(), false), 0.0, Act::None)?;
        Ok(out)
    }

    /// `residual += x16 @ W^T + b`
    pub fn forward16_residual(&self, gpu: &Gpu, x16: &Bf16, residual: &Tensor) -> Result<()> {
        assert_eq!((residual.rows(), residual.cols()), (x16.rows, self.out_features()));
        self.forward_lt(gpu, x16, (residual.ptr(), false), 1.0, Act::None)?;
        Ok(())
    }

    /// `out = act(x16 @ W^T + b) + beta * out` through cuBLASLt. Returns whether the
    /// epilogue applied `act`; if not, only the bias was added
    fn forward_lt(&self, gpu: &Gpu, x16: &Bf16, out: (u64, bool), beta: f32, act: Act) -> Result<bool> {
        let LinearWeights::Bf16 { w: w16, b: b16 } = &self.w else { bail!("linear has no BF16 weights") };
        assert_eq!(x16.cols, self.in_features());
        let (m, n, k) = (x16.rows, self.out_features(), self.in_features());
        let bias = if out.1 { b16.as_ref().map_or(0, Buf::ptr) } else { self.b.as_ref().map_or(0, Tensor::ptr) };
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
    pub act: Act,
}

impl Mlp {
    pub fn load(gpu: &Gpu, wts: &Weights, prefixes: &[String], act: Act) -> Result<Self> {
        let layers = prefixes.iter().map(|p| Linear::load(gpu, wts, p)).collect::<Result<_>>()?;
        Ok(Self { layers, act })
    }

    /// `prefix.layer1 .. prefix.layerN`
    pub fn load_numbered(gpu: &Gpu, wts: &Weights, prefix: &str, n: usize, act: Act) -> Result<Self> {
        let prefixes: Vec<String> = (1..=n).map(|i| format!("{prefix}.layer{i}")).collect();
        Self::load(gpu, wts, &prefixes, act)
    }

    /// `prefix.layers.0 .. prefix.layers.{N-1}`
    pub fn load_indexed(gpu: &Gpu, wts: &Weights, prefix: &str, n: usize, act: Act) -> Result<Self> {
        let prefixes: Vec<String> = (0..n).map(|i| format!("{prefix}.layers.{i}")).collect();
        Self::load(gpu, wts, &prefixes, act)
    }

    /// `prefix.proj_in`, `prefix.layers.*`, `prefix.proj_out`
    pub fn load_in_out(gpu: &Gpu, wts: &Weights, prefix: &str, n: usize, act: Act) -> Result<Self> {
        let mut prefixes = vec![format!("{prefix}.proj_in")];
        prefixes.extend((0..n - 2).map(|i| format!("{prefix}.layers.{i}")));
        prefixes.push(format!("{prefix}.proj_out"));
        Self::load(gpu, wts, &prefixes, act)
    }

    pub fn forward(&self, gpu: &Gpu, x: &Tensor) -> Result<Tensor> {
        let last = self.layers.len() - 1;
        let mut h = self.layers[0].forward_act(gpu, x, if last == 0 { Act::None } else { self.act })?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            h = layer.forward_act(gpu, &h, if i == last { Act::None } else { self.act })?;
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
        assert_eq!(x.cols(), self.w.len());
        launch!(gpu, "layernorm_inplace", (x.rows() as u32, 1, 1), B256, 0; x, &self.w, &self.b, x.cols(), self.eps)
    }

    /// Normalized rows in BF16, as the input of a BF16 linear
    pub fn forward16(&self, gpu: &Gpu, x: &Tensor) -> Result<Bf16> {
        assert_eq!(x.cols(), self.w.len());
        let out = Bf16::empty(gpu, x.rows(), x.cols())?;
        launch!(gpu, "layernorm_bf16", (x.rows() as u32, 1, 1), B256, 0;
            &out, x, &self.w, &self.b, x.cols(), self.eps)?;
        Ok(out)
    }

    /// `x += add16` in place, then [`Self::forward16`] of the sum, in one pass
    pub fn add_forward16(&self, gpu: &Gpu, x: &Tensor, add16: &Bf16) -> Result<Bf16> {
        assert_eq!(x.cols(), self.w.len());
        assert_eq!((add16.rows, add16.cols), (x.rows(), x.cols()));
        let out = Bf16::empty(gpu, x.rows(), x.cols())?;
        launch!(gpu, "add_layernorm_bf16", (x.rows() as u32, 1, 1), B256, 0;
            &out, x, add16, &self.w, &self.b, x.cols(), self.eps)?;
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
    /// u8 per key, nonzero means the key is padding: `lk` bytes, or 0 for none
    pub key_pad: u64,
    pub causal: bool,
    pub rpb: Option<BoxRpb>,
}

/// Dynamic shared memory of `flash_attn_f32_d64`: 64-row Q, K (then P) and V tiles with
/// rows of 64, 68 and 64 floats, small enough for two blocks an SM on parts with 100 KB.
/// The kernel traps if a launch passes less
const F32_D64_SMEM_BYTES: u32 = (64 * 64 + 64 * 68 + 64 * 64) * 4;

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
/// which made the fusion encoder's self-attention L2-bound. TF32 mode keeps K and V in
/// FP32 and uses its own 128-row kernel. The BF16 encoder's global attention measured
/// slower in 128-row blocks
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
    // Mat views carry no allocation size, so each one's slice is the extent its geometry
    // addresses: `batch` entries of `rows` rows of `hd` values
    let extent = |m: Mat, rows: usize| match batch * rows {
        0 => 0,
        _ => (batch - 1) * m.stride + (rows - 1) * m.ld + hd,
    };
    // u8 per key, `lk` of them, or none
    let key_pad = match mask.key_pad {
        0 => Arg::EMPTY,
        p => Arg::slice(p, lk),
    };
    // the box RPB's dx and dy are `[heads, lq - 1, side]` each, heads being the batch
    let rpb_len = |r: BoxRpb| batch * (lq - 1) * r.side;

    if f32_flash_ok(gpu, &[q, k, v, out], hd, &mask) {
        let scale_log2 = std::f32::consts::LOG2_E / (hd as f32).sqrt();
        let view = |(m, rows): (Mat, usize)| [Arg::slice(m.ptr, extent(m, rows)), Arg::from(m.ld), Arg::from(m.stride)];
        let args = [(out, lq), (q, lq), (k, lk), (v, lk)]
            .into_iter()
            .flat_map(view)
            .chain([Arg::from(lq), Arg::from(lk), Arg::from(scale_log2)])
            .collect();
        let grid = (lq.div_ceil(64) as u32, batch as u32, 1);
        return gpu.launch("flash_attn_f32_d64", grid, (128, 1, 1), F32_D64_SMEM_BYTES, args);
    }

    let tf32 = tf32_flash_ok(gpu, &[q, k, v, out], hd, &mask);
    if tf32 || flash_ok(gpu, &[q, k, v, out], hd, &mask) {
        let tall = !q.bf16 && hd == 32 && mask.rpb.is_none() && lq >= TALL_MIN && lk >= TALL_MIN;
        let kernel = match (tf32, q.bf16, hd, mask.rpb) {
            (true, _, 32, Some(_)) => "flash_attn_tf32_rpb_d32",
            (true, _, 32, None) if tall => "flash_attn_tf32_tall_d32",
            (true, _, 32, None) => "flash_attn_tf32_d32",
            (true, _, _, _) => "flash_attn_tf32_d64",
            (false, true, _, _) => "flash_attn_bf16_d64",
            (false, false, 32, Some(_)) => "flash_attn_rpb_d32",
            (false, false, 32, None) => "flash_attn_d32",
            _ => "flash_attn_d64",
        };
        let scale_log2 = std::f32::consts::LOG2_E / (hd as f32).sqrt();
        let view = |(m, rows): (Mat, usize)| [Arg::slice(m.ptr, extent(m, rows)), Arg::from(m.ld), Arg::from(m.stride)];
        if tall && !tf32 {
            // K then V as FP16 [lk, batch * hd] each in one buffer, exactly as the kernel
            // would round them while staging; one output slice lets each thread own one
            // 8-value group of it
            let n = lk * batch * hd;
            let kv = gpu.alloc_bytes(2 * n * 2)?;
            launch!(gpu, "kv_to_16", ew(2 * n / 8), B256, 0;
                kv.slice::<[u16; 8]>(), Arg::slice(k.ptr, extent(k, lk)), k.ld, k.stride,
                Arg::slice(v.ptr, extent(v, lk)), v.ld, v.stride, batch, hd)?;
            let packed = |ptr: u64| (Mat::new(ptr, batch * hd).stride(hd), lk);
            let tail = [Arg::from(lq), Arg::from(lk), Arg::from(scale_log2), key_pad];
            let args = [(out, lq), (q, lq), packed(kv.ptr()), packed(kv.ptr() + (n * 2) as u64)]
                .into_iter()
                .flat_map(view)
                .chain(tail)
                .collect();
            let grid = (lq.div_ceil(128) as u32, batch as u32, 1);
            return gpu.launch("flash_attn_kv16_d32", grid, (256, 1, 1), 0, args);
        }

        // the tall TF32 kernel takes 128 query rows a block, with 256 threads
        let rows = if tall { 128 } else { 64 };
        let grid = (lq.div_ceil(rows) as u32, batch as u32, 1);
        let mut args = Vec::with_capacity(16);
        for m in [(out, lq), (q, lq), (k, lk), (v, lk)] {
            args.extend(view(m));
        }
        args.extend([Arg::from(lq), Arg::from(lk), Arg::from(scale_log2)]);

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
                args.extend([Arg::slice(r.dx, rpb_len(r)), Arg::slice(r.dy, rpb_len(r)), Arg::from(r.side)]);
            }
            None => args.push(key_pad),
        }
        args.extend([part.as_ref().map_or(Arg::EMPTY, Arg::from), Arg::from(chunk)]);
        gpu.launch(kernel, (grid.0, grid.1, splits as u32), (rows as u32 * 2, 1, 1), 0, args)?;
        let Some(part) = part else { return Ok(()) };
        // the kernel takes the split count from the length of `part`
        let warps = batch * lq;
        return launch!(gpu, "flash_combine_d32", (blocks(warps * 32, 256), 1, 1), B256, 0;
            Arg::slice(out.ptr, extent(out, lq)), out.ld, out.stride, &part, batch, lq);
    }

    let (bias, bias_div) = match mask.rpb {
        Some(r) => {
            assert!(mask.bias == 0 && lk == r.side * r.side, "box RPB covers a side x side key grid");
            let bias = gpu.empty(&[batch, lq, lk])?;
            launch!(gpu, "rpb_combine", ew(bias.len()), B256, 0;
                &bias, Arg::slice(r.dx, rpb_len(r)), Arg::slice(r.dy, rpb_len(r)), lq - 1, r.side)?;
            (Some(bias), 1)
        }
        None => (None, mask.bias_div.max(1)),
    };
    let bias_arg = match (&bias, mask.bias) {
        (Some(b), _) => Arg::from(b),
        (None, 0) => Arg::EMPTY,
        (None, p) => Arg::slice(p, batch.div_ceil(bias_div) * lq * lk),
    };
    let scores = gpu.empty(&[batch, lq, lk])?;
    let scale = 1.0 / (hd as f32).sqrt();
    let s = Mat::new(scores.ptr(), lk).stride(lq * lk);
    gpu.gemm(lq, lk, hd, q, k.t(), s, batch, scale, 0.0)?;
    launch!(gpu, "softmax", ((batch * lq) as u32, 1, 1), B256, 0;
        &scores, lq, bias_arg, bias_div, key_pad, mask.causal as i32)?;
    gpu.gemm(lq, hd, lk, s, v, out, batch, 1.0, 0.0)
}

/// FP32 mode fuses unmasked head-dim-64 attention, the backbone's, on the FP32 pipes; the
/// detector's masked and head-dim-32 attentions stay materialized. The kernel loads and
/// stores in 16-byte chunks
fn f32_flash_ok(gpu: &Gpu, [q, k, v, out]: &[Mat; 4], hd: usize, mask: &Mask) -> bool {
    let aligned =
        |m: &Mat| !m.trans && m.ptr.is_multiple_of(16) && m.ld.is_multiple_of(4) && m.stride.is_multiple_of(4);
    gpu.precision() == Precision::Fp32
        && !q.bf16
        && hd == 64
        && mask.rpb.is_none()
        && mask.key_pad == 0
        && mask.bias == 0
        && !mask.causal
        && [q, k, v, out].into_iter().all(aligned)
}

/// TF32 mode fuses attention for FP32 q/k/v with head dims 32 and 64 on TF32 tensor
/// cores, with the same masks as BF16 mode's fused kernel: key padding, or the box RPB at
/// head dim 32 without key padding. Additive bias and causal masks stay materialized. The
/// kernels load q/k/v in 16-byte chunks and store float2
fn tf32_flash_ok(gpu: &Gpu, [q, k, v, out]: &[Mat; 4], hd: usize, mask: &Mask) -> bool {
    let aligned = |m: &Mat, elems: usize| {
        !m.trans
            && m.ptr.is_multiple_of(elems as u64 * 4)
            && m.ld.is_multiple_of(elems)
            && m.stride.is_multiple_of(elems)
    };
    gpu.precision() == Precision::Tf32
        && !q.bf16
        && (hd == 32 || hd == 64)
        && (mask.rpb.is_none() || (hd == 32 && mask.key_pad == 0))
        && mask.bias == 0
        && !mask.causal
        && aligned(out, 2)
        && [q, k, v].iter().all(|m| aligned(m, 4))
}

/// The fused kernel covers BF16 mode for FP32 q/k/v with head dims 32 and 64, which it
/// rounds to FP16 for `mma.sync`, and BF16 q/k/v with head dim 64, without additive bias
/// or causal masking, except the box RPB, which it fuses for FP32 q/k/v with head dim 32
/// and no key padding. It loads q/k/v in 16-byte chunks and stores float2. BF16 q/k/v
/// must use it, since the materialized path reads FP32
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

/// Dynamic shared memory of `flash_attn_f16_d256`: Q, K and V tiles of 64 rows of 264
/// FP16 values
const FLASH_D256_SMEM: u32 = 3 * 64 * 264 * 2;
/// Most key splits of the head-dim-256 attentions (`flash_attn_f16_d256` and
/// `flash_attn_f16_qk256_v64`)
const D256_MAX_SPLITS: usize = 8;

/// Single-head attention with head dim 256 over FP16 `q [lq, 256]` and `k`, `v [lk, 256]`
/// (rows packed), through the fused kernel, which needs a GPU with 99 KB of shared memory
/// per block (sm_86 and up)
pub fn attention_f16_d256(gpu: &Gpu, q: &Buf, k: &Buf, v: &Buf, lq: usize, lk: usize) -> Result<Tensor> {
    let kernel = SplitAttention { name: "flash_attn_f16_d256", combine: "flash_combine_d256", dv: 256, rows: 64 };
    kernel.run(gpu, q, k, v, lq, lk, (128, FLASH_D256_SMEM))
}

/// Dynamic shared memory of `flash_attn_f16_qk256_v64`: two K tiles of 64 rows of 264
/// FP16 values and two V tiles of 64 rows of 72
const FLASH_MEM_SMEM: u32 = 2 * 64 * 264 * 2 + 2 * 64 * 72 * 2;

/// Single-head attention of FP16 `q [lq, 256]` over keys `k [lk, 256]` and 64-wide values
/// `v [lk, 64]` (rows packed), `[lq, 64]` out: the tracker's memory cross-attention with
/// `v_proj` folded into the output projection. Needs 84 KB of shared memory per block
pub fn attention_f16_qk256_v64(gpu: &Gpu, q: &Buf, k: &Buf, v: &Buf, lq: usize, lk: usize) -> Result<Tensor> {
    let kernel = SplitAttention { name: "flash_attn_f16_qk256_v64", combine: "flash_combine_d64", dv: 64, rows: 128 };
    kernel.run(gpu, q, k, v, lq, lk, (256, FLASH_MEM_SMEM))
}

/// A fused single-head attention with 256-wide queries and keys that runs one block per
/// SM, so it splits the keys across blocks and merges the splits after
struct SplitAttention {
    name: &'static str,
    combine: &'static str,
    /// value and output width
    dv: usize,
    /// query rows per block
    rows: usize,
}

impl SplitAttention {
    #[allow(clippy::too_many_arguments)]
    fn run(&self, gpu: &Gpu, q: &Buf, k: &Buf, v: &Buf, lq: usize, lk: usize, block: (u32, u32)) -> Result<Tensor> {
        const D: usize = 256;
        let dv = self.dv;
        assert!(lk > 0 && q.bytes() >= lq * D * 2 && k.bytes() >= lk * D * 2 && v.bytes() >= lk * dv * 2);
        let out = gpu.empty(&[lq, dv])?;
        // one block per SM at this much shared memory, so split the keys into the count
        // whose grid wastes the least of its last wave, the fewest on ties
        let qblocks = lq.div_ceil(self.rows);
        let cost = |s: usize| (qblocks * s).div_ceil(gpu.sms()) as f64 / s as f64;
        let splits = (1..=D256_MAX_SPLITS)
            .filter(|s| lk >= s * 64)
            .fold(1, |best, s| if cost(s) < cost(best) { s } else { best });
        let chunk = lk.div_ceil(splits).next_multiple_of(64);
        let splits = lk.div_ceil(chunk);
        let part = if splits > 1 { Some(gpu.empty(&[splits, 1, lq, dv + 2])?) } else { None };

        let view = |ptr: u64, rows: usize, w: usize| [Arg::slice(ptr, rows * w), Arg::from(w), Arg::from(0usize)];
        let mut args: Vec<Arg> = [(out.ptr(), lq, dv), (q.ptr(), lq, D), (k.ptr(), lk, D), (v.ptr(), lk, dv)]
            .into_iter()
            .flat_map(|(p, r, w)| view(p, r, w))
            .collect();
        let scale_log2 = std::f32::consts::LOG2_E / (D as f32).sqrt();
        args.extend([Arg::from(lq), Arg::from(lk), Arg::from(scale_log2)]);
        args.extend([part.as_ref().map_or(Arg::EMPTY, Arg::from), Arg::from(chunk)]);
        gpu.launch(self.name, (qblocks as u32, 1, splits as u32), (block.0, 1, 1), block.1, args)?;
        if let Some(part) = part {
            // the kernel takes the split count from the length of `part`
            launch!(gpu, self.combine, (blocks(lq * 32, 256), 1, 1), B256, 0; &out, dv, 0usize, &part, 1usize, lq)?;
        }
        Ok(out)
    }
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
                Some(to_bf16(gpu, &rows, rows.len(), 1, rows.len())?)
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
        // writes every element of xp, the padding as zeros
        launch!(gpu, "pad1", ew(rows * c), B256, 0; xp.slice::<f32>(), x, w, c)?;
        let yp = gpu.empty(&[h * wp, o])?;
        for t in 0..9 {
            let (dy, dx) = (t / 3, t % 3);
            let input = Mat::new(xp.ptr() + ((dy * wp + dx) * c * 4) as u64, c);
            let weight = Mat::new(self.w.at(t * o * c), c).t();
            let beta = if t == 0 { 0.0 } else { 1.0 };
            gpu.linear_f32((h * wp, o, c), input.ptr, c, weight.ptr, yp.ptr(), beta, 0, Epilogue::None)?;
        }
        let y = gpu.empty(&[h * w, o])?;
        launch!(gpu, "crop_bias", ew(y.len()), B256, 0; &y, &yp, &self.b, w, o)?;
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
            launch!(gpu, "upsample2_add", ew(up.len()), B256, 0; &up, fine, coarse, w / 2, c)?;
            return self.forward(gpu, &up, h, w);
        };

        assert!(norm.is_none_or(|n| (c / n.groups).is_multiple_of(8)), "im2col normalizes 8 channels of one group");
        // empty stats mean no norm
        let (stats, gw, gb) = norm
            .map_or((Arg::EMPTY, Arg::EMPTY, Arg::EMPTY), |n| (Arg::from(&n.stats), Arg::from(n.w), Arg::from(n.b)));
        let (groups, eps, act) = norm.map_or((1, 0.0, Act::None), |n| (n.groups, n.eps, n.act));
        self.im2col_gemm(gpu, w16, h, w, |cols, p0, n| {
            launch!(gpu, "im2col3x3_up_bf16", ew(n * 9 * c / 8), B256, 0;
                cols, fine, coarse, stats, gw, gb, groups, eps, act, w, c, p0)
        })
    }

    fn forward_im2col(&self, gpu: &Gpu, x: &Tensor, w16: &Buf, h: usize, w: usize) -> Result<Tensor> {
        let c = self.w.shape()[2];
        self.im2col_gemm(
            gpu,
            w16,
            h,
            w,
            |cols, p0, n| launch!(gpu, "im2col3x3_bf16", ew(n * 9 * c / 8), B256, 0; cols, x, w, c, p0),
        )
    }

    /// One K = 9C GEMM per chunk of output pixels, instead of nine GEMMs that each read
    /// and rewrite the whole FP32 output. `im2col(cols, p0, n)` writes the BF16 rows of
    /// output pixels `[p0, p0 + n)` into `cols`, a slice of exactly their `n * 9C / 8`
    /// groups of eight values
    fn im2col_gemm(
        &self,
        gpu: &Gpu,
        w16: &Buf,
        h: usize,
        w: usize,
        im2col: impl Fn(Arg, usize, usize) -> Result<()>,
    ) -> Result<Tensor> {
        let (o, c) = (self.w.shape()[1], self.w.shape()[2]);
        assert!(c.is_multiple_of(8), "im2col copies 8 channels at a time");
        let (pixels, k) = (h * w, 9 * c);
        let y = gpu.empty(&[pixels, o])?;
        let cols = gpu.alloc_bytes(pixels.min(IM2COL_ROWS) * k * 2)?;
        for p0 in (0..pixels).step_by(IM2COL_ROWS) {
            let n = IM2COL_ROWS.min(pixels - p0);
            // the kernel writes the whole slice, so it is the chunk's rows, not the buffer
            im2col(Arg::slice(cols.ptr(), n * k / 8), p0, n)?;
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
        launch!(gpu, "convt2x2_scatter", ew(y.len()), B256, 0; &y, &g, &self.b, w, o)?;
        Ok(y)
    }

    /// BF16 mode from a BF16 input to `act(y)` in BF16, for a following BF16 GEMM: the
    /// values [`Self::forward`] then `bias_act` and a BF16 conversion would give, without
    /// the FP32 map between them
    pub fn forward16(&self, gpu: &Gpu, x16: &Bf16, h: usize, w: usize, act: Act) -> Result<Bf16> {
        let w16 = self.w16.as_ref().ok_or_else(|| eyre!("transposed conv has no BF16 weights"))?;
        let (c, n) = (self.w.shape()[0], self.w.shape()[1]);
        assert_eq!((x16.rows, x16.cols), (h * w, c));
        let g = gpu.empty(&[h * w, n])?;
        let (input, weight) = (Mat::new(x16.buf.ptr(), c).bf16(), Mat::new(w16.ptr(), n).bf16());
        gpu.gemm(h * w, n, c, input, weight, Mat::new(g.ptr(), n), 1, 1.0, 0.0)?;
        let o = n / 4;
        let y = Bf16::empty(gpu, 4 * h * w, o)?;
        launch!(gpu, "convt2x2_scatter_bf16", ew(4 * h * w * o), B256, 0; &y, &g, &self.b, w, o, act)?;
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
    act: Act,
) -> Result<PendingNorm<'a>> {
    let (hw, c) = (x.rows(), x.cols());
    let chunk = 256usize;
    let blocks = hw.div_ceil(chunk);
    let part = gpu.empty(&[groups * blocks, 2])?;
    launch!(gpu, "group_stats", (blocks as u32, groups as u32, 1), B256, 0;
        &part, x, c, groups, chunk)?;
    let stats = gpu.empty(&[groups, 2])?;
    launch!(gpu, "group_stats_reduce", (groups as u32, 1, 1), B256, 0; &stats, &part)?;
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
    act: Act,
}

impl PendingNorm<'_> {
    /// Normalizes `x`, the tensor the statistics came from, in place
    pub fn apply(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        launch!(gpu, "group_norm_apply", ew(x.len()), B256, 0;
            x, &self.stats, self.w, self.b, x.cols(), self.groups, self.eps, self.act)
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
        Precision::Bf16 => Some(to_bf16(gpu, w, w.len(), 1, w.len())?),
    })
}

/// `rows` rows of `cols` floats of `src` (row stride `ld`) -> contiguous BF16 `[rows, cols]`
pub fn to_bf16(gpu: &Gpu, src: &Tensor, cols: usize, rows: usize, ld: usize) -> Result<Buf> {
    let out = gpu.alloc_bytes(rows * cols * 2)?;
    launch!(gpu, "f32_to_bf16", ew(rows * cols), B256, 0; out.slice::<u16>(), src, cols, ld)?;
    Ok(out)
}

pub fn upload_i32(gpu: &Gpu, v: Vec<i32>) -> Result<Buf> {
    gpu.upload_buf(v)
}

#[cfg(test)]
mod tests {
    use super::Act;

    /// The kernels take the activation as a bare `i32`, so a renumbering on either side
    /// would silently apply the wrong one
    #[test]
    fn act_numbers_match_the_kernels() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/crates/sam3rs-kernels/src/common.rs");
        let source = std::fs::read_to_string(path).unwrap();
        let mut kernel: Vec<(&str, i32)> = source
            .lines()
            .filter_map(|line| line.strip_prefix("pub const ACT_")?.strip_suffix(';')?.split_once(": i32 = "))
            .map(|(name, value)| (name, value.parse().unwrap()))
            .collect();
        kernel.sort_by_key(|&(_, value)| value);

        // the kernels have no constant for `Act::None`: any other number is the identity
        let host = [("RELU", Act::Relu), ("GELU", Act::Gelu), ("GELU_TANH", Act::GeluTanh), ("SIGMOID", Act::Sigmoid)];
        let host: Vec<(&str, i32)> = host.iter().map(|&(name, act)| (name, act as i32)).collect();
        assert_eq!(kernel, host);
        assert!(kernel.iter().all(|&(_, value)| value != Act::None as i32));
    }
}
