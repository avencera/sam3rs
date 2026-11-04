//! Layers shared by every stage on Metal: linears, norms and convolutions over
//! channels-last `[rows, features]` tensors
//!
//! Every GEMM runs through the `matmul2d` tensor-op kernels in `kernels/gemm.metal` with
//! FP32 accumulation; a layer's weights are FP32, or BF16 when its stage opts in with
//! [`weight_dtype`] in [`Precision::Bf16`]

use std::f32::consts::LOG2_E;

use eyre::{Result, bail, ensure};

use crate::metal::gpu::{Arg, DType, Gpu, Tensor, groups};
use crate::precision::Precision;
use crate::weights::Weights;

/// Threads in a GEMM threadgroup (4 SIMD groups) and the output tile it computes
const GEMM_THREADS: usize = 128;
const GEMM_TILE: usize = 64;

/// Threads of the one-threadgroup-per-row kernels
const ROW_THREADS: usize = 256;

/// An activation a kernel applies; the discriminants are the kernels' `ACT_*` numbers
/// in `kernels/common.metal`
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

impl From<Act> for Arg<'_> {
    fn from(act: Act) -> Self {
        Arg::I32(act as i32)
    }
}

/// The element type of the weights and activations of a stage that opts into BF16 (the
/// vision backbone and necks): BF16 when the model runs in BF16, FP32 otherwise. The
/// bias, norm parameters and the residual stream stay FP32 either way
pub fn weight_dtype(gpu: &Gpu) -> DType {
    match gpu.precision() {
        Precision::Bf16 => DType::Bf16,
        Precision::Fp32 | Precision::Tf32 => DType::F32,
    }
}

/// `x` converted to `dtype`, or `x` itself when it already is
pub fn convert(gpu: &Gpu, x: Tensor, dtype: DType) -> Result<Tensor> {
    let name = match (x.dtype(), dtype) {
        (a, b) if a == b => return Ok(x),
        (DType::F32, DType::Bf16) => "f32_to_bf16",
        (DType::Bf16, DType::F32) => "bf16_to_f32",
        (a, b) => bail!("no conversion from {a:?} to {b:?}"),
    };
    let out = gpu.empty(x.shape(), dtype)?;
    gpu.dispatch_1d(name, out.len(), &[out.arg(), x.arg()])?;
    Ok(out)
}

/// `x += y` over FP32 `x` and FP32 or BF16 `y`
pub fn add_assign(gpu: &Gpu, x: &Tensor, y: &Tensor) -> Result<()> {
    ensure!(x.len() == y.len(), "add_assign of {:?} and {:?}", x.shape(), y.shape());
    let name = match y.dtype() {
        DType::Bf16 => "add_assign_bf16",
        _ => "add_assign",
    };
    gpu.dispatch_1d(name, x.len(), &[x.arg(), y.arg()])
}

/// `x = act(x + bias)` in place over FP32 `x [rows, cols]`
pub fn bias_act(gpu: &Gpu, x: &Tensor, bias: Option<&Tensor>, act: Act) -> Result<()> {
    let b = bias.map_or_else(|| x.buf.arg(), Tensor::arg);
    let args = [x.arg(), b, x.cols().into(), u32::from(bias.is_some()).into(), act.into()];
    gpu.dispatch_1d("bias_act", x.len(), &args)
}

/// The GEMM kernel for inputs and weights of `ti` writing `to`
fn gemm_kernel(prefix: &'static str, ti: DType, to: DType) -> Result<&'static str> {
    Ok(match (prefix, ti, to) {
        ("linear", DType::F32, DType::F32) => "linear_f32_f32",
        ("linear", DType::Bf16, DType::Bf16) => "linear_bf16_bf16",
        ("linear", DType::Bf16, DType::F32) => "linear_bf16_f32",
        ("convt2x2", DType::F32, DType::F32) => "convt2x2_f32_f32",
        ("convt2x2", DType::Bf16, DType::Bf16) => "convt2x2_bf16_bf16",
        ("conv3x3", DType::F32, DType::F32) => "conv3x3_f32_f32",
        ("conv3x3", DType::Bf16, DType::F32) => "conv3x3_bf16_f32",
        _ => bail!("no {prefix} kernel from {ti:?} to {to:?}"),
    })
}

/// `nn.Linear`: `y = act(x . w^T + b)`
pub struct Linear {
    /// `[out, in]`, FP32 or BF16
    w: Tensor,
    /// FP32 `[out]`, or `[r, out]` whose row `i % r` output row `i` adds
    b: Option<Tensor>,
}

impl Linear {
    pub fn new(w: Tensor, b: Option<Tensor>) -> Self {
        Self { w, b }
    }

    /// One GEMM for several FP32 linears that read the same input, their outputs side by
    /// side in `parts` order. Parts flagged `shifted` read the input plus a constant
    /// `c [r, in]` instead, folded into an `[r, out]` bias as `(x + c) W^T + b = x W^T +
    /// (c W^T + b)`, so an input of stacked `r`-row blocks adds `c` to each block; the
    /// fold rounds differently from adding first, within FP32 noise
    pub fn fuse_shifted(gpu: &Gpu, parts: &[(&Linear, bool)], c: &Tensor) -> Result<Self> {
        ensure!(!parts.is_empty(), "nothing to fuse");
        let (k, r) = (parts[0].0.in_features(), c.rows());
        ensure!(c.cols() == k && c.dtype() == DType::F32, "shift {:?} for inputs of {k}", c.shape());
        let n: usize = parts.iter().map(|(l, _)| l.out_features()).sum();
        let w = gpu.empty(&[n, k], DType::F32)?;
        let mut bias = vec![0f32; r * n];
        let mut col = 0;
        for (lin, shifted) in parts {
            let o = lin.out_features();
            ensure!(lin.in_features() == k && lin.w.dtype() == DType::F32, "fusing linear {:?}", lin.w.shape());
            ensure!(lin.bias_rows()? <= 1, "fusing a linear whose bias is already a matrix");
            gpu.dispatch_1d("copy_f32", lin.w.len(), &[w.at(col * k), lin.w.arg()])?;
            // each block row of this part's bias columns: c W^T + b when shifted, else b
            let rows = if *shifted {
                gpu.download(&lin.forward(gpu, c, Act::None)?)?
            } else {
                let b = lin.b.as_ref().map_or_else(|| Ok(vec![0f32; o]), |b| gpu.download(b))?;
                b.repeat(r)
            };
            for (i, row) in rows.chunks_exact(o).enumerate() {
                bias[i * n + col..][..o].copy_from_slice(row);
            }
            col += o;
        }
        Ok(Self::new(w, Some(gpu.upload_f32(&bias, &[r, n])?)))
    }

    /// The bias as the kernels take it: rows of `out`, 0 without one
    fn bias_rows(&self) -> Result<usize> {
        let n = self.out_features();
        let Some(b) = &self.b else { return Ok(0) };
        ensure!(b.len().is_multiple_of(n) && b.len() > 0, "bias {:?} for {n} outputs", b.shape());
        Ok(b.len() / n)
    }

    /// Loads `prefix.weight` and the optional `prefix.bias`, the weights as `dtype` (see
    /// [`weight_dtype`]) and the bias as FP32
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        let w = wts.get_as(gpu, &format!("{prefix}.weight"), dtype)?;
        // 1x1 conv weights [O, C, 1, 1] are a linear's [O, C]
        let (o, len) = (w.shape()[0], w.len());
        let w = w.reshape(&[o, len / o]);
        let bias = format!("{prefix}.bias");
        let b = if wts.contains(&bias) { Some(wts.get(gpu, &bias)?) } else { None };
        Ok(Self::new(w, b))
    }

    /// Fuses separate projections that read the same input, such as q/k/v
    pub fn load_concat(gpu: &Gpu, wts: &Weights, prefixes: &[String], dtype: DType) -> Result<Self> {
        let names = |s: &str| prefixes.iter().map(|p| format!("{p}.{s}")).collect::<Vec<_>>();
        let w = wts.concat_as(gpu, &names("weight"), dtype)?;
        Ok(Self::new(w, Some(wts.concat_as(gpu, &names("bias"), DType::F32)?)))
    }

    pub fn out_features(&self) -> usize {
        self.w.rows()
    }

    pub fn in_features(&self) -> usize {
        self.w.cols()
    }

    /// `act(x . w^T + b)` as a new `[rows, out]` tensor of `out` type; `x` must have the
    /// weights' element type
    pub fn forward_to(&self, gpu: &Gpu, x: &Tensor, act: Act, out: DType) -> Result<Tensor> {
        let y = gpu.empty(&[x.rows(), self.out_features()], out)?;
        self.run(gpu, x, &y, act, false)?;
        Ok(y)
    }

    /// [`Self::forward_to`] writing the weights' element type
    pub fn forward(&self, gpu: &Gpu, x: &Tensor, act: Act) -> Result<Tensor> {
        self.forward_to(gpu, x, act, self.w.dtype())
    }

    /// `y += x . w^T + b` into an existing `[rows, out]` tensor
    pub fn forward_residual(&self, gpu: &Gpu, x: &Tensor, y: &Tensor) -> Result<()> {
        ensure!(
            y.shape() == [x.rows(), self.out_features()],
            "residual {:?} for linear {:?}",
            y.shape(),
            self.w.shape()
        );
        self.run(gpu, x, y, Act::None, true)
    }

    fn run(&self, gpu: &Gpu, x: &Tensor, y: &Tensor, act: Act, accumulate: bool) -> Result<()> {
        let (m, n, k) = (x.rows(), self.out_features(), self.in_features());
        ensure!(x.cols() == k, "linear input {:?} vs weight {:?}", x.shape(), self.w.shape());
        ensure!(x.dtype() == self.w.dtype(), "linear input {:?} vs weights {:?}", x.dtype(), self.w.dtype());
        // an absent bias binds any buffer; the kernel never reads it
        let b = self.b.as_ref().map_or_else(|| self.w.buf.arg(), Tensor::arg);
        let bias_rows = self.bias_rows()?;
        let tiles = groups(n, GEMM_TILE) * groups(m, GEMM_TILE);
        let (splits, kc) = k_splits(tiles, k);
        if splits > 1 {
            let part = gpu.empty(&[splits * m, n], DType::F32)?;
            let gemm = match x.dtype() {
                DType::Bf16 => "linear_splitk_bf16",
                _ => "linear_splitk_f32",
            };
            let args = [x.arg(), self.w.arg(), part.arg(), m.into(), n.into(), k.into(), kc.into()];
            gpu.dispatch(gemm, [groups(n, GEMM_TILE), groups(m, GEMM_TILE), splits], [GEMM_THREADS, 1, 1], &args)?;
            let reduce = match y.dtype() {
                DType::Bf16 => "linear_splitk_reduce_bf16",
                _ => "linear_splitk_reduce_f32",
            };
            let args = [
                y.arg(),
                part.arg(),
                b,
                m.into(),
                n.into(),
                n.into(),
                splits.into(),
                bias_rows.into(),
                act.into(),
                u32::from(accumulate).into(),
            ];
            return gpu.dispatch_1d(reduce, m * n, &args);
        }
        let name = gemm_kernel("linear", x.dtype(), y.dtype())?;
        let args = [
            x.arg(),
            self.w.arg(),
            b,
            y.arg(),
            m.into(),
            n.into(),
            k.into(),
            k.into(),
            n.into(),
            bias_rows.into(),
            act.into(),
            u32::from(accumulate).into(),
        ];
        gpu.dispatch(name, [groups(n, GEMM_TILE), groups(m, GEMM_TILE), 1], [GEMM_THREADS, 1, 1], &args)
    }
}

/// Fewest K elements one split of a split-K GEMM takes, so its partial tile costs less
/// than the K range it covers
const SPLIT_K_MIN: usize = 256;

/// How a GEMM of `tiles` output tiles over `k` splits along K: `(splits, chunk)`, one
/// split when the tiles already fill the GPU (the [`KV_TARGET_GROUPS`] threadgroups
/// split-KV attention aims for) or K is too short, else enough to reach it, with chunks
/// a multiple of 32
fn k_splits(tiles: usize, k: usize) -> (usize, usize) {
    if tiles >= KV_TARGET_GROUPS || k < 2 * SPLIT_K_MIN {
        return (1, k);
    }
    let want = KV_TARGET_GROUPS.div_ceil(tiles).min(k / SPLIT_K_MIN);
    let chunk = k.div_ceil(want).next_multiple_of(32);
    (k.div_ceil(chunk), chunk)
}

/// `nn.LayerNorm` over the last dimension, with FP32 statistics and parameters
pub struct LayerNorm {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self { w: wts.get(gpu, &format!("{prefix}.weight"))?, b: wts.get(gpu, &format!("{prefix}.bias"))?, eps })
    }

    fn rows(&self, gpu: &Gpu, name: &'static str, args: &[Arg], rows: usize) -> Result<()> {
        gpu.dispatch(name, [rows, 1, 1], [ROW_THREADS, 1, 1], args)
    }

    /// The norm of FP32 `x` as a new tensor of `out` type (FP32 or BF16)
    pub fn forward_to(&self, gpu: &Gpu, x: &Tensor, out: DType) -> Result<Tensor> {
        let y = gpu.empty(x.shape(), out)?;
        let name = match out {
            DType::Bf16 => "layernorm_bf16",
            _ => "layernorm",
        };
        let args = [y.arg(), x.arg(), self.w.arg(), self.b.arg(), x.cols().into(), self.eps.into()];
        self.rows(gpu, name, &args, x.rows())?;
        Ok(y)
    }

    pub fn forward_inplace(&self, gpu: &Gpu, x: &Tensor) -> Result<()> {
        let args = [x.arg(), self.w.arg(), self.b.arg(), x.cols().into(), self.eps.into()];
        self.rows(gpu, "layernorm_inplace", &args, x.rows())
    }

    /// `x = act(norm(x))` in place for rows of at most 256 FP32 values, a SIMD group a
    /// row: the narrow, many-row maps a threadgroup a row leaves mostly idle
    pub fn forward_act_narrow(&self, gpu: &Gpu, x: &Tensor, act: Act) -> Result<()> {
        ensure!(x.cols() <= 256 && x.dtype() == DType::F32, "narrow LayerNorm of {:?}", x.shape());
        let args = [x.arg(), self.w.arg(), self.b.arg(), x.rows().into(), x.cols().into(), self.eps.into(), act.into()];
        gpu.dispatch("layernorm_act_simd", [groups(x.rows(), 8), 1, 1], [256, 1, 1], &args)
    }

    /// `x += add` in place, then the norm of the sum as a new tensor of `add`'s type: a
    /// residual branch's add folded into the norm that reads the sum
    pub fn add_forward(&self, gpu: &Gpu, x: &Tensor, add: &Tensor) -> Result<Tensor> {
        ensure!(x.shape() == add.shape(), "add_layernorm of {:?} and {:?}", x.shape(), add.shape());
        let y = gpu.empty(x.shape(), add.dtype())?;
        let name = match add.dtype() {
            DType::Bf16 => "add_layernorm_bf16",
            _ => "add_layernorm_f32",
        };
        let args = [y.arg(), x.arg(), add.arg(), self.w.arg(), self.b.arg(), x.cols().into(), self.eps.into()];
        self.rows(gpu, name, &args, x.rows())?;
        Ok(y)
    }
}

/// ConvTranspose2d with kernel 2 and stride 2, as one GEMM whose epilogue adds the bias,
/// applies the activation and scatters each column to its output pixel
pub struct ConvT2x2 {
    /// `[4 * O, C]`: PyTorch's `[C, O, 2, 2]` transposed tap-major (`dy * 2 + dx`), so the
    /// GEMM reads it like a linear's weights
    w: Tensor,
    b: Tensor,
}

impl ConvT2x2 {
    /// Loads the weights as `dtype`, which is also the input and output type
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        let (w, shape) = wts.host(&format!("{prefix}.weight"))?;
        let c = shape[0];
        let n = w.len() / c;
        let o = n / 4;
        // PyTorch's column j = o * 4 + tap becomes row tap * O + o, so a GEMM tile's
        // columns are consecutive channels of one output pixel and its stores coalesce
        let mut t = vec![0f32; w.len()];
        for ci in 0..c {
            for oc in 0..o {
                for tap in 0..4 {
                    t[(tap * o + oc) * c + ci] = w[ci * n + oc * 4 + tap];
                }
            }
        }
        Ok(Self { w: gpu.upload_as(&t, &[n, c], dtype)?, b: wts.get(gpu, &format!("{prefix}.bias"))? })
    }

    /// `x [h * w, C]` of the weights' type -> `act(convT(x))` `[2h * 2w, O]` of the same
    pub fn forward(&self, gpu: &Gpu, x: &Tensor, h: usize, w: usize, act: Act) -> Result<Tensor> {
        self.forward_add(gpu, x, h, w, None, act)
    }

    /// [`Self::forward`] with FP32 `add` added before the activation: one `[2h * 2w, O]`
    /// map, or a map of a fraction of the rows repeated, for `h` stacking several images
    pub fn forward_add(
        &self,
        gpu: &Gpu,
        x: &Tensor,
        h: usize,
        w: usize,
        add: Option<&Tensor>,
        act: Act,
    ) -> Result<Tensor> {
        let (n, c) = (self.w.rows(), self.w.cols());
        ensure!(x.shape() == [h * w, c], "convT input {:?} for {h}x{w}x{c}", x.shape());
        let dtype = self.w.dtype();
        let y = gpu.empty(&[4 * h * w, n / 4], dtype)?;
        if let Some(a) = add {
            ensure!(
                a.dtype() == DType::F32 && a.cols() == n / 4 && y.len().is_multiple_of(a.len()),
                "convT addend {:?} for output {:?}",
                a.shape(),
                y.shape()
            );
        }
        let m = h * w;
        let args = [
            x.arg(),
            self.w.arg(),
            self.b.arg(),
            y.arg(),
            m.into(),
            n.into(),
            c.into(),
            w.into(),
            act.into(),
            add.map_or_else(|| self.b.arg(), Tensor::arg),
            add.map_or(0, Tensor::len).into(),
        ];
        let name = gemm_kernel("convt2x2", dtype, dtype)?;
        gpu.dispatch(name, [groups(n, GEMM_TILE), groups(m, GEMM_TILE), 1], [GEMM_THREADS, 1, 1], &args)?;
        Ok(y)
    }
}

/// 3x3 conv, stride 1, padding 1, channels-last, FP32 in and out; BF16 weights multiply a
/// BF16 copy of the input
pub struct Conv3x3 {
    /// `[9, O, C]`, tap-major, FP32 or BF16
    taps: Tensor,
    b: Tensor,
}

impl Conv3x3 {
    /// Loads the weights as `dtype`, the type the GEMM multiplies in
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        let (w, shape) = wts.host(&format!("{prefix}.weight"))?;
        let (o, c) = (shape[0], shape[1]);
        ensure!(shape[2..] == [3, 3], "{prefix}: a 3x3 conv has weights {shape:?}");
        let mut taps = vec![0f32; w.len()];
        for oc in 0..o {
            for ic in 0..c {
                for t in 0..9 {
                    taps[(t * o + oc) * c + ic] = w[(oc * c + ic) * 9 + t];
                }
            }
        }
        Ok(Self { taps: gpu.upload_as(&taps, &[9, o, c], dtype)?, b: wts.get(gpu, &format!("{prefix}.bias"))? })
    }

    /// FP32 `x [h * w, C]` -> FP32 `[h * w, O]`
    pub fn forward(&self, gpu: &Gpu, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (o, c) = (self.taps.shape()[1], self.taps.shape()[2]);
        ensure!(x.shape() == [h * w, c], "conv3x3 input {:?} for {h}x{w}x{c}", x.shape());
        let ti = self.taps.dtype();
        let xp = gpu.empty(&[(h + 2) * (w + 2) + 2, c], ti)?;
        let pad = if ti == DType::Bf16 { "pad1_f32_bf16" } else { "pad1_f32_f32" };
        gpu.dispatch_1d(pad, xp.len(), &[xp.arg(), x.arg(), h.into(), w.into(), c.into()])?;
        let y = gpu.empty(&[h * w, o], DType::F32)?;
        let args = [xp.arg(), self.taps.arg(), self.b.arg(), y.arg(), h.into(), w.into(), c.into(), o.into()];
        let rows = h * (w + 2);
        let name = gemm_kernel("conv3x3", ti, DType::F32)?;
        gpu.dispatch(name, [groups(o, GEMM_TILE), groups(rows, GEMM_TILE), 1], [GEMM_THREADS, 1, 1], &args)?;
        Ok(y)
    }
}

/// A strided row-major view for [`attention`]: batch entry `b` starts `offset + b *
/// stride` elements into `t`, with rows `ld` elements apart
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub t: &'a Tensor,
    pub offset: usize,
    pub ld: usize,
    pub stride: usize,
}

impl<'a> View<'a> {
    /// The whole of `t` as one `[rows, cols]` entry
    pub fn rows(t: &'a Tensor) -> Self {
        Self { t, offset: 0, ld: t.cols(), stride: 0 }
    }

    /// Head `h` of `heads * hd` columns from column `col` of `t`, heads being the batch:
    /// the per-head view of a (possibly fused) projection's output
    pub fn heads(t: &'a Tensor, col: usize, hd: usize) -> Self {
        Self { t, offset: col, ld: t.cols(), stride: hd }
    }

    /// The view from row `rows` on
    pub fn skip(self, rows: usize) -> Self {
        Self { offset: self.offset + rows * self.ld, ..self }
    }

    /// The extent the view's geometry reaches, which must lie inside `t`
    fn check(&self, batch: usize, rows: usize, width: usize) -> Result<()> {
        let end = self.offset + (batch - 1) * self.stride + (rows - 1) * self.ld + width;
        ensure!(rows == 0 || end <= self.t.len(), "attention view reaches {end} of {:?}", self.t.shape());
        ensure!(self.t.dtype() == DType::F32, "attention reads FP32, got {:?}", self.t.dtype());
        Ok(())
    }
}

/// Softmax masking options for [`attention`]
#[derive(Clone, Copy, Default)]
pub struct Mask<'a> {
    /// u8 per key, nonzero for padding, shared by every batch entry
    pub key_pad: Option<&'a Tensor>,
    /// Masks keys after the query's row
    pub causal: bool,
    pub rpb: Option<BoxRpb<'a>>,
}

/// Box relative position bias in factored form: for head `h`, query row `1 + q` and key
/// `y * side + x`, the bias is `dy[h, q, y] + dx[h, q, x]`, and row 0 gets none
#[derive(Clone, Copy)]
pub struct BoxRpb<'a> {
    /// `[heads, lq - 1, side]`
    pub dx: &'a Tensor,
    /// `[heads, lq - 1, side]`
    pub dy: &'a Tensor,
    pub side: usize,
}

/// Query rows per attention threadgroup, and its threads
const ATTN_ROWS: usize = 32;
const ATTN_THREADS: usize = 128;
/// Query rows a threadgroup of the tensor-op attention covers: 4 SIMD groups of 16
const TOPS_ROWS: usize = 64;

/// Scaled dot-product attention over `batch` independent entries of FP32 views, query
/// and key width `hd`, value and output width `dv`, into `out`
#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    q: View,
    k: View,
    v: View,
    out: View,
    batch: usize,
    lq: usize,
    lk: usize,
    (hd, dv): (usize, usize),
    mask: Mask,
) -> Result<()> {
    if batch == 0 || lq == 0 {
        return Ok(());
    }
    ensure!(lk > 0, "attention over no keys");
    let name = match (hd, dv) {
        (16, 16) => "attn_d16",
        (32, 32) => "attn_d32",
        (64, 64) => "attn_d64",
        (256, 256) => "attn_d256",
        (256, 64) => "attn_d256_v64",
        _ => bail!("no attention kernel for head dim {hd} and value width {dv}"),
    };
    q.check(batch, lq, hd)?;
    k.check(batch, lk, hd)?;
    v.check(batch, lk, dv)?;
    out.check(batch, lq, dv)?;
    let mut flags = 0u32;
    if let Some(pad) = mask.key_pad {
        ensure!(pad.dtype() == DType::U8 && pad.len() >= lk, "key padding {:?} for {lk} keys", pad.shape());
        flags |= 1;
    }
    if mask.causal {
        flags |= 2;
    }
    let side = match mask.rpb {
        Some(r) => {
            ensure!(lk == r.side * r.side, "box RPB covers a {0}x{0} key grid, not {lk} keys", r.side);
            ensure!(r.dx.len() == batch * (lq - 1) * r.side && r.dy.len() == r.dx.len(), "box RPB shape");
            flags |= 4;
            r.side
        }
        None => 0,
    };
    let scale_log2 = LOG2_E / (hd as f32).sqrt();
    let strides = [out.ld, out.stride, q.ld, q.stride, k.ld, k.stride, v.ld, v.stride];
    // unmasked attention runs on tensor ops: the detector's 5184-token self-attentions
    // took 38 ms rather than 72 on the fragment kernel, and the tracker's memory attention
    // 0.6 s rather than 2.7 per object at 36k keys before its value fold
    // head dim and value width 256 run as scores GEMM, row softmax and P@V GEMM: the fused
    // kernel's P@V over 64-key tiles held it at ~2.6 TFLOPS whatever the tile or SIMD
    // group count, so the tracker's 5184-token self-attention took 13.6-14.0 ms fused
    // against 6.3-6.6 unfused (M4 Pro, 2026-11-04)
    // head dim 256 over 64-wide values (the tracker's memory attention) shares each score
    // tile across 4 SIMD groups and splits keys: 23.4 against 26.7 ms at 36k keys, and
    // batched objects keep that rate per entry where the per-group kernel fell to 48 ms
    if (hd, dv, flags) == (256, 64, 0) {
        return attention_coop(gpu, q, k, v, out, batch, lq, lk, scale_log2);
    }
    if (hd, dv, flags) == (256, 256, 0) {
        return attention_unfused(gpu, q, k, v, out, batch, lq, lk, (hd, dv), scale_log2);
    }
    let tops = match (hd, dv, flags) {
        (32, 32, 0) => Some("attn_tops_d32"),
        (64, 64, 0) => Some("attn_tops_d64"),
        _ => None,
    };
    if let Some(tops) = tops {
        let mut args = vec![
            out.t.at(out.offset),
            q.t.at(q.offset),
            k.t.at(k.offset),
            v.t.at(v.offset),
            lq.into(),
            lk.into(),
            scale_log2.into(),
        ];
        args.extend(strides.map(Arg::from));
        return gpu.dispatch(tops, [groups(lq, TOPS_ROWS), batch, 1], [ATTN_THREADS, 1, 1], &args);
    }

    // absent masks bind the output buffer; the kernel never reads them
    let unused = out.t.buf.arg();
    let args = [
        out.t.at(out.offset),
        q.t.at(q.offset),
        k.t.at(k.offset),
        v.t.at(v.offset),
        mask.key_pad.map_or(unused, Tensor::arg),
        mask.rpb.map_or(unused, |r| r.dx.arg()),
        mask.rpb.map_or(unused, |r| r.dy.arg()),
        lq.into(),
        lk.into(),
        scale_log2.into(),
        flags.into(),
        side.into(),
    ];
    let blocks = groups(lq, ATTN_ROWS) * batch;
    let splits = kv_splits(blocks, lk);
    if splits == 1 {
        let args: Vec<Arg> =
            args.into_iter().chain(strides.map(Arg::from)).chain([unused, unused, lk.into()]).collect();
        return gpu.dispatch(name, [groups(lq, ATTN_ROWS), batch, 1], [ATTN_THREADS, 1, 1], &args);
    }

    let kchunk = lk.div_ceil(splits).next_multiple_of(KV_ALIGN);
    let splits = lk.div_ceil(kchunk);
    let part = gpu.empty(&[splits * batch * lq, dv], DType::F32)?;
    let ml = gpu.empty(&[splits * batch * lq, 2], DType::F32)?;
    let args: Vec<Arg> =
        args.into_iter().chain(strides.map(Arg::from)).chain([part.arg(), ml.arg(), kchunk.into()]).collect();
    gpu.dispatch(name, [groups(lq, ATTN_ROWS), batch, splits], [ATTN_THREADS, 1, 1], &args)?;
    let args = [
        out.t.at(out.offset),
        part.arg(),
        ml.arg(),
        splits.into(),
        batch.into(),
        lq.into(),
        dv.into(),
        out.ld.into(),
        out.stride.into(),
    ];
    gpu.dispatch_1d("attn_combine", batch * lq * dv, &args)
}

/// Query rows per `attn_coop` threadgroup
const COOP_ROWS: usize = 64;
/// Threadgroups `attn_coop` splits keys to reach: 4 SIMD groups each, so about 16 a
/// core; it gained nothing past 4 splits of 81 query blocks (324 threadgroups)
const COOP_TARGET_GROUPS: usize = 320;
/// Fewest keys per `attn_coop` split, a multiple of its 64-key tile
const COOP_MIN_CHUNK: usize = 1024;

/// Unmasked attention of head dim 256 over 64-wide values on `attn_coop`, split over keys
/// when the query blocks alone leave the GPU underfilled
#[allow(clippy::too_many_arguments)]
fn attention_coop(
    gpu: &Gpu,
    q: View,
    k: View,
    v: View,
    out: View,
    batch: usize,
    lq: usize,
    lk: usize,
    scale_log2: f32,
) -> Result<()> {
    let blocks = groups(lq, COOP_ROWS) * batch;
    let splits = COOP_TARGET_GROUPS.div_ceil(blocks).min(lk / COOP_MIN_CHUNK).max(1);
    let kchunk = lk.div_ceil(splits).next_multiple_of(COOP_ROWS);
    let splits = lk.div_ceil(kchunk);
    let dv = 64;
    let (part, ml) = if splits > 1 {
        (
            Some(gpu.empty(&[splits * batch * lq, dv], DType::F32)?),
            Some(gpu.empty(&[splits * batch * lq, 2], DType::F32)?),
        )
    } else {
        (None, None)
    };
    // the unsplit kernel never touches the partials, so they bind the output
    let unused = out.t.buf.arg();
    let strides = [out.ld, out.stride, q.ld, q.stride, k.ld, k.stride, v.ld, v.stride];
    let args: Vec<Arg> = [
        out.t.at(out.offset),
        q.t.at(q.offset),
        k.t.at(k.offset),
        v.t.at(v.offset),
        lq.into(),
        lk.into(),
        scale_log2.into(),
    ]
    .into_iter()
    .chain(strides.map(Arg::from))
    .chain([part.as_ref().map_or(unused, Tensor::arg), ml.as_ref().map_or(unused, Tensor::arg), kchunk.into()])
    .collect();
    gpu.dispatch("attn_coop_d256_v64", [groups(lq, COOP_ROWS), batch, splits], [128, 1, 1], &args)?;
    let (Some(part), Some(ml)) = (part, ml) else {
        return Ok(());
    };
    let args = [
        out.t.at(out.offset),
        part.arg(),
        ml.arg(),
        splits.into(),
        batch.into(),
        lq.into(),
        dv.into(),
        out.ld.into(),
        out.stride.into(),
    ];
    gpu.dispatch_1d("attn_combine", batch * lq * dv, &args)
}

/// Most score elements the unfused attention holds at once, 256 MB: longer inputs run in
/// query chunks
const UNFUSED_SCORES: usize = 64 << 20;

/// Unmasked attention as two GEMMs around a row softmax, through a `[rows, lk]` score
/// buffer per query chunk; see `attn_scores` in gemm.metal
#[allow(clippy::too_many_arguments)]
fn attention_unfused(
    gpu: &Gpu,
    q: View,
    k: View,
    v: View,
    out: View,
    batch: usize,
    lq: usize,
    lk: usize,
    (hd, dv): (usize, usize),
    scale_log2: f32,
) -> Result<()> {
    let chunk = (UNFUSED_SCORES / lk).clamp(GEMM_TILE, lq.next_multiple_of(GEMM_TILE)) / GEMM_TILE * GEMM_TILE;
    let s = gpu.empty(&[chunk.min(lq), lk], DType::F32)?;
    let inv = gpu.empty(&[chunk.min(lq)], DType::F32)?;
    for b in 0..batch {
        for r0 in (0..lq).step_by(chunk) {
            let rows = chunk.min(lq - r0);
            let qa = q.t.at(q.offset + b * q.stride + r0 * q.ld);
            let args = [
                s.arg(),
                qa,
                k.t.at(k.offset + b * k.stride),
                rows.into(),
                lk.into(),
                hd.into(),
                q.ld.into(),
                k.ld.into(),
                scale_log2.into(),
            ];
            gpu.dispatch(
                "attn_scores",
                [groups(lk, GEMM_TILE), groups(rows, GEMM_TILE), 1],
                [GEMM_THREADS, 1, 1],
                &args,
            )?;
            gpu.dispatch("attn_softmax_rows", [rows, 1, 1], [SOFTMAX_THREADS, 1, 1], &[s.arg(), inv.arg(), lk.into()])?;
            let oa = out.t.at(out.offset + b * out.stride + r0 * out.ld);
            let args = [
                oa,
                s.arg(),
                v.t.at(v.offset + b * v.stride),
                inv.arg(),
                rows.into(),
                lk.into(),
                dv.into(),
                v.ld.into(),
                out.ld.into(),
            ];
            gpu.dispatch("attn_pv", [groups(dv, GEMM_TILE), groups(rows, GEMM_TILE), 1], [GEMM_THREADS, 1, 1], &args)?;
        }
    }
    Ok(())
}

/// Threads of `attn_softmax_rows`, one threadgroup a row
const SOFTMAX_THREADS: usize = 256;

/// Threadgroups the fragment attention aims for before it splits keys: about 8 for each
/// core of a 20-core M4 Pro
const KV_TARGET_GROUPS: usize = 160;
/// Fewest keys a split takes, so its partial rows cost less than the keys they cover
const KV_MIN_CHUNK: usize = 256;
/// Key chunks start on a multiple of every key tile the kernels use (8, 16 and 32)
const KV_ALIGN: usize = 32;

/// How many key ranges `blocks` query blocks over `lk` keys split into: one when the
/// blocks already fill the GPU, else enough to reach [`KV_TARGET_GROUPS`]
fn kv_splits(blocks: usize, lk: usize) -> usize {
    if blocks >= KV_TARGET_GROUPS {
        return 1;
    }
    KV_TARGET_GROUPS.div_ceil(blocks).min(lk / KV_MIN_CHUNK).max(1)
}

/// Multi-head attention over `[tokens, heads * hd]` row-major q, k and v views (possibly
/// columns of a wider fused projection), merged heads `[lq, heads * hd]` out
pub fn mha(gpu: &Gpu, q: View, k: View, v: View, (lq, lk): (usize, usize), heads: usize, mask: Mask) -> Result<Tensor> {
    let hd = q.stride;
    let out = gpu.empty(&[lq, heads * hd], DType::F32)?;
    attention(gpu, q, k, v, View::heads(&out, 0, hd), heads, lq, lk, (hd, hd), mask)?;
    Ok(out)
}

/// A stack of linears with an activation between them (not after the last)
pub struct Mlp {
    pub layers: Vec<Linear>,
    pub act: Act,
}

impl Mlp {
    pub fn load(gpu: &Gpu, wts: &Weights, prefixes: &[String], act: Act) -> Result<Self> {
        let layers = prefixes.iter().map(|p| Linear::load(gpu, wts, p, DType::F32)).collect::<Result<_>>()?;
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
        let act = |i: usize| if i == last { Act::None } else { self.act };
        let mut h = self.layers[0].forward(gpu, x, act(0))?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            h = layer.forward(gpu, &h, act(i))?;
        }
        Ok(h)
    }
}

/// Multi-head attention with separate q/k/v/o projections (`prefix.{q,k,v,o}_proj`),
/// FP32. The projections may narrow to an internal width, as in SAM's downsampled
/// attention
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub heads: usize,
}

impl Attention {
    pub fn load(gpu: &Gpu, wts: &Weights, prefix: &str, heads: usize) -> Result<Self> {
        let l = |n: &str| Linear::load(gpu, wts, &format!("{prefix}.{n}"), DType::F32);
        Ok(Self { q: l("q_proj")?, k: l("k_proj")?, v: l("v_proj")?, o: l("o_proj")?, heads })
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
        let attn = self.attend(gpu, q_in, k_in, v_in, 1, mask)?;
        self.o.forward_residual(gpu, &attn, residual)
    }

    /// [`Self::forward`] without a mask over `batch` independent sequences stacked along
    /// the rows of the inputs: sequence `b` is rows `b * lq..` of `q_in` and `b * lk..`
    /// of `k_in` and `v_in`, and only attends within itself
    pub fn forward_batch(
        &self,
        gpu: &Gpu,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        batch: usize,
    ) -> Result<Tensor> {
        let attn = self.attend(gpu, q_in, k_in, v_in, batch, Mask::default())?;
        self.o.forward(gpu, &attn, Act::None)
    }

    /// [`Self::forward_residual`] over `batch` stacked sequences, as in
    /// [`Self::forward_batch`]
    pub fn forward_residual_batch(
        &self,
        gpu: &Gpu,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        batch: usize,
        residual: &Tensor,
    ) -> Result<()> {
        let attn = self.attend(gpu, q_in, k_in, v_in, batch, Mask::default())?;
        self.o.forward_residual(gpu, &attn, residual)
    }

    /// Merged-head attention output `[batch * lq, dim]`
    fn attend(
        &self,
        gpu: &Gpu,
        q_in: &Tensor,
        k_in: &Tensor,
        v_in: &Tensor,
        batch: usize,
        mask: Mask,
    ) -> Result<Tensor> {
        let q = self.q.forward(gpu, q_in, Act::None)?;
        let k = self.k.forward(gpu, k_in, Act::None)?;
        let v = self.v.forward(gpu, v_in, Act::None)?;
        self.attend_projected(gpu, (&q, 0), (&k, 0), (&v, 0), batch, mask)
    }

    /// [`Self::attend`] from projections already made: q, k and v are this module's
    /// width of columns from the given column of each tensor, so they can be slices of
    /// one fused projection. Each sequence's heads are one launch's batch entries, so
    /// `batch` sequences take `batch` launches over row ranges, with no reshuffling copies
    pub fn attend_projected(
        &self,
        gpu: &Gpu,
        (q, qc): (&Tensor, usize),
        (k, kc): (&Tensor, usize),
        (v, vc): (&Tensor, usize),
        batch: usize,
        mask: Mask,
    ) -> Result<Tensor> {
        let (dim, heads) = (self.q.out_features(), self.heads);
        let hd = dim / heads;
        ensure!(
            qc + dim <= q.cols() && kc + dim <= k.cols() && vc + dim <= v.cols(),
            "attention columns past the projections"
        );
        ensure!(q.rows().is_multiple_of(batch) && k.rows().is_multiple_of(batch), "rows of {batch} sequences");
        ensure!(k.rows() == v.rows(), "{} keys, {} values", k.rows(), v.rows());
        let (lq, lk) = (q.rows() / batch, k.rows() / batch);
        let out = gpu.empty(&[q.rows(), dim], DType::F32)?;
        for b in 0..batch {
            let (qv, kv, vv) = (View::heads(q, qc, hd), View::heads(k, kc, hd), View::heads(v, vc, hd));
            let ov = View::heads(&out, 0, hd);
            let (qv, kv, vv, ov) = (qv.skip(b * lq), kv.skip(b * lk), vv.skip(b * lk), ov.skip(b * lq));
            attention(gpu, qv, kv, vv, ov, heads, lq, lk, (hd, hd), mask)?;
        }
        Ok(out)
    }
}

/// `[n, d]` rows of `a` followed by rows of `b`, FP32
pub fn concat_rows(gpu: &Gpu, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure!(a.cols() == b.cols(), "concat_rows of {:?} and {:?}", a.shape(), b.shape());
    let out = gpu.empty(&[a.rows() + b.rows(), a.cols()], DType::F32)?;
    gpu.dispatch_1d("copy_f32", a.len(), &[out.arg(), a.arg()])?;
    gpu.dispatch_1d("copy_f32", b.len(), &[out.at(a.len()), b.arg()])?;
    Ok(out)
}

/// Rows `idx` of FP32 `x [rows, cols]`
pub fn gather_rows(gpu: &Gpu, x: &Tensor, idx: &[i32]) -> Result<Tensor> {
    let cols = x.cols();
    ensure!(idx.iter().all(|&i| (i as usize) < x.rows()), "gather of rows {idx:?} from {:?}", x.shape());
    let idx = gpu.upload(idx, &[idx.len()], DType::I32)?;
    let out = gpu.empty(&[idx.len(), cols], DType::F32)?;
    gpu.dispatch_1d("gather_rows", out.len(), &[out.arg(), x.arg(), idx.arg(), cols.into()])?;
    Ok(out)
}
