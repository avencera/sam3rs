//! Elementwise kernels: one thread per output element
//!
//! Every kernel here writes only the element at its own thread index, so outputs are
//! `DisjointSlice`s and the borrow checker rules out two threads writing one element.
//! In-place kernels take the buffer once, as the output; none takes the same buffer as
//! both input and output. Index math is 32-bit: the host launches one thread per
//! element and keeps every buffer under 2^32 elements

use cuda_device::{DisjointSlice, kernel, launch_contract, thread};

use crate::common::{Bf16, Elem, activate, math, pack_bf16};

/// `x[r, c] = act(x[r, c] + bias[c])` in place, for rows of `bias.len()`; an empty
/// `bias` only applies the activation
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn bias_act(mut x: DisjointSlice<f32>, bias: &[f32], act: i32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let Some(mut x) = x.element_thread32(t) else { return };
    let mut v = x.read();
    if !bias.is_empty() {
        v += bias[(i % bias.len() as u32) as usize];
    }
    x.write(activate(v, act));
}

/// `out = a + b`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn add(mut out: DisjointSlice<f32>, a: &[f32], b: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    if let Some(mut out) = out.element_thread32(t) {
        out.write(a[i] + b[i]);
    }
}

/// `x += b`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn add_assign(mut x: DisjointSlice<f32>, b: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    if let Some(mut x) = x.element_thread32(t) {
        x.write(x.read() + b[i]);
    }
}

/// `x += b` with a BF16 `b`, the output of a BF16-out GEMM
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn add_assign_bf16(mut x: DisjointSlice<f32>, b: &[Bf16]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    if let Some(mut x) = x.element_thread32(t) {
        x.write(x.read() + b[i].widen());
    }
}

/// `x[r, c] += row[c]` in place, `row` broadcast over the rows of `x`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn add_row(mut x: DisjointSlice<f32>, row: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let c = t.get() % row.len() as u32;
    if let Some(mut x) = x.element_thread32(t) {
        x.write(x.read() + row[c as usize]);
    }
}

/// `x = clamp(x, lo, hi)` in place
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn clamp(mut x: DisjointSlice<f32>, lo: f32, hi: f32) {
    let t = thread::index_1d_u32(lc);
    if let Some(mut x) = x.element_thread32(t) {
        x.write(math::fminf(math::fmaxf(x.read(), lo), hi));
    }
}

/// `mask[i] = x[i] > threshold`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn threshold_u8(mut mask: DisjointSlice<u8>, x: &[f32], threshold: f32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    if let Some(mut mask) = mask.element_thread32(t) {
        mask.write((x[i] > threshold) as u8);
    }
}

/// `out[i] = src[idx[i]]` over rows of `dim` floats
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn gather_rows(mut out: DisjointSlice<f32>, src: &[f32], idx: &[i32], dim: u32) {
    let t = thread::index_1d_u32(lc);
    let (i, c) = (t.get() / dim, t.get() % dim);
    if let Some(mut out) = out.element_thread32(t) {
        out.write(src[(idx[i as usize] as u32 * dim + c) as usize]);
    }
}

/// `x [rows, cols] -> out [cols, rows]`, each thread gathering its output element
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn transpose(mut out: DisjointSlice<f32>, x: &[f32], cols: u32) {
    let t = thread::index_1d_u32(lc);
    let rows = x.len() as u32 / cols;
    let (c, r) = (t.get() / rows, t.get() % rows);
    if let Some(mut out) = out.element_thread32(t) {
        out.write(x[(r * cols + c) as usize]);
    }
}

/// Rows of `cols` floats at row stride `ld` in `src`, into contiguous BF16 `[rows, cols]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn f32_to_bf16(mut dst: DisjointSlice<Bf16>, src: &[f32], cols: u32, ld: u32) {
    let t = thread::index_1d_u32(lc);
    let (r, c) = (t.get() / cols, t.get() % cols);
    if let Some(mut dst) = dst.element_thread32(t) {
        dst.write(Bf16(pack_bf16(src[(r * ld + c) as usize], 0.0) as u16));
    }
}
