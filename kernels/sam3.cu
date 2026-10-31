// SAM 3 kernels. Feature maps are channels-last ([H*W, C] row-major) so 1x1 convs and
// linears are plain GEMMs; everything here is the glue cuBLAS does not cover

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define DEV __device__ __forceinline__

DEV float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

DEV float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// block-wide sum; `scratch` holds 32 floats
DEV float block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = lane < nwarps ? scratch[lane] : 0.f;
    return warp_sum(v);
}

DEV float block_max(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = lane < nwarps ? scratch[lane] : -INFINITY;
    return warp_max(v);
}

enum Act { ACT_NONE = 0, ACT_RELU = 1, ACT_GELU = 2, ACT_GELU_TANH = 3, ACT_SIGMOID = 4 };

DEV float activate(float x, int act) {
    switch (act) {
        case ACT_RELU: return fmaxf(x, 0.f);
        case ACT_GELU: return 0.5f * x * (1.f + erff(x * 0.70710678118654752f));
        case ACT_GELU_TANH: {
            float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.f + tanhf(u));
        }
        case ACT_SIGMOID: return 1.f / (1.f + expf(-x));
        default: return x;
    }
}

// x[r, c] = act(x[r, c] + bias[c]); bias may be null
extern "C" __global__ void bias_act(float* x, const float* bias, long long rows, int cols, int act) {
    long long n = rows * cols;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        float v = x[i];
        if (bias) v += bias[i % cols];
        x[i] = activate(v, act);
    }
}

// out = a + b, elementwise
extern "C" __global__ void add(float* out, const float* a, const float* b, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = a[i] + b[i];
}

// out[r, c] = a[r, c] + b[c % b_len... ] with b broadcast over rows: b has `cols` entries
extern "C" __global__ void add_row(float* out, const float* a, const float* b, long long rows, int cols) {
    long long n = rows * cols;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = a[i] + b[i % cols];
}

// LayerNorm over the last dim, one block per row; out may alias x. If `residual` is
// non-null it is added to x first and the sum is written back to `residual`
DEV void store(float* p, float v) { *p = v; }
DEV void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// one block per row; the bf16 variant feeds a BF16 GEMM directly
template <typename T>
DEV void layernorm_row(T* out, const float* x, const float* w, const float* b, int cols, float eps) {
    __shared__ float scratch[32];
    long long row = blockIdx.x;
    const float* xr = x + row * cols;
    T* yr = out + row * cols;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c];
    float mean = block_sum(s, scratch) / cols;
    float v = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = xr[c] - mean;
        v += d * d;
    }
    float rstd = rsqrtf(block_sum(v, scratch) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float y = (xr[c] - mean) * rstd;
        store(yr + c, w ? y * w[c] + (b ? b[c] : 0.f) : y);
    }
}

extern "C" __global__ void layernorm(float* out, const float* x, const float* w, const float* b, int cols, float eps) {
    layernorm_row(out, x, w, b, cols, eps);
}

extern "C" __global__ void layernorm_bf16(__nv_bfloat16* out, const float* x, const float* w, const float* b, int cols,
                                          float eps) {
    layernorm_row(out, x, w, b, cols, eps);
}

// Row softmax over x[rows, n] in place. Row r is query q = r % lq of batch g = r / lq.
// Optional additive bias[g / bias_div][q][j], key mask (nonzero = masked) shared by all
// rows, and causal masking (j > q)
extern "C" __global__ void softmax(float* x, int n, int lq, const float* bias, int bias_div,
                                   const uint8_t* key_mask, int causal) {
    __shared__ float scratch[32];
    long long r = blockIdx.x;
    int q = r % lq;
    long long g = r / lq;
    float* xr = x + r * n;
    const float* br = bias ? bias + ((g / bias_div) * lq + q) * (long long)n : nullptr;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = xr[j];
        if (br) v += br[j];
        if ((key_mask && key_mask[j]) || (causal && j > q)) v = -INFINITY;
        xr[j] = v;
        m = fmaxf(m, v);
    }
    m = block_max(m, scratch);
    float s = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float e = m == -INFINITY ? 0.f : __expf(xr[j] - m);
        xr[j] = e;
        s += e;
    }
    s = block_sum(s, scratch);
    float inv = s > 0.f ? 1.f / s : 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) xr[j] *= inv;
}

// ---------------------------------------------------------------------------------------
// preprocessing

// Antialiased bilinear resize along one axis (PyTorch's _upsample_bilinear2d_aa).
// src [C, in_len along axis, other], weights precomputed on the host per output index:
// taps[o] = (start, count), w[o * max_taps + k]
extern "C" __global__ void resample_rows(float* dst, const float* src, int channels, int in_h, int w,
                                         int out_h, const int* starts, const int* counts, const float* weights,
                                         int max_taps) {
    long long n = (long long)channels * out_h * w;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int x = i % w;
        int y = (i / w) % out_h;
        int c = i / ((long long)w * out_h);
        int s = starts[y], cnt = counts[y];
        const float* col = src + (long long)c * in_h * w + x;
        float acc = 0.f;
        for (int k = 0; k < cnt; k++) acc += weights[y * max_taps + k] * col[(long long)(s + k) * w];
        dst[i] = acc;
    }
}

extern "C" __global__ void resample_cols(float* dst, const float* src, int rows, int in_w, int out_w,
                                         const int* starts, const int* counts, const float* weights, int max_taps) {
    long long n = (long long)rows * out_w;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int x = i % out_w;
        long long r = i / out_w;
        int s = starts[x], cnt = counts[x];
        const float* row = src + r * in_w;
        float acc = 0.f;
        for (int k = 0; k < cnt; k++) acc += weights[x * max_taps + k] * row[s + k];
        dst[i] = acc;
    }
}

// uint8 HWC image -> float CHW planes
extern "C" __global__ void hwc_u8_to_chw(float* dst, const uint8_t* src, int h, int w) {
    long long n = (long long)h * w * 3;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = i % 3;
        long long p = i / 3;
        dst[c * (long long)h * w + p] = src[i];
    }
}

// the processor resizes a uint8 tensor, so values are rounded back to uint8 before
// scaling to [-1, 1]
extern "C" __global__ void quantize_normalize(float* x, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        float v = fminf(fmaxf(rintf(x[i]), 0.f), 255.f);
        x[i] = (v / 255.f - 0.5f) / 0.5f;
    }
}

// ---------------------------------------------------------------------------------------
// vision backbone

// pixels [3, S, S] -> patches [(S/p)^2, 3*p*p], column order matching the conv weight
extern "C" __global__ void patchify(float* dst, const float* src, int size, int patch) {
    int grid = size / patch;
    int k = 3 * patch * patch;
    long long n = (long long)grid * grid * k;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int col = i % k;
        long long tok = i / k;
        int c = col / (patch * patch), ky = (col / patch) % patch, kx = col % patch;
        int ty = tok / grid, tx = tok % grid;
        dst[i] = src[(long long)c * size * size + (long long)(ty * patch + ky) * size + tx * patch + kx];
    }
}

// x[y*grid + x, :] += pos[(y % tile) * tile + x % tile, :] (absolute position embedding
// tiled from the 24x24 pretraining grid)
extern "C" __global__ void add_pos_tiled(float* x, const float* pos, int grid, int tile, int dim) {
    long long n = (long long)grid * grid * dim;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int d = i % dim;
        long long tok = i / dim;
        int y = tok / grid, xx = tok % grid;
        x[i] += pos[((y % tile) * tile + xx % tile) * (long long)dim + d];
    }
}

// qkv [grid*grid, 3*heads*hd] -> q, k, v [windows*heads, win*win, hd], partitioning into
// windows (win == grid for global attention) and applying interleaved axial RoPE to q
// and k. cos/sin tables are [win*win, hd/2]
template <typename In, typename Out>
DEV void split_rope(Out* q, Out* k, Out* v, const In* qkv, const float* cos_t, const float* sin_t, int grid, int win,
                    int heads, int hd) {
    int nwx = grid / win;
    int dim = heads * hd;
    long long n = (long long)grid * grid * dim / 2;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int pair = i % (dim / 2);
        long long tok = i / (dim / 2);
        int h = pair / (hd / 2), p = pair % (hd / 2);
        int y = tok / grid, x = tok % grid;
        int wy = y / win, wx = x / win, ly = y % win, lx = x % win;
        int wtok = ly * win + lx;
        long long dst = (((long long)(wy * nwx + wx) * heads + h) * win * win + wtok) * hd + 2 * p;
        const In* src = qkv + tok * 3 * dim + h * hd + 2 * p;
        float c = cos_t[wtok * (hd / 2) + p], s = sin_t[wtok * (hd / 2) + p];
        float q0 = (float)src[0], q1 = (float)src[1];
        store(q + dst, q0 * c - q1 * s);
        store(q + dst + 1, q0 * s + q1 * c);
        float k0 = (float)src[dim], k1 = (float)src[dim + 1];
        store(k + dst, k0 * c - k1 * s);
        store(k + dst + 1, k0 * s + k1 * c);
        store(v + dst, (float)src[2 * dim]);
        store(v + dst + 1, (float)src[2 * dim + 1]);
    }
}

extern "C" __global__ void vit_split_rope(float* q, float* k, float* v, const float* qkv, const float* cos_t,
                                          const float* sin_t, int grid, int win, int heads, int hd) {
    split_rope(q, k, v, qkv, cos_t, sin_t, grid, win, heads, hd);
}

extern "C" __global__ void vit_split_rope_bf16(__nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
                                               const __nv_bfloat16* qkv, const float* cos_t, const float* sin_t,
                                               int grid, int win, int heads, int hd) {
    split_rope(q, k, v, qkv, cos_t, sin_t, grid, win, heads, hd);
}

// o [windows*heads, win*win, hd] -> [grid*grid, heads*hd], undoing the window partition
template <typename T>
DEV void merge_heads(T* out, const float* o, int grid, int win, int heads, int hd) {
    int nwx = grid / win;
    int dim = heads * hd;
    long long n = (long long)grid * grid * dim;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int d = i % hd;
        int h = (i / hd) % heads;
        long long tok = i / dim;
        int y = tok / grid, x = tok % grid;
        int wy = y / win, wx = x / win, ly = y % win, lx = x % win;
        store(out + i, o[(((long long)(wy * nwx + wx) * heads + h) * win * win + ly * win + lx) * hd + d]);
    }
}

extern "C" __global__ void vit_merge_heads(float* out, const float* o, int grid, int win, int heads, int hd) {
    merge_heads(out, o, grid, win, heads, hd);
}

extern "C" __global__ void vit_merge_heads_bf16(__nv_bfloat16* out, const float* o, int grid, int win, int heads, int hd) {
    merge_heads(out, o, grid, win, heads, hd);
}

// ---------------------------------------------------------------------------------------
// convolutions as GEMMs

// ConvTranspose2d(k=2, s=2): g [H*W, O*4] holds x @ W for W [C, O, 2, 2]; scatter to
// y [2H*2W, O] and add bias
extern "C" __global__ void convt2x2_scatter(float* y, const float* g, const float* bias, int h, int w, int o) {
    long long n = (long long)4 * h * w * o;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int oc = i % o;
        long long p = i / o;
        int yy = p / (2 * w), xx = p % (2 * w);
        int sy = yy >> 1, sx = xx >> 1, ky = yy & 1, kx = xx & 1;
        y[i] = g[((long long)sy * w + sx) * o * 4 + oc * 4 + ky * 2 + kx] + (bias ? bias[oc] : 0.f);
    }
}

// x [H*W, C] -> zero-padded xp [(H+2)*(W+2) + 2, C]; xp must be zeroed first. With the
// padded row width, the input of every 3x3 tap is one contiguous matrix at a fixed offset
template <typename T>
DEV void pad1_impl(T* xp, const float* x, int h, int w, int c) {
    long long n = (long long)h * w * c;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int ch = i % c;
        long long p = i / c;
        int yy = p / w, xx = p % w;
        store(xp + ((long long)(yy + 1) * (w + 2) + xx + 1) * c + ch, x[i]);
    }
}

extern "C" __global__ void pad1(float* xp, const float* x, int h, int w, int c) { pad1_impl(xp, x, h, w, c); }


__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// 3x3, padding 1 im2col to BF16 for output pixels [p0, p0 + n): out[p - p0, t * C + ic]
// is x at pixel p shifted by tap t = dy * 3 + dx (offsets dy - 1, dx - 1), zero outside
// the image, so the conv is one GEMM with K = 9C. C must be a multiple of 8
extern "C" __global__ void im2col3x3_bf16(__nv_bfloat16* out, const float* x, int h, int w, int c, long long p0,
                                          long long n) {
    int c8 = c / 8;
    long long total = n * 9 * c8;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int ch = (int)(i % c8) * 8;
        long long r = i / c8;
        int t = (int)(r % 9);
        long long p = p0 + r / 9;
        int yy = (int)(p / w) + t / 3 - 1, xx = (int)(p % w) + t % 3 - 1;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
            const float4* src = reinterpret_cast<const float4*>(x + ((long long)yy * w + xx) * c + ch);
            float4 a = src[0], b = src[1];
            v = make_uint4(pack_bf16(a.x, a.y), pack_bf16(a.z, a.w), pack_bf16(b.x, b.y), pack_bf16(b.z, b.w));
        }
        *reinterpret_cast<uint4*>(out + r * c + ch) = v;
    }
}

// im2col3x3_bf16 of fine + up2(coarse) without materializing that sum: the input at fine
// pixel (y, x) is fine[y, x] + coarse[y / 2, x / 2], as upsample2_add computes it. With
// stats, coarse first gets the group norm and activation group_norm_apply would apply,
// so neither the coarse map's normalized copy nor the upsampled sum is ever written.
// h and w are the fine map's (even) size; groups must divide C into multiples of 8
extern "C" __global__ void im2col3x3_up_bf16(__nv_bfloat16* out, const float* fine, const float* coarse,
                                             const float* stats, const float* gw, const float* gb, int groups,
                                             float eps, int act, int h, int w, int c, long long p0, long long n) {
    int c8 = c / 8, cg = c / groups;
    float count = (float)((long long)(h / 2) * (w / 2) * cg);
    long long total = n * 9 * c8;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int ch = (int)(i % c8) * 8;
        long long r = i / c8;
        int t = (int)(r % 9);
        long long p = p0 + r / 9;
        int yy = (int)(p / w) + t / 3 - 1, xx = (int)(p % w) + t % 3 - 1;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
            const float4* f = reinterpret_cast<const float4*>(fine + ((long long)yy * w + xx) * c + ch);
            const float4* q =
                reinterpret_cast<const float4*>(coarse + ((long long)(yy >> 1) * (w >> 1) + (xx >> 1)) * c + ch);
            float4 f0 = f[0], f1 = f[1], q0 = q[0], q1 = q[1];
            float a[8] = {f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w};
            float u[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            if (stats) {
                int g = ch / cg;
                float mean = stats[g * 2] / count;
                float var = fmaxf(stats[g * 2 + 1] / count - mean * mean, 0.f);
#pragma unroll
                for (int k = 0; k < 8; k++) u[k] = activate((u[k] - mean) * rsqrtf(var + eps) * gw[ch + k] + gb[ch + k], act);
            }
#pragma unroll
            for (int k = 0; k < 8; k++) a[k] = a[k] + u[k];
            v = make_uint4(pack_bf16(a[0], a[1]), pack_bf16(a[2], a[3]), pack_bf16(a[4], a[5]), pack_bf16(a[6], a[7]));
        }
        *reinterpret_cast<uint4*>(out + r * c + ch) = v;
    }
}

// yp [H*(W+2), O] (3x3 conv over padded rows) -> y [H*W, O] + bias
extern "C" __global__ void crop_bias(float* y, const float* yp, const float* bias, int h, int w, int o) {
    long long n = (long long)h * w * o;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int oc = i % o;
        long long p = i / o;
        int yy = p / w, xx = p % w;
        y[i] = yp[((long long)yy * (w + 2) + xx) * o + oc] + (bias ? bias[oc] : 0.f);
    }
}

// x [H*W, C] -> [C, H*W]
extern "C" __global__ void transpose(float* out, const float* x, long long rows, int cols) {
    long long n = rows * cols;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = i % cols;
        long long r = i / cols;
        out[(long long)c * rows + r] = x[i];
    }
}

// ---------------------------------------------------------------------------------------
// text encoder

// out[t] = tok_table[ids[t]] + pos_table[t]
extern "C" __global__ void text_embed(float* out, const float* tok_table, const float* pos_table, const int* ids,
                                      int len, int dim) {
    long long n = (long long)len * dim;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int d = i % dim, t = i / dim;
        out[i] = tok_table[(long long)ids[t] * dim + d] + pos_table[(long long)t * dim + d];
    }
}

// ---------------------------------------------------------------------------------------
// detector

// torchvision roi_align (aligned=False, spatial_scale=1, adaptive sampling ratio) over a
// channels-last feature map; out [n_boxes, C * pool * pool] in (c, ph, pw) order, the
// flattening the 7x7 pooling conv weight expects. boxes are xyxy in feature pixels
DEV float roi_bilinear(const float* f, int h, int w, int c, int ch, float y, float x) {
    if (y < -1.f || y > h || x < -1.f || x > w) return 0.f;
    if (y <= 0.f) y = 0.f;
    if (x <= 0.f) x = 0.f;
    int y0 = (int)y, x0 = (int)x, y1, x1;
    if (y0 >= h - 1) { y1 = y0 = h - 1; y = (float)y0; } else y1 = y0 + 1;
    if (x0 >= w - 1) { x1 = x0 = w - 1; x = (float)x0; } else x1 = x0 + 1;
    float ly = y - y0, lx = x - x0, hy = 1.f - ly, hx = 1.f - lx;
    float v1 = f[((long long)y0 * w + x0) * c + ch], v2 = f[((long long)y0 * w + x1) * c + ch];
    float v3 = f[((long long)y1 * w + x0) * c + ch], v4 = f[((long long)y1 * w + x1) * c + ch];
    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

extern "C" __global__ void roi_align(float* out, const float* feat, const float* boxes, int n_boxes, int h, int w,
                                     int c, int pool) {
    long long n = (long long)n_boxes * c * pool * pool;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int pw = i % pool, ph = (i / pool) % pool;
        int ch = (i / (pool * pool)) % c;
        int b = i / ((long long)pool * pool * c);
        const float* box = boxes + b * 4;
        float sx = box[0], sy = box[1];
        float rw = fmaxf(box[2] - sx, 1.f), rh = fmaxf(box[3] - sy, 1.f);
        float bh = rh / pool, bw = rw / pool;
        int gh = (int)ceilf(rh / pool), gw = (int)ceilf(rw / pool);
        float count = fmaxf((float)(gh * gw), 1.f);
        float acc = 0.f;
        for (int iy = 0; iy < gh; iy++) {
            float y = sy + ph * bh + (iy + .5f) * bh / gh;
            for (int ix = 0; ix < gw; ix++) {
                float x = sx + pw * bw + (ix + .5f) * bw / gw;
                acc += roi_bilinear(feat, h, w, c, ch, y, x);
            }
        }
        out[i] = acc / count;
    }
}

// log-scaled distances from each grid coordinate to a box's two edges along one axis:
// out[(q * side + i) * 2 + e] for reference boxes cxcywh [nq, 4]; axis 0 = x, 1 = y
extern "C" __global__ void rpb_deltas(float* out, const float* boxes, int nq, int side, int axis) {
    long long n = (long long)nq * side * 2;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int e = i % 2;
        int pos = (i / 2) % side;
        int q = i / (2 * side);
        const float* b = boxes + q * 4;
        float center = b[axis], size = b[axis + 2];
        float edge = e == 0 ? center - 0.5f * size : center + 0.5f * size;
        float d = ((float)pos / side - edge) * 8.f;
        float s = (d > 0.f) - (d < 0.f);
        out[i] = s * log2f(fabsf(d) + 1.f) / 3.f;
    }
}

// attention bias [heads, 1 + nq, side * side]: row 0 (presence token) is zero, row 1 + q
// is dy[h, q, y] + dx[h, q, x]
extern "C" __global__ void rpb_combine(float* bias, const float* dx, const float* dy, int nq, int side, int heads) {
    long long hw = (long long)side * side;
    long long n = (long long)heads * (nq + 1) * hw;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int p = i % hw;
        int row = (i / hw) % (nq + 1);
        int h = i / (hw * (nq + 1));
        if (row == 0) {
            bias[i] = 0.f;
        } else {
            int q = row - 1, y = p / side, x = p % side;
            long long base = ((long long)h * nq + q) * side;
            bias[i] = dy[base + y] + dx[base + x];
        }
    }
}

// gen_sineembed_for_position for cxcywh boxes [n, 4] -> [n, 512] (y, x, w, h blocks of 128)
extern "C" __global__ void box_sine_embed(float* out, const float* boxes, int n) {
    long long total = (long long)n * 512;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        int f = i % 128, block = (i / 128) % 4;
        int q = i / 512;
        // block order y, x, w, h maps to box components 1, 0, 2, 3
        int comp = block == 0 ? 1 : (block == 1 ? 0 : block);
        float v = boxes[q * 4 + comp] * 6.283185307179586f;
        float dim_t = powf(10000.f, (float)(2 * (f / 2)) / 128.f);
        float a = v / dim_t;
        out[i] = (f % 2 == 0) ? sinf(a) : cosf(a);
    }
}

DEV float inverse_sigmoid(float x) {
    x = fminf(fmaxf(x, 0.f), 1.f);
    float x1 = fmaxf(x, 1e-3f), x2 = fmaxf(1.f - x, 1e-3f);
    return logf(x1 / x2);
}

// out = sigmoid(delta + inverse_sigmoid(ref)), elementwise over [n, 4]
extern "C" __global__ void box_refine(float* out, const float* ref, const float* delta, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * 4; i += gridDim.x * blockDim.x) {
        float v = delta[i] + inverse_sigmoid(ref[i]);
        out[i] = 1.f / (1.f + expf(-v));
    }
}

// masked mean over rows of x [len, dim] (mask nonzero = padding) -> out [dim]
extern "C" __global__ void masked_mean(float* out, const float* x, const uint8_t* mask, int len, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float s = 0.f, cnt = 0.f;
    for (int t = 0; t < len; t++)
        if (!mask[t]) { s += x[(long long)t * dim + d]; cnt += 1.f; }
    out[d] = s / fmaxf(cnt, 1.f);
}

// scores[q] = clamp(dot(hs[q], p) * scale, -clampv, clampv), one warp per query
extern "C" __global__ void dot_scores(float* scores, const float* hs, const float* p, int nq, int dim, float scale,
                                      float clampv) {
    int q = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (q >= nq) return;
    float s = 0.f;
    for (int d = lane; d < dim; d += 32) s += hs[(long long)q * dim + d] * p[d];
    s = warp_sum(s);
    if (lane == 0) scores[q] = fminf(fmaxf(s * scale, -clampv), clampv);
}

// GroupNorm statistics for channels-last x [hw, c]: one block per (group, chunk) writes
// its partial sums to part[(group * gridDim.x + chunk) * 2 + {0, 1}]
extern "C" __global__ void group_stats(float* part, const float* x, long long hw, int c, int groups, int chunk) {
    __shared__ float scratch[32];
    int g = blockIdx.y;
    int cg = c / groups;
    long long start = (long long)blockIdx.x * chunk;
    long long end = min(start + chunk, hw);
    float s = 0.f, s2 = 0.f;
    for (long long p = start; p < end; p++) {
        for (int k = threadIdx.x; k < cg; k += blockDim.x) {
            float v = x[p * c + g * cg + k];
            s += v;
            s2 += v * v;
        }
    }
    s = block_sum(s, scratch);
    s2 = block_sum(s2, scratch);
    if (threadIdx.x == 0) {
        long long i = ((long long)g * gridDim.x + blockIdx.x) * 2;
        part[i] = s;
        part[i + 1] = s2;
    }
}

// Sums group_stats' partials into stats[group * 2 + {0, 1}], one block per group. A fixed
// order instead of atomics, so the statistics don't change with block scheduling: atomic
// adds flipped mask pixels between runs whose only difference was timing
extern "C" __global__ void group_stats_reduce(float* stats, const float* part, int blocks) {
    __shared__ float scratch[32];
    const float* p = part + (long long)blockIdx.x * blocks * 2;
    float s = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        s += p[i * 2];
        s2 += p[i * 2 + 1];
    }
    s = block_sum(s, scratch);
    s2 = block_sum(s2, scratch);
    if (threadIdx.x == 0) {
        stats[blockIdx.x * 2] = s;
        stats[blockIdx.x * 2 + 1] = s2;
    }
}

extern "C" __global__ void group_norm_apply(float* x, const float* stats, const float* w, const float* b, long long hw,
                                            int c, int groups, float eps, int act) {
    long long n = hw * c;
    int cg = c / groups;
    float count = (float)(hw * cg);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int ch = i % c, g = ch / cg;
        float mean = stats[g * 2] / count;
        float var = fmaxf(stats[g * 2 + 1] / count - mean * mean, 0.f);
        float v = (x[i] - mean) * rsqrtf(var + eps) * w[ch] + b[ch];
        x[i] = activate(v, act);
    }
}

// out [2h * 2w, c] = fine + nearest-upsampled coarse [h * w, c]
extern "C" __global__ void upsample2_add(float* out, const float* fine, const float* coarse, int h, int w, int c) {
    long long n = (long long)4 * h * w * c;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int ch = i % c;
        long long p = i / c;
        int y = p / (2 * w), x = p % (2 * w);
        out[i] = fine[i] + coarse[((long long)(y >> 1) * w + (x >> 1)) * c + ch];
    }
}

// bilinear resize (align_corners=False, no antialias) of n planes [n, ih, iw] -> [n, oh, ow]
extern "C" __global__ void resize_bilinear(float* out, const float* in, int n, int ih, int iw, int oh, int ow) {
    long long total = (long long)n * oh * ow;
    float sh = (float)ih / oh, sw = (float)iw / ow;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        int x = i % ow, y = (i / ow) % oh;
        long long p = i / ((long long)ow * oh);
        float fy = fmaxf((y + 0.5f) * sh - 0.5f, 0.f), fx = fmaxf((x + 0.5f) * sw - 0.5f, 0.f);
        int y0 = (int)fy, x0 = (int)fx;
        int y1 = y0 + (y0 < ih - 1), x1 = x0 + (x0 < iw - 1);
        float ly = fy - y0, lx = fx - x0;
        const float* src = in + p * ih * iw;
        out[i] = (1.f - ly) * ((1.f - lx) * src[y0 * iw + x0] + lx * src[y0 * iw + x1]) +
                 ly * ((1.f - lx) * src[y1 * iw + x0] + lx * src[y1 * iw + x1]);
    }
}

// mask[i] = x[i] > threshold
extern "C" __global__ void threshold_u8(uint8_t* mask, const float* x, long long n, float threshold) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        mask[i] = x[i] > threshold;
}

// rows[i] = src[idx[i]] for row-major [*, dim]
extern "C" __global__ void gather_rows(float* out, const float* src, const int* idx, int n, int dim) {
    long long total = (long long)n * dim;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x)
        out[i] = src[(long long)idx[i / dim] * dim + i % dim];
}

// ---------------------------------------------------------------------------------------
// SAM 2 point decoder

// non-overlapping 2x2 patches of channels-last x [h * w, c] -> [(h/2) * (w/2), c * 4] in
// (c, ky, kx) order, so a stride-2 2x2 conv is one GEMM with the [O, C, 2, 2] weight
extern "C" __global__ void patchify2x2(float* out, const float* x, int h, int w, int c) {
    int oh = h / 2, ow = w / 2;
    long long n = (long long)oh * ow * c * 4;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int col = i % (c * 4);
        long long p = i / (c * 4);
        int ch = col / 4, ky = (col / 2) % 2, kx = col % 2;
        int y = p / ow, xx = p % ow;
        out[i] = x[((long long)(2 * y + ky) * w + 2 * xx + kx) * c + ch];
    }
}

extern "C" __global__ void clamp(float* x, long long n, float lo, float hi) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        x[i] = fminf(fmaxf(x[i], lo), hi);
}

// ---------------------------------------------------------------------------------------
// fused attention (FlashAttention-2 style) for the BF16 precision mode
//
// FP32 q/k/v are rounded to BF16 (V to FP16) as they are staged in shared memory, scores
// and the softmax run in FP32 with an online max, and P @ V runs on FP16 tensor cores,
// accumulating each 64-key tile in FP16 before adding it to FP32 totals. Each warp owns
// 16 query rows; a block walks all keys in tiles of 64, so the [lq, lk] score matrix
// never reaches global memory

constexpr int FA_WARPS = 4;
constexpr int FA_BN = 64;

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// FP16 inputs and FP16 accumulation: twice the FP32-accumulate rate on GeForce parts
__device__ __forceinline__ void mma_f16(uint32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// rows [0, ROWS) of a [ROWS, D] tile -> bf16 smem with row stride lds; rows past `valid`
// are zero. FP32 sources are rounded here; 16-bit sources copy 16 bytes at a time
template <int D, int ROWS = 64>
__device__ __forceinline__ void fa_stage(__nv_bfloat16* dst, int lds, const float* src, long long ld, int valid) {
    for (int i = threadIdx.x; i < ROWS * D / 4; i += blockDim.x) {
        int r = i / (D / 4), c = (i % (D / 4)) * 4;
        float4 x = r < valid ? *reinterpret_cast<const float4*>(src + r * ld + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        uint32_t* d = reinterpret_cast<uint32_t*>(dst + r * lds + c);
        d[0] = pack_bf16(x.x, x.y);
        d[1] = pack_bf16(x.z, x.w);
    }
}

template <int D, int ROWS = 64>
__device__ __forceinline__ void fa_stage(__nv_bfloat16* dst, int lds, const __nv_bfloat16* src, long long ld,
                                         int valid) {
    for (int i = threadIdx.x; i < ROWS * D / 8; i += blockDim.x) {
        int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 x = r < valid ? *reinterpret_cast<const uint4*>(src + r * ld + c) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dst + r * lds + c) = x;
    }
}

// the same into FP16, for V
template <int D, int ROWS = 64>
__device__ __forceinline__ void fa_stage(__half* dst, int lds, const float* src, long long ld, int valid) {
    for (int i = threadIdx.x; i < ROWS * D / 4; i += blockDim.x) {
        int r = i / (D / 4), c = (i % (D / 4)) * 4;
        float4 x = r < valid ? *reinterpret_cast<const float4*>(src + r * ld + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        uint32_t* d = reinterpret_cast<uint32_t*>(dst + r * lds + c);
        d[0] = pack_f16(x.x, x.y);
        d[1] = pack_f16(x.z, x.w);
    }
}

template <int D, int ROWS = 64>
__device__ __forceinline__ void fa_stage(__half* dst, int lds, const __nv_bfloat16* src, long long ld, int valid) {
    for (int i = threadIdx.x; i < ROWS * D / 8; i += blockDim.x) {
        int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 x = r < valid ? *reinterpret_cast<const uint4*>(src + r * ld + c) : make_uint4(0, 0, 0, 0);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&x);
        uint4 y;
        uint32_t* yy = reinterpret_cast<uint32_t*>(&y);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(h[j]);
            yy[j] = pack_f16(f.x, f.y);
        }
        *reinterpret_cast<uint4*>(dst + r * lds + c) = y;
    }
}

template <int D, int ROWS = 64>
__device__ __forceinline__ void fa_stage(__half* dst, int lds, const __half* src, long long ld, int valid) {
    for (int i = threadIdx.x; i < ROWS * D / 8; i += blockDim.x) {
        int r = i / (D / 8), c = (i % (D / 8)) * 8;
        uint4 x = r < valid ? *reinterpret_cast<const uint4*>(src + r * ld + c) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dst + r * lds + c) = x;
    }
}

// Batch entry blockIdx.y reads q + y * q_stride etc.; row i of each view starts at
// i * ld. scale_log2 = log2(e) / sqrt(D). Keys with key_pad[j] != 0 are skipped; rows
// with no valid key produce zeros, like the softmax kernel
//
// With RPB, the box relative position bias of the DETR decoder's image cross-attention is
// added to the scores without materializing it: for head h = blockIdx.y, query row 1 + q
// and key y * side + x it is dy[h, q, y] + dx[h, q, x], and row 0 (presence token) gets
// none. side must be even, so each thread's key pair shares a grid row
//
// With `part`, block z covers keys [z * kv_chunk, (z + 1) * kv_chunk) only and writes its
// unnormalized accumulators, row max and row sum to part[z][batch][row][D + 2] for
// flash_combine, so few query rows over many keys still fill the GPU
//
// WARPS sets the query rows per block, 16 per warp: every block re-reads all of K and V,
// so taller blocks cut that traffic. TK and TV let K and V arrive already in 16 bits
template <int D, typename T, bool RPB = false, int WARPS = FA_WARPS, typename TK = T, typename TV = T>
__device__ void flash_attn(float* o, long long o_ld, long long o_stride, const T* q, long long q_ld,
                           long long q_stride, const TK* k, long long k_ld, long long k_stride, const TV* v,
                           long long v_ld, long long v_stride, int lq, int lk, float scale_log2,
                           const uint8_t* key_pad, const float* rpb_dx = nullptr, const float* rpb_dy = nullptr,
                           int side = 0, float* part = nullptr, int kv_chunk = 0) {
    constexpr int BM = 16 * WARPS;
    constexpr int LDS = D + 8;  // padding keeps fragment loads bank-conflict free
    __shared__ __align__(16) __nv_bfloat16 qs[BM * LDS];
    __shared__ __align__(16) __nv_bfloat16 ks[FA_BN * LDS];
    __shared__ __align__(16) __half vs[FA_BN * LDS];

    int b = blockIdx.y, q0 = blockIdx.x * BM;
    q += b * q_stride + q0 * q_ld;
    k += b * k_stride;
    v += b * v_stride;
    o += b * o_stride;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane / 4, t = lane % 4, wr = warp * 16;

    fa_stage<D, BM>(qs, LDS, q, q_ld, lq - q0);
    __syncthreads();
    uint32_t qa[D / 16][4];
#pragma unroll
    for (int kk = 0; kk < D / 16; kk++) {
        const __nv_bfloat16* p = qs + (wr + g) * LDS + kk * 16 + t * 2;
        qa[kk][0] = *reinterpret_cast<const uint32_t*>(p);
        qa[kk][1] = *reinterpret_cast<const uint32_t*>(p + 8 * LDS);
        qa[kk][2] = *reinterpret_cast<const uint32_t*>(p + 8);
        qa[kk][3] = *reinterpret_cast<const uint32_t*>(p + 8 * LDS + 8);
    }

    float acc[D / 8][4] = {};
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    // per fragment row: this head's dx and dy rows, or null for the presence token and
    // rows past lq
    const float* rdx[2] = {nullptr, nullptr};
    const float* rdy[2] = {nullptr, nullptr};
    if constexpr (RPB) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int row = q0 + wr + g + r * 8;
            if (row >= 1 && row < lq) {
                long long base = ((long long)b * (lq - 1) + row - 1) * side;
                rdx[r] = rpb_dx + base;
                rdy[r] = rpb_dy + base;
            }
        }
    }
    int k_begin = 0, k_end = lk;
    if (part) {
        k_begin = blockIdx.z * kv_chunk;
        k_end = min(lk, k_begin + kv_chunk);
    }
    for (int k0 = k_begin; k0 < k_end; k0 += FA_BN) {
        __syncthreads();
        fa_stage<D>(ks, LDS, k + k0 * k_ld, k_ld, k_end - k0);
        fa_stage<D>(vs, LDS, v + k0 * v_ld, v_ld, k_end - k0);
        __syncthreads();

        float s[FA_BN / 8][4] = {};
#pragma unroll
        for (int j = 0; j < FA_BN / 8; j += 2) {
#pragma unroll
            for (int kk = 0; kk < D / 16; kk++) {
                // matrices: keys j*8.. at k and k+8, then keys (j+1)*8.. at k and k+8
                int mi = lane / 8, ri = lane % 8;
                const __nv_bfloat16* p = ks + ((j + mi / 2) * 8 + ri) * LDS + kk * 16 + (mi & 1) * 8;
                uint32_t bf[4];
                ldmatrix_x4(bf, p);
                mma_bf16(s[j], qa[kk], bf);
                mma_bf16(s[j + 1], qa[kk], bf + 2);
            }
        }

        float corr[2];
        // a whole tile of real keys without padding or bias: the softmax needs no masking
        // or infinity guards, and the scale folds into the exponent's FMA. A row max of
        // -inf only comes from the first tile, where corr = ex2(-inf) = 0 is right
        if (!RPB && !key_pad && k0 + FA_BN <= k_end) {
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int j = 0; j < FA_BN / 8; j++) {
                mx[0] = fmaxf(mx[0], fmaxf(s[j][0], s[j][1]));
                mx[1] = fmaxf(mx[1], fmaxf(s[j][2], s[j][3]));
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
                mx[r] = fmaxf(m[r], mx[r] * scale_log2);
                corr[r] = ex2(m[r] - mx[r]);
                m[r] = mx[r];
                l[r] *= corr[r];
            }
#pragma unroll
            for (int j = 0; j < FA_BN / 8; j++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    float p = ex2(fmaf(s[j][e], scale_log2, -m[e / 2]));
                    s[j][e] = p;
                    l[e / 2] += p;
                }
            }
        } else {
            float mx[2] = {m[0], m[1]};
            // the grid position (gy, gx) of this thread's first key pair, k0 + j * 8 + t * 2,
            // tracked as j advances instead of dividing per score
            int gy = 0, gx = 0;
            if constexpr (RPB) {
                gy = k0 / side;
                gx = k0 - gy * side + t * 2;
                while (gx >= side) gx -= side, gy++;
            }
#pragma unroll
            for (int j = 0; j < FA_BN / 8; j++) {
                float2 bias[2] = {};
                if constexpr (RPB) {
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        if (rdx[r] && k0 + j * 8 + t * 2 < lk) {
                            float dy = __ldg(rdy[r] + gy);
                            float2 dx = __ldg(reinterpret_cast<const float2*>(rdx[r] + gx));
                            bias[r] = make_float2(dy + dx.x, dy + dx.y);
                        }
                    }
                    gx += 8;
                    while (gx >= side) gx -= side, gy++;
                }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    int col = k0 + j * 8 + t * 2 + (e & 1);
                    float x = s[j][e] * scale_log2;
                    // scaled here rather than with the loads, so it contracts into the same FMA
                    if constexpr (RPB) x += (e & 1 ? bias[e / 2].y : bias[e / 2].x) * 1.4426950408889634f;
                    if (col >= k_end || (key_pad && key_pad[col])) x = -INFINITY;
                    s[j][e] = x;
                    mx[e / 2] = fmaxf(mx[e / 2], x);
                }
            }
#pragma unroll
            for (int r = 0; r < 2; r++) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
                corr[r] = mx[r] == -INFINITY ? 1.f : ex2(m[r] - mx[r]);
                m[r] = mx[r];
                l[r] *= corr[r];
            }
#pragma unroll
            for (int j = 0; j < FA_BN / 8; j++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    float p = m[e / 2] == -INFINITY ? 0.f : ex2(s[j][e] - m[e / 2]);
                    s[j][e] = p;
                    l[e / 2] += p;
                }
            }
        }
#pragma unroll
        for (int dj = 0; dj < D / 8; dj++) {
            acc[dj][0] *= corr[0];
            acc[dj][1] *= corr[0];
            acc[dj][2] *= corr[1];
            acc[dj][3] *= corr[1];
        }

        // the score fragments of two adjacent 8-key tiles form the A fragment of P. P @ V
        // runs in FP16 with FP16 accumulation over this tile's 64 keys (p <= 1, so the
        // partial sums stay within 64 |v|), then joins the FP32 accumulator
        uint32_t pacc[D / 8][2] = {};
#pragma unroll
        for (int kk = 0; kk < FA_BN / 16; kk++) {
            uint32_t pa[4] = {pack_f16(s[2 * kk][0], s[2 * kk][1]), pack_f16(s[2 * kk][2], s[2 * kk][3]),
                              pack_f16(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                              pack_f16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int dj = 0; dj < D / 8; dj += 2) {
                int mi = lane / 8, ri = lane % 8;
                const __half* p = vs + (kk * 16 + (mi & 1) * 8 + ri) * LDS + (dj + mi / 2) * 8;
                uint32_t bf[4];
                ldmatrix_x4_trans(bf, p);
                mma_f16(pacc[dj], pa, bf);
                mma_f16(pacc[dj + 1], pa, bf + 2);
            }
        }
#pragma unroll
        for (int dj = 0; dj < D / 8; dj++) {
            float2 lo = __half22float2(*reinterpret_cast<__half2*>(&pacc[dj][0]));
            float2 hi = __half22float2(*reinterpret_cast<__half2*>(&pacc[dj][1]));
            acc[dj][0] += lo.x;
            acc[dj][1] += lo.y;
            acc[dj][2] += hi.x;
            acc[dj][3] += hi.y;
        }
    }

#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffff, l[r], 2);
    }
    if (part) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int row = q0 + wr + g + r * 8;
            if (row >= lq) continue;
            float* out = part + (((long long)blockIdx.z * gridDim.y + b) * lq + row) * (D + 2);
#pragma unroll
            for (int dj = 0; dj < D / 8; dj++) {
                *reinterpret_cast<float2*>(out + dj * 8 + t * 2) = make_float2(acc[dj][2 * r], acc[dj][2 * r + 1]);
            }
            if (t == 0) {
                out[D] = m[r];
                out[D + 1] = l[r];
            }
        }
        return;
    }
#pragma unroll
    for (int r = 0; r < 2; r++) l[r] = l[r] > 0.f ? 1.f / l[r] : 0.f;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        int row = q0 + wr + g + r * 8;
        if (row >= lq) continue;
        float* out = o + row * o_ld;
#pragma unroll
        for (int dj = 0; dj < D / 8; dj++) {
            *reinterpret_cast<float2*>(out + dj * 8 + t * 2) =
                make_float2(acc[dj][2 * r] * l[r], acc[dj][2 * r + 1] * l[r]);
        }
    }
}

#define FLASH_ATTN(NAME, D, T)                                                                                     \
    extern "C" __global__ void __launch_bounds__(FA_WARPS * 32)                                                    \
        NAME(float* o, long long o_ld, long long o_stride, const T* q, long long q_ld, long long q_stride, const T* k, \
             long long k_ld, long long k_stride, const T* v, long long v_ld, long long v_stride, int lq, int lk,     \
             float scale_log2, const uint8_t* key_pad, float* part, int kv_chunk) {                                \
        flash_attn<D, T>(o, o_ld, o_stride, q, q_ld, q_stride, k, k_ld, k_stride, v, v_ld, v_stride, lq, lk,       \
                         scale_log2, key_pad, nullptr, nullptr, 0, part, kv_chunk);                                \
    }
FLASH_ATTN(flash_attn_d32, 32, float)
FLASH_ATTN(flash_attn_d64, 64, float)
FLASH_ATTN(flash_attn_bf16_d64, 64, __nv_bfloat16)

extern "C" __global__ void __launch_bounds__(FA_WARPS * 32)
    flash_attn_rpb_d32(float* o, long long o_ld, long long o_stride, const float* q, long long q_ld, long long q_stride,
                       const float* k, long long k_ld, long long k_stride, const float* v, long long v_ld,
                       long long v_stride, int lq, int lk, float scale_log2, const float* rpb_dx, const float* rpb_dy,
                       int side, float* part, int kv_chunk) {
    flash_attn<32, float, true>(o, o_ld, o_stride, q, q_ld, q_stride, k, k_ld, k_stride, v, v_ld, v_stride, lq, lk,
                                scale_log2, nullptr, rpb_dx, rpb_dy, side, part, kv_chunk);
}

// FP32 q with K and V converted once by kv_to_16, in 128-row blocks
extern "C" __global__ void __launch_bounds__(256)
    flash_attn_kv16_d32(float* o, long long o_ld, long long o_stride, const float* q, long long q_ld, long long q_stride,
                        const __nv_bfloat16* k, long long k_ld, long long k_stride, const __half* v, long long v_ld,
                        long long v_stride, int lq, int lk, float scale_log2, const uint8_t* key_pad) {
    flash_attn<32, float, false, 8, __nv_bfloat16, __half>(o, o_ld, o_stride, q, q_ld, q_stride, k, k_ld, k_stride, v,
                                                           v_ld, v_stride, lq, lk, scale_log2, key_pad);
}

// batch strided FP32 views k and v [batch][lk, hd] -> contiguous kd (BF16) and vd (FP16)
// [lk, batch * hd], 8 values per thread, for flash_attn_kv16: rounding them once instead
// of in every query block
extern "C" __global__ void kv_to_16(__nv_bfloat16* kd, __half* vd, const float* k, long long k_ld, long long k_stride,
                                    const float* v, long long v_ld, long long v_stride, int lk, int batch, int hd) {
    long long n = (long long)lk * batch * hd / 8;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        long long e = i * 8;
        int d = e % hd, b = (e / hd) % batch;
        long long row = e / ((long long)hd * batch);
        const float* ks = k + b * k_stride + row * k_ld + d;
        const float* vs = v + b * v_stride + row * v_ld + d;
        float4 k0 = *reinterpret_cast<const float4*>(ks), k1 = *reinterpret_cast<const float4*>(ks + 4);
        float4 v0 = *reinterpret_cast<const float4*>(vs), v1 = *reinterpret_cast<const float4*>(vs + 4);
        *reinterpret_cast<uint4*>(kd + e) = make_uint4(pack_bf16(k0.x, k0.y), pack_bf16(k0.z, k0.w),
                                                       pack_bf16(k1.x, k1.y), pack_bf16(k1.z, k1.w));
        *reinterpret_cast<uint4*>(vd + e) = make_uint4(pack_f16(v0.x, v0.y), pack_f16(v0.z, v0.w),
                                                       pack_f16(v1.x, v1.y), pack_f16(v1.z, v1.w));
    }
}

// merges the key splits of a split flash_attn: one warp per (batch, row), lane d < D
// owns output column d. Rows with no valid key in any split produce zeros
template <int D>
__device__ void flash_combine(float* o, long long o_ld, long long o_stride, const float* part, int splits, int batch,
                              int lq) {
    int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (w >= batch * lq) return;
    int b = w / lq, row = w % lq;
    long long step = (long long)batch * lq * (D + 2);
    const float* p = part + (long long)w * (D + 2);
    float mx = -INFINITY;
    for (int z = 0; z < splits; z++) mx = fmaxf(mx, p[z * step + D]);
    float l = 0.f, acc[D / 32] = {};
    if (mx != -INFINITY) {
        for (int z = 0; z < splits; z++) {
            const float* ps = p + z * step;
            float mz = ps[D];
            if (mz == -INFINITY) continue;
            float c = exp2f(mz - mx);
            l += c * ps[D + 1];
#pragma unroll
            for (int i = 0; i < D / 32; i++) acc[i] += c * ps[lane + i * 32];
        }
    }
    float inv = l > 0.f ? 1.f / l : 0.f;
    float* out = o + b * o_stride + row * o_ld;
#pragma unroll
    for (int i = 0; i < D / 32; i++) out[lane + i * 32] = acc[i] * inv;
}

extern "C" __global__ void flash_combine_d32(float* o, long long o_ld, long long o_stride, const float* part,
                                             int splits, int batch, int lq) {
    flash_combine<32>(o, o_ld, o_stride, part, splits, batch, lq);
}

// rows of `cols` floats with row stride `ld` -> contiguous bf16 [rows, cols]
extern "C" __global__ void f32_to_bf16(__nv_bfloat16* dst, const float* src, long long rows, int cols, long long ld) {
    long long n = rows * cols;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        dst[i] = __float2bfloat16(src[(i / cols) * ld + i % cols]);
}
