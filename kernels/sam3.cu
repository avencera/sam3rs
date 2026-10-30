// SAM 3 kernels. Feature maps are channels-last ([H*W, C] row-major) so 1x1 convs and
// linears are plain GEMMs; everything here is the glue cuBLAS does not cover

#include <cstdint>
#include <cuda_bf16.h>

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
extern "C" __global__ void layernorm(float* out, const float* x, const float* w, const float* b, int cols, float eps) {
    __shared__ float scratch[32];
    long long row = blockIdx.x;
    const float* xr = x + row * cols;
    float* yr = out + row * cols;
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
        yr[c] = w ? y * w[c] + (b ? b[c] : 0.f) : y;
    }
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
extern "C" __global__ void vit_split_rope(float* q, float* k, float* v, const float* qkv, const float* cos_t,
                                          const float* sin_t, int grid, int win, int heads, int hd) {
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
        const float* src = qkv + tok * 3 * dim + h * hd + 2 * p;
        float c = cos_t[wtok * (hd / 2) + p], s = sin_t[wtok * (hd / 2) + p];
        float q0 = src[0], q1 = src[1];
        q[dst] = q0 * c - q1 * s;
        q[dst + 1] = q0 * s + q1 * c;
        float k0 = src[dim], k1 = src[dim + 1];
        k[dst] = k0 * c - k1 * s;
        k[dst + 1] = k0 * s + k1 * c;
        v[dst] = src[2 * dim];
        v[dst + 1] = src[2 * dim + 1];
    }
}

// o [windows*heads, win*win, hd] -> [grid*grid, heads*hd], undoing the window partition
extern "C" __global__ void vit_merge_heads(float* out, const float* o, int grid, int win, int heads, int hd) {
    int nwx = grid / win;
    int dim = heads * hd;
    long long n = (long long)grid * grid * dim;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int d = i % hd;
        int h = (i / hd) % heads;
        long long tok = i / dim;
        int y = tok / grid, x = tok % grid;
        int wy = y / win, wx = x / win, ly = y % win, lx = x % win;
        out[i] = o[(((long long)(wy * nwx + wx) * heads + h) * win * win + ly * win + lx) * hd + d];
    }
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
extern "C" __global__ void pad1(float* xp, const float* x, int h, int w, int c) {
    long long n = (long long)h * w * c;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int ch = i % c;
        long long p = i / c;
        int yy = p / w, xx = p % w;
        xp[((long long)(yy + 1) * (w + 2) + xx + 1) * c + ch] = x[i];
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
// is dy[q, y, h] + dx[q, x, h]
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
            bias[i] = dy[((long long)q * side + y) * heads + h] + dx[((long long)q * side + x) * heads + h];
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

// GroupNorm statistics for channels-last x [hw, c]: one block per (group, chunk) adds
// partial sums into stats[group * 2 + {0, 1}] (zeroed first)
extern "C" __global__ void group_stats(float* stats, const float* x, long long hw, int c, int groups, int chunk) {
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
        atomicAdd(&stats[g * 2], s);
        atomicAdd(&stats[g * 2 + 1], s2);
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
