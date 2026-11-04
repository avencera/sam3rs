// Text embedding, detector geometry, scoring and the pixel decoder's group norm. One
// thread per output element unless a kernel says otherwise; the semantics follow the
// CUDA kernels in crates/sam3rs-kernels/src/{detector,norm,conv}.rs

// out[t, d] = tok[ids[t], d] + pos[t, d] over [ids.len(), dim]
kernel void text_embed(device float* p_out, device const float* p_tok, device const float* p_pos,
                       device const int* p_ids, constant uint& dim, uint i [[thread_position_in_grid]]) {
    uint row = i / dim, d = i % dim;
    p_out[i] = p_tok[(long)p_ids[row] * dim + d] + p_pos[i];
}

// torchvision's bilinear sample of channel ch at (y, x) of a channels-last [h, w, c] map
inline float roi_bilinear(device const float* f, int h, int w, uint c, uint ch, float y, float x) {
    if (y < -1.0f || y > float(h) || x < -1.0f || x > float(w)) return 0.0f;
    if (y <= 0.0f) y = 0.0f;
    if (x <= 0.0f) x = 0.0f;
    int y0 = int(y), x0 = int(x), y1, x1;
    if (y0 >= h - 1) {
        y0 = h - 1;
        y1 = y0;
        y = float(y0);
    } else {
        y1 = y0 + 1;
    }
    if (x0 >= w - 1) {
        x0 = w - 1;
        x1 = x0;
        x = float(x0);
    } else {
        x1 = x0 + 1;
    }
    float ly = y - float(y0), lx = x - float(x0);
    float hy = 1.0f - ly, hx = 1.0f - lx;
    float v1 = f[(uint(y0) * uint(w) + uint(x0)) * c + ch];
    float v2 = f[(uint(y0) * uint(w) + uint(x1)) * c + ch];
    float v3 = f[(uint(y1) * uint(w) + uint(x0)) * c + ch];
    float v4 = f[(uint(y1) * uint(w) + uint(x1)) * c + ch];
    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

// torchvision roi_align (aligned=False, spatial_scale=1, adaptive sampling ratio) over a
// channels-last [h, w, c] map: out [n_boxes, c * pool * pool] in (c, ph, pw) order, the
// flattening the pooling projection expects; boxes are xyxy in feature pixels
kernel void roi_align(device float* p_out, device const float* p_feat, device const float* p_boxes,
                      constant uint& h, constant uint& w, constant uint& c, constant uint& pool,
                      uint i [[thread_position_in_grid]]) {
    uint pw = i % pool, ph = (i / pool) % pool, ch = (i / (pool * pool)) % c, b = i / (pool * pool * c);
    float sx = p_boxes[b * 4], sy = p_boxes[b * 4 + 1];
    float rw = fmax(p_boxes[b * 4 + 2] - sx, 1.0f);
    float rh = fmax(p_boxes[b * 4 + 3] - sy, 1.0f);
    float bh = rh / float(pool), bw = rw / float(pool);
    int gh = int(ceil(rh / float(pool))), gw = int(ceil(rw / float(pool)));
    float count = fmax(float(gh * gw), 1.0f);
    float acc = 0.0f;
    for (int iy = 0; iy < gh; iy++) {
        float y = sy + float(ph) * bh + (float(iy) + 0.5f) * bh / float(gh);
        for (int ix = 0; ix < gw; ix++) {
            float x = sx + float(pw) * bw + (float(ix) + 0.5f) * bw / float(gw);
            acc += roi_bilinear(p_feat, int(h), int(w), c, ch, y, x);
        }
    }
    p_out[i] = acc / count;
}

// log-scaled distances from each grid coordinate to a box's two edges along one axis:
// out[(q * side + pos) * 2 + edge] for cxcywh boxes [nq, 4]; axis 0 is x, 1 is y
// The box relative position bias MLP, fused with its input: for query q and grid position
// pos along `axis`, the log-scaled distances from pos to the box's two edges on that axis
// go through Linear(2, hidden), ReLU and Linear(hidden, RPB_HEADS) in registers, and the
// biases are stored head-major as [RPB_HEADS, rows] so the attention kernel reads a
// query's biases contiguously. One thread per (q, pos) row; the hidden layer never
// leaves registers
constant constexpr uint RPB_HEADS = 8;

kernel void rpb_mlp(device float* p_out, device const float* p_boxes, device const float* p_w1,
                    device const float* p_b1, device const float* p_w2, device const float* p_b2,
                    constant uint& side, constant uint& axis, constant uint& hidden, constant uint& rows,
                    uint i [[thread_position_in_grid]]) {
    uint pos = i % side, q = i / side;
    float center = p_boxes[q * 4 + axis], size = p_boxes[q * 4 + axis + 2];
    float d[2];
    for (uint e = 0; e < 2; ++e) {
        float edge = e == 0 ? center - 0.5f * size : center + 0.5f * size;
        float v = (float(pos) / float(side) - edge) * 8.0f;
        float s = float(v > 0.0f) - float(v < 0.0f);
        d[e] = s * precise::log2(fabs(v) + 1.0f) / 3.0f;
    }
    float acc[RPB_HEADS];
    for (uint h = 0; h < RPB_HEADS; ++h) acc[h] = p_b2[h];
    for (uint j = 0; j < hidden; ++j) {
        float v = fmax(fma(p_w1[2 * j], d[0], fma(p_w1[2 * j + 1], d[1], p_b1[j])), 0.0f);
        for (uint h = 0; h < RPB_HEADS; ++h) acc[h] = fma(p_w2[h * hidden + j], v, acc[h]);
    }
    for (uint h = 0; h < RPB_HEADS; ++h) p_out[h * rows + i] = acc[h];
}

// gen_sineembed_for_position for cxcywh boxes [n, 4] -> [n, 512], in y, x, w, h blocks
// of 128 interleaved sin/cos features
kernel void box_sine_embed(device float* p_out, device const float* p_boxes, uint i [[thread_position_in_grid]]) {
    uint f = i % 128, block = (i / 128) % 4, q = i / 512;
    // block order y, x, w, h maps to box components 1, 0, 2, 3
    uint comp = block == 0 ? 1 : (block == 1 ? 0 : block);
    float v = p_boxes[q * 4 + comp] * (2.0f * M_PI_F);
    float dim_t = precise::pow(10000.0f, float(2 * (f / 2)) / 128.0f);
    float a = v / dim_t;
    p_out[i] = f % 2 == 0 ? precise::sin(a) : precise::cos(a);
}

inline float inverse_sigmoid(float x) {
    x = fmin(fmax(x, 0.0f), 1.0f);
    return precise::log(fmax(x, 1e-3f) / fmax(1.0f - x, 1e-3f));
}

// out = sigmoid(delta + inverse_sigmoid(reference)) over [n, 4]
kernel void box_refine(device float* p_out, device const float* p_ref, device const float* p_delta,
                       uint i [[thread_position_in_grid]]) {
    float v = p_delta[i] + inverse_sigmoid(p_ref[i]);
    p_out[i] = 1.0f / (1.0f + precise::exp(-v));
}

// out [dim] = mean of the rows of x [len, dim] whose mask byte is zero (nonzero is padding)
kernel void masked_mean(device float* p_out, device const float* p_x, device const uchar* p_mask,
                        constant uint& dim, constant uint& len, uint i [[thread_position_in_grid]]) {
    float s = 0.0f, cnt = 0.0f;
    for (uint t = 0; t < len; t++) {
        if (p_mask[t] == 0) {
            s += p_x[t * dim + i];
            cnt += 1.0f;
        }
    }
    p_out[i] = s / fmax(cnt, 1.0f);
}

// scores[q] = clamp(dot(hs[q], p) * scale, -clampv, clampv), one SIMD group per query
// over hs [n, dim] and p [dim]
kernel void dot_scores(device float* p_scores, device const float* p_hs, device const float* p_p, constant uint& n,
                       constant uint& dim, constant float& scale, constant float& clampv,
                       uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint simds [[simdgroups_per_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    uint q = group * simds + simd;
    // the whole SIMD group leaves together, so simd_sum below sees every lane
    if (q >= n) return;
    float s = 0.0f;
    for (uint d = lane; d < dim; d += 32) s += p_hs[q * dim + d] * p_p[d];
    s = simd_sum(s);
    if (lane == 0) p_scores[q] = fmin(fmax(s * scale, -clampv), clampv);
}

// out [2h * 2w, c] = fine + nearest-upsampled coarse [h * w, c], with w the coarse width
kernel void upsample2_add(device float* p_out, device const float* p_fine, device const float* p_coarse,
                          constant uint& w, constant uint& c, uint i [[thread_position_in_grid]]) {
    uint ch = i % c, p = i / c;
    uint y = p / (2 * w), x = p % (2 * w);
    p_out[i] = p_fine[i] + p_coarse[((y >> 1) * w + (x >> 1)) * c + ch];
}

// GroupNorm partial sums of channels-last x [hw, c]: threadgroup (chunk, group) writes
// part[(group * chunks + chunk) * 2 + {0, 1}], the sum and sum of squares of its pixels.
// SIMD group s takes pixels start + s, start + s + simds, ... and lanes the group's
// channels, so each load is a contiguous row segment
kernel void group_stats(device float* p_part, device const float* p_x, constant uint& hw, constant uint& c,
                        constant uint& groups, constant uint& chunk, uint3 tg [[threadgroup_position_in_grid]],
                        uint3 ntg [[threadgroups_per_grid]], uint3 tpt [[threads_per_threadgroup]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    uint g = tg.y, cg = c / groups;
    uint simd = tid / 32, lane = tid % 32, simds = tpt.x / 32;
    uint start = tg.x * chunk, end = min(start + chunk, hw);
    device const float* x = p_x + g * cg;
    float s = 0.0f, s2 = 0.0f;
    for (uint p = start + simd; p < end; p += simds) {
        for (uint k = lane; k < cg; k += 32) {
            float v = x[p * c + k];
            s += v;
            s2 += v * v;
        }
    }
    s = group_sum(s, scratch, tid, tpt.x);
    s2 = group_sum(s2, scratch, tid, tpt.x);
    if (tid == 0) {
        uint at = (g * ntg.x + tg.x) * 2;
        p_part[at] = s;
        p_part[at + 1] = s2;
    }
}

// sums group_stats' `blocks` partial pairs per group into stats [groups, 2], one
// threadgroup per group, in a fixed order so the statistics don't vary between runs
kernel void group_stats_reduce(device float* p_stats, device const float* p_part, constant uint& blocks,
                               uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint threads [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* pairs = p_part + g * 2 * blocks;
    float s = 0.0f, s2 = 0.0f;
    for (uint i = tid; i < blocks; i += threads) {
        s += pairs[i * 2];
        s2 += pairs[i * 2 + 1];
    }
    s = group_sum(s, scratch, tid, threads);
    s2 = group_sum(s2, scratch, tid, threads);
    if (tid == 0) {
        p_stats[g * 2] = s;
        p_stats[g * 2 + 1] = s2;
    }
}

// GroupNorm of x [hw, c] in place from group_stats_reduce's sums, with affine weights and
// an activation
kernel void group_norm_apply(device float* p_x, device const float* p_stats, device const float* p_w,
                             device const float* p_b, constant uint& hw, constant uint& c, constant uint& groups,
                             constant float& eps, constant int& act, uint i [[thread_position_in_grid]]) {
    uint ch = i % c, cg = c / groups, g = ch / cg;
    float count = float(hw * cg);
    float mean = p_stats[g * 2] / count;
    float var = fmax(p_stats[g * 2 + 1] / count - mean * mean, 0.0f);
    float v = (p_x[i] - mean) * rsqrt(var + eps) * p_w[ch] + p_b[ch];
    p_x[i] = activate(v, act);
}
