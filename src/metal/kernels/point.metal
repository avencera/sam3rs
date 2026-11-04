// SAM 2-style point decoder: mask prompt patches, row broadcasts, the hypernetwork mask
// product and the low-res hole fill

// x[i] += row[i % n]: one row (or a block of rows) repeated over every copy in x
kernel void add_tiled(device float* p_x, device const float* p_row, constant uint& n,
                      uint i [[thread_position_in_grid]]) {
    p_x[i] += p_row[i % n];
}

// out[i] = row[i % n]: the no-mask embedding broadcast, as zeros plus the row
kernel void tile_rows(device float* p_out, device const float* p_row, constant uint& n,
                      uint i [[thread_position_in_grid]]) {
    p_out[i] = p_row[i % n];
}

// x = clamp(x, lo, hi) in place, as fminf(fmaxf(x, lo), hi): NaN becomes lo
kernel void clamp_f32(device float* p_x, constant float& lo, constant float& hi, uint i [[thread_position_in_grid]]) {
    p_x[i] = fmin(fmax(p_x[i], lo), hi);
}

// x [h, w, c] channels-last -> 2x2 patches [(h / 2) * (w / 2), c * 4], columns in the
// conv weight's (channel, ky, kx) order
kernel void patchify2x2(device float* p_out, device const float* p_x, constant uint& w, constant uint& c,
                        uint i [[thread_position_in_grid]]) {
    uint col = i % (c * 4), p = i / (c * 4);
    uint ch = col / 4, ky = (col / 2) % 2, kx = col % 2;
    uint ow = w / 2;
    uint y = p / ow, xx = p % ow;
    p_out[i] = p_x[((2 * y + ky) * w + 2 * xx + kx) * c + ch];
}

// Hole fill of low-res masks, union-find over a batch of [masks, side, side] logits in
// four launches after ECL-CC: point each background pixel (logit <= 0, so NaN is
// foreground) at its first earlier background neighbor, union it with the rest, count
// each root's pixels, then set the pixels of roots of at most max_area to +10. A parent
// always has a smaller index, so chains end at a root; `holes_find` also points each
// pixel it passes at its grandparent. Races change the trees but not the components, and
// the counts are integers, so the output doesn't depend on scheduling

constant constexpr uint HOLES_FOREGROUND = 0xffffffffu;

inline bool holes_background(float v) {
    return v <= 0.0f;
}

// the earlier 8-connected background neighbors of pixel i, smallest index first:
// north-west, north, north-east, west; each neighbor pair is the earlier neighbor of
// exactly one of the two
inline void holes_neighbors(device const float* x, uint i, uint side, thread uint* out) {
    uint c = i % side, r = (i / side) % side;
    for (int k = 0; k < 4; k++) out[k] = HOLES_FOREGROUND;
    if (r > 0) {
        uint up = i - side;
        if (c > 0 && holes_background(x[up - 1])) out[0] = up - 1;
        if (holes_background(x[up])) out[1] = up;
        if (c + 1 < side && holes_background(x[up + 1])) out[2] = up + 1;
    }
    if (c > 0 && holes_background(x[i - 1])) out[3] = i - 1;
}

// the root of a's tree with intermediate pointer jumping; a racing store can undo
// another thread's link only on a pixel that thread found was no longer a root, so it
// goes on to join the trees itself
inline uint holes_find(device atomic_uint* labels, uint a) {
    uint parent = atomic_load_explicit(&labels[a], memory_order_relaxed);
    while (true) {
        uint next = atomic_load_explicit(&labels[parent], memory_order_relaxed);
        if (next >= parent) return parent;
        atomic_store_explicit(&labels[a], next, memory_order_relaxed);
        a = parent;
        parent = next;
    }
}

// joins two trees, hanging the larger root under the smaller; a lost race leaves the
// old parent detached, so the loop joins it in turn
inline void holes_union(device atomic_uint* labels, uint a, uint b) {
    while (true) {
        a = holes_find(labels, a);
        b = holes_find(labels, b);
        if (a == b) return;
        uint hi = max(a, b), lo = min(a, b);
        uint old = atomic_fetch_min_explicit(&labels[hi], lo, memory_order_relaxed);
        if (old == hi) return;
        a = old;
        b = lo;
    }
}

// labels each pixel and zeroes its area slot, which `holes_count` adds into
kernel void holes_init(device uint* p_labels, device uint* p_area, device const float* p_x, constant uint& side,
                       uint i [[thread_position_in_grid]]) {
    p_area[i] = 0;
    if (!holes_background(p_x[i])) {
        p_labels[i] = HOLES_FOREGROUND;
        return;
    }
    uint near[4];
    holes_neighbors(p_x, i, side, near);
    uint first = i;
    for (int k = 0; k < 4; k++) first = min(first, near[k]);
    p_labels[i] = first;
}

// unions each background pixel with its other earlier background neighbors, which
// covers every 8-connected pair once; the first is already its parent
kernel void holes_merge(device atomic_uint* p_labels, device const float* p_x, constant uint& side,
                        uint i [[thread_position_in_grid]]) {
    if (!holes_background(p_x[i])) return;
    uint near[4];
    holes_neighbors(p_x, i, side, near);
    bool linked = false;
    for (int k = 0; k < 4; k++) {
        if (near[k] == HOLES_FOREGROUND) continue;
        if (linked) holes_union(p_labels, i, near[k]);
        linked = true;
    }
}

// points each background pixel at its root and counts the root's pixels; walks without
// shortcuts, since the trees no longer change and a shortcut store from a stale read
// could overwrite a root another thread just wrote
kernel void holes_count(device atomic_uint* p_labels, device atomic_uint* p_area, uint i [[thread_position_in_grid]]) {
    uint a = atomic_load_explicit(&p_labels[i], memory_order_relaxed);
    if (a == HOLES_FOREGROUND) return;
    a = i;
    while (true) {
        uint parent = atomic_load_explicit(&p_labels[a], memory_order_relaxed);
        if (parent == a) break;
        a = parent;
    }
    atomic_store_explicit(&p_labels[i], a, memory_order_relaxed);
    atomic_fetch_add_explicit(&p_area[a], 1u, memory_order_relaxed);
}

kernel void holes_fill(device float* p_x, device const uint* p_labels, device const uint* p_area,
                       constant uint& max_area, uint i [[thread_position_in_grid]]) {
    uint label = p_labels[i];
    if (label != HOLES_FOREGROUND && p_area[label] <= max_area) p_x[i] = 10.0f;
}

// Each prompt's best multimask candidate, picked on the GPU so a click doesn't wait for
// the host between passes: out[b, :] = low[b * 3 + best] with best the first argmax of
// iou[b, 1..4] (mask tokens 1 to 3, the candidates low holds), clamped to [lo, hi] when
// `clamp` is set (NaN to lo) and copied as is otherwise
kernel void best_mask(device float* p_out, device const float* p_low, device const float* p_iou, constant uint& side,
                      constant uint& clamp, constant float& lo, constant float& hi, uint i [[thread_position_in_grid]]) {
    uint b = i / side, p = i % side;
    device const float* iou = p_iou + b * 4;
    uint best = 1;
    for (uint j = 2; j < 4; j++) {
        if (iou[j] > iou[best]) best = j;
    }
    float v = p_low[((long)b * 3 + best - 1) * side + p];
    p_out[i] = clamp != 0 ? fmin(fmax(v, lo), hi) : v;
}

// LayerNorm then `act` over rows of x [rows, cols] in place, one SIMD group a row and 8
// rows a threadgroup: each lane keeps its columns (lane + 32 i, cols at most 32 *
// LN_SIMD_VALS) in registers through both statistics passes and the write, so a row is
// read once and no threadgroup barrier runs. The decoder's upscaled map has 64 channels,
// which a threadgroup a row left mostly idle
constant constexpr uint LN_SIMD_VALS = 8;
constant constexpr uint LN_SIMD_ROWS = 8;

kernel void layernorm_act_simd(device float* p_x, device const float* p_w, device const float* p_b,
                               constant uint& rows, constant uint& cols, constant float& eps, constant int& act,
                               uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * LN_SIMD_ROWS + sg;
    if (r >= rows) return;
    device float* x = p_x + (long)r * cols;
    float v[LN_SIMD_VALS];
    float s = 0.0f;
    for (uint i = 0; i < LN_SIMD_VALS; i++) {
        uint c = lane + 32 * i;
        v[i] = c < cols ? x[c] : 0.0f;
        s += v[i];
    }
    const float mean = simd_sum(s) / float(cols);
    float q = 0.0f;
    for (uint i = 0; i < LN_SIMD_VALS; i++) {
        float d = lane + 32 * i < cols ? v[i] - mean : 0.0f;
        q += d * d;
    }
    const float rstd = rsqrt(simd_sum(q) / float(cols) + eps);
    for (uint i = 0; i < LN_SIMD_VALS; i++) {
        uint c = lane + 32 * i;
        if (c < cols) x[c] = activate((v[i] - mean) * rstd * p_w[c] + p_b[c], act);
    }
}

// The decoder's token MLP heads in one launch, a threadgroup of 256 threads per (head h =
// tg.x, prompt b = tg.y): head h runs weight slot meta.slot, 3 linears 256 -> 256 -> 256
// -> out on token row meta.row of the prompt's n tokens, ReLU between and meta.act after,
// into column meta.col of row b of output meta.dst (out0 or out1, rows ld0 or ld1 apart).
// A slot packs W1 [256, 256], b1, W2, b2, W3 [256, 256] of which the first `out` rows
// are used, b3. Each SIMD group computes outputs sg, sg + 8, ... with its lanes across
// the inputs, so weight rows read coalesced
constant constexpr uint HEAD_D = 256;
constant constexpr uint HEAD_SLOT = 3 * HEAD_D * HEAD_D + 3 * HEAD_D;

struct HeadMeta {
    uint slot, row, out, act, dst, col;
};

inline void head_layer(device const float* w, device const float* bias, threadgroup const float* in,
                       threadgroup float* out, uint outs, int act, ushort sg, ushort lane) {
    for (uint j = sg; j < outs; j += HEAD_D / 32) {
        device const float* wr = w + j * HEAD_D;
        float acc = 0.0f;
        for (uint i = lane; i < HEAD_D; i += 32) acc = fma(wr[i], in[i], acc);
        acc = simd_sum(acc);
        if (lane == 0) out[j] = activate(acc + bias[j], act);
    }
}

kernel void decoder_heads(device float* p_out0, device float* p_out1, device const float* p_tokens,
                          device const float* p_w, device const int* p_meta, constant uint& n,
                          constant uint& ld0, constant uint& ld1,
                          uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float a[HEAD_D], h[HEAD_D];
    const uint b = tg.y;
    // the host uploads the rows as I32, which the argument check matches to int
    const HeadMeta meta = ((device const HeadMeta*)p_meta)[tg.x];
    device const float* w = p_w + (long)meta.slot * HEAD_SLOT;
    a[tid] = p_tokens[((long)b * n + meta.row) * HEAD_D + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    head_layer(w, w + HEAD_D * HEAD_D, a, h, HEAD_D, ACT_RELU, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w += HEAD_D * HEAD_D + HEAD_D;
    head_layer(w, w + HEAD_D * HEAD_D, h, a, HEAD_D, ACT_RELU, sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w += HEAD_D * HEAD_D + HEAD_D;
    head_layer(w, w + HEAD_D * HEAD_D, a, h, meta.out, int(meta.act), sg, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < meta.out) {
        if (meta.dst == 0) p_out0[(long)b * ld0 + meta.col + tid] = h[tid];
        else p_out1[(long)b * ld1 + meta.col + tid] = h[tid];
    }
}

// Every mask of every prompt in one pass over the upscaled map: out[b, j, p] = hyper[b,
// j] . up[b, p] for j < masks, hyper rows hyper_ld apart per prompt and c wide
kernel void mask_dots(device float* p_out, device const float* p_hyper, device const float* p_up, constant uint& side,
                      constant uint& c, constant uint& masks, constant uint& hyper_ld,
                      uint i [[thread_position_in_grid]]) {
    const uint b = i / side, p = i % side;
    device const float* u = p_up + (long)i * c;
    device const float* hb = p_hyper + (long)b * hyper_ld;
    for (uint j = 0; j < masks; j++) {
        float acc = 0.0f;
        for (uint k = 0; k < c; k++) acc = fma(hb[j * c + k], u[k], acc);
        p_out[((long)b * masks + j) * side + p] = acc;
    }
}
