// One thread per output element; the host dispatches exactly `n` threads

// uint8 HWC image into float CHW planes
kernel void hwc_u8_to_chw(device float* p_dst, device const uchar* p_src, constant uint& plane,
                          uint i [[thread_position_in_grid]]) {
    uint c = i / plane, p = i % plane;
    p_dst[i] = float(p_src[p * 3 + c]);
}

// antialiased bilinear resize along rows: [channels, in_h, w] -> [channels, out_h, w],
// with taps per output row; sums in tap order like the CUDA kernel
kernel void resample_rows(device float* p_dst, device const float* p_src, device const int* p_starts,
                          device const int* p_counts, device const float* p_weights, constant uint& w,
                          constant uint& in_h, constant uint& out_h, constant uint& max_taps,
                          uint i [[thread_position_in_grid]]) {
    uint x = i % w, y = (i / w) % out_h, c = i / (w * out_h);
    uint start = p_starts[y], cnt = p_counts[y];
    device const float* col = p_src + (c * in_h + start) * w + x;
    device const float* taps = p_weights + y * max_taps;
    float acc = 0.0f;
    for (uint k = 0; k < cnt; k++) acc += taps[k] * col[k * w];
    p_dst[i] = acc;
}

// the same along columns: [rows, in_w] -> [rows, out_w]
kernel void resample_cols(device float* p_dst, device const float* p_src, device const int* p_starts,
                          device const int* p_counts, device const float* p_weights, constant uint& in_w,
                          constant uint& out_w, constant uint& max_taps, uint i [[thread_position_in_grid]]) {
    uint x = i % out_w, r = i / out_w;
    uint start = p_starts[x], cnt = p_counts[x];
    device const float* row = p_src + r * in_w + start;
    device const float* taps = p_weights + x * max_taps;
    float acc = 0.0f;
    for (uint k = 0; k < cnt; k++) acc += taps[k] * row[k];
    p_dst[i] = acc;
}

// rounds to uint8 and scales to [-1, 1] in place: the processor resizes a uint8 tensor
kernel void quantize_normalize(device float* p_x, uint i [[thread_position_in_grid]]) {
    float v = fmin(fmax(rint(p_x[i]), 0.0f), 255.0f);
    p_x[i] = (v / 255.0f - 0.5f) / 0.5f;
}

// pixels [3, size, size] -> patches [(size / patch)^2, 3 * patch * patch], columns in the
// conv weight's order
kernel void patchify(device float* p_dst, device const float* p_src, constant uint& size, constant uint& patch,
                     uint i [[thread_position_in_grid]]) {
    uint grid = size / patch, k = 3 * patch * patch;
    uint tok = i / k, col = i % k;
    uint c = col / (patch * patch), ky = (col / patch) % patch, kx = col % patch;
    uint ty = tok / grid, tx = tok % grid;
    p_dst[i] = p_src[c * size * size + (ty * patch + ky) * size + (tx * patch + kx)];
}

// x[y * grid + x, :] += pos[(y % tile) * tile + x % tile, :]: the absolute position
// embedding tiled from the 24x24 pretraining grid
kernel void add_pos_tiled(device float* p_x, device const float* p_pos, constant uint& grid, constant uint& tile,
                          constant uint& dim, uint i [[thread_position_in_grid]]) {
    uint d = i % dim, tok = i / dim;
    uint y = tok / grid, xx = tok % grid;
    p_x[i] += p_pos[((y % tile) * tile + xx % tile) * dim + d];
}

kernel void add_assign(device float* p_x, device const float* p_b, uint i [[thread_position_in_grid]]) {
    p_x[i] += p_b[i];
}

kernel void add_assign_bf16(device float* p_x, device const bfloat* p_b, uint i [[thread_position_in_grid]]) {
    p_x[i] += float(p_b[i]);
}

kernel void add(device float* p_out, device const float* p_a, device const float* p_b,
                uint i [[thread_position_in_grid]]) {
    p_out[i] = p_a[i] + p_b[i];
}

kernel void f32_to_bf16(device bfloat* p_dst, device const float* p_src, uint i [[thread_position_in_grid]]) {
    p_dst[i] = bfloat(p_src[i]);
}

kernel void bf16_to_f32(device float* p_dst, device const bfloat* p_src, uint i [[thread_position_in_grid]]) {
    p_dst[i] = float(p_src[i]);
}

// x = act(x + bias[col]) in place; an empty bias is skipped (has_bias 0)
kernel void bias_act(device float* p_x, device const float* p_bias, constant uint& cols, constant uint& has_bias,
                     constant int& act, uint i [[thread_position_in_grid]]) {
    float v = p_x[i];
    if (has_bias != 0) v += p_bias[i % cols];
    p_x[i] = activate(v, act);
}

// [rows, cols] -> [cols, rows]
kernel void transpose(device float* p_out, device const float* p_x, constant uint& rows, constant uint& cols,
                      uint i [[thread_position_in_grid]]) {
    uint r = i % rows, c = i / rows;
    p_out[i] = p_x[r * cols + c];
}

kernel void copy_f32(device float* p_dst, device const float* p_src, uint i [[thread_position_in_grid]]) {
    p_dst[i] = p_src[i];
}

// out[r, :] = x[idx[r], :]
kernel void gather_rows(device float* p_out, device const float* p_x, device const int* p_idx, constant uint& cols,
                        uint i [[thread_position_in_grid]]) {
    p_out[i] = p_x[(long)p_idx[i / cols] * cols + i % cols];
}
