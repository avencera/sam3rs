// Normalizations: one threadgroup per row, its threads taking columns tid, tid + threads,
// ... as the CUDA kernels do, with the two-pass mean and variance

// LayerNorm statistics of one row as (mean, rstd); `load(c)` reads column c
template <typename Load>
inline float2 ln_stats(Load load, uint cols, float eps, threadgroup float* scratch, uint tid, uint threads) {
    float s = 0.0f;
    for (uint c = tid; c < cols; c += threads) s += load(c);
    float mean = group_sum(s, scratch, tid, threads) / float(cols);
    float v = 0.0f;
    for (uint c = tid; c < cols; c += threads) {
        float d = load(c) - mean;
        v += d * d;
    }
    float rstd = rsqrt(group_sum(v, scratch, tid, threads) / float(cols) + eps);
    return float2(mean, rstd);
}

// out = LayerNorm(x) per row, FP32 or BF16 out; x may be written by the caller first
template <typename TO>
inline void ln_row(device const float* x, device TO* out, device const float* w, device const float* b, uint cols,
                   float eps, threadgroup float* scratch, uint tid, uint threads) {
    float2 st = ln_stats([&](uint c) { return x[c]; }, cols, eps, scratch, tid, threads);
    for (uint c = tid; c < cols; c += threads) out[c] = TO((x[c] - st.x) * st.y * w[c] + b[c]);
}

kernel void layernorm(device float* p_out, device const float* p_x, device const float* p_w,
                      device const float* p_b, constant uint& cols, constant float& eps,
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    ln_row(p_x + row * cols, p_out + row * cols, p_w, p_b, cols, eps, scratch, tid, threads);
}

kernel void layernorm_bf16(device bfloat* p_out, device const float* p_x, device const float* p_w,
                           device const float* p_b, constant uint& cols, constant float& eps,
                           uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    ln_row(p_x + row * cols, p_out + row * cols, p_w, p_b, cols, eps, scratch, tid, threads);
}

kernel void layernorm_inplace(device float* p_x, device const float* p_w, device const float* p_b,
                              constant uint& cols, constant float& eps, uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device float* x = p_x + row * cols;
    float2 st = ln_stats([&](uint c) { return x[c]; }, cols, eps, scratch, tid, threads);
    for (uint c = tid; c < cols; c += threads) x[c] = (x[c] - st.x) * st.y * p_w[c] + p_b[c];
}

// x += add (a residual branch in T), then LayerNorm of the sum into T; each thread adds
// the columns it later normalizes, so the statistics need no extra barrier
template <typename T>
kernel void add_layernorm(device T* p_out, device float* p_x, device const T* p_add, device const float* p_w,
                          device const float* p_b, constant uint& cols, constant float& eps,
                          uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device float* x = p_x + row * cols;
    device const T* a = p_add + row * cols;
    for (uint c = tid; c < cols; c += threads) x[c] += float(a[c]);
    ln_row(x, p_out + row * cols, p_w, p_b, cols, eps, scratch, tid, threads);
}

#define ADD_LN(name, T)                                                                                          \
    template [[host_name(name)]] kernel void add_layernorm<T>(device T*, device float*, device const T*,        \
                                                              device const float*, device const float*,        \
                                                              constant uint&, constant float&, uint, uint, uint);
ADD_LN("add_layernorm_bf16", bfloat)
ADD_LN("add_layernorm_f32", float)
