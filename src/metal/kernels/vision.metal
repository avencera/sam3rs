// Vision backbone layout kernels: the BF16 blocks run in window-major token order, where
// each 24x24 window's tokens are contiguous rows in raster order inside it

constant constexpr uint VIT_GRID = 72;
constant constexpr uint VIT_WIN = 24;

inline uint window_to_raster(uint r) {
    constexpr uint NWX = VIT_GRID / VIT_WIN;
    uint w = r / (VIT_WIN * VIT_WIN), l = r % (VIT_WIN * VIT_WIN);
    return ((w / NWX) * VIT_WIN + l / VIT_WIN) * VIT_GRID + (w % NWX) * VIT_WIN + l % VIT_WIN;
}

inline uint raster_to_window(uint tok) {
    constexpr uint NWX = VIT_GRID / VIT_WIN;
    uint y = tok / VIT_GRID, x = tok % VIT_GRID;
    return ((y / VIT_WIN) * NWX + x / VIT_WIN) * VIT_WIN * VIT_WIN + (y % VIT_WIN) * VIT_WIN + x % VIT_WIN;
}

// reorders rows of x [72^2, cols] from raster to window-major order with to_window
// nonzero, and back otherwise
kernel void vit_window_order(device float* p_out, device const float* p_x, constant uint& cols,
                             constant uint& to_window, uint i [[thread_position_in_grid]]) {
    uint row = i / cols, c = i % cols;
    uint src = to_window != 0 ? window_to_raster(row) : raster_to_window(row);
    p_out[i] = p_x[src * cols + c];
}

// rotates the q and k columns of qkv [tokens, 3 * dim] by RoPE into q and k [tokens, dim],
// one pair of each per thread, in FP32 with one rounding; windowed blocks use the window
// tables
template <typename T>
kernel void vit_rope_qk(device T* p_q, device T* p_k, device const T* p_qkv, device const float* p_cos,
                        device const float* p_sin, constant uint& dim, constant uint& hd, constant uint& windowed,
                        uint p [[thread_position_in_grid]]) {
    uint row = p / (dim / 2), c = (p % (dim / 2)) * 2;
    uint pos = windowed != 0 ? row % (VIT_WIN * VIT_WIN) : row;
    uint tab = pos * (hd / 2) + (c % hd) / 2;
    float cs = p_cos[tab], sn = p_sin[tab];
    device const T* src = p_qkv + row * 3 * dim + c;
    float qa = src[0], qb = src[1], ka = src[dim], kb = src[dim + 1];
    p_q[row * dim + c] = T(qa * cs - qb * sn);
    p_q[row * dim + c + 1] = T(qa * sn + qb * cs);
    p_k[row * dim + c] = T(ka * cs - kb * sn);
    p_k[row * dim + c + 1] = T(ka * sn + kb * cs);
}

#define VIT_ROPE_QK(name, T)                                                                                     \
    template [[host_name(name)]] kernel void vit_rope_qk<T>(device T*, device T*, device const T*,               \
                                                            device const float*, device const float*,           \
                                                            constant uint&, constant uint&, constant uint&, uint);
VIT_ROPE_QK("vit_rope_qk_bf16", bfloat)
VIT_ROPE_QK("vit_rope_qk_f32", float)

// ViT self-attention, head dim 64, over window-major tokens, on Metal 4 tensor ops: each
// SIMD group runs its own BQ query rows against BK-key tiles read straight from device
// memory, S = q k^T into a cooperative tensor, the online softmax on it in registers
// (row max and sum through reduce_rows), then O += P v with P converted in place to the
// left operand. Windowed blocks attend within their 576-token window, global ones over
// every token. q and k come rotated (`vit_rope_qk`), v from the qkv GEMM's output; scores
// are scaled by `scale` (log2(e) / 8) in FP32
template <typename T, int BQ, int BK, int SGS>
kernel void vit_attn_tops(device T* p_out, device T* p_q, device T* p_k, device T* p_qkv,
                          constant uint& windowed, constant uint& tokens, constant float& scale,
                          uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float rows_tg[SGS][BQ];
    threadgroup float* rows = rows_tg[sg];
    const int head = tg.y, q0 = (tg.x * SGS + sg) * BQ;
    const int k0 = windowed != 0 ? (q0 / VIT_WIN_TOKENS) * VIT_WIN_TOKENS : 0;
    const int nkeys = windowed != 0 ? VIT_WIN_TOKENS : int(tokens);
    const int n = int(tokens);

    mat<T> tq(p_q + head * VIT_HD, dextents<int32_t, 2>(VIT_HD, n), array<int, 2>{1, VIT_DIM});
    mat<T> tk(p_k + head * VIT_HD, dextents<int32_t, 2>(VIT_HD, n), array<int, 2>{1, VIT_DIM});
    mat<T> tv(p_qkv + 2 * VIT_DIM + head * VIT_HD, dextents<int32_t, 2>(VIT_HD, n), array<int, 2>{1, 3 * VIT_DIM});
    constexpr auto dqk = matmul2d_descriptor(BQ, BK, VIT_HD, false, true, false);
    constexpr auto dpv =
        matmul2d_descriptor(BQ, VIT_HD, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dqk, execution_simdgroup> qk;
    matmul2d<dpv, execution_simdgroup> pv;

    auto mq = tq.template slice<VIT_HD, BQ>(0, q0);
    auto mk0 = tk.template slice<VIT_HD, BK>(0, k0);
    auto mv0 = tv.template slice<VIT_HD, BK>(0, k0);
    auto s = qk.template get_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto rmax = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto rsum = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto m = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto l = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto p0 = pv.template get_left_input_cooperative_tensor<float, T, float>(s);
    auto o = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(mv0), float>();
    for (uint16_t i = 0; i < m.get_capacity(); ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    for (uint16_t i = 0; i < o.get_capacity(); ++i) o[i] = 0.0f;

    for (int kt = 0; kt < nkeys; kt += BK) {
        auto mk = tk.template slice<VIT_HD, BK>(0, k0 + kt);
        auto mv = tv.template slice<VIT_HD, BK>(0, k0 + kt);
        qk.run(mq, mk, s);
        for (uint16_t i = 0; i < s.get_capacity(); ++i) s[i] *= scale;
        reduce_rows(s, rmax, reduction_operation::max, -INFINITY);
        // rmax becomes the correction of the old rows once m holds the new max
        for (uint16_t i = 0; i < m.get_capacity(); ++i) {
            if (!m.is_valid_element(i)) continue;
            float mn = fmax(m[i], rmax[i]);
            rmax[i] = exp2(m[i] - mn);
            m[i] = mn;
        }
        for (auto it = s.begin(); it != s.end(); it++) {
            if (!it.is_valid_element()) continue;
            auto mi = m.map_iterator(it);
            *it = exp2(*it - *mi);
        }
        reduce_rows(s, rsum, reduction_operation::sum, 0.0f);
        for (uint16_t i = 0; i < l.get_capacity(); ++i) {
            if (l.is_valid_element(i)) l[i] = l[i] * rmax[i] + rsum[i];
        }
        // O's rows sit in another op's layout, so the corrections go through threadgroup
        // memory, indexed by row
        for (uint16_t i = 0; i < rmax.get_capacity(); ++i) {
            if (rmax.is_valid_element(i)) rows[rmax.get_multidimensional_index(i)[0]] = rmax[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < o.get_capacity(); ++i) {
            if (o.is_valid_element(i)) o[i] *= rows[o.get_multidimensional_index(i)[1]];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto p = pv.template get_left_input_cooperative_tensor<float, T, float>(s);
        pv.run(p, mv, o);
    }

    for (uint16_t i = 0; i < l.get_capacity(); ++i) {
        if (l.is_valid_element(i)) rows[l.get_multidimensional_index(i)[0]] = 1.0f / l[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < o.get_capacity(); ++i) {
        if (!o.is_valid_element(i)) continue;
        auto idx = o.get_multidimensional_index(i);
        p_out[(long)(q0 + idx[1]) * VIT_DIM + head * VIT_HD + idx[0]] = T(o[i] * rows[idx[1]]);
    }
}

#define VIT_ATTN_TOPS(name, T, BQ, BK, SGS)                                                                       \
    template [[host_name(name)]] kernel void vit_attn_tops<T, BQ, BK, SGS>(                                       \
        device T*, device T*, device T*, device T*, constant uint&, constant uint&, constant float&, uint2, ushort);
// 16 query rows a SIMD group and 32-key tiles measured fastest of 16/32 x 32/64 on M4 Pro
// (job 0f06df2b: 125 ms of attention per encode against 145-158 for the others)
VIT_ATTN_TOPS("vit_attn_bf16", bfloat, 16, 32, 4)
VIT_ATTN_TOPS("vit_attn_f32", float, 16, 32, 4)
