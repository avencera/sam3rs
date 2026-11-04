// GEMMs on Metal 4 tensor ops (matmul2d), with the epilogue applied to the FP32
// accumulators in registers through a cooperative tensor. Inputs are row-major with a
// leading dimension; weights are nn.Linear's [N, K]. One threadgroup of 4 SIMD groups
// computes a BM x BN tile

constant constexpr int GEMM_BM = 64;
constant constexpr int GEMM_BN = 64;
constant constexpr int GEMM_SG = 4;

template <typename T>
using mat = tensor<device T, dextents<int32_t, 2>, tensor_inline>;

// x [m, k] (leading dim ldx) . w[n, k]^T for the tile at (m0, n0), handed to `epi(row,
// col, value)` for every in-bounds element
template <typename TI, typename Epi>
inline void gemm_tile(device TI* x, device TI* w, int m, int n, int k, int ldx, int ldw, int m0, int n0, Epi epi) {
    mat<TI> tx(x, dextents<int32_t, 2>(k, m), array<int, 2>{1, ldx});
    mat<TI> tw(w, dextents<int32_t, 2>(k, n), array<int, 2>{1, ldw});
    constexpr auto desc = matmul2d_descriptor(GEMM_BM, GEMM_BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<GEMM_SG>> op;
    auto mx = tx.slice(0, m0);
    auto mw = tw.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    op.run(mx, mw, acc);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], row = m0 + idx[1];
        if (row < m && col < n) epi(row, col, acc[i]);
    }
}

// y[row, col] = act(x . w^T + bias) (+ y when accumulate). The bias is `bias_rows` rows
// of n, row r of y taking row r % bias_rows: none at 0, nn.Linear's vector at 1, and at
// more a matrix, such as a constant input's projection folded in
template <typename TI, typename TO>
kernel void linear(device TI* p_x [[buffer(0)]], device TI* p_w [[buffer(1)]], device const float* p_bias [[buffer(2)]],
                   device TO* p_y [[buffer(3)]], constant uint& m [[buffer(4)]], constant uint& n [[buffer(5)]],
                   constant uint& k [[buffer(6)]], constant uint& ldx [[buffer(7)]], constant uint& ldy [[buffer(8)]],
                   constant uint& bias_rows [[buffer(9)]], constant int& act [[buffer(10)]],
                   constant uint& accumulate [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]]) {
    gemm_tile(p_x, p_w, m, n, k, ldx, k, tg.y * GEMM_BM, tg.x * GEMM_BN, [&](int row, int col, float v) {
        // the vector case skips the integer modulo
        if (bias_rows == 1) v += p_bias[col];
        else if (bias_rows != 0) v += p_bias[(long)(row % bias_rows) * n + col];
        v = activate(v, act);
        device TO* dst = p_y + (long)row * ldy + col;
        if (accumulate != 0) v += float(*dst);
        *dst = TO(v);
    });
}

template [[host_name("linear_f32_f32")]] kernel void linear<float, float>(
    device float*, device float*, device const float*, device float*, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant int&, constant uint&, uint2);
template [[host_name("linear_bf16_bf16")]] kernel void linear<bfloat, bfloat>(
    device bfloat*, device bfloat*, device const float*, device bfloat*, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&, constant int&, constant uint&, uint2);
template [[host_name("linear_bf16_f32")]] kernel void linear<bfloat, float>(
    device bfloat*, device bfloat*, device const float*, device float*, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&, constant int&, constant uint&, uint2);

// Split-K GEMM for grids too small to fill the GPU, such as the text encoder's 32-token
// linears: threadgroup z sums K range [z * kc, min(k, (z + 1) * kc)) of x . w^T into its
// own FP32 partial [m, n] of p_part; linear_splitk_reduce then adds the partials, bias,
// activation and residual
template <typename TI>
kernel void linear_splitk(device TI* p_x [[buffer(0)]], device TI* p_w [[buffer(1)]], device float* p_part [[buffer(2)]],
                          constant uint& m [[buffer(3)]], constant uint& n [[buffer(4)]], constant uint& k [[buffer(5)]],
                          constant uint& kc [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]]) {
    uint k0 = tg.z * kc;
    int klen = int(min(kc, k - k0));
    device float* part = p_part + (long)tg.z * m * n;
    gemm_tile(p_x + k0, p_w + k0, m, n, klen, k, k, tg.y * GEMM_BM, tg.x * GEMM_BN,
              [&](int row, int col, float v) { part[(long)row * n + col] = v; });
}

template [[host_name("linear_splitk_f32")]] kernel void linear_splitk<float>(
    device float*, device float*, device float*, constant uint&, constant uint&, constant uint&, constant uint&, uint3);
template [[host_name("linear_splitk_bf16")]] kernel void linear_splitk<bfloat>(
    device bfloat*, device bfloat*, device float*, constant uint&, constant uint&, constant uint&, constant uint&, uint3);

// y[i] = act(sum over splits of p_part + bias) (+ y when accumulate), one thread per
// element, the bias as in `linear`
template <typename TO>
kernel void linear_splitk_reduce(device TO* p_y [[buffer(0)]], device const float* p_part [[buffer(1)]],
                                 device const float* p_bias [[buffer(2)]], constant uint& m [[buffer(3)]],
                                 constant uint& n [[buffer(4)]], constant uint& ldy [[buffer(5)]],
                                 constant uint& splits [[buffer(6)]], constant uint& bias_rows [[buffer(7)]],
                                 constant int& act [[buffer(8)]], constant uint& accumulate [[buffer(9)]],
                                 uint i [[thread_position_in_grid]]) {
    uint row = i / n, col = i % n;
    long mn = (long)m * n;
    float v = 0.0f;
    for (uint z = 0; z < splits; ++z) v += p_part[z * mn + i];
    if (bias_rows == 1) v += p_bias[col];
    else if (bias_rows != 0) v += p_bias[(long)(row % bias_rows) * n + col];
    v = activate(v, act);
    device TO* dst = p_y + (long)row * ldy + col;
    if (accumulate != 0) v += float(*dst);
    *dst = TO(v);
}

template [[host_name("linear_splitk_reduce_f32")]] kernel void linear_splitk_reduce<float>(
    device float*, device const float*, device const float*, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant int&, constant uint&, uint);
template [[host_name("linear_splitk_reduce_bf16")]] kernel void linear_splitk_reduce<bfloat>(
    device bfloat*, device const float*, device const float*, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant int&, constant uint&, uint);

// ConvTranspose2d, kernel 2 stride 2, as a GEMM whose epilogue adds the bias, applies the
// activation and scatters each output column j = (dy * 2 + dx) * O + o of input pixel
// (y, x) to output pixel (2y + dy, 2x + dx), channel o; w is the weight transposed to
// [4 * O, C], tap-major, so a tile's columns store to consecutive channels. With add_len
// nonzero, output element i adds p_add[i % add_len] before the activation, such as a
// skip connection repeated over stacked images
template <typename TI, typename TO>
kernel void convt2x2(device TI* p_x [[buffer(0)]], device TI* p_w [[buffer(1)]], device const float* p_bias [[buffer(2)]],
                     device TO* p_y [[buffer(3)]], constant uint& m [[buffer(4)]], constant uint& n [[buffer(5)]],
                     constant uint& k [[buffer(6)]], constant uint& width [[buffer(7)]], constant int& act [[buffer(8)]],
                     device const float* p_add [[buffer(9)]], constant uint& add_len [[buffer(10)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
    uint o = n / 4;
    gemm_tile(p_x, p_w, m, n, k, k, k, tg.y * GEMM_BM, tg.x * GEMM_BN, [&](int row, int col, float v) {
        uint tap = col / o, oc = col % o, dy = tap / 2, dx = tap % 2;
        uint y = row / width, x = row % width;
        long i = ((long)(2 * y + dy) * 2 * width + 2 * x + dx) * o + oc;
        v += p_bias[oc];
        if (add_len != 0) v += p_add[i % add_len];
        p_y[i] = TO(activate(v, act));
    });
}

template [[host_name("convt2x2_f32_f32")]] kernel void convt2x2<float, float>(
    device float*, device float*, device const float*, device float*, constant uint&, constant uint&, constant uint&,
    constant uint&, constant int&, device const float*, constant uint&, uint2);
template [[host_name("convt2x2_bf16_bf16")]] kernel void convt2x2<bfloat, bfloat>(
    device bfloat*, device bfloat*, device const float*, device bfloat*, constant uint&, constant uint&,
    constant uint&, constant uint&, constant int&, device const float*, constant uint&, uint2);

// 3x3 conv, stride 1, padding 1, channels-last, as nine accumulating GEMMs over shifted
// row windows of the zero-padded input xp [(h + 2) * (w + 2) + 2, C]: output row r of the
// padded-width map (y = r / (w + 2), x = r % (w + 2)) reads xp rows r + dy * (w + 2) + dx.
// Columns x >= w are padding and never stored. Taps are [9, O, C]
template <typename TI, typename TO>
kernel void conv3x3(device TI* p_xp [[buffer(0)]], device TI* p_taps [[buffer(1)]], device const float* p_bias [[buffer(2)]],
                    device TO* p_y [[buffer(3)]], constant uint& h [[buffer(4)]], constant uint& w [[buffer(5)]],
                    constant uint& c [[buffer(6)]], constant uint& o [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]]) {
    int wp = w + 2, rows = h * wp, xrows = (h + 2) * wp + 2;
    int m0 = tg.y * GEMM_BM, n0 = tg.x * GEMM_BN;
    constexpr auto desc = matmul2d_descriptor(GEMM_BM, GEMM_BN, static_cast<int>(dynamic_extent), false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<GEMM_SG>> op;
    mat<TI> tx(p_xp, dextents<int32_t, 2>(c, xrows), array<int, 2>{1, (int)c});
    mat<TI> tw0(p_taps, dextents<int32_t, 2>(c, o), array<int, 2>{1, (int)c});
    auto acc = op.template get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw0.slice(0, 0)), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (int t = 0; t < 9; t++) {
        int shift = (t / 3) * wp + t % 3;
        mat<TI> tw(p_taps + (long)t * o * c, dextents<int32_t, 2>(c, o), array<int, 2>{1, (int)c});
        auto mx = tx.slice(0, m0 + shift);
        auto mw = tw.slice(0, n0);
        op.run(mx, mw, acc);
    }
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], r = m0 + idx[1];
        int y = r / wp, x = r % wp;
        if (r < rows && x < (int)w && col < (int)o) p_y[(long)(y * w + x) * o + col] = TO(acc[i] + p_bias[col]);
    }
}

template [[host_name("conv3x3_f32_f32")]] kernel void conv3x3<float, float>(
    device float*, device float*, device const float*, device float*, constant uint&, constant uint&, constant uint&,
    constant uint&, uint2);
template [[host_name("conv3x3_bf16_f32")]] kernel void conv3x3<bfloat, float>(
    device bfloat*, device bfloat*, device const float*, device float*, constant uint&, constant uint&, constant uint&,
    constant uint&, uint2);

// zero-pads x [h * w, c] into xp [(h + 2) * (w + 2) + 2, c] (see conv3x3), converting to TO
template <typename TI, typename TO>
kernel void pad1(device TO* p_xp, device const TI* p_x, constant uint& h, constant uint& w, constant uint& c,
                 uint i [[thread_position_in_grid]]) {
    uint wp = w + 2, ch = i % c, r = i / c;
    uint y = r / wp, x = r % wp;
    bool inside = y >= 1 && y <= h && x >= 1 && x <= w;
    p_xp[i] = inside ? TO(p_x[((y - 1) * w + x - 1) * c + ch]) : TO(0.0f);
}

template [[host_name("pad1_f32_f32")]] kernel void pad1<float, float>(device float*, device const float*, constant uint&,
                                                                       constant uint&, constant uint&, uint);
template [[host_name("pad1_f32_bf16")]] kernel void pad1<float, bfloat>(device bfloat*, device const float*,
                                                                         constant uint&, constant uint&, constant uint&,
                                                                         uint);

// Unfused attention for long head dims (the tracker's 256-wide one-head self-attention),
// where the fused kernel's P @ V on 64-key tiles ran at a third of the GEMM rate: scores
// S = q . k^T * scale_log2 into a [lq, lk] buffer, `attn_softmax_rows` turns each row into
// softmax numerators and its reciprocal sum, then `attn_pv` multiplies by v on long-K
// tiles and scales each row. q and k rows are ld_q and ld_k apart
kernel void attn_scores(device float* p_s [[buffer(0)]], device float* p_q [[buffer(1)]], device float* p_k [[buffer(2)]],
                        constant uint& lq [[buffer(3)]], constant uint& lk [[buffer(4)]], constant uint& d [[buffer(5)]],
                        constant uint& ld_q [[buffer(6)]], constant uint& ld_k [[buffer(7)]],
                        constant float& scale_log2 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]]) {
    gemm_tile(p_q, p_k, lq, lk, d, ld_q, ld_k, tg.y * GEMM_BM, tg.x * GEMM_BN,
              [&](int row, int col, float v) { p_s[(long)row * lk + col] = v * scale_log2; });
}

// out[row, col] = (p . v)[row, col] * inv_l[row], p [lq, lk] row-major, v [lk, dv] with
// rows ld_v apart, out rows ld_o apart
kernel void attn_pv(device float* p_o [[buffer(0)]], device float* p_p [[buffer(1)]], device float* p_v [[buffer(2)]],
                    device const float* p_inv [[buffer(3)]], constant uint& lq [[buffer(4)]], constant uint& lk [[buffer(5)]],
                    constant uint& dv [[buffer(6)]], constant uint& ld_v [[buffer(7)]], constant uint& ld_o [[buffer(8)]],
                    uint2 tg [[threadgroup_position_in_grid]]) {
    mat<float> tp(p_p, dextents<int32_t, 2>(lk, lq), array<int, 2>{1, int(lk)});
    mat<float> tv(p_v, dextents<int32_t, 2>(dv, lk), array<int, 2>{1, int(ld_v)});
    constexpr auto desc = matmul2d_descriptor(GEMM_BM, GEMM_BN, static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<GEMM_SG>> op;
    const int m0 = tg.y * GEMM_BM, n0 = tg.x * GEMM_BN;
    auto mp = tp.slice(0, m0);
    auto mv = tv.slice(n0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mp), decltype(mv), float>();
    op.run(mp, mv, acc);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto idx = acc.get_multidimensional_index(i);
        int col = n0 + idx[0], row = m0 + idx[1];
        if (row < int(lq) && col < int(dv)) p_o[(long)row * ld_o + col] = acc[i] * p_inv[row];
    }
}

constant constexpr uint SOFTMAX_THREADS = 256;

// One threadgroup per row of [rows, cols] scaled scores: s <- 2^(s - max) in place and
// inv_l[row] = 1 / sum, the row read twice (the second time from cache)
kernel void attn_softmax_rows(device float* p_s [[buffer(0)]], device float* p_inv [[buffer(1)]],
                              constant uint& cols [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[SOFTMAX_THREADS / 32];
    device float* s = p_s + (long)row * cols;
    float mx = -INFINITY;
    for (uint i = tid; i < cols; i += SOFTMAX_THREADS) mx = fmax(mx, s[i]);
    mx = simd_max(mx);
    if (lane == 0) part[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = part[0];
    for (uint i = 1; i < SOFTMAX_THREADS / 32; i++) mx = fmax(mx, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < cols; i += SOFTMAX_THREADS) {
        float e = exp2(s[i] - mx);
        s[i] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (lane == 0) part[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < SOFTMAX_THREADS / 32; i++) t += part[i];
        p_inv[row] = 1.0f / t;
    }
}
