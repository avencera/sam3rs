// Fused attention on simdgroup matrices: online softmax in FP32 registers, scores and
// P @ V on 8x8 FP32 fragments. A thread of an 8x8 fragment holds row fm, columns fn and
// fn + 1 (the layout simdgroup_matrix uses on Apple GPUs), so the 4 threads of a row
// differ in lane bits 0 and 3 and row reductions are two xor shuffles

struct FragPos {
    short fm, fn;
};

inline FragPos frag_pos(ushort lane) {
    short qid = lane / 4;
    return {short((qid & 4) + (lane / 2) % 4), short((qid & 2) * 2 + (lane % 2) * 2)};
}

inline float row_max(float v) {
    v = fmax(v, simd_shuffle_xor(v, 1));
    return fmax(v, simd_shuffle_xor(v, 8));
}

inline float row_sum(float v) {
    v += simd_shuffle_xor(v, 1);
    return v + simd_shuffle_xor(v, 8);
}

constant constexpr int VIT_DIM = 1024;
constant constexpr int VIT_HD = 64;
constant constexpr int VIT_WIN_TOKENS = 576;
constant constexpr int ATT_BQ = 32;
constant constexpr int ATT_BK = 32;

// General attention over strided [rows, width] views, one batch entry (head) per grid y:
// entry b of q starts at p_q + b * bs_q with rows ld_q apart, and likewise k, v and out.
// FP32 throughout, which costs Apple GPUs nothing over 16-bit math. Masks, by `flags`:
// ATT_KEY_PAD skips keys whose p_key_pad byte is nonzero (shared by every entry),
// ATT_CAUSAL keys after the query row, and ATT_RPB adds the factored box bias
// dy[b, q - 1, y] + dx[b, q - 1, x] for key y * side + x to every query row q but row 0.
// A row with every key masked comes out as zeros.
//
// Split-KV: with more than one threadgroup along grid z, threadgroup z takes keys
// [z * kchunk, (z + 1) * kchunk) and writes its unnormalized O rows to p_part
// [z, b, lq, DV] and each row's (max, sum) to p_ml [z, b, lq, 2], for `attn_combine`.
// A few queries over many keys (the point decoder's tokens over 5184 image keys) would
// otherwise run on a handful of threadgroups
constant constexpr uint ATT_KEY_PAD = 1;
constant constexpr uint ATT_CAUSAL = 2;
constant constexpr uint ATT_RPB = 4;

template <int D, int DV, int BK>
kernel void attn(device float* p_out, device const float* p_q, device const float* p_k, device const float* p_v,
                 device const uchar* p_key_pad, device const float* p_dx, device const float* p_dy,
                 constant uint& lq, constant uint& lk, constant float& scale_log2, constant uint& flags,
                 constant uint& side, constant uint& ld_o, constant uint& bs_o, constant uint& ld_q,
                 constant uint& bs_q, constant uint& ld_k, constant uint& bs_k, constant uint& ld_v,
                 constant uint& bs_v, device float* p_part, device float* p_ml, constant uint& kchunk,
                 uint3 tg [[threadgroup_position_in_grid]], uint3 grid [[threadgroups_per_grid]],
                 ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int QW = D < 64 ? D : 64;
    threadgroup float stage[4 * 8 * QW];
    threadgroup float ks[BK * D];
    threadgroup float vs[BK * DV];
    const uint b = tg.y, q0 = tg.x * 32 + sg * 8;
    device const float* q = p_q + (long)b * bs_q;
    device const float* k = p_k + (long)b * bs_k;
    device const float* v = p_v + (long)b * bs_v;
    device float* out = p_out + (long)b * bs_o;

    // q fragments, staged QW columns at a time through this SIMD group's slice, with the
    // softmax scale folded in; rows past lq are zeros
    simdgroup_matrix<float, 8, 8> qf[D / 8];
    threadgroup float* st = stage + sg * 8 * QW;
    for (int c0 = 0; c0 < D; c0 += QW) {
        for (int i = lane; i < 8 * QW; i += 32) {
            uint r = i / QW, c = i % QW, row = q0 + r;
            st[i] = row < lq ? q[(long)row * ld_q + c0 + c] * scale_log2 : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < QW / 8; kk++) simdgroup_load(qf[c0 / 8 + kk], st + kk * 8, QW);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_matrix<float, 8, 8> o[DV / 8];
    for (int d = 0; d < DV / 8; d++) o[d] = simdgroup_matrix<float, 8, 8>(0.0f);
    float m = -INFINITY, l = 0.0f;
    FragPos fp = frag_pos(lane);
    const uint qrow = q0 + fp.fm;
    const bool pad = (flags & ATT_KEY_PAD) != 0, causal = (flags & ATT_CAUSAL) != 0;
    const bool rpb = (flags & ATT_RPB) != 0 && qrow >= 1 && qrow < lq;
    device const float* dx = p_dx + ((long)b * (lq - 1) + qrow - 1) * side;
    device const float* dy = p_dy + ((long)b * (lq - 1) + qrow - 1) * side;

    const uint k_begin = tg.z * kchunk, k_end = min(lk, k_begin + kchunk);
    for (uint kt = k_begin; kt < k_end; kt += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < BK * D; i += 128) {
            uint key = kt + i / D;
            ks[i] = key < k_end ? k[(long)key * ld_k + i % D] : 0.0f;
        }
        for (int i = tid; i < BK * DV; i += 128) {
            uint key = kt + i / DV;
            vs[i] = key < k_end ? v[(long)key * ld_v + i % DV] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> s[BK / 8];
        for (int j = 0; j < BK / 8; j++) {
            s[j] = simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kk = 0; kk < D / 8; kk++) {
                simdgroup_matrix<float, 8, 8> kf;
                simdgroup_load(kf, ks + j * 8 * D + kk * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qf[kk], kf, s[j]);
            }
        }
        float mx = -INFINITY;
        for (int j = 0; j < BK / 8; j++) {
            thread auto& e = s[j].thread_elements();
            for (int t = 0; t < 2; t++) {
                uint key = kt + j * 8 + fp.fn + t;
                float x = e[t];
                if (rpb && key < k_end) x += M_LOG2E_F * (dy[key / side] + dx[key % side]);
                bool dead = key >= k_end || (pad && p_key_pad[key] != 0) || (causal && key > qrow);
                e[t] = dead ? -INFINITY : x;
                mx = fmax(mx, e[t]);
            }
        }
        float m_new = fmax(m, row_max(mx));
        // a row with no live key so far keeps a zero base, so exp2 gives zeros, not NaN
        float base = m_new == -INFINITY ? 0.0f : m_new;
        float factor = exp2(m - base);
        float sum = 0.0f;
        for (int j = 0; j < BK / 8; j++) {
            thread auto& e = s[j].thread_elements();
            e[0] = exp2(e[0] - base);
            e[1] = exp2(e[1] - base);
            sum += e[0] + e[1];
        }
        l = l * factor + row_sum(sum);
        m = m_new;
        for (int d = 0; d < DV / 8; d++) {
            thread auto& e = o[d].thread_elements();
            e[0] *= factor;
            e[1] *= factor;
            for (int j = 0; j < BK / 8; j++) {
                simdgroup_matrix<float, 8, 8> vf;
                simdgroup_load(vf, vs + j * 8 * DV + d * 8, DV);
                simdgroup_multiply_accumulate(o[d], s[j], vf, o[d]);
            }
        }
    }

    if (qrow >= lq) return;
    if (grid.z > 1) {
        const long row = ((long)tg.z * grid.y + b) * lq + qrow;
        for (int d = 0; d < DV / 8; d++) {
            auto e = o[d].thread_elements();
            device float* dst = p_part + row * DV + d * 8 + fp.fn;
            dst[0] = e[0];
            dst[1] = e[1];
        }
        if (fp.fn == 0) {
            p_ml[row * 2] = m;
            p_ml[row * 2 + 1] = l;
        }
        return;
    }
    float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int d = 0; d < DV / 8; d++) {
        auto e = o[d].thread_elements();
        device float* dst = out + (long)qrow * ld_o + d * 8 + fp.fn;
        dst[0] = e[0] * inv;
        dst[1] = e[1] * inv;
    }
}

#define ATTN(name, D, DV, BK)                                                                                     \
    template [[host_name(name)]] kernel void attn<D, DV, BK>(                                                     \
        device float*, device const float*, device const float*, device const float*, device const uchar*,       \
        device const float*, device const float*, constant uint&, constant uint&, constant float&, constant uint&, \
        constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,           \
        constant uint&, constant uint&, constant uint&, device float*, device float*, constant uint&, uint3, uint3, \
        ushort, ushort, ushort);
ATTN("attn_d16", 16, 16, 32)
ATTN("attn_d32", 32, 32, 32)
ATTN("attn_d64", 64, 64, 32)
ATTN("attn_d256", 256, 256, 8)
ATTN("attn_d256_v64", 256, 64, 16)

// Unmasked attention on Metal 4 tensor ops (`matmul2d`), the long self-attentions' path:
// the scheme of `vit_attn_tops` over the strided views of `attn`, FP32 in and out, values
// DV wide. Each SIMD group owns BQ query rows; tensor extents zero-fill reads past lq and
// lk, and keys past lk are masked to -inf in the last tile, so any lengths work. P @ V
// runs in value chunks at most 64 wide, one accumulator each: a single 16 x 256 one gave
// garbage on M4 Pro (macOS 27), while 64-wide ones match the reference
template <int D, int DV, int BQ, int BK, int SGS>
kernel void attn_tops(device float* p_out, device float* p_q, device float* p_k, device float* p_v,
                      constant uint& lq, constant uint& lk, constant float& scale_log2, constant uint& ld_o,
                      constant uint& bs_o, constant uint& ld_q, constant uint& bs_q, constant uint& ld_k,
                      constant uint& bs_k, constant uint& ld_v, constant uint& bs_v,
                      uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float rows_tg[SGS][BQ];
    threadgroup float* rows = rows_tg[sg];
    const int b = tg.y, q0 = (tg.x * SGS + sg) * BQ;
    const int nq = int(lq), nk = int(lk);
    // only SIMD-group barriers below, so a group past the last query can leave
    if (q0 >= nq) return;

    mat<float> tq(p_q + (long)b * bs_q, dextents<int32_t, 2>(D, nq), array<int, 2>{1, int(ld_q)});
    mat<float> tk(p_k + (long)b * bs_k, dextents<int32_t, 2>(D, nk), array<int, 2>{1, int(ld_k)});
    constexpr int DC = DV < 64 ? DV : 64, NC = DV / DC;
    mat<float> tv(p_v + (long)b * bs_v, dextents<int32_t, 2>(DV, nk), array<int, 2>{1, int(ld_v)});
    constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
    constexpr auto dpv =
        matmul2d_descriptor(BQ, DC, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dqk, execution_simdgroup> qk;
    matmul2d<dpv, execution_simdgroup> pv;

    auto mq = tq.template slice<D, BQ>(0, q0);
    auto mk0 = tk.template slice<D, BK>(0, 0);
    auto mv0 = tv.template slice<DC, BK>(0, 0);
    auto s = qk.template get_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto rmax = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto rsum = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto m = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto l = qk.template get_row_reduction_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto p0 = pv.template get_left_input_cooperative_tensor<float, float, float>(s);
    // named accumulators, since cooperative tensors can't form arrays; unused ones fold away
    auto o0 = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(mv0), float>();
    auto o1 = o0, o2 = o0, o3 = o0;
    static_assert(NC <= 4, "values at most 256 wide");
    auto each = [&](auto f) {
        f(o0, 0);
        if constexpr (NC > 1) f(o1, 1);
        if constexpr (NC > 2) f(o2, 2);
        if constexpr (NC > 3) f(o3, 3);
    };
    for (uint16_t i = 0; i < m.get_capacity(); ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    each([&](thread auto& o, int) {
        for (uint16_t i = 0; i < o.get_capacity(); ++i) o[i] = 0.0f;
    });

    for (int kt = 0; kt < nk; kt += BK) {
        auto mk = tk.template slice<D, BK>(0, kt);
        qk.run(mq, mk, s);
        for (uint16_t i = 0; i < s.get_capacity(); ++i) s[i] *= scale_log2;
        if (kt + BK > nk) {
            for (uint16_t i = 0; i < s.get_capacity(); ++i) {
                if (s.is_valid_element(i) && kt + s.get_multidimensional_index(i)[0] >= nk) s[i] = -INFINITY;
            }
        }
        // the first tile holds a live key for every row, so m is finite from then on
        reduce_rows(s, rmax, reduction_operation::max, -INFINITY);
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
        for (uint16_t i = 0; i < rmax.get_capacity(); ++i) {
            if (rmax.is_valid_element(i)) rows[rmax.get_multidimensional_index(i)[0]] = rmax[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        each([&](thread auto& o, int) {
            for (uint16_t i = 0; i < o.get_capacity(); ++i) {
                if (o.is_valid_element(i)) o[i] *= rows[o.get_multidimensional_index(i)[1]];
            }
        });
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto p = pv.template get_left_input_cooperative_tensor<float, float, float>(s);
        each([&](thread auto& o, int c) {
            auto mv = tv.template slice<DC, BK>(c * DC, kt);
            pv.run(p, mv, o);
        });
    }

    for (uint16_t i = 0; i < l.get_capacity(); ++i) {
        if (l.is_valid_element(i)) rows[l.get_multidimensional_index(i)[0]] = 1.0f / l[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device float* out = p_out + (long)b * bs_o;
    each([&](thread auto& o, int c) {
        for (uint16_t i = 0; i < o.get_capacity(); ++i) {
            if (!o.is_valid_element(i)) continue;
            auto idx = o.get_multidimensional_index(i);
            int row = q0 + idx[1];
            if (row < nq) out[(long)row * ld_o + c * DC + idx[0]] = o[i] * rows[idx[1]];
        }
    });
}

#define ATTN_TOPS(name, D, DV, BQ, BK, SGS)                                                                           \
    template [[host_name(name)]] kernel void attn_tops<D, DV, BQ, BK, SGS>(                                           \
        device float*, device float*, device float*, device float*, constant uint&, constant uint&,             \
        constant float&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,        \
        constant uint&, constant uint&, constant uint&, uint2, ushort);
// 16 x 32 tiles on 4 SIMD groups, as the ViT: 16 x 64, 32 x 32, 32 x 64, 16 x 128 and 8
// SIMD groups tied or lost on the detector (2026-11-03 sweep, M4 Pro)
ATTN_TOPS("attn_tops_d32", 32, 32, 16, 32, 4)
ATTN_TOPS("attn_tops_d64", 64, 64, 16, 32, 4)
// head dim 256 runs on `attn_coop` (64-wide values) or unfused (see `attention` in ops.rs)

// Unmasked attention for wide heads (D = 256): the 4 SIMD groups of a threadgroup compute
// each 64-query x 64-key score tile as one `matmul2d`, the GEMM's shape, into threadgroup
// memory; each group then runs the online softmax on its 16 rows there and multiplies
// those rows of P by the value tile into its own accumulators. A SIMD group on its own
// 16 rows (`attn_tops`) held QK near 4.5 TFLOPS on these shapes against about 6.9 for the
// 64 x 64 GEMM tile. Same arguments as `attn_tops`, plus the split-KV partials of `attn`:
// with threadgroups along z, group z takes keys [z kchunk, (z + 1) kchunk) and writes
// unnormalized O and (max, sum) for `attn_combine`. With 81 query blocks it needs about
// 4 splits to fill the GPU: 31.2 ms unsplit, 24.2 at 4 splits, 23.9 at 8 or 16 for the
// tracker's 5184 x 36352 memory attention (M4 Pro, 2026-11-04)
constant constexpr int COOP_SG = 4;
constant constexpr int COOP_BQ = 16 * COOP_SG;
constant constexpr int COOP_BK = 64;

template <int D, int DV>
kernel void attn_coop(device float* p_out, device float* p_q, device float* p_k, device float* p_v,
                      constant uint& lq, constant uint& lk, constant float& scale_log2, constant uint& ld_o,
                      constant uint& bs_o, constant uint& ld_q, constant uint& bs_q, constant uint& ld_k,
                      constant uint& bs_k, constant uint& ld_v, constant uint& bs_v, device float* p_part,
                      device float* p_ml, constant uint& kchunk, uint3 tg [[threadgroup_position_in_grid]],
                      uint3 grid [[threadgroups_per_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_tile[COOP_BQ * COOP_BK];
    threadgroup float rows_tg[COOP_BQ];
    threadgroup float* rows = rows_tg + sg * 16;
    const int b = tg.y, q0 = tg.x * COOP_BQ;
    const int nq = int(lq);
    // this group's keys as their own tensors, so the ragged end zero-fills at the extent
    const int k0 = int(tg.z * kchunk), nk = min(int(lk) - k0, int(kchunk));

    mat<float> tq(p_q + (long)b * bs_q, dextents<int32_t, 2>(D, nq), array<int, 2>{1, int(ld_q)});
    mat<float> tk(p_k + (long)b * bs_k + (long)k0 * ld_k, dextents<int32_t, 2>(D, nk), array<int, 2>{1, int(ld_k)});
    constexpr int DC = DV < 64 ? DV : 64, NC = DV / DC;
    mat<float> tv(p_v + (long)b * bs_v + (long)k0 * ld_v, dextents<int32_t, 2>(DV, nk), array<int, 2>{1, int(ld_v)});
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> ts(s_tile, dextents<int32_t, 2>(COOP_BK, COOP_BQ),
                                                                      array<int, 2>{1, COOP_BK});
    constexpr auto dqk = matmul2d_descriptor(COOP_BQ, COOP_BK, D, false, true, false);
    constexpr auto dpv =
        matmul2d_descriptor(16, DC, COOP_BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dqk, execution_simdgroups<COOP_SG>> qk;
    matmul2d<dpv, execution_simdgroup> pv;

    auto mq = tq.template slice<D, COOP_BQ>(0, q0);
    auto mk0 = tk.template slice<D, COOP_BK>(0, 0);
    auto mp = ts.template slice<COOP_BK, 16>(0, sg * 16);
    auto mv0 = tv.template slice<DC, COOP_BK>(0, 0);
    auto s = qk.template get_destination_cooperative_tensor<decltype(mq), decltype(mk0), float>();
    auto o0 = pv.template get_destination_cooperative_tensor<decltype(mp), decltype(mv0), float>();
    auto o1 = o0, o2 = o0, o3 = o0;
    static_assert(NC <= 4, "values at most 256 wide");
    auto each = [&](auto f) {
        f(o0, 0);
        if constexpr (NC > 1) f(o1, 1);
        if constexpr (NC > 2) f(o2, 2);
        if constexpr (NC > 3) f(o3, 3);
    };
    each([&](thread auto& o, int) {
        for (uint16_t i = 0; i < o.get_capacity(); ++i) o[i] = 0.0f;
    });
    // every lane keeps its group's 16 row maxima and sums, uniform across the group
    float m[16], l[16];
    for (int r = 0; r < 16; r++) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    for (int kt = 0; kt < nk; kt += COOP_BK) {
        auto mk = tk.template slice<D, COOP_BK>(0, kt);
        qk.run(mq, mk, s);
        for (uint16_t i = 0; i < s.get_capacity(); ++i) {
            if (!s.is_valid_element(i)) continue;
            auto idx = s.get_multidimensional_index(i);
            s_tile[idx[1] * COOP_BK + idx[0]] = kt + idx[0] < nk ? s[i] * scale_log2 : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the first tile holds a live key for every row, so m is finite from then on
        threadgroup float* st = s_tile + sg * 16 * COOP_BK;
        for (int r = 0; r < 16; r++) {
            float v0 = st[r * COOP_BK + lane], v1 = st[r * COOP_BK + 32 + lane];
            float mn = fmax(m[r], simd_max(fmax(v0, v1)));
            float alpha = exp2(m[r] - mn);
            v0 = exp2(v0 - mn);
            v1 = exp2(v1 - mn);
            st[r * COOP_BK + lane] = v0;
            st[r * COOP_BK + 32 + lane] = v1;
            l[r] = l[r] * alpha + simd_sum(v0 + v1);
            m[r] = mn;
            if (lane == 0) rows[r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        each([&](thread auto& o, int) {
            for (uint16_t i = 0; i < o.get_capacity(); ++i) {
                if (o.is_valid_element(i)) o[i] *= rows[o.get_multidimensional_index(i)[1]];
            }
        });
        each([&](thread auto& o, int c) {
            auto mv = tv.template slice<DC, COOP_BK>(c * DC, kt);
            pv.run(mp, mv, o);
        });
        // the next tile's scores overwrite P
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int r0 = q0 + sg * 16;
    const bool split = grid.z > 1;
    // a split keeps O unnormalized and hands (max, sum) to attn_combine
    const long prow = ((long)tg.z * grid.y + b) * nq;
    if (lane < 16) {
        float mr = 0.0f, lr = 0.0f;
        for (int r = 0; r < 16; r++) {
            mr = r == lane ? m[r] : mr;
            lr = r == lane ? l[r] : lr;
        }
        rows[lane] = split ? 1.0f : 1.0f / lr;
        if (split && r0 + lane < nq) {
            p_ml[(prow + r0 + lane) * 2] = mr;
            p_ml[(prow + r0 + lane) * 2 + 1] = lr;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device float* out = split ? p_part + prow * DV : p_out + (long)b * bs_o;
    const long ld = split ? DV : ld_o;
    each([&](thread auto& o, int c) {
        for (uint16_t i = 0; i < o.get_capacity(); ++i) {
            if (!o.is_valid_element(i)) continue;
            auto idx = o.get_multidimensional_index(i);
            int row = r0 + idx[1];
            if (row < nq) out[(long)row * ld + c * DC + idx[0]] = o[i] * rows[idx[1]];
        }
    });
}

template [[host_name("attn_coop_d256_v64")]] kernel void attn_coop<256, 64>(
    device float*, device float*, device float*, device float*, constant uint&, constant uint&, constant float&,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, device float*, device float*, constant uint&, uint3, uint3, ushort, ushort);

// Merges the split-KV partials of `attn`: out[b, row] = sum_z 2^(m_z - M) O_z / sum_z
// 2^(m_z - M) l_z with M the largest m_z, one thread per output element; rows whose keys
// were all masked come out as zeros, as in the unsplit kernel
kernel void attn_combine(device float* p_out, device const float* p_part, device const float* p_ml,
                         constant uint& splits, constant uint& batch, constant uint& lq, constant uint& dv,
                         constant uint& ld_o, constant uint& bs_o, uint i [[thread_position_in_grid]]) {
    const uint col = i % dv, row = (i / dv) % lq, b = i / (dv * lq);
    const long stride = (long)batch * lq;
    const long r0 = (long)b * lq + row;
    float mx = -INFINITY;
    for (uint z = 0; z < splits; z++) mx = fmax(mx, p_ml[(r0 + z * stride) * 2]);
    float num = 0.0f, den = 0.0f;
    if (mx != -INFINITY) {
        for (uint z = 0; z < splits; z++) {
            const long r = r0 + z * stride;
            float w = exp2(p_ml[r * 2] - mx);
            num += w * p_part[r * dv + col];
            den += w * p_ml[r * 2 + 1];
        }
    }
    p_out[(long)b * bs_o + (long)row * ld_o + col] = den > 0.0f ? num / den : 0.0f;
}

