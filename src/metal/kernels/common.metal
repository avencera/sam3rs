// Shared header of the sam3rs Metal kernels, compiled first. Pointer parameters are
// named `p_*`: the host's argument check reads the names to tell a pointer to one float
// from a `constant float&` scalar
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int ACT_NONE = 0;
constant constexpr int ACT_RELU = 1;
constant constexpr int ACT_GELU = 2;
constant constexpr int ACT_GELU_TANH = 3;
constant constexpr int ACT_SIGMOID = 4;

// erf in single precision, as glibc's and CUDA's erff (Cephes-style rational
// approximations, error under 1 ulp in [-4, 4]); MSL has no erf
inline float erf_f32(float x) {
    float ax = fabs(x);
    float r;
    if (ax < 0.84375f) {
        float z = x * x;
        float p = 1.28379166e-01f + z * (-3.25042099e-01f + z * (-2.84817486e-02f + z * (-5.77027029e-03f + z * -2.37630166e-05f)));
        float q = 1.0f + z * (3.97917223e-01f + z * (6.50222499e-02f + z * (5.08130628e-03f + z * (1.32494738e-04f + z * -3.96022827e-06f))));
        return x + x * (p / q);
    } else if (ax < 1.25f) {
        float s = ax - 1.0f;
        float p = -2.36211857e-03f + s * (4.14856106e-01f + s * (-3.72207481e-01f + s * (3.18346625e-01f + s * (-1.10894695e-01f + s * (3.54783051e-02f + s * -2.16637559e-03f)))));
        float q = 1.0f + s * (1.06420882e-01f + s * (5.40397942e-01f + s * (7.18286544e-02f + s * (1.26171216e-01f + s * (1.36370836e-02f + s * 1.19844996e-02f)))));
        r = 8.45062911e-01f + p / q;
        return x < 0.0f ? -r : r;
    } else if (ax >= 4.0f) {
        return x < 0.0f ? -1.0f : 1.0f;
    }
    float s = 1.0f / (ax * ax);
    float rr, ss;
    if (ax < 2.85714281f) {
        rr = -9.86494403e-03f + s * (-6.93858570e-01f + s * (-1.05586262e+01f + s * (-6.23753357e+01f + s * (-1.62396660e+02f + s * (-1.84605083e+02f + s * (-8.12874375e+01f + s * -9.81432915e+00f))))));
        ss = 1.0f + s * (1.96512716e+01f + s * (1.37657754e+02f + s * (4.34565880e+02f + s * (6.45387268e+02f + s * (4.29008148e+02f + s * (1.08635002e+02f + s * (6.57024983e+00f + s * -6.04244141e-02f)))))));
    } else {
        rr = -9.86494310e-03f + s * (-7.99283247e-01f + s * (-1.77579546e+01f + s * (-1.60636383e+02f + s * (-6.37566467e+02f + s * (-1.02509509e+03f + s * -4.83519199e+02f)))));
        ss = 1.0f + s * (3.03380604e+01f + s * (3.25792511e+02f + s * (1.53672925e+03f + s * (3.19985815e+03f + s * (2.55305054e+03f + s * (4.74528534e+02f + s * -2.24409524e+01f))))));
    }
    // split ax so exp(-z^2) keeps its precision
    float z = as_type<float>(as_type<uint>(ax) & 0xfffff000u);
    r = exp(-z * z - 0.5625f) * exp((z - ax) * (z + ax) + rr / ss);
    r = 1.0f - r / ax;
    return x < 0.0f ? -r : r;
}

inline float activate(float x, int act) {
    switch (act) {
    case ACT_RELU:
        return fmax(x, 0.0f);
    case ACT_GELU:
        return 0.5f * x * (1.0f + erf_f32(x * M_SQRT1_2_F));
    case ACT_GELU_TANH: {
        float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + precise::tanh(u));
    }
    case ACT_SIGMOID:
        return 1.0f / (1.0f + exp(-x));
    default:
        return x;
    }
}

// threadgroup-wide sum of one value per thread; `scratch` holds 32 floats and every
// thread of the group calls it
inline float group_sum(float v, threadgroup float* scratch, uint tid, uint threads) {
    v = simd_sum(v);
    uint lane = tid % 32, warp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint warps = (threads + 31) / 32;
    v = lane < warps ? scratch[lane] : 0.0f;
    return simd_sum(v);
}

inline float group_max(float v, threadgroup float* scratch, uint tid, uint threads) {
    v = simd_max(v);
    uint lane = tid % 32, warp = tid / 32;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[warp] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint warps = (threads + 31) / 32;
    v = lane < warps ? scratch[lane] : -INFINITY;
    return simd_max(v);
}
