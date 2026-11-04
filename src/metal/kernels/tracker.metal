// Video tracker: axial RoPE, the memory encoder's convs and norms, and the mask inputs
// and overlap rule between the decoder and the memory encoder

// x [n, dim] with its first rope_rows rows rotated at position row % positions, as
// upstream's `apply_rotary_enc` complex product (re + i im)(cos + i sin); cos and sin
// are [positions, dim / 2]
kernel void rope_rows(device float* p_out, device const float* p_x, device const float* p_cos,
                      device const float* p_sin, constant uint& dim, constant uint& rope_rows,
                      constant uint& positions, uint i [[thread_position_in_grid]]) {
    uint row = i / dim, c = i % dim;
    if (row >= rope_rows) {
        p_out[i] = p_x[i];
        return;
    }
    uint half_dim = dim / 2;
    uint k = (row % positions) * half_dim + c / 2;
    uint base = row * dim + (c & ~1u);
    float re = p_x[base], im = p_x[base + 1];
    p_out[i] = (c & 1) == 0 ? re * p_cos[k] - im * p_sin[k] : re * p_sin[k] + im * p_cos[k];
}

// 3x3 conv, stride 2, padding 1: x [ih * iw, cin] -> out [(ih / 2) * (iw / 2), cout]
// with PyTorch's [cout, cin, 3, 3] weights, summed in (ky, kx, ic) order from the bias;
// the downsampler's first layers have too few channels for a GEMM to pay for an im2col
kernel void conv3x3_s2(device float* p_out, device const float* p_x, device const float* p_w,
                       device const float* p_bias, constant uint& ih, constant uint& iw, constant uint& cin,
                       constant uint& cout, uint i [[thread_position_in_grid]]) {
    uint ow = iw / 2;
    uint oc = i % cout, p = i / cout;
    uint oy = p / ow, ox = p % ow;
    float acc = p_bias[oc];
    for (uint ky = 0; ky < 3; ky++) {
        int iy = int(2 * oy + ky) - 1;
        if (iy < 0 || uint(iy) >= ih) continue;
        for (uint kx = 0; kx < 3; kx++) {
            int ix = int(2 * ox + kx) - 1;
            if (ix < 0 || uint(ix) >= iw) continue;
            device const float* src = p_x + (uint(iy) * iw + uint(ix)) * cin;
            for (uint ic = 0; ic < cin; ic++) acc += p_w[((oc * cin + ic) * 3 + ky) * 3 + kx] * src[ic];
        }
    }
    p_out[i] = acc;
}

// the im2col of a 3x3 stride-2 conv with padding 1: x [ih * iw, cin] ->
// [(ih / 2) * (iw / 2), cin * 9], columns in PyTorch's (ic, ky, kx) weight order
kernel void im2col3x3_s2(device float* p_out, device const float* p_x, constant uint& ih, constant uint& iw,
                         constant uint& cin, uint i [[thread_position_in_grid]]) {
    uint k = cin * 9, ow = iw / 2;
    uint p = i / k, col = i % k;
    uint ic = col / 9, tap = col % 9;
    uint oy = p / ow, ox = p % ow;
    int iy = int(2 * oy + tap / 3) - 1, ix = int(2 * ox + tap % 3) - 1;
    bool inside = iy >= 0 && uint(iy) < ih && ix >= 0 && uint(ix) < iw;
    p_out[i] = inside ? p_x[(uint(iy) * iw + uint(ix)) * cin + ic] : 0.0f;
}

// LayerNorm then `act` over each row of x [rows, cols] in place, one thread per row: the
// downsampler's maps have 4 to 64 channels, too few for a threadgroup per row. Two
// passes and rsqrt, as the threadgroup LayerNorm
kernel void layernorm_act_rows(device float* p_x, device const float* p_w, device const float* p_b,
                               constant uint& cols, constant float& eps, constant int& act,
                               uint r [[thread_position_in_grid]]) {
    device float* row = p_x + (long)r * cols;
    float s = 0.0f;
    for (uint c = 0; c < cols; c++) s += row[c];
    float mean = s / float(cols);
    float v = 0.0f;
    for (uint c = 0; c < cols; c++) {
        float d = row[c] - mean;
        v += d * d;
    }
    float rstd = rsqrt(v / float(cols) + eps);
    for (uint c = 0; c < cols; c++) row[c] = activate((row[c] - mean) * rstd * p_w[c] + p_b[c], act);
}

// depthwise 7x7 conv, padding 3, of a channels-last map x [h * w, c] with PyTorch's
// [c, 1, 7, 7] weights
kernel void dwconv7x7(device float* p_out, device const float* p_x, device const float* p_w,
                      device const float* p_bias, constant uint& h, constant uint& w, constant uint& c,
                      uint i [[thread_position_in_grid]]) {
    uint ch = i % c, p = i / c;
    uint y = p / w, xx = p % w;
    float acc = p_bias[ch];
    for (uint ky = 0; ky < 7; ky++) {
        int iy = int(y + ky) - 3;
        if (iy < 0 || uint(iy) >= h) continue;
        for (uint kx = 0; kx < 7; kx++) {
            int ix = int(xx + kx) - 3;
            if (ix < 0 || uint(ix) >= w) continue;
            acc += p_w[ch * 49 + ky * 7 + kx] * p_x[(uint(iy) * w + uint(ix)) * c + ch];
        }
    }
    p_out[i] = acc;
}

// the memory encoder's mask input: sigmoid(x) * 20 - 10, or with binarize (frames the
// user clicked on) (x > 0) * 20 - 10
kernel void mask_for_mem(device float* p_out, device const float* p_x, constant uint& binarize,
                         uint i [[thread_position_in_grid]]) {
    float v = p_x[i];
    float m = binarize != 0 ? float(v > 0.0f) : 1.0f / (1.0f + exp(-v));
    p_out[i] = m * 20.0f - 10.0f;
}

// upstream's `_apply_non_overlapping_constraints` over planes x [objects, n]: at each
// pixel the object with the highest logit (the first on ties) keeps it, and the others
// are clamped to at most -10
kernel void nonoverlap(device float* p_out, device const float* p_x, constant uint& objects, constant uint& n,
                       uint i [[thread_position_in_grid]]) {
    uint o = i / n, p = i % n;
    uint best = 0;
    for (uint k = 1; k < objects; k++) {
        if (p_x[(long)k * n + p] > p_x[(long)best * n + p]) best = k;
    }
    float v = p_x[i];
    p_out[i] = best == o ? v : fmin(v, -10.0f);
}

// bilinear resize (align_corners=False, no antialias) of planes [n, ih, iw] to
// [n, oh, ow], the interpolation of `resize_threshold_u8` without the threshold
kernel void resize_bilinear(device float* p_out, device const float* p_in, constant uint& ih, constant uint& iw,
                            constant uint& oh, constant uint& ow, uint i [[thread_position_in_grid]]) {
    float sh = float(ih) / float(oh), sw = float(iw) / float(ow);
    uint x = i % ow, y = (i / ow) % oh, p = i / (ow * oh);
    float fy = fmax((float(y) + 0.5f) * sh - 0.5f, 0.0f);
    float fx = fmax((float(x) + 0.5f) * sw - 0.5f, 0.0f);
    uint y0 = uint(fy), x0 = uint(fx);
    uint y1 = y0 + uint(y0 < ih - 1), x1 = x0 + uint(x0 < iw - 1);
    float ly = fy - float(y0), lx = fx - float(x0);
    device const float* in = p_in + (long)p * ih * iw;
    p_out[i] = (1.0f - ly) * ((1.0f - lx) * in[y0 * iw + x0] + lx * in[y0 * iw + x1]) +
               ly * ((1.0f - lx) * in[y1 * iw + x0] + lx * in[y1 * iw + x1]);
}
