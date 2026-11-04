// Mask postprocessing shared by the detector and the point decoder

// bilinear resize (align_corners=False, no antialias) of logit planes [n, ih, iw] to
// [n, oh, ow], thresholded to uint8 in the same pass: sigmoid(x) > 0.5 is x > 0. The
// interpolation matches the CUDA resize_bilinear op for op, so the bits match too
kernel void resize_threshold_u8(device uchar* p_out, device const float* p_in, constant uint& ih, constant uint& iw,
                                constant uint& oh, constant uint& ow, uint i [[thread_position_in_grid]]) {
    float sh = float(ih) / float(oh), sw = float(iw) / float(ow);
    uint x = i % ow, y = (i / ow) % oh, p = i / (ow * oh);
    float fy = fmax((float(y) + 0.5f) * sh - 0.5f, 0.0f);
    float fx = fmax((float(x) + 0.5f) * sw - 0.5f, 0.0f);
    uint y0 = uint(fy), x0 = uint(fx);
    uint y1 = y0 + uint(y0 < ih - 1), x1 = x0 + uint(x0 < iw - 1);
    float ly = fy - float(y0), lx = fx - float(x0);
    device const float* in = p_in + (long)p * ih * iw;
    float v = (1.0f - ly) * ((1.0f - lx) * in[y0 * iw + x0] + lx * in[y0 * iw + x1]) +
              ly * ((1.0f - lx) * in[y1 * iw + x0] + lx * in[y1 * iw + x1]);
    p_out[(long)p * oh * ow + y * ow + x] = uchar(v > 0.0f);
}
