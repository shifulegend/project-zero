#ifndef TN_CORE_FP16_H
#define TN_CORE_FP16_H

#include <stdint.h>
#include <string.h>

/*
 * IEEE 754 binary16 -> binary32, exact for every one of the 65536 inputs
 * (zeros, normals, subnormals, infinities, NaNs with their payload bits).
 * The one shared implementation: kernels and loaders used to carry fifteen
 * private copies, and several of them turned fp16 subnormals into ~0 instead
 * of converting them (tests/test_fp16.c checks all 65536 values).
 * Pure integer code, so the result is identical on every compiler and CPU.
 */
static inline float tn_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t u;
    if (exp == 0x1fu) { /* inf / NaN: keep the payload */
        u = sign | 0x7f800000u | (mant << 13);
    } else if (exp != 0u) { /* normal */
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    } else if (mant == 0u) { /* signed zero */
        u = sign;
    } else { /* subnormal: renormalise */
        exp = 1u;
        while (!(mant & 0x400u)) {
            mant <<= 1;
            exp--;
        }
        u = sign | ((exp + 112u) << 23) | ((mant & 0x3ffu) << 13);
    }
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}


/* Reads the little-endian binary16 value at `p` (no alignment needed) as a float: the scale at the
 * start of a GGUF quantisation block. */
static inline float tn_fp16_load(const void *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return tn_fp16_to_f32(h);
}

#endif /* TN_CORE_FP16_H */
