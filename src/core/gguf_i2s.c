#include "core/gguf_i2s.h"
#include <math.h>
#include <string.h>

size_t tn_i2s_tensor_bytes(size_t n_elems) { return n_elems / 4 + 32; }

int tn_i2s_repack(uint8_t *dst, float *scale_out, const uint8_t *src,
                  int d_rows, int n_inner) {
    if (!dst || !scale_out || !src || d_rows <= 0 || n_inner <= 0 || n_inner % TN_I2S_BLOCK != 0)
        return -1;
    size_t n = (size_t)d_rows * (size_t)n_inner;
    float scale;
    memcpy(&scale, src + n / 4, sizeof(scale));
    if (!isfinite(scale)) return -1;
    int bad = 0;
    for (size_t b = 0; b < n / TN_I2S_BLOCK; b++) {
        const uint8_t *in = src + b * 32;
        /* 128 consecutive weights b*128 .. b*128+127 == 32 output bytes; blocks
         * never straddle rows because n_inner % 128 == 0. */
        uint8_t *out = dst + b * 32;
        memset(out, 0, 32);
        for (int k = 0; k < 32; k++) {
            uint8_t byte = in[k];
            for (int g = 0; g < 4; g++) {
                uint8_t code = (uint8_t)((byte >> (6 - 2 * g)) & 3);
                bad |= (code == 3);
                size_t j = (size_t)(k + 32 * g);          /* index within block */
                out[j >> 2] |= (uint8_t)(code << ((j & 3) << 1));
            }
        }
    }
    *scale_out = scale;
    return bad ? -1 : 0;
}

void tn_i2s_pack_reference(uint8_t *dst, const int8_t *vals, size_t n, float scale) {
    memset(dst, 0, tn_i2s_tensor_bytes(n));
    for (size_t i = 0; i < n / TN_I2S_BLOCK; i++) {
        for (int j = 0; j < TN_I2S_BLOCK; j++) {
            int group_idx = j / 32, group_pos = j % 32;
            uint8_t q = (uint8_t)(vals[i * TN_I2S_BLOCK + (size_t)j] + 1);
            dst[i * 32 + (size_t)group_pos] |= (uint8_t)(q << (6 - 2 * group_idx));
        }
    }
    memcpy(dst + n / 4, &scale, sizeof(scale));
}
