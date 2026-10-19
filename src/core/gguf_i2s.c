#include "core/gguf_i2s.h"
#include <math.h>
#include <stdint.h>
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
    /* Output byte m of a block holds within-block weights 4m..4m+3, which
     * all sit in group g = m/8 of input bytes k = 4(m%8)..4(m%8)+3. So each
     * 32-bit word of 4 input bytes yields one output byte per group: pick
     * the group's 2-bit field from every byte (bits 0,8,16,24), then fold
     * them down to bits 0,2,4,6. Word-at-a-time instead of per field: the
     * per-field loop made BitNet-2B loading take seconds (2026-10-19). */
    uint32_t bad = 0;
    for (size_t b = 0; b < n / TN_I2S_BLOCK; b++) {
        const uint8_t *in = src + b * 32;
        /* blocks never straddle rows because n_inner % 128 == 0 */
        uint8_t *out = dst + b * 32;
        for (int q = 0; q < 8; q++) {
            const uint8_t *w4 = in + 4 * q;
            uint32_t w = (uint32_t)w4[0] | ((uint32_t)w4[1] << 8) |
                         ((uint32_t)w4[2] << 16) | ((uint32_t)w4[3] << 24);
            bad |= w & (w >> 1) & 0x55555555u;          /* any code 3 (0b11) */
            for (int g = 0; g < 4; g++) {
                uint32_t f = (w >> (6 - 2 * g)) & 0x03030303u;
                f |= f >> 6;
                out[8 * g + q] = (uint8_t)((f | (f >> 12)) & 0xFFu);
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
