/* tn_fp16_to_f32 must be exact for all 65536 half-precision bit patterns. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/fp16.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

static uint32_t bits_of(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

int main(void) {
    int bad_value = 0, bad_special = 0, bad_sub = 0, subs = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        int s = (int)(h >> 15), e = (int)((h >> 10) & 0x1f), m = (int)(h & 0x3ff);
        float got = tn_fp16_to_f32((uint16_t)h);
        if (e == 31) {
            if (m == 0) {
                if (!(isinf(got) && (signbit(got) != 0) == s)) bad_special++;
            } else {
                if (!isnan(got) || ((bits_of(got) & 0x7fffffu) != ((uint32_t)m << 13)))
                    bad_special++;
            }
            continue;
        }
        /* exact value: normal (1024+m)*2^(e-25), subnormal m*2^-24; both exactly representable in
         * float */
        double v = (e == 0) ? ldexp((double)m, -24) : ldexp((double)(1024 + m), e - 25);
        if (s) v = -v;
        if (bits_of(got) != bits_of((float)v)) {
            bad_value++;
            if (e == 0 && m != 0) bad_sub++;
        }
        if (e == 0 && m != 0) subs++;
    }
    CHECK(bad_value == 0, "all 63488 finite values (normals, subnormals, +-0) are bit-exact");
    CHECK(bad_sub == 0 && subs == 2046, "all 2046 subnormals convert correctly (not flushed)");
    CHECK(bad_special == 0, "all infinities and NaNs (payload kept) are correct");
    CHECK(tn_fp16_to_f32(0x0001) == ldexpf(1.0f, -24), "smallest subnormal is 2^-24");
    CHECK(bits_of(tn_fp16_to_f32(0x8000)) == 0x80000000u, "negative zero keeps its sign");
    printf(g_fail ? "fp16: %d FAILED\n" : "fp16: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
