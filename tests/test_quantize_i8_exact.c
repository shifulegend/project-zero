/*
 * test_quantize_i8_exact.c — the activation quantiser's specification (include/math/quantize_i8.h)
 *
 * 2026-11-18: the int8 quantiser truncated instead of rounding (3.7 % layer error instead of 0.4 %,
 * +9.7 % perplexity on BitNet 2B; docs/rca/2026-11-18-ternary-output-correctness.md). The old test
 * allowed an error of one full step, and every tier truncated identically, so nothing failed.
 * This test pins the specification:
 *   1. the scalar function rounds to nearest, ties to even (explicit tie vectors);
 *   2. its error never exceeds half a step against an exact double-precision value;
 *   3. every tier equals the scalar oracle BIT FOR BIT on random, adversarial and ragged-length input.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/platform.h"
#include "math/quantize_i8.h"
#include "test_harness.h"

typedef float (*quantize_fn)(const float *, int8_t *, int);

typedef struct {
    const char *name;
    quantize_fn fn;
} Tier;

static unsigned rng_state = 12345u;
static float rnd(void) { /* uniform in [-1, 1) */
    rng_state = rng_state * 1103515245u + 12345u;
    return (float)((rng_state >> 8) & 0xFFFFu) / 32768.0f - 1.0f;
}

static int tiers(Tier *t) {
    int n  = 0;
    t[n++] = (Tier){"neon", quantize_row_to_i8_neon}; /* scalar on non-NEON builds */
#if TN_HAS_AVX2
    t[n++] = (Tier){"avx2", quantize_row_to_i8_avx2};
#endif
#if TN_HAS_AVX512
    t[n++] = (Tier){"avx512", quantize_row_to_i8_avx512};
#endif
    return n;
}

/* abs_max = 127 makes inv_scale exactly 1.0f, so x[i] * inv_scale is x[i]: exact ties are testable. */
static void test_ties_round_to_even(void) {
    float x[16]           = {127.0f, 0.5f,  1.5f,  2.5f,   3.5f,   -0.5f,  -1.5f,   -2.5f,
                             -3.5f,  0.49f, 0.51f, -0.49f, -0.51f, 126.5f, -126.5f, 0.0f};
    const int8_t want[16] = {127, 0, 2, 2, 4, 0, -2, -2, -4, 0, 1, 0, -1, 126, -126, 0};
    int8_t q[16];
    TEST_ASSERT_FLOAT_EQ(quantize_row_to_i8(x, q, 16), 1.0f, 1e-7f, "ties: scale is abs_max/127");
    int ok = 1;
    for (int i = 0; i < 16; i++)
        if (q[i] != want[i]) {
            ok = 0;
            printf("    scalar x=%g got %d want %d\n", x[i], q[i], want[i]);
        }
    TEST_ASSERT(ok, "scalar rounds to nearest, ties to even (truncation would give 0 1 2 3 for the "
                    "first four)");
    Tier t[4];
    int nt = tiers(t);
    for (int k = 0; k < nt; k++) {
        int8_t qt[16];
        t[k].fn(x, qt, 16);
        TEST_ASSERT(memcmp(q, qt, 16) == 0, t[k].name);
    }
}

static void test_zero_and_single(void) {
    float z[37] = {0};
    int8_t q[37];
    memset(q, 7, sizeof q);
    TEST_ASSERT(quantize_row_to_i8(z, q, 37) == 0.0f, "all-zero: scale 0");
    int ok = 1;
    for (int i = 0; i < 37; i++) ok &= (q[i] == 0);
    TEST_ASSERT(ok, "all-zero: q all zero");
    float one = -3.25f;
    int8_t q1;
    float s = quantize_row_to_i8(&one, &q1, 1);
    TEST_ASSERT(q1 == -127, "single element maps to +-127");
    TEST_ASSERT_FLOAT_EQ(s * q1, one, 1e-5f, "single element reconstructs");
}

/* Error against the exact value x * 127 / abs_max computed in double. Rounding to nearest cannot be
 * further than 0.5 (plus a hair for the float arithmetic); truncation reaches 1.0. */
static void test_half_step_error_bound(void) {
    enum { N = 2560 };
    static float x[N];
    static int8_t q[N];
    double worst = 0.0;
    for (int trial = 0; trial < 20; trial++) {
        for (int i = 0; i < N; i++)
            x[i] = rnd() * (trial % 3 == 0 ? 0.01f : 1.0f) * (i % 97 == 0 ? 8.0f : 1.0f);
        float scale = quantize_row_to_i8(x, q, N);
        double amax = 0;
        for (int i = 0; i < N; i++)
            if (fabs((double)x[i]) > amax) amax = fabs((double)x[i]);
        for (int i = 0; i < N; i++) {
            double e = fabs((double)q[i] - (double)x[i] * 127.0 / amax);
            if (e > worst) worst = e;
        }
        (void)scale;
    }
    printf("    worst |q - x*127/absmax| = %.4f steps\n", worst);
    TEST_ASSERT(worst <= 0.5 + 1e-3,
                "quantisation error is at most half a step (truncation gives up to 1.0)");
}

static void check_tier_equals_oracle(const Tier *t, const float *base, int n, const char *what) {
    int8_t qo[512], qt[512];
    float so = quantize_row_to_i8(base, qo, n);
    float st = t->fn(base, qt, n);
    int same = (so == st) && memcmp(qo, qt, (size_t)n) == 0;
    if (!same)
        printf("    %s: %s differs from the oracle (n=%d, scale %g vs %g)\n", what, t->name, n, so,
               st);
    TEST_ASSERT(same, what);
}

static void test_tiers_equal_oracle_bit_for_bit(void) {
    Tier t[4];
    int nt = tiers(t);
    static float buf[512 + 8];
    for (int k = 0; k < nt; k++) {
        /* every length 1..300 (vector bodies and scalar tails), random data, unaligned start */
        for (int n = 1; n <= 300; n++) {
            for (int i = 0; i < n + 1; i++) buf[i] = rnd() * 3.0f;
            check_tier_equals_oracle(&t[k], buf + 1, n, "random, unaligned");
        }
        /* dynamic range: tiny values next to a large outlier; ties made at several scales */
        for (int n = 1; n <= 300; n += 7) {
            for (int i = 0; i < n; i++) buf[i] = (i % 11 == 0 ? 1.0f : 1e-4f) * rnd();
            buf[n / 2] = 5000.0f;
            check_tier_equals_oracle(&t[k], buf, n, "outlier");
        }
        for (int n = 1; n <= 200;
             n += 3) { /* abs_max 127: every x is its own product, ties included */
            for (int i = 0; i < n; i++) buf[i] = (float)((i * 7) % 255 - 127) * 0.5f;
            buf[0] = 127.0f;
            check_tier_equals_oracle(&t[k], buf, n, "ties at abs_max 127");
        }
        for (int i = 0; i < 64; i++) buf[i] = (i & 1) ? -0.0f : 0.0f; /* signed zeros */
        check_tier_equals_oracle(&t[k], buf, 64, "signed zeros");
    }
}

int main(void) {
    printf("=== quantize_i8 specification ===\n");
    RUN_TEST(test_ties_round_to_even);
    RUN_TEST(test_zero_and_single);
    RUN_TEST(test_half_step_error_bound);
    RUN_TEST(test_tiers_equal_oracle_bit_for_bit);
    TEST_SUMMARY();
}
