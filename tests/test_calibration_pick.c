/*
 * tests/test_calibration_pick.c — which SIMD tier calibration may choose, and
 * the cache key (2026-11-18).
 *
 * The old sweep measured each tier in its own block and picked "avx2" on an
 * AVX-512 VNNI host from noise. Candidates are now measured in interleaved
 * rounds and a tier replaces "auto" only if it wins a one-sided sign test on the
 * paired samples AND clears the median margin. The decision is a pure function,
 * so noise, ties and clear wins are tested with synthetic series.
 */
#include "test_harness.h"
#include "core/calibration.h"
#include "core/hardware_profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define R TN_CALIB_ROUNDS

static unsigned lcg(unsigned *s) {
    *s = *s * 1103515245u + 12345u;
    return (*s >> 16) & 0x7FFF;
}

/* Deterministic noise: base * (1 + amp * u), u in [-1, 1]. */
static void series(double *out, double base, double amp, unsigned seed) {
    for (int r = 0; r < R; r++)
        out[r] = base * (1.0 + amp * (((double)lcg(&seed) / 16383.5) - 1.0));
}

static void test_wins_needed(void) {
    TEST_ASSERT_EQ(tn_calib_wins_needed(15), 13,
                   "15 rounds: 13 wins (p = 0.0037; 12 wins is p = 0.0176)");
    TEST_ASSERT_EQ(tn_calib_wins_needed(9), 9,
                   "9 rounds: all 9 (p = 0.0020; 8 of 9 is p = 0.0195)");
    TEST_ASSERT(tn_calib_wins_needed(3) == 4,
                "3 rounds can never be significant (unreachable count)");
    TEST_ASSERT(tn_calib_wins_needed(0) == 1 && tn_calib_wins_needed(99) == 100,
                "out-of-range round counts never pass");
}

static void test_noise_never_wins(void) {
    /* Two candidates with the SAME mean and +/-20% noise: out of 200 seeds the
     * challenger must (almost) never replace auto. */
    int overrides = 0;
    for (unsigned seed = 1; seed <= 200; seed++) {
        static double s[2][TN_CALIB_MAX_ROUNDS];
        bool avail[2] = {true, true};
        series(s[0], 100.0, 0.20, seed * 2);
        series(s[1], 100.0, 0.20, seed * 2 + 1);
        if (tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])s, avail, 2, R,
                               TN_CALIB_OVERRIDE_MARGIN) != 0)
            overrides++;
    }
    printf("  noise-only overrides: %d of 200\n", overrides);
    TEST_ASSERT(overrides <= 2,
                "an identical candidate replaced auto in at most 1% of noisy trials");
}

static void test_clear_win_and_clear_loss(void) {
    static double s[3][TN_CALIB_MAX_ROUNDS];
    bool avail[3] = {true, true, true};
    series(s[0], 100.0, 0.10, 11);
    series(s[1], 140.0, 0.10, 12); /* 40% faster */
    series(s[2], 30.0, 0.10, 13);  /* 3x slower */
    TEST_ASSERT_EQ(tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])s, avail, 3, R,
                                      TN_CALIB_OVERRIDE_MARGIN),
                   1, "a clearly faster candidate wins");
    avail[1] = false;
    TEST_ASSERT_EQ(tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])s, avail, 3, R,
                                      TN_CALIB_OVERRIDE_MARGIN),
                   0, "a clearly slower candidate never wins; unavailable ones are ignored");
}

static void test_margin_and_ties(void) {
    static double s[2][TN_CALIB_MAX_ROUNDS];
    bool avail[2] = {true, true};
    for (int r = 0; r < R; r++) {
        s[0][r] = 100.0 + r;
        s[1][r] = 102.0 + r;
    } /* wins every round, +2% */
    TEST_ASSERT_EQ(tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])s, avail, 2, R,
                                      TN_CALIB_OVERRIDE_MARGIN),
                   0, "winning every round by 2% is below the 5% margin: auto stays");
    for (int r = 0; r < R; r++) {
        s[0][r] = 100.0 + r;
        s[1][r] = 100.0 + r;
    }
    TEST_ASSERT_EQ(tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])s, avail, 2, R,
                                      TN_CALIB_OVERRIDE_MARGIN),
                   0, "an exact tie keeps auto");
    for (int r = 0; r < R; r++) {
        s[0][r] = 100.0;
        s[1][r] = (r < 11) ? 200.0 : 50.0;
    } /* median 200 but only 11 wins */
    TEST_ASSERT_EQ(
        tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])s, avail, 2, R,
                           TN_CALIB_OVERRIDE_MARGIN),
        0, "11 of 15 wins is not significant even with a high median (12 and 13 are not either)");
    TEST_ASSERT_EQ(tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])s, avail, 2, 3,
                                      TN_CALIB_OVERRIDE_MARGIN),
                   0, "too few rounds never override");
}

/* The cache key: a result from a build with other tiers compiled in, or another
 * version, is never reused. HOME points at a scratch directory. */
static void test_cache_key(void) {
    char dir[] = "/tmp/pz_calib_test_XXXXXX";
    if (!mkdtemp(dir)) {
        printf("mkdtemp failed -- skipping\n");
        return;
    }
    setenv("HOME", dir, 1);

    TnHardwareProfile hw;
    memset(&hw, 0, sizeof(hw));
    hw.physical_cores = 4;
    hw.logical_cores  = 4;
    hw.l2_cache_bytes = 1u << 21;
    hw.l3_cache_bytes = 1u << 25;

    static TnCalibrationResult res;
    memset(&res, 0, sizeof(res));
    res.valid          = 1;
    res.physical_cores = 4;
    res.logical_cores  = 4;
    res.l2_cache_bytes = hw.l2_cache_bytes;
    res.l3_cache_bytes = hw.l3_cache_bytes;
    res.isa_mask       = tn_calibration_isa_mask();
    snprintf(res.best_simd_name, sizeof(res.best_simd_name), "auto");
    /* cpu_model must equal this host's: tn_calibrate fills it, we ask the same source */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f))
            if (!strncmp(line, "model name", 10)) {
                char *p = strchr(line, ':');
                if (!p) break;
                p++;
                while (*p == ' ') p++;
                size_t n = strlen(p);
                if (n && p[n - 1] == '\n') p[n - 1] = 0;
                snprintf(res.cpu_model, sizeof(res.cpu_model), "%s", p);
                break;
            }
        fclose(f);
    }

    tn_calibration_save(&res);
    static TnCalibrationResult back;
    TEST_ASSERT(tn_calibration_load(&back, &hw), "a result saved by this build and host loads");

    res.isa_mask ^= 1u << 7; /* a build with another tier set */
    tn_calibration_save(&res);
    TEST_ASSERT(!tn_calibration_load(&back, &hw),
                "a result from a different ISA mask is not reused");

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
    if (system(cmd) != 0) printf("cleanup of %s failed\n", dir);
}

static void test_act_quant_of_tier(void) {
    TnCalibrationResult r;
    memset(&r, 0, sizeof(r));
    TEST_ASSERT(tn_calibration_act_quant(&r) == NULL, "invalid result: no decision");
    r.valid = 1;
    snprintf(r.best_simd_name, sizeof(r.best_simd_name), "auto");
    TEST_ASSERT(tn_calibration_act_quant(&r) == NULL, "auto keeps the dispatcher's choice");
    snprintf(r.best_simd_name, sizeof(r.best_simd_name), "vnni");
    TEST_ASSERT(strcmp(tn_calibration_act_quant(&r), "int8") == 0, "vnni quantises activations");
    snprintf(r.best_simd_name, sizeof(r.best_simd_name), "vnni256");
    TEST_ASSERT(strcmp(tn_calibration_act_quant(&r), "int8") == 0, "vnni256 quantises activations");
    snprintf(r.best_simd_name, sizeof(r.best_simd_name), "avx2");
    TEST_ASSERT(strcmp(tn_calibration_act_quant(&r), "float") == 0, "avx2 keeps float activations");
    snprintf(r.best_simd_name, sizeof(r.best_simd_name), "scalar");
    TEST_ASSERT(strcmp(tn_calibration_act_quant(&r), "float") == 0,
                "scalar keeps float activations");
}

static void test_act_timing(void) {
    TnCalibrationResult r;
    memset(&r, 0, sizeof(r));
    double a = 0, b = 0;
    TEST_ASSERT_EQ(tn_calibration_act_timing(&r, 0, &a, &b), 0, "invalid result");
    r.valid = 1;
    TEST_ASSERT_EQ(tn_calibration_act_timing(&r, 0, &a, &b), 0, "nothing measured");
    r.simd_tokps[1] = 70;
    r.simd_tokps[2] = 90;
    r.simd_tokps[3] = 85;
    TEST_ASSERT_EQ(tn_calibration_act_timing(&r, 0, &a, &b), 0, "no int8 tier measured");
    r.simd_tokps[4] = 116;
    r.simd_tokps[5] = 110;
    TEST_ASSERT_EQ(tn_calibration_act_timing(&r, 0, &a, &b), 1, "both groups measured");
    TEST_ASSERT(a == 116 && b == 90, "best of each group");
    r.simd_tokps[0] = 120;
    TEST_ASSERT_EQ(tn_calibration_act_timing(&r, 1, &a, &b), 1, "auto counted with the int8 tiers");
    TEST_ASSERT(a == 120 && b == 90, "auto is int8 here");
    TEST_ASSERT_EQ(tn_calibration_act_timing(&r, 0, &a, &b), 1,
                   "auto counted with the float tiers");
    TEST_ASSERT(a == 116 && b == 120, "auto is float here");
}

int main(void) {
    RUN_TEST(test_wins_needed);
    RUN_TEST(test_noise_never_wins);
    RUN_TEST(test_clear_win_and_clear_loss);
    RUN_TEST(test_margin_and_ties);
    RUN_TEST(test_cache_key);
    RUN_TEST(test_act_quant_of_tier);
    RUN_TEST(test_act_timing);
    TEST_SUMMARY();
}
