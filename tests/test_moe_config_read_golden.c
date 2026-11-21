/*
 * tests/test_moe_config_read_golden.c — moe_config_read over 400,000 generated .bin MoE headers:
 * the 48-byte layout, the legacy 32-byte one, and every size from 0 to 60 bytes, read at offsets
 * 0..4, with each field drawn from the values at and around its limits (0, 1, 2, 64, 65, 512, 513,
 * negatives, INT_MIN / INT_MAX). Each header sits in a heap block of exactly the size handed to
 * the reader, so a read past the end is an ASan failure. The hash covers the return code, the
 * offset and every byte of the config (pre-filled with 0xAA so padding and untouched fields count),
 * recorded from the reader before it was split into steps.
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/moe_config.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x6849be49df33f361ull
#define N_CASES 400000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0xdeadbeefcafef00dull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const int32_t VALS[] = {0,   1,   2,   3,   7,  8,  63,      64,      65,
                               100, 511, 512, 513, -1, -2, INT_MIN, INT_MAX, 4096};

static int32_t pick(void) {
    return VALS[rnd(sizeof VALS / sizeof VALS[0])];
}

static int g_ok, g_mla;

static void one_case(void) {
    unsigned size = rnd(61);
    size_t offset = rnd(5);
    uint8_t *data = (uint8_t *)malloc(size ? size : 1);
    /* Mostly plausible headers: valid core fields, then one field perturbed. */
    int32_t f[12] = {8, 2, 16, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    if (rnd(3) != 0) {
        f[5] = (int32_t)rnd(3);
        f[6] = 4;
        if (rnd(2)) {
            f[7]  = 1;
            f[8]  = 32;
            f[9]  = 8;
            f[10] = 4;
            f[11] = 8;
        }
    }
    unsigned nperturb = rnd(3);
    for (unsigned k = 0; k < nperturb; k++) {
        unsigned slot = rnd(12);
        f[slot]       = pick();
    }
    if (rnd(8) == 0) {
        unsigned slot = rnd(12);
        f[slot]       = pick();
    }
    for (unsigned i = 0; i < size; i++) data[i] = (uint8_t)rnd(256);
    for (unsigned i = 0; i < 12; i++) {
        size_t at = offset + (size_t)i * 4;
        if (at + 4 <= size) memcpy(data + at, &f[i], 4);
    }
    MoEConfig mc;
    memset(&mc, 0xAA, sizeof mc);
    size_t off = offset;
    int rc     = (int)moe_config_read(&mc, data, size, &off);
    g_ok += rc == 0;
    g_mla += rc == 0 && mc.mla.enabled;
    mix(&rc, sizeof rc);
    mix(&off, sizeof off);
    mix(&mc, sizeof mc);
    free(data);
}

static void test_moe_config_read_golden(void) {
    for (int t = 0; t < N_CASES; t++) one_case();
    if (getenv("MOE_READ_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d accepted, %d with MLA)\n", (unsigned long long)g_hash, g_ok,
               g_mla);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "header reading matches the recorded hash");
}

int main(void) {
    printf("=== moe_config_read golden ===\n");
    RUN_TEST(test_moe_config_read_golden);
    TEST_SUMMARY();
}
