/*
 * tests/test_config_validate_golden.c — pz_config_validate for every key of the schema against a
 * large set of values: each key's own bounds and the values just outside them, malformed and
 * out-of-range numbers (empty, signs, hex, exponents, nan, inf, trailing text, overflow), booleans,
 * every choice of every key and strings that merely start or end like one, newlines, values at and
 * over the length limit, and a missing key or value. Error text is captured in buffers of 0, 8,
 * 64 and 400 bytes (pre-filled, so a write past the end shows), and the hash covers the return
 * code and every buffer byte. Recorded from the validator before it was split into per-type steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/pz_config.h"
#include "test_harness.h"

#define EXPECTED_HASH 0xe7a3ec6ab45baf56ull

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static const char *const FIXED[] = {
    "",
    "0",
    "1",
    "-1",
    "+5",
    "007",
    "3.14",
    "-0.5",
    "1e3",
    "1e999",
    "-1e999",
    "nan",
    "inf",
    "-inf",
    "0x10",
    " 5",
    "5 ",
    "5x",
    "x5",
    "9223372036854775807",
    "9223372036854775808",
    "-9223372036854775809",
    "99999999999999999999",
    ".5",
    "5.",
    "true",
    "false",
    "True",
    "TRUE",
    "yes",
    "on",
    "tru",
    "falsey",
    "a\nb",
    "\n",
    "a b",
    "q8_0",
    "f16",
    "i8",
    "i4",
    "auto",
    "none",
    "off",
    "x",
    "|",
    "a|b",
};

static void run_one(const PzCfgKey *k, const char *value) {
    static const size_t SIZES[] = {0, 8, 64, 400};
    for (size_t s = 0; s < sizeof SIZES / sizeof SIZES[0]; s++) {
        char buf[400];
        memset(buf, 0x5a, sizeof buf);
        size_t n = SIZES[s];
        int rc   = pz_config_validate(k, value, n ? buf : NULL, n);
        mix(&rc, sizeof rc);
        mix(buf, sizeof buf);
    }
}

static void run_key(const PzCfgKey *k) {
    char v[600];
    for (size_t i = 0; i < sizeof FIXED / sizeof FIXED[0]; i++) run_one(k, FIXED[i]);
    /* the key's own bounds and values around them */
    double probes[] = {k->min,       k->max,       k->min - 1,           k->max + 1,
                       k->min + 0.5, k->max - 0.5, (k->min + k->max) / 2};
    for (size_t i = 0; i < sizeof probes / sizeof probes[0]; i++) {
        snprintf(v, sizeof v, "%g", probes[i]);
        run_one(k, v);
        snprintf(v, sizeof v, "%lld", (long long)probes[i]);
        run_one(k, v);
    }
    if (k->choices) {
        const char *p = k->choices;
        while (*p) {
            const char *bar = strchr(p, '|');
            size_t l        = bar ? (size_t)(bar - p) : strlen(p);
            snprintf(v, sizeof v, "%.*s", (int)l, p);
            run_one(k, v);
            snprintf(v, sizeof v, "%.*sx", (int)l, p);
            run_one(k, v);
            snprintf(v, sizeof v, "x%.*s", (int)l, p);
            run_one(k, v);
            if (l > 1) {
                snprintf(v, sizeof v, "%.*s", (int)(l - 1), p);
                run_one(k, v);
            }
            if (!bar) break;
            p = bar + 1;
        }
    }
    char *big = (char *)malloc(PZ_CFG_MAX_VALUE + 8);
    for (size_t len = PZ_CFG_MAX_VALUE - 2; len <= PZ_CFG_MAX_VALUE + 1; len++) {
        memset(big, '7', len);
        big[len] = '\0';
        run_one(k, big);
        memset(big, 'a', len);
        big[len] = '\0';
        run_one(k, big);
    }
    free(big);
}

static void test_validate_golden(void) {
    int n;
    const PzCfgKey *schema = pz_config_schema(&n);
    for (int i = 0; i < n; i++) run_key(&schema[i]);
    /* missing key or value */
    char buf[64];
    memset(buf, 0x5a, sizeof buf);
    int rc = pz_config_validate(NULL, "1", buf, sizeof buf);
    mix(&rc, sizeof rc);
    mix(buf, sizeof buf);
    memset(buf, 0x5a, sizeof buf);
    rc = pz_config_validate(&schema[0], NULL, buf, sizeof buf);
    mix(&rc, sizeof rc);
    mix(buf, sizeof buf);
    if (getenv("CONFIG_VALIDATE_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d keys)\n", (unsigned long long)g_hash, n);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "validation matches the recorded hash");
}

int main(void) {
    printf("=== pz_config_validate golden ===\n");
    RUN_TEST(test_validate_golden);
    TEST_SUMMARY();
}
