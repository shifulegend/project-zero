/*
 * tests/test_pz_config_parse_golden.c — pz_config_parse over 60,000 generated config texts
 * (comments, tables, flat keys, scalars, quoted strings with escapes, lists, malformed lines,
 * unknown keys, trailing junk, CRLF), hashing the loaded count, every warning and the resulting
 * settings. The hash was recorded from the function before it was split into smaller ones.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/pz_config.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x8da516003865696dull
#define N_TEXTS 60000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}
static void mix_str(const char *s) {
    mix(s, strlen(s));
    mix("\n", 1);
}
static void warn_hash(const char *m) {
    mix_str("W:");
    mix_str(m);
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const k_keys[]   = {"threads",      "top-p",
                                       "model",        "kv-type",
                                       "stop",         "cors-origin",
                                       "no-calibrate", "quality-tolerance",
                                       "ctx_size",     "seed",
                                       "nope",         "temperature",
                                       "Threads",      "",
                                       "a b",          "port"};
static const char *const k_values[] = {"8",
                                       "0",
                                       "-1",
                                       "1e3",
                                       "0.95",
                                       "true",
                                       "false",
                                       "yes",
                                       "f32",
                                       "i4",
                                       "i3",
                                       "\"x\"",
                                       "\"a b\"",
                                       "\"esc\\n\\t\\\"q\\\"\"",
                                       "\"bad\\q\"",
                                       "\"unterminated",
                                       "abc",
                                       "8x",
                                       "1024",
                                       "99999999999999999999",
                                       "[]",
                                       "[\"a\"]",
                                       "[\"a\", \"b\"]",
                                       "[a,b,c]",
                                       "[\"a\",]",
                                       "[,]",
                                       "[\"a\" \"b\"]",
                                       "[\"a\"",
                                       "\"a\" # note",
                                       "8 # note",
                                       "8 junk",
                                       "[1, 2] junk",
                                       "[\"x\"] # c",
                                       "{}",
                                       "'single'",
                                       "\"\"",
                                       ""};
static const char *const k_seps[]   = {" = ", "=", "\t=\t", "  =  ", " ", " == ", ": "};
static const char *const k_other[]  = {"",
                                       "# comment",
                                       "   # indented comment",
                                       "[table]",
                                       "  [other] # t",
                                       "just words",
                                       "= 5",
                                       "key",
                                       "key =",
                                       "\t",
                                       "   "};
static const char *const k_eol[]    = {"\n", "\r\n", "\n\n", "\n   \n"};

static void gen_line(char *out, size_t cap) {
    out[0] = '\0';
    size_t n;
    if (rnd(5) == 0) {
        snprintf(out, cap, "%s", k_other[rnd(sizeof k_other / sizeof *k_other)]);
        return;
    }
    if (rnd(6) == 0) strncat(out, rnd(2) ? "  " : "\t", cap - strlen(out) - 1);
    strncat(out, k_keys[rnd(sizeof k_keys / sizeof *k_keys)], cap - strlen(out) - 1);
    strncat(out, k_seps[rnd(sizeof k_seps / sizeof *k_seps)], cap - strlen(out) - 1);
    n = 1 + rnd(3) * (rnd(4) == 0);
    for (size_t i = 0; i < n; i++) {
        if (i) strncat(out, rnd(2) ? ", " : " ", cap - strlen(out) - 1);
        strncat(out, k_values[rnd(sizeof k_values / sizeof *k_values)], cap - strlen(out) - 1);
    }
}

static void test_parse_golden(void) {
    for (int t = 0; t < N_TEXTS; t++) {
        char text[1024];
        text[0]     = '\0';
        unsigned nl = 1 + rnd(6);
        for (unsigned i = 0; i < nl; i++) {
            char line[200];
            gen_line(line, sizeof line);
            strncat(text, line, sizeof text - strlen(text) - 1);
            strncat(text, k_eol[rnd(sizeof k_eol / sizeof *k_eol)], sizeof text - strlen(text) - 1);
        }
        /* sometimes no final newline */
        if (rnd(7) == 0) text[strlen(text) > 0 ? strlen(text) - 1 : 0] = '\0';
        PzConfig c;
        pz_config_init(&c);
        int loaded = pz_config_parse(&c, text, warn_hash);
        mix(&loaded, sizeof loaded);
        char *dump = pz_config_format(&c);
        mix_str(dump ? dump : "(null)");
        free(dump);
        pz_config_free(&c);
    }
    PzConfig c;
    pz_config_init(&c);
    TEST_ASSERT(pz_config_parse(NULL, "a=1", NULL) == 0, "NULL config");
    TEST_ASSERT(pz_config_parse(&c, NULL, NULL) == 0, "NULL text");
    pz_config_free(&c);
    if (getenv("PZ_CONFIG_GOLDEN_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "config parse results match the recorded hash");
}

int main(void) {
    printf("=== pz_config_parse golden ===\n");
    RUN_TEST(test_parse_golden);
    TEST_SUMMARY();
}
