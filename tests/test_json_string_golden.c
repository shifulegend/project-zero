/*
 * tests/test_json_string_golden.c — json_parse_string / json_parse_string_heap over 200,000 generated
 * JSON strings built from escapes (simple, \/, \uXXXX, surrogate pairs, lone and malformed ones),
 * raw UTF-8, control bytes and truncations, parsed into buffers of 1..64 bytes. Each input sits in a
 * heap block of exactly its length + NUL, so a read past the end is an ASan failure. The hash covers
 * the return code, how far the cursor moved and every byte of the destination (pre-filled with 0xAA),
 * recorded from the parser before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/json_scan.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x217bf40b033fcd80ull
#define N_CASES 200000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x1234567887654321ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const PIECES[] = {
    "a",
    "Z",
    " ",
    "hello",
    "\\n",
    "\\t",
    "\\r",
    "\\b",
    "\\f",
    "\\\"",
    "\\\\",
    "\\/",
    "\\u0041",
    "\\u00e9",
    "\\u20ac",
    "\\ud83d\\ude00",
    "\\ud83d",
    "\\ude00",
    "\\ud83dx",
    "\\ud83d\\u0041",
    "\\uZZZZ",
    "\\u12",
    "\\u",
    "\\x",
    "\\0",
    "\xc3\xa9",
    "\xe2\x82\xac",
    "\xf0\x9f\x98\x80",
    "\x01",
    "\x7f",
    "\\uD83D\\uDE00",
    "\\uDBFF\\uDFFF",
    "\\u0000",
    "\\uFFFF",
    "\\u007F",
};
#define N_PIECES (sizeof PIECES / sizeof PIECES[0])

static void one_case(void) {
    char text[512];
    size_t n = 0;
    if (rnd(12) != 0) text[n++] = '"'; /* sometimes no opening quote */
    unsigned pieces = rnd(10);
    for (unsigned i = 0; i < pieces; i++) {
        const char *s = PIECES[rnd(N_PIECES)];
        size_t l      = strlen(s);
        memcpy(text + n, s, l);
        n += l;
    }
    unsigned end = rnd(10);
    if (end < 7)
        text[n++] = '"'; /* terminated */
    else if (end == 7)
        text[n++] = '\\'; /* trailing backslash */
    /* else: unterminated */
    if (rnd(4) == 0) text[n++] = 'x'; /* something after */
    if (rnd(5) == 0) {                /* cut anywhere */
        size_t cut = rnd((unsigned)n + 1);
        n          = cut;
    }
    char *in = (char *)malloc(n + 1);
    memcpy(in, text, n);
    in[n] = '\0';

    static const size_t CAPS[] = {1, 2, 3, 4, 5, 8, 16, 64};
    size_t cap                 = CAPS[rnd(8)];
    unsigned char dst[64];
    memset(dst, 0xAA, sizeof dst);
    const char *p = in;
    int rc        = json_parse_string(&p, (char *)dst, cap);
    size_t used   = (size_t)(p - in);
    mix(&rc, sizeof rc);
    mix(&used, sizeof used);
    mix(dst, cap + 1 < sizeof dst ? cap + 1 : sizeof dst); /* decoded bytes + the byte after */

    p        = in;
    char *hp = json_parse_string_heap(&p);
    used     = (size_t)(p - in);
    int ok   = hp != NULL;
    mix(&ok, sizeof ok);
    mix(&used, sizeof used);
    if (hp) mix(hp, strlen(hp) + 1);
    free(hp);
    free(in);
}

static void test_json_string_golden(void) {
    for (int t = 0; t < N_CASES; t++) one_case();
    if (getenv("JSON_STRING_GOLDEN_PRINT"))
        printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "string parsing matches the recorded hash");
}

int main(void) {
    printf("=== json_parse_string golden ===\n");
    RUN_TEST(test_json_string_golden);
    TEST_SUMMARY();
}
