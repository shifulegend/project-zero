/*
 * tests/test_thought_filter_golden.c — thought_filter_process over 100,000 generated token
 * streams: whole tags, partial and broken tags (`<`, `<th`, `<thinx>`, `</thi`, `</think `), tags
 * split across tokens, text, newlines and empty tokens, fed through output buffers of 1..24 bytes
 * (the small ones hit every bounds check). After every call the hash covers the return value, the
 * output text and the filter's whole state (state, tag position, thinking-token count, tag
 * buffer), recorded from the filter before thought_filter_process was split into one function per
 * state.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "reasoning/thought_filter.h"
#include "test_harness.h"

#define EXPECTED_HASH 0xcf6b335c89c8ca0cull
#define N_STREAMS 100000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0xa5a5a5a5c3c3c3c3ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const TOKENS[] = {
    "<think>",
    "</think>",
    "<",
    "<t",
    "<th",
    "<thi",
    "<thin",
    "<think",
    "think>",
    "hink>",
    "</",
    "</t",
    "</th",
    "</thi",
    "</thin",
    "</think",
    "ink>",
    ">",
    "<thinx>",
    "<thinkk>",
    "</thinx>",
    "</think >",
    "<<think>",
    "<</think>",
    "a",
    "hello",
    " world",
    "\n",
    "",
    "x<y",
    "a<b>c",
    "<think>hi",
    "</think>bye",
    "ab</think>cd",
    "<think>x</think>y",
    "<tool>",
    "</s>",
    "0123456789012345678",
};

static void test_thought_filter_golden(void) {
    char out[32];
    for (int s = 0; s < N_STREAMS; s++) {
        ThoughtFilter f;
        thought_filter_init(&f);
        size_t cap = 1 + rnd(24);
        for (unsigned k = 1 + rnd(8); k > 0; k--) {
            const char *tok = TOKENS[rnd(sizeof TOKENS / sizeof TOKENS[0])];
            memset(out, 0x7f, sizeof out);
            int r = thought_filter_process(&f, tok, out, cap);
            mix(&r, sizeof r);
            mix(out, cap);
            mix(&f, sizeof f);
        }
    }
    if (getenv("THOUGHT_FILTER_GOLDEN_PRINT"))
        printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "filter output and state match the recorded hash");
}

int main(void) {
    printf("=== thought filter golden ===\n");
    RUN_TEST(test_thought_filter_golden);
    TEST_SUMMARY();
}
