/*
 * tests/test_pretokenize_golden.c — every pre-tokenizer type over 60,000 generated texts each,
 * built from pieces that hit every character-class pattern (digits, runs of punctuation and of
 * the symbols $ + < = > ^ ~ |, spaces, newlines, trailing blanks, CJK / Thai / Hangul, Latin,
 * Greek, Cyrillic, Armenian, Georgian, fullwidth forms, quotes, emoji, malformed UTF-8). Each
 * text sits in a heap block of exactly its length so a read past the end is an ASan failure.
 * The hash covers the piece count and every piece's start and length, recorded from the
 * pre-tokenizer before cls_match was split into one function per pattern.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "tokenizer/pretokenize.h"

#define EXPECTED_HASH 0xc59a01fd85f1f7aeull
#define N_CASES 60000
#define MAX_PIECES 512

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x0fedcba987654321ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const BITS[] = {
    "a",
    "Zebra",
    "hello",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\n\n",
    "0",
    "12",
    "123",
    "1234567",
    "3.14",
    "!",
    "?!?",
    "...",
    "$",
    "+",
    "<=>",
    "^~|",
    "$+<=>^~|",
    "(",
    ")",
    "'s",
    "'re",
    "\"",
    "-",
    "_",
    "/",
    ":",
    "@",
    "#",
    "{}",
    "\xE4\xB8\xAD\xE6\x96\x87",
    "\xE3\x81\x82",
    "\xED\x95\x9C\xEA\xB8\x80",
    "\xE0\xB8\x81\xE0\xB8\x82",
    "\xCE\xB1\xCE\xB2",
    "\xD0\x96\xD0\xB6",
    "\xD4\xB1",
    "\xE1\x82\xA0",
    "\xC3\xA9",
    "\xC3\x97",
    "\xC3\xB7",
    "\xEF\xBC\xA1",
    "\xEF\xBC\x81",
    "\xEF\xBC\x9A",
    "\xE2\x80\x98",
    "\xE3\x80\x80",
    "\xE3\x80\x82",
    "\xC2\xA0",
    "\xF0\x9F\x98\x80",
    "\xE2\x82\xAC",
    "\xC2\xB5",
    "\xFF",
    "\xE6\x9D",
    "\xC3",
};

static void one_case(PretokType t) {
    char buf[600];
    size_t n   = 0;
    unsigned k = 1 + rnd(14);
    while (k-- > 0) {
        const char *b = BITS[rnd(sizeof BITS / sizeof BITS[0])];
        size_t bl     = strlen(b);
        if (n + bl >= sizeof buf) break;
        memcpy(buf + n, b, bl);
        n += bl;
    }
    char *text = (char *)malloc(n ? n : 1);
    memcpy(text, buf, n);
    size_t st[MAX_PIECES], ln[MAX_PIECES];
    int np = pretokenize(t, text, n, st, ln, MAX_PIECES);
    mix(&np, sizeof np);
    for (int i = 0; i < np; i++) {
        mix(&st[i], sizeof st[i]);
        mix(&ln[i], sizeof ln[i]);
    }
    free(text);
}

static void test_pretokenize_golden(void) {
    static const PretokType TYPES[] = {PRETOK_NONE,         PRETOK_DEFAULT,       PRETOK_GPT2,
                                       PRETOK_LLAMA3,       PRETOK_QWEN2,         PRETOK_SMOLLM,
                                       PRETOK_DEEPSEEK_LLM, PRETOK_DEEPSEEK_CODER};
    for (size_t ti = 0; ti < sizeof TYPES / sizeof TYPES[0]; ti++)
        for (int c = 0; c < N_CASES; c++) one_case(TYPES[ti]);
    if (getenv("PRETOK_GOLDEN_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "pre-tokenization matches the recorded hash");
}

int main(void) {
    printf("=== pretokenize golden ===\n");
    RUN_TEST(test_pretokenize_golden);
    TEST_SUMMARY();
}
