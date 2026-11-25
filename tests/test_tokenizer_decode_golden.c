/*
 * tests/test_tokenizer_decode_golden.c — tokenizer_decode over an in-memory vocabulary of pieces
 * that hit every branch: raw byte tokens `<0xHH>` (upper and lower case hex digits, non-hex
 * digits, too short, wrong closing character), pieces with the byte-level BPE markers (the 2-byte
 * G-dot space marker, a literal space, multi-byte UTF-8), empty and NULL pieces, and out-of-range
 * or negative token ids, each decoded after BOS and after other tokens, with and without a BOS id
 * set. The hash covers every returned string, recorded from the function before it was split.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "tokenizer/tokenizer.h"

#define EXPECTED_HASH 0x5362c78f417a5fe0ull

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static const char *const PIECES[] = {
    "<0x00>",
    "<0x41>",
    "<0x7F>",
    "<0xFF>",
    "<0xab>",
    "<0xff>",
    "<0xcf>",
    "<0xfa>",
    "<0xEF>",
    "<0xAB>",
    "<0xaB>",
    "<0xGG>",
    "<0x4G>",
    "<0xZ1>",
    "<0x4>",
    "<0x41",
    "<0x41x",
    "<0x41>>",
    "<x41>",
    "<0X41>",
    "<0x0A>",
    "<0x20>",
    "\xC4\xA0hello",
    "\xC4\xA0",
    "\xC4",
    "\xC4\xA1x",
    " hello",
    " ",
    "  two",
    "hello",
    "world",
    "\xE4\xB8\xAD\xE6\x96\x87",
    "\xC4\xA0\xE4\xB8\xAD",
    "\xC4\x8A",
    "\xC4\xA0\xC4\xA0",
    "",
    "a",
    "ab",
    "<unk>",
    "</s>",
    "<s>",
    "\xC3\xA9",
    "\xC4\xA0\xC3\xA9",
    NULL,
};
#define N_PIECES ((int)(sizeof PIECES / sizeof PIECES[0]))

static void decode_all(Tokenizer *t, int bos_id) {
    t->bos_token_id = bos_id;
    for (int prev = -2; prev < N_PIECES + 1; prev++) {
        for (int tok = -2; tok < N_PIECES + 2; tok++) {
            const char *out = tokenizer_decode(t, prev, tok);
            int len         = out ? (int)strlen(out) : -1;
            mix(&len, sizeof len);
            if (out) mix(out, (size_t)len + 1);
        }
    }
}

static void test_decode_golden(void) {
    Tokenizer t;
    memset(&t, 0, sizeof t);
    t.vocab      = (char **)PIECES;
    t.vocab_size = N_PIECES;
    for (int bos = -1; bos < 6; bos++) decode_all(&t, bos);
    /* no vocabulary at all, and no tokenizer */
    Tokenizer empty;
    memset(&empty, 0, sizeof empty);
    const char *e1 = tokenizer_decode(&empty, 0, 0);
    const char *e2 = tokenizer_decode(NULL, 0, 0);
    mix(e1, strlen(e1) + 1);
    mix(e2, strlen(e2) + 1);
    if (getenv("TOKENIZER_DECODE_GOLDEN_PRINT"))
        printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "decoded text matches the recorded hash");
}

int main(void) {
    printf("=== tokenizer_decode golden ===\n");
    RUN_TEST(test_decode_golden);
    TEST_SUMMARY();
}
