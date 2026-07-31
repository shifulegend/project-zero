#include "test_harness.h"
#include "tokenizer/tokenizer.h"
#include "core/error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Helper: write a synthetic tokenizer binary file ---------- */

/*
 * Binary format:
 *   [vocab_size: int] [max_token_len: int]
 *   for each token:
 *     [score: float] [len: int] [bytes: char[len]]
 *
 * We build a minimal vocab:
 *   0: "<unk>"    score -1.0
 *   1: "<s>"      score -1.0  (BOS)
 *   2: "</s>"     score -1.0  (EOS)
 *   3-258: single bytes 0x00-0xFF  score -100.0
 *          (only printable ASCII 32-126 + newline for simplicity)
 *   Then merge tokens:
 *   259: "ab"     score 10.0
 *   260: "cd"     score 9.0
 *   261: "abcd"   score 20.0
 *   262: " h"     score 5.0
 *   263: " he"    score 6.0
 *   264: " hel"   score 7.0
 *   265: " hello" score 8.0
 *   266: "<0x0A>" score -1.0  (raw byte token for newline)
 *
 *   GPT-2 byte-level-BPE-encoded pieces (as a real GGUF vocab would store
 *   them, unlike the literal-text merge tokens above — see tokenizer_decode.c
 *   for the reverse mapping these exercise):
 *   267: 0xC4 0xA0 "world"  (Ġworld — byte-level space marker + "world")
 *   268: 0xC2 0xA9          (byte-level encoding of raw byte 0xA9, "©")
 *   269: 0xC3 0xA2          (byte-level encoding of raw byte 0xE2)
 *   270: 0xC4 0xBE          (byte-level encoding of raw byte 0x9C)
 *   271: 0xC4 0xA7          (byte-level encoding of raw byte 0x85)
 *   269-271 concatenated decode to the raw UTF-8 bytes of U+2705 (✅),
 *   split across 3 separate tokens the way a real BPE tokenizer would.
 *   272: 0xC4 0xA0 "hi"     (Ġhi — for the byte-level BOS-leading-space case)
 */

#define SPECIAL_TOKENS 3
#define BYTE_TOKENS 256
#define MERGE_TOKENS 14
#define TEST_VOCAB_SIZE (SPECIAL_TOKENS + BYTE_TOKENS + MERGE_TOKENS)
#define TEST_MAX_TOKEN_LEN 16

static const char *test_tokenizer_path = "/tmp/tn_test_tokenizer.bin";

static void write_token_len(FILE *fp, float score, const char *str, int len) {
    fwrite(&score, sizeof(float), 1, fp);
    fwrite(&len, sizeof(int), 1, fp);
    if (len > 0) fwrite(str, 1, (size_t)len, fp);
}

static void write_token(FILE *fp, float score, const char *str) {
    write_token_len(fp, score, str, (int)strlen(str));
}

static int create_test_tokenizer_file(void) {
    FILE *fp = fopen(test_tokenizer_path, "wb");
    if (!fp) return 0;

    int vocab_size = TEST_VOCAB_SIZE;
    int max_token_len = TEST_MAX_TOKEN_LEN;
    fwrite(&vocab_size, sizeof(int), 1, fp);
    fwrite(&max_token_len, sizeof(int), 1, fp);

    /* Special tokens: 0, 1, 2 */
    write_token(fp, -1.0f, "<unk>");
    write_token(fp, -1.0f, "<s>");
    write_token(fp, -1.0f, "</s>");

    /* Single byte tokens: indices 3..258 map to bytes 0x00..0xFF */
    for (int b = 0; b < 256; b++) {
        char single[1] = { (char)b };
        write_token_len(fp, -100.0f, single, 1);
    }

    /* Merge tokens */
    write_token(fp, 10.0f, "ab");       /* 259 */
    write_token(fp, 9.0f,  "cd");       /* 260 */
    write_token(fp, 20.0f, "abcd");     /* 261 */
    write_token(fp, 5.0f,  " h");       /* 262 */
    write_token(fp, 6.0f,  " he");      /* 263 */
    write_token(fp, 7.0f,  " hel");     /* 264 */
    write_token(fp, 8.0f,  " hello");   /* 265 */
    write_token(fp, -1.0f, "<0x0A>");   /* 266 */

    /* GPT-2 byte-level-BPE-encoded pieces — see header comment above. */
    write_token(fp, -1.0f, "\xC4\xA0world");  /* 267: Ġworld */
    write_token(fp, -1.0f, "\xC2\xA9");       /* 268: byte 0xA9 */
    write_token(fp, -1.0f, "\xC3\xA2");       /* 269: byte 0xE2 (emoji byte 1/3) */
    write_token(fp, -1.0f, "\xC4\xBE");       /* 270: byte 0x9C (emoji byte 2/3) */
    write_token(fp, -1.0f, "\xC4\xA7");       /* 271: byte 0x85 (emoji byte 3/3) */
    write_token(fp, -1.0f, "\xC4\xA0hi");     /* 272: Ġhi */

    fclose(fp);
    return 1;
}

/* ---------- Tests ---------- */

static void test_tokenizer_load_basic(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "tokenizer_load should succeed");
    TEST_ASSERT_EQ(t.vocab_size, TEST_VOCAB_SIZE, "vocab_size should match");
    TEST_ASSERT_EQ(t.max_token_len, TEST_MAX_TOKEN_LEN, "max_token_len should match");
    TEST_ASSERT(t.sorted == 1, "sorted index should be built");

    /* Check special tokens */
    TEST_ASSERT(strcmp(t.vocab[0], "<unk>") == 0, "token 0 is <unk>");
    TEST_ASSERT(strcmp(t.vocab[1], "<s>") == 0, "token 1 is <s>");
    TEST_ASSERT(strcmp(t.vocab[2], "</s>") == 0, "token 2 is </s>");

    /* Check a single-byte token (e.g., 'a' = 0x61 -> index 3 + 0x61 = 100) */
    int a_idx = 3 + 'a';
    TEST_ASSERT(strcmp(t.vocab[a_idx], "a") == 0, "byte token for 'a'");

    /* Check merge tokens */
    TEST_ASSERT(strcmp(t.vocab[259], "ab") == 0, "merge token 'ab'");
    TEST_ASSERT_FLOAT_EQ(t.vocab_scores[259], 10.0f, 0.001f, "score for 'ab'");
    TEST_ASSERT(strcmp(t.vocab[261], "abcd") == 0, "merge token 'abcd'");
    TEST_ASSERT_FLOAT_EQ(t.vocab_scores[261], 20.0f, 0.001f, "score for 'abcd'");

    tokenizer_free(&t);
}

static void test_tokenizer_load_bad_path(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, "/tmp/nonexistent_tokenizer_file.bin");
    TEST_ASSERT_EQ(err, TN_ERR_TOKENIZER_LOAD, "load from bad path should fail");
}

static void test_tokenizer_encode_single_chars(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for encode test");

    /* Encode "xy" — no merges possible, should be two single-byte tokens */
    int tokens[64];
    int n = tokenizer_encode(&t, "xy", strlen("xy"), tokens, 64);
    TEST_ASSERT_EQ(n, 2, "encode 'xy' -> 2 tokens");
    TEST_ASSERT_EQ(tokens[0], 3 + 'x', "'x' maps to byte token");
    TEST_ASSERT_EQ(tokens[1], 3 + 'y', "'y' maps to byte token");

    tokenizer_free(&t);
}

static void test_tokenizer_encode_bpe_merge(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for BPE merge test");

    /* Encode "ab" — should merge to token 259 ("ab", score 10.0) */
    int tokens[64];
    int n = tokenizer_encode(&t, "ab", strlen("ab"), tokens, 64);
    TEST_ASSERT_EQ(n, 1, "encode 'ab' -> 1 token (merged)");
    TEST_ASSERT_EQ(tokens[0], 259, "'ab' merges to token 259");

    tokenizer_free(&t);
}

static void test_tokenizer_encode_cascading_merge(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for cascading merge test");

    /* Encode "abcd":
     * Initial: 'a' 'b' 'c' 'd' (4 byte tokens)
     * Highest merge: "abcd" (score 20.0) — but first we need "ab" and "cd"
     * Round 1: merge "ab" (score 10.0) -> [ab] c d  (3 tokens)
     *   Actually: best pair scoring: ab=10, cd=9 -> merge ab first
     * Round 2: merge "cd" (score 9.0) -> [ab] [cd] (2 tokens)
     * Round 3: merge "abcd" (score 20.0) -> [abcd] (1 token)
     */
    int tokens[64];
    int n = tokenizer_encode(&t, "abcd", strlen("abcd"), tokens, 64);
    TEST_ASSERT_EQ(n, 1, "encode 'abcd' -> 1 token (cascading merge)");
    TEST_ASSERT_EQ(tokens[0], 261, "'abcd' merges to token 261");

    tokenizer_free(&t);
}

static void test_tokenizer_encode_partial_merge(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for partial merge test");

    /* Encode "abz" — "ab" merges (score 10.0), 'z' stays */
    int tokens[64];
    int n = tokenizer_encode(&t, "abz", strlen("abz"), tokens, 64);
    TEST_ASSERT_EQ(n, 2, "encode 'abz' -> 2 tokens");
    TEST_ASSERT_EQ(tokens[0], 259, "first token is 'ab'");
    TEST_ASSERT_EQ(tokens[1], 3 + 'z', "second token is 'z'");

    tokenizer_free(&t);
}

static void test_tokenizer_encode_empty(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for empty encode test");

    int tokens[64];
    int n = tokenizer_encode(&t, "", strlen(""), tokens, 64);
    TEST_ASSERT_EQ(n, 0, "encode '' -> 0 tokens");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_basic(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for decode test");

    /* Decode token 259 ("ab") */
    const char *s = tokenizer_decode(&t, -1, 259);
    TEST_ASSERT(strcmp(s, "ab") == 0, "decode token 259 -> 'ab'");

    /* Decode single byte token */
    s = tokenizer_decode(&t, -1, 3 + 'z');
    TEST_ASSERT(strcmp(s, "z") == 0, "decode byte token for 'z'");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_raw_byte(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for raw byte decode test");

    /* Decode <0x0A> (newline) — token 266 */
    const char *s = tokenizer_decode(&t, -1, 266);
    TEST_ASSERT(s[0] == '\n' && s[1] == '\0', "decode <0x0A> -> newline");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_after_bos(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for BOS decode test");

    /* Token 265 is " hello". After BOS (token 1), leading space stripped. */
    const char *s = tokenizer_decode(&t, 1, 265);
    TEST_ASSERT(strcmp(s, "hello") == 0, "leading space stripped after BOS");

    /* Same token after non-BOS: space preserved */
    s = tokenizer_decode(&t, 259, 265);
    TEST_ASSERT(strcmp(s, " hello") == 0, "leading space preserved after non-BOS");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_out_of_range(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for out-of-range decode test");

    const char *s = tokenizer_decode(&t, -1, -1);
    TEST_ASSERT(strcmp(s, "") == 0, "decode -1 -> empty string");

    s = tokenizer_decode(&t, -1, 999999);
    TEST_ASSERT(strcmp(s, "") == 0, "decode huge token -> empty string");

    tokenizer_free(&t);
}

static void test_tokenizer_roundtrip(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for roundtrip test");

    const char *input = "abcdab";
    int tokens[64];
    int n = tokenizer_encode(&t, input, strlen(input), tokens, 64);
    TEST_ASSERT(n > 0, "roundtrip encode should succeed");

    /* Decode all tokens and concatenate */
    char output[256];
    output[0] = '\0';
    for (int i = 0; i < n; i++) {
        const char *piece = tokenizer_decode(&t, -1, tokens[i]);
        strcat(output, piece);
    }
    TEST_ASSERT(strcmp(output, input) == 0, "roundtrip: decode(encode(x)) == x");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_byte_level_space(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for byte-level space decode test");

    /* Token 267 is byte-level-BPE "Ġworld" (0xC4 0xA0 + "world"). The Ġ
     * marker must decode to a real space, not pass through as two literal
     * bytes 0xC4 0xA0 (the old clean_bpe_string-only path handled this one
     * correctly too, via its hardcoded Ġ special case — this confirms the
     * general reverse-table decode preserves that). */
    const char *s = tokenizer_decode(&t, -1, 267);
    TEST_ASSERT(strcmp(s, " world") == 0, "byte-level Ġworld decodes to ' world'");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_byte_level_latin1(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for byte-level latin1 decode test");

    /* Token 268 is byte-level-BPE encoding of raw byte 0xA9: two UTF-8 bytes
     * 0xC2 0xA9 in the vocab piece. The old code had no general reverse
     * mapping and would have passed both bytes through unchanged (wrong —
     * doubled/mangled output); the fix must collapse them back to the
     * single raw byte 0xA9. */
    const char *s = tokenizer_decode(&t, -1, 268);
    TEST_ASSERT(strlen(s) == 1 && (unsigned char)s[0] == 0xA9,
                "byte-level piece for raw byte 0xA9 decodes to single byte 0xA9");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_byte_level_multibyte_emoji(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for byte-level emoji decode test");

    /* Tokens 269, 270, 271 are the byte-level-BPE encodings of the 3 raw
     * UTF-8 bytes (0xE2, 0x9C, 0x85) that together spell U+2705 (✅) — the
     * real mojibake case found via screenshot review of the model's actual
     * output. Each token decodes independently (no cross-call state), and
     * concatenating the 3 pieces must reassemble the original valid UTF-8
     * emoji bytes exactly, the same way a real streaming decode loop would
     * concatenate them. */
    char out[16] = {0};
    strcat(out, tokenizer_decode(&t, -1, 269));
    strcat(out, tokenizer_decode(&t, -1, 270));
    strcat(out, tokenizer_decode(&t, -1, 271));

    TEST_ASSERT(strlen(out) == 3, "3 byte-level tokens decode to 3 raw bytes total");
    TEST_ASSERT((unsigned char)out[0] == 0xE2 && (unsigned char)out[1] == 0x9C &&
                (unsigned char)out[2] == 0x85,
                "concatenated byte-level tokens reassemble raw UTF-8 emoji bytes E2 9C 85");

    tokenizer_free(&t);
}

static void test_tokenizer_decode_byte_level_after_bos(void) {
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, test_tokenizer_path);
    TEST_ASSERT_EQ(err, TN_OK, "load for byte-level BOS decode test");

    /* Token 272 is byte-level-BPE "Ġhi" (0xC4 0xA0 + "hi"). After BOS, the
     * Ġ marker itself must be stripped (2 raw bytes), not just its first
     * byte — the old piece[0]==' ' check never matched this at all since a
     * byte-level piece's leading byte is 0xC4, not a literal space. */
    const char *s = tokenizer_decode(&t, 1, 272);
    TEST_ASSERT(strcmp(s, "hi") == 0, "byte-level Ġhi after BOS strips the 2-byte marker -> 'hi'");

    /* Non-BOS: marker decodes to a normal leading space. */
    s = tokenizer_decode(&t, 259, 272);
    TEST_ASSERT(strcmp(s, " hi") == 0, "byte-level Ġhi after non-BOS decodes to ' hi'");

    tokenizer_free(&t);
}

static void test_tokenizer_free_idempotent(void) {
    Tokenizer t;
    memset(&t, 0, sizeof(t));
    /* Double free should not crash */
    tokenizer_free(&t);
    tokenizer_free(&t);
    TEST_ASSERT(1, "double free does not crash");
}

/* ---------- Main ---------- */

int main(void) {
    if (!create_test_tokenizer_file()) {
        printf("FATAL: could not create test tokenizer file\n");
        return 1;
    }

    RUN_TEST(test_tokenizer_load_basic);
    RUN_TEST(test_tokenizer_load_bad_path);
    RUN_TEST(test_tokenizer_encode_single_chars);
    RUN_TEST(test_tokenizer_encode_bpe_merge);
    RUN_TEST(test_tokenizer_encode_cascading_merge);
    RUN_TEST(test_tokenizer_encode_partial_merge);
    RUN_TEST(test_tokenizer_encode_empty);
    RUN_TEST(test_tokenizer_decode_basic);
    RUN_TEST(test_tokenizer_decode_raw_byte);
    RUN_TEST(test_tokenizer_decode_after_bos);
    RUN_TEST(test_tokenizer_decode_out_of_range);
    RUN_TEST(test_tokenizer_decode_byte_level_space);
    RUN_TEST(test_tokenizer_decode_byte_level_latin1);
    RUN_TEST(test_tokenizer_decode_byte_level_multibyte_emoji);
    RUN_TEST(test_tokenizer_decode_byte_level_after_bos);
    RUN_TEST(test_tokenizer_roundtrip);
    RUN_TEST(test_tokenizer_free_idempotent);

    /* Clean up test file */
    remove(test_tokenizer_path);

    TEST_SUMMARY();
}
