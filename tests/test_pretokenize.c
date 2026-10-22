/*
 * tests/test_pretokenize.c — BPE pre-tokenization (tokenizer/pretokenize.c):
 * fragment boundaries for each llama.cpp pipeline (default, gpt2, llama3,
 * qwen2, smollm) on contractions, digits, punctuation, whitespace runs and
 * UTF-8, plus name mapping and capacity handling. Expected splits follow
 * llama.cpp's unicode_regex_split_custom_* behaviour.
 */
#include "test_harness.h"
#include "tokenizer/pretokenize.h"
#include <string.h>

/* Joins fragments with '|' for compact comparison. */
static const char *split(PretokType t, const char *s) {
    static char out[1024];
    size_t st[256], ln[256];
    int n = pretokenize(t, s, strlen(s), st, ln, 256);
    size_t k = 0;
    out[0] = '\0';
    for (int i = 0; i < n && k + 2 < sizeof(out); i++) {
        if (i) out[k++] = '|';
        memcpy(out + k, s + st[i], ln[i]);
        k += ln[i];
    }
    out[k] = '\0';
    return n < 0 ? "<error>" : out;
}

#define EXPECT(type, in, want) do { \
    const char *_g = split(type, in); \
    TEST_ASSERT(strcmp(_g, want) == 0, #type ": " #in); \
    if (strcmp(_g, want) != 0) printf("    got  [%s]\n    want [%s]\n", _g, want); \
} while (0)

static void test_gpt2(void) {
    EXPECT(PRETOK_GPT2, "I'm here", "I|'m| here");
    EXPECT(PRETOK_GPT2, "they'll go", "they|'ll| go");
    EXPECT(PRETOK_GPT2, "a  b", "a| | b");                 /* \s+(?!\S) keeps one space for the word */
    EXPECT(PRETOK_GPT2, "x 12345 y", "x| 12345| y");
    EXPECT(PRETOK_GPT2, "Hi!!  (ok)", "Hi|!!| | (|ok|)");
    EXPECT(PRETOK_GPT2, "?\n\nB", "?|\n|\n|B");
}

static void test_llama3(void) {
    EXPECT(PRETOK_LLAMA3, "?\n\nBIT", "?\n\n|BIT");          /* punctuation absorbs trailing newlines */
    EXPECT(PRETOK_LLAMA3, "x 12345", "x| |123|45");
    EXPECT(PRETOK_LLAMA3, "I'M here", "I|'M| here");        /* contractions case-insensitive */
    EXPECT(PRETOK_LLAMA3, "a\n\n  b", "a|\n\n| | b");
}

static void test_qwen2_smollm_default(void) {
    EXPECT(PRETOK_QWEN2, "x 123", "x| |1|2|3");              /* one digit per piece */
    EXPECT(PRETOK_SMOLLM, "x 123 y", "x| |1|2|3| y");
    EXPECT(PRETOK_DEFAULT, "What?\n\nBIT", "What|?|\n|\n|BIT");
    EXPECT(PRETOK_DEFAULT, "v 12345", "v| |123|45");
    EXPECT(PRETOK_DEFAULT, "a+=b", "a|+=|b");                /* [\p{P}$+<=>^~|]+ first */
}

static void test_utf8(void) {
    EXPECT(PRETOK_GPT2, "café naïve", "café| naïve");
    EXPECT(PRETOK_LLAMA3, "東京は首都", "東京は首都");
    EXPECT(PRETOK_GPT2, "a — b", "a| —| b");                 /* em dash is punctuation */
    EXPECT(PRETOK_GPT2, "go 🚀!", "go| 🚀!");                 /* symbol + punct share [^\s\p{L}\p{N}]+ */
    /* A truncated UTF-8 tail must not be split mid-sequence nor read past. */
    const char bad[] = { 'a', (char)0xE6, (char)0x9D, 0 };
    size_t st[8], ln[8];
    int n = pretokenize(PRETOK_GPT2, bad, 3, st, ln, 8);
    size_t total = 0;
    for (int i = 0; i < n; i++) total += ln[i];
    TEST_ASSERT(n > 0 && total == 3, "fragments cover malformed UTF-8 exactly");
}

static void test_api(void) {
    TEST_ASSERT_EQ(pretok_type_from_name(NULL), PRETOK_DEFAULT, "absent key -> default");
    TEST_ASSERT_EQ(pretok_type_from_name("smollm"), PRETOK_SMOLLM, "smollm");
    TEST_ASSERT_EQ(pretok_type_from_name("qwen2"), PRETOK_QWEN2, "qwen2");
    TEST_ASSERT_EQ(pretok_type_from_name("llama-bpe"), PRETOK_LLAMA3, "llama-bpe");
    TEST_ASSERT_EQ(pretok_type_from_name("olmo"), PRETOK_GPT2, "olmo");
    TEST_ASSERT_EQ(pretok_type_from_name("brand-new"), PRETOK_DEFAULT, "unknown -> default");
    size_t st[2], ln[2];
    TEST_ASSERT_EQ(pretokenize(PRETOK_GPT2, "a b c d", 7, st, ln, 2), -1, "too few slots -> -1");
    TEST_ASSERT_EQ(pretokenize(PRETOK_NONE, "a b", 3, st, ln, 2), 1, "none = one fragment");
    TEST_ASSERT_EQ(pretokenize(PRETOK_GPT2, "", 0, st, ln, 2), 0, "empty input");
}

int main(void) {
    printf("=== BPE Pre-tokenization Tests ===\n");
    RUN_TEST(test_gpt2);
    RUN_TEST(test_llama3);
    RUN_TEST(test_qwen2_smollm_default);
    RUN_TEST(test_utf8);
    RUN_TEST(test_api);
    TEST_SUMMARY();
}
