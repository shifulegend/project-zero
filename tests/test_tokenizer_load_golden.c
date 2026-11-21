/*
 * tests/test_tokenizer_load_golden.c — tokenizer_load (the legacy .bin vocabulary format) over
 * 6,000 generated files written to a temporary directory: valid vocabularies of 1..40 tokens that
 * include (or leave out) the BOS / EOS names the loader looks for, token lengths at, over and under
 * the declared maximum, negative and zero sizes, vocabularies past the size limits, and files cut
 * short at any byte. Each result is the return code and, when it loaded, every field of the
 * tokenizer: sizes, every token and score, the sorted index, BOS / EOS ids and the
 * end-of-generation list; failed and loaded tokenizers are freed (leaks fail under ASan). Hash
 * recorded from the loader before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "test_harness.h"
#include "tokenizer/tokenizer.h"

#define EXPECTED_HASH 0x26c1e6d90ac176b4ull
#define N_CASES 6000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x1f83d9abfb41bd6bull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const NAMES[] = {
    "<|begin_of_text|>",
    "<s>",
    "<bos>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "</s>",
    "<eos>",
    "[EOS]",
    "<|endoftext|>",
    "<|im_start|>",
    "<|eom_id|>",
    "<|end|>",
    "<unk>",
    "a",
    "b",
    "hello",
    " world",
    "\xE4\xB8\xAD",
    "",
    "tok",
    "the",
    "ing",
    "er",
};

static char g_path[64];

static void put_int(FILE *f, int v) {
    fwrite(&v, sizeof v, 1, f);
}

static void write_token(FILE *f, int max_len, int damaged) {
    const char *name = NAMES[rnd(sizeof NAMES / sizeof NAMES[0])];
    int len          = (int)strlen(name);
    float score      = (float)rnd(2000) / 10.0f - 100.0f;
    unsigned damage  = damaged ? rnd(4) : 99;
    if (damage == 0) len = max_len + 1; /* over the declared maximum */
    if (damage == 1) len = -1;
    if (damage == 2) len = max_len; /* exactly the declared maximum: still valid */
    fwrite(&score, sizeof score, 1, f);
    put_int(f, len);
    if (damage == 2) {
        for (int i = 0; i < len; i++) fputc('x', f);
    } else if (len > 0 && damage > 1) {
        fwrite(name, 1, strlen(name), f);
    }
}

/* A header that is mostly valid: some sizes are zero, negative or past the limits. */
static void write_header(FILE *f, int *vocab, int *max_len) {
    *vocab     = 1 + (int)rnd(40);
    *max_len   = 18 + (int)rnd(30);
    unsigned k = rnd(30);
    if (k == 0) *vocab = 0;
    if (k == 1) *vocab = -5;
    if (k == 2) *vocab = 1000001;
    if (k == 3) *max_len = 0;
    if (k == 4) *max_len = 4097;
    if (k == 5) *max_len = -1;
    if (k == 6 || k == 7) *max_len = 4096; /* the largest accepted */
    put_int(f, *vocab);
    put_int(f, *max_len);
}

static void write_file(void) {
    FILE *f = fopen(g_path, "wb");
    int vocab, max_len;
    write_header(f, &vocab, &max_len);
    int n   = vocab > 0 && vocab < 100 ? vocab : 3;
    int bad = rnd(4) == 0 ? (int)rnd((unsigned)n) : -1; /* the one damaged token, if any */
    for (int i = 0; i < n; i++) write_token(f, max_len, i == bad);
    fclose(f);
    if (rnd(5) == 0) { /* cut short */
        FILE *g = fopen(g_path, "rb");
        fseek(g, 0, SEEK_END);
        long size = ftell(g);
        fclose(g);
        if (size > 0 && truncate(g_path, (off_t)rnd((unsigned)size)) != 0) return;
    }
}

static void hash_loaded(const Tokenizer *t) {
    mix(&t->vocab_size, sizeof t->vocab_size);
    mix(&t->max_token_len, sizeof t->max_token_len);
    mix(&t->bos_token_id, sizeof t->bos_token_id);
    mix(&t->eos_token_id, sizeof t->eos_token_id);
    mix(&t->add_bos_token, sizeof t->add_bos_token);
    mix(&t->sorted, sizeof t->sorted);
    mix(&t->n_eos, sizeof t->n_eos);
    mix(t->eos_list, sizeof(int) * (size_t)t->n_eos);
    mix(t->vocab_scores, sizeof(float) * (size_t)t->vocab_size);
    mix(t->sorted_vocab_indices, sizeof(int) * (size_t)t->vocab_size);
    for (int i = 0; i < t->vocab_size; i++) mix(t->vocab[i], strlen(t->vocab[i]) + 1);
}

static int g_ok;

static void one_case(void) {
    write_file();
    Tokenizer t;
    memset(&t, 0xAB, sizeof t);
    int rc = (int)tokenizer_load(&t, g_path);
    mix(&rc, sizeof rc);
    if (rc == TN_OK) {
        g_ok++;
        hash_loaded(&t);
        tokenizer_free(&t);
    }
}

static void test_tokenizer_load_golden(void) {
    snprintf(g_path, sizeof g_path, "/tmp/pz_tok_golden_XXXXXX");
    int fd = mkstemp(g_path);
    if (fd < 0) {
        TEST_ASSERT(0, "temporary file");
        return;
    }
    close(fd);
    for (int i = 0; i < N_CASES; i++) one_case();
    /* a path that does not exist */
    Tokenizer t;
    int rc = (int)tokenizer_load(&t, "/nonexistent/pz_tokenizer.bin");
    mix(&rc, sizeof rc);
    unlink(g_path);
    if (getenv("TOKENIZER_LOAD_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d of %d loaded)\n", (unsigned long long)g_hash, g_ok, N_CASES);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "loaded tokenizers match the recorded hash");
}

int main(void) {
    printf("=== tokenizer_load golden ===\n");
    RUN_TEST(test_tokenizer_load_golden);
    TEST_SUMMARY();
}
