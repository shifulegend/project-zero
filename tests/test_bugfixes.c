/**
 * test_bugfixes.c — Regression tests for QA audit bug fixes
 *
 * Tests:
 *  1. QA-BUG-001: sample_top_k with k > 1024 clamps safely (no stack overflow)
 *  2. QA-BUG-002: thought_filter_process with small output buffer (no buffer overflow)
 *  3. QA-BUG-003: MappedFile struct uses pointer-sized handle on Windows (compile-time)
 *  4. QA-ISS-004: byte_decode_buf is thread-local (compile-time, verified by build)
 *  5. PRE10-BUG-001: mapped_file_open fd/flock leak on zero-size file
 *  6. PRE10-BUG-002: tokenizer_load FILE handle leak on validation failure
 *  7. PRE10-BUG-003: tokenizer_decode NULL dereference when piece is NULL
 */

#include "sampling/rng.h"
#include "sampling/sampling.h"
#include "reasoning/thought_filter.h"
#include "memory/mapped_file.h"
#include "tokenizer/tokenizer.h"
#include "test_harness.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <fcntl.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

/* ================================================================
 * QA-BUG-001: top_k stack overflow regression
 * ================================================================ */

static void test_topk_large_k_clamps(void) {
  /* Create a small vocab with known logits */
  float logits[32];
  for (int i = 0; i < 32; i++) {
    logits[i] = (float)i;
  }

  unsigned long long rng;
  rng_seed(&rng, 42);

  /* k=5000 exceeds both TOP_K_MAX(1024) and vocab_size(32).
   * Before the fix, this would overflow the stack-allocated arrays.
   * After the fix, k is clamped first to vocab_size then to TOP_K_MAX. */
  int result = sample_top_k(logits, 32, 5000, &rng);
  TEST_ASSERT(result >= 0 && result < 32, "top_k with k=5000 returns valid index");
}

static void test_topk_exactly_1024(void) {
  /* k=1024 with large vocab should work without clamping to smaller value */
  float logits[2048];
  for (int i = 0; i < 2048; i++) {
    logits[i] = (float)(i % 100);
  }

  unsigned long long rng;
  rng_seed(&rng, 99);

  int result = sample_top_k(logits, 2048, 1024, &rng);
  TEST_ASSERT(result >= 0 && result < 2048, "top_k with k=1024 returns valid index");
}

static void test_topk_above_1024_below_vocab(void) {
  /* k=2000 exceeds TOP_K_MAX but not vocab_size — should be clamped to 1024 */
  float logits[4096];
  for (int i = 0; i < 4096; i++) {
    logits[i] = (float)(i % 50) * 0.1f;
  }

  unsigned long long rng;
  rng_seed(&rng, 7);

  int result = sample_top_k(logits, 4096, 2000, &rng);
  TEST_ASSERT(result >= 0 && result < 4096, "top_k with k=2000 returns valid index (clamped to 1024)");
}

static void test_topk_zero_and_negative(void) {
  float logits[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  unsigned long long rng;
  rng_seed(&rng, 1);

  /* k=0 should be clamped to 1 */
  int r1 = sample_top_k(logits, 8, 0, &rng);
  TEST_ASSERT(r1 >= 0 && r1 < 8, "top_k with k=0 returns valid index");

  /* k=-5 should be clamped to 1 */
  rng_seed(&rng, 1);
  int r2 = sample_top_k(logits, 8, -5, &rng);
  TEST_ASSERT(r2 >= 0 && r2 < 8, "top_k with k=-5 returns valid index");
}

/* ================================================================
 * QA-BUG-002: thought_filter buffer overflow regression
 * ================================================================ */

static void test_filter_tiny_output_buffer(void) {
  ThoughtFilter f;
  thought_filter_init(&f);

  /* Output buffer of size 1 — can only hold null terminator */
  char out[1];
  bool show = thought_filter_process(&f, "Hello", out, sizeof(out));
  /* Should not crash, and should not produce output since buffer is too small */
  TEST_ASSERT(out[0] == '\0', "size-1 buffer stays null-terminated");
  (void)show;
}

static void test_filter_zero_output_buffer(void) {
  ThoughtFilter f;
  thought_filter_init(&f);

  char out[1] = {'X'};
  /* output_buf_size=0 should return false immediately without touching buffer */
  bool show = thought_filter_process(&f, "Hello", out, 0);
  TEST_ASSERT(!show, "zero-size buffer returns false");
}

static void test_filter_small_buffer_passthrough(void) {
  ThoughtFilter f;
  thought_filter_init(&f);

  /* Buffer of 4 bytes: can hold 3 chars + null */
  char out[4];
  bool show = thought_filter_process(&f, "ABCDEFGH", out, sizeof(out));
  TEST_ASSERT(show, "small buffer should still output what fits");
  /* Verify no overflow: out must be null-terminated within bounds */
  TEST_ASSERT(strlen(out) < sizeof(out), "output fits within buffer");
}

static void test_filter_small_buffer_tag_flush(void) {
  ThoughtFilter f;
  thought_filter_init(&f);

  /* Feed a '<' followed by non-tag character — the buffered '<' must be flushed.
   * With a tiny output buffer, the flush must respect bounds. */
  char out[3];
  bool show = thought_filter_process(&f, "<Z", out, sizeof(out));
  TEST_ASSERT(show, "tag mismatch flush produces output");
  TEST_ASSERT(strlen(out) < sizeof(out), "flushed tag respects buffer bounds");
}

static void test_filter_normal_with_size(void) {
  ThoughtFilter f;
  thought_filter_init(&f);

  char out[256];
  bool show = thought_filter_process(&f, "Hello world", out, sizeof(out));
  TEST_ASSERT(show, "normal text passes through");
  TEST_ASSERT(strcmp(out, "Hello world") == 0, "output matches input");
}

static void test_filter_think_block_with_size(void) {
  ThoughtFilter f;
  thought_filter_init(&f);

  char out[256];
  /* Feed <think> tag */
  bool show = thought_filter_process(&f, "<think>", out, sizeof(out));
  TEST_ASSERT(!show, "<think> tag is consumed");
  TEST_ASSERT(f.state == FILTER_THINKING, "state is THINKING after <think>");

  /* Feed hidden content */
  show = thought_filter_process(&f, "hidden reasoning", out, sizeof(out));
  TEST_ASSERT(!show, "thinking content is hidden");

  /* Feed </think> tag */
  show = thought_filter_process(&f, "</think>", out, sizeof(out));
  TEST_ASSERT(!show, "</think> tag is consumed");
  TEST_ASSERT(f.state == FILTER_OUTPUT, "state is OUTPUT after </think>");

  /* Feed visible content */
  show = thought_filter_process(&f, "visible", out, sizeof(out));
  TEST_ASSERT(show, "post-think content is visible");
  TEST_ASSERT(strcmp(out, "visible") == 0, "output is correct");
}

/* ================================================================
 * QA-BUG-003: MappedFile handle type (compile-time check)
 * ================================================================ */

static void test_mapped_file_struct_sizes(void) {
  /* Verify basic struct layout expectations */
  MappedFile mf;
  memset(&mf, 0, sizeof(mf));

  TEST_ASSERT(sizeof(mf.data) == sizeof(void *), "data is pointer-sized");
  TEST_ASSERT(sizeof(mf.size) == sizeof(size_t), "size is size_t");

#ifdef _WIN32
  /* On Windows, handle must be pointer-sized (not int-sized) */
  TEST_ASSERT(sizeof(mf.handle) == sizeof(void *),
              "Windows handle is pointer-sized (not truncated to int)");
#else
  TEST_ASSERT(sizeof(mf.fd) == sizeof(int), "POSIX fd is int-sized");
#endif
}

/* ================================================================
 * PRE10-BUG-001: mapped_file_open fd leak on zero-size file
 * Before fix: TN_CHECK(file_size > 0) did a bare return, leaking fd
 * and the shared flock. Now the error path closes fd + releases flock.
 * ================================================================ */

#if !defined(_WIN32)
static void test_mmap_zero_size_no_fd_leak(void) {
  /* Create a zero-byte file */
  FILE *f = fopen("/tmp/tn_test_empty.bin", "w");
  TEST_ASSERT(f != NULL, "create temp file");
  fclose(f);

  /* Measure fd usage before */
  int before_fd = open("/dev/null", O_RDONLY);
  close(before_fd);

  /* Open zero-size file — should fail WITHOUT leaking fd */
  MappedFile mf;
  TernaryError err = mapped_file_open(&mf, "/tmp/tn_test_empty.bin");
  TEST_ASSERT(err != TN_OK, "zero-size file returns error");

  /* Measure fd usage after */
  int after_fd = open("/dev/null", O_RDONLY);
  close(after_fd);

  TEST_ASSERT(after_fd == before_fd,
              "no file descriptor leaked on zero-size file");

  unlink("/tmp/tn_test_empty.bin");
}
#endif

/* ================================================================
 * PRE10-BUG-002: tokenizer_load FILE handle leak on bad validation
 * Before fix: TN_CHECK on lines 87-88 returned without fclose(fp),
 * leaking the FILE handle when vocab_size or max_token_len is invalid.
 * ================================================================ */

#if !defined(_WIN32)
static void test_tokenizer_load_no_file_leak(void) {
  /* Create a file with invalid vocab_size (negative) */
  FILE *f = fopen("/tmp/tn_test_bad_tok.bin", "wb");
  TEST_ASSERT(f != NULL, "create temp tok file");
  int bad_vocab = -5;
  int max_tok = 100;
  fwrite(&bad_vocab, sizeof(int), 1, f);
  fwrite(&max_tok, sizeof(int), 1, f);
  fclose(f);

  int before_fd = open("/dev/null", O_RDONLY);
  close(before_fd);

  Tokenizer t;
  TernaryError err = tokenizer_load(&t, "/tmp/tn_test_bad_tok.bin");
  TEST_ASSERT(err != TN_OK, "invalid vocab_size returns error");

  int after_fd = open("/dev/null", O_RDONLY);
  close(after_fd);

  TEST_ASSERT(after_fd == before_fd,
              "no FILE handle leaked on invalid vocab_size");

  unlink("/tmp/tn_test_bad_tok.bin");
}

static void test_tokenizer_load_bad_max_token_len(void) {
  /* Create a file with valid vocab_size but invalid max_token_len */
  FILE *f = fopen("/tmp/tn_test_bad_tok2.bin", "wb");
  TEST_ASSERT(f != NULL, "create temp tok file");
  int vocab = 100;
  int bad_max = -1;
  fwrite(&vocab, sizeof(int), 1, f);
  fwrite(&bad_max, sizeof(int), 1, f);
  fclose(f);

  int before_fd = open("/dev/null", O_RDONLY);
  close(before_fd);

  Tokenizer t;
  TernaryError err = tokenizer_load(&t, "/tmp/tn_test_bad_tok2.bin");
  TEST_ASSERT(err != TN_OK, "invalid max_token_len returns error");

  int after_fd = open("/dev/null", O_RDONLY);
  close(after_fd);

  TEST_ASSERT(after_fd == before_fd,
              "no FILE handle leaked on invalid max_token_len");

  unlink("/tmp/tn_test_bad_tok2.bin");
}
#endif

/* ================================================================
 * PRE10-BUG-003: tokenizer_decode NULL piece dereference
 * Before fix: if t->vocab[token] was NULL and prev_token == 1 (BOS),
 * line 42 accessed piece[0] without NULL check -> SIGSEGV.
 * ================================================================ */

static void test_decode_null_piece_no_crash(void) {
  /* Build a minimal tokenizer with a NULL vocab entry */
  Tokenizer t;
  memset(&t, 0, sizeof(t));
  t.vocab_size = 3;
  t.vocab = (char **)calloc(3, sizeof(char *));
  t.vocab[0] = strdup("hello");
  t.vocab[1] = NULL; /* simulates corrupt/missing entry */
  t.vocab[2] = strdup("world");

  /* prev_token=1 (BOS) + NULL piece = the exact crash path */
  const char *result = tokenizer_decode(&t, 1, 1);
  TEST_ASSERT(result != NULL, "NULL piece returns non-NULL (empty string)");
  TEST_ASSERT(strlen(result) == 0, "NULL piece returns empty string");

  /* Non-BOS path with NULL piece should also be safe */
  result = tokenizer_decode(&t, 0, 1);
  TEST_ASSERT(result != NULL, "NULL piece with non-BOS also safe");

  free(t.vocab[0]);
  free(t.vocab[2]);
  free(t.vocab);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
  /* QA-BUG-001: top_k stack overflow */
  RUN_TEST(test_topk_large_k_clamps);
  RUN_TEST(test_topk_exactly_1024);
  RUN_TEST(test_topk_above_1024_below_vocab);
  RUN_TEST(test_topk_zero_and_negative);

  /* QA-BUG-002: thought_filter buffer overflow */
  RUN_TEST(test_filter_tiny_output_buffer);
  RUN_TEST(test_filter_zero_output_buffer);
  RUN_TEST(test_filter_small_buffer_passthrough);
  RUN_TEST(test_filter_small_buffer_tag_flush);
  RUN_TEST(test_filter_normal_with_size);
  RUN_TEST(test_filter_think_block_with_size);

  /* QA-BUG-003: MappedFile handle type */
  RUN_TEST(test_mapped_file_struct_sizes);

  /* PRE10-BUG-001: mapped_file fd leak on zero-size file */
#if !defined(_WIN32)
  RUN_TEST(test_mmap_zero_size_no_fd_leak);
#endif

  /* PRE10-BUG-002: tokenizer_load FILE handle leak */
#if !defined(_WIN32)
  RUN_TEST(test_tokenizer_load_no_file_leak);
  RUN_TEST(test_tokenizer_load_bad_max_token_len);
#endif

  /* PRE10-BUG-003: tokenizer_decode NULL piece dereference */
  RUN_TEST(test_decode_null_piece_no_crash);

  TEST_SUMMARY();
}
