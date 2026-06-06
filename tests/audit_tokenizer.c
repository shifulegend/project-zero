#include "test_harness.h"
#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Audit Test: Tok-01 (Short Hex Token OOB Read)
 * The decode logic checks piece[5] to identify hex bytes without
 * checking if strlen(piece) >= 6. We craft a vocabulary with a short
 * "<0x" token to trigger an OOB read.
 */
static void aud_tok_short_hex(void) {
  Tokenizer t;
  memset(&t, 0, sizeof(t));

  t.vocab_size = 3;
  t.vocab = malloc(3 * sizeof(char *));
  t.vocab_scores = malloc(3 * sizeof(float));

  t.vocab[0] = strdup("<unk>");
  t.vocab[1] = strdup("<s>");
  t.vocab[2] = strdup("<0x"); // Malformed short token resembling hex byte

  t.vocab_scores[0] = 0.0f;
  t.vocab_scores[1] = 0.0f;
  t.vocab_scores[2] = 0.0f;

  /* If the code reads piece[5] on an isolated 3-char string, ASAN will catch
   * it. */
  const char *decoded = tokenizer_decode(&t, 1, 2);

  TEST_ASSERT(decoded != NULL, "Decode did not crash gracefully");

  free(t.vocab[0]);
  free(t.vocab[1]);
  free(t.vocab[2]);
  free(t.vocab);
  free(t.vocab_scores);
}

int main(void) {
  RUN_TEST(aud_tok_short_hex);
  TEST_SUMMARY();
}
