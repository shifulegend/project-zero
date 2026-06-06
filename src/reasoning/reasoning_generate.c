#include "reasoning/prompt_inject.h"
#include "reasoning/thought_filter.h"
#include "sampling/rng.h"
#include "sampling/sampling.h"
#include "transformer/forward.h"
#include "transformer/generate.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

/**
 * Reasoning-aware generation wrapper.
 *
 * Wraps the standard generate() loop with:
 * 1. Prompt injection (adds <think> instruction)
 * 2. ThoughtFilter on every decoded token
 * 3. "Thinking..." indicator during hidden reasoning
 * 4. Reports think-token count for diagnostics
 *
 * NOTE: This is a high-level orchestrator. It re-implements the generation
 * loop to integrate the thought filter on each token. For a model that
 * supports <think> tags, this will hide the reasoning from the user while
 * showing only the final answer.
 */
void generate_with_reasoning(Config *p, TransformerWeights *w, RunState *s,
                             Tokenizer *t, ThreadPool *tp, const char *prompt,
                             int max_tokens, float temperature, float top_p) {
  /* Step 1: Inject reasoning prompt */
  char augmented_prompt[4096];
  inject_reasoning_prompt(augmented_prompt, prompt, sizeof(augmented_prompt));

  /* Step 2: Initialize thought filter */
  ThoughtFilter filter;
  thought_filter_init(&filter);

  /* Step 3: Encode the augmented prompt */
  int prompt_tokens[512];
  int n_prompt = tokenizer_encode(t, augmented_prompt, strlen(augmented_prompt), prompt_tokens, 512);

  if (n_prompt <= 0) {
    fprintf(stderr, "[reasoning] Error: could not encode prompt\n");
    return;
  }

  /* Step 4: Generation loop  */
  int token = prompt_tokens[0];
  int pos = 0;
  int printed_thinking = 0;

  /* Initialize RNG once — PRE10-BUG-005: previously re-seeded inside
   * the loop with (step * 1337), breaking stochastic sampling. */
  unsigned long long rng_state;
  rng_seed(&rng_state, (unsigned long long)time(NULL) ^ 0xDEADBEEFULL);

  /* Initialize sliding window for generation */
  sw_init(&s->sw, s->max_seq_len, n_prompt);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int step = 0; step < max_tokens; step++) {
    /* Forward pass */
    float *logits = transformer_forward(token, pos, p, w, s, NULL, tp);

    /* Determine next token */
    int next;
    if (step + 1 < n_prompt && step + 1 < 512) {
      /* Still in prompt — use the known prompt token */
      next = prompt_tokens[step + 1];
    } else {
      /* Sample from logits */
      if (temperature <= 0.0f || temperature == 1e-9f) {
        next = sample_argmax(logits, p->vocab_size);
      } else {
        apply_temperature(logits, p->vocab_size, temperature);
        if (top_p > 0.0f && top_p < 1.0f) {
          next = sample_top_p(logits, p->vocab_size, top_p, &rng_state);
        } else {
          next = sample_argmax(logits, p->vocab_size);
        }
      }
    }

    /* Decode the token and pass through thought filter */
    if (step >= n_prompt - 1) {
      const char *piece = tokenizer_decode(t, token, next);
      if (piece) {
        char output[256];
        bool show = thought_filter_process(&filter, piece, output, sizeof(output));

        if (filter.state == FILTER_THINKING && !printed_thinking) {
          fprintf(stderr, "Thinking...\n");
          printed_thinking = 1;
        }

        if (show) {
          printf("%s", output);
          fflush(stdout);
        }
      }
    }

    /* Check for EOS — all IDs come from tokenizer vocab scan / GGUF metadata */
    {
      int is_eos = 0;
      for (int _ei = 0; _ei < t->n_eos && !is_eos; _ei++)
        if (next == t->eos_list[_ei]) is_eos = 1;
      if (!is_eos && t->eos_token_id >= 0 && next == t->eos_token_id) is_eos = 1;
      if (is_eos) break;
    }

    token = next;
    pos++;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\n");
  fprintf(stderr, "[reasoning] %d tokens generated in %.2fs (%.1f tok/s)\n",
          pos + 1, elapsed, (pos + 1) / elapsed);
  fprintf(stderr, "[reasoning] %d tokens consumed by <think> reasoning\n",
          thought_filter_think_count(&filter));
}
