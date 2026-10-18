#ifndef TN_GEN_PARAMS_H
#define TN_GEN_PARAMS_H

/*
 * gen_params.h — every per-request generation knob in one value.
 *
 * Filled from CLI flags (main.c), from server-side defaults + the JSON
 * request (src/api/), or from the web UI (which only talks to the API).
 * Nothing here is a compile-time constant: defaults are documented in
 * gen_params_default() and every field is overridable per request.
 */

#include <stdbool.h>
#include "sampling/sampler_chain.h"
#include "sampling/watermark.h"
#include "kv_cache/prefix_cache.h"
#include "kv_cache/prompt_store.h"

#define GEN_MAX_STOP 8

typedef enum {
    GEN_FINISH_STOP = 0,  /* EOS or a stop sequence */
    GEN_FINISH_LENGTH,    /* max_tokens or context length reached */
    GEN_FINISH_CANCELLED, /* the token callback asked to stop */
    GEN_FINISH_ERROR      /* template/tokenization failure, nothing generated */
} GenFinishReason;

/* Filled by generate_ex() when GenerationParams.stats is non-NULL. */
typedef struct {
    int             prompt_tokens;
    int             completion_tokens;
    int             cached_tokens;  /* prompt tokens reused from the prefix cache */
    GenFinishReason finish_reason;
    double          tokens_per_sec; /* 0 when fewer than 2 tokens */
    /* Sum of log p(token) over emitted tokens under the raw model
     * distribution; only filled when GenerationParams.compute_logprobs. */
    double          sum_logprob;
} GenerationStats;

/* "stop" | "length" | "cancelled" | "error" — OpenAI finish_reason strings. */
const char *gen_finish_reason_str(GenFinishReason r);

typedef struct {
    int             max_tokens;
    SamplerSettings sampler;
    long long       seed;          /* < 0 = time-based (non-reproducible) */
    bool            json_mode;     /* Phase 20 grammar-constrained JSON   */
    WatermarkConfig watermark;     /* Phase 33, disabled by default        */

    /* Conversation for chat-template models. When n_messages > 0 the GGUF
     * chat template renders this whole history (system/user/assistant
     * turns) instead of the single `prompt` argument. Borrowed pointers. */
    const char *const *roles;
    const char *const *contents;
    int                n_messages;

    /* Optional system prompt, prepended as a "system" turn when the
     * template path renders a single prompt (ignored if `roles` already
     * starts with a system message). Borrowed. */
    const char *system_prompt;

    /* OpenAI-style stop strings (borrowed). */
    const char *stop[GEN_MAX_STOP];
    int         n_stop;

    /* Phase 21.5: KV prefix reuse across calls on the same RunState. NULL =
     * disabled (always recompute the whole prompt). Owned by the caller
     * (the API server keeps one per RunState). */
    PrefixCache *prefix_cache;
    /* Phase 26: multi-entry snapshot store consulted when the live prefix
     * cache matches less than a stored sequence. Requires prefix_cache. */
    PromptStore *prompt_store;

    /* Phase 30: accumulate GenerationStats.sum_logprob (costs one logits
     * copy + log-sum-exp per generated token; off by default). */
    bool compute_logprobs;

    /* Optional out-param: token counts + why generation ended. */
    GenerationStats *stats;
} GenerationParams;

/* max_tokens 256, neutral sampler (sampler_settings_default), random seed,
 * no JSON mode, watermark disabled, no messages/system prompt/stop strings. */
void gen_params_default(GenerationParams *gp);

#endif /* TN_GEN_PARAMS_H */
