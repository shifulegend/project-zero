#ifndef TN_GENERATE_H
#define TN_GENERATE_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"

/**
 * Text generation loop.
 *
 * Encodes the prompt, runs the forward pass for each position,
 * samples the next token, decodes and prints it, and stops
 * on EOS or max_tokens.
 *
 * @param cfg         Model configuration
 * @param w           Transformer weights
 * @param s           Run state (modified in-place)
 * @param mc          MoE config — NULL or zero-init for dense (BitNet/Llama) models
 * @param tok         Tokenizer
 * @param tp          Thread pool (NULL for single-threaded)
 * @param prompt      Input prompt string (NULL or "" for unconditional generation)
 * @param max_tokens  Maximum number of tokens to generate
 * @param temperature Sampling temperature (0.0 = greedy argmax)
 * @param top_p       Nucleus sampling threshold (1.0 = disabled)
 */
void generate(const Config *cfg, const TransformerWeights *w, RunState *s,
              const MoEConfig *mc,
              Tokenizer *tok, ThreadPool *tp, const char *prompt,
              int max_tokens, float temperature, float top_p);

/*
 * Phase 21 — Token callback type.
 * Phase 22: now returns int so a caller (e.g. the HTTP API's cancel
 * endpoint) can request early termination.
 *
 * Called once for every decoded non-EOS token during generation.
 * @param piece    Decoded token text (may be a UTF-8 fragment, multi-byte).
 *                 The pointer is valid only for the duration of the call.
 * @param userdata Opaque pointer passed through from generate_with_callback().
 * @return         0 to continue generating, nonzero to stop early (as if
 *                 EOS had been reached).
 */
typedef int (*TokenCallback)(const char *piece, void *userdata);

/**
 * Callback-based variant of generate().
 *
 * Identical to generate() but instead of writing to stdout, delivers each
 * decoded token text piece to @p callback with @p userdata.  Used by the
 * OpenAI-compatible API server (Phase 21) to stream tokens over SSE without
 * modifying core inference logic.
 */
void generate_with_callback(const Config *cfg, const TransformerWeights *w,
                             RunState *s, const MoEConfig *mc,
                             Tokenizer *tok, ThreadPool *tp, const char *prompt,
                             int max_tokens, float temperature, float top_p,
                             TokenCallback callback, void *userdata);

#endif /* TN_GENERATE_H */
