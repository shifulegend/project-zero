#ifndef AGENT_OUTPUT_INJECT_H
#define AGENT_OUTPUT_INJECT_H

#include "core/config.h"
#include "core/weights.h"
#include "core/run_state.h"
#include "tokenizer/tokenizer.h"
#include "threading/thread_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tokenize `text` and run transformer_forward() for each token to populate the
 * KV cache in `s`. Returns number of tokens injected or negative on error. */
int inject_text_into_kv(const char *text, const Config *cfg, const TransformerWeights *w, RunState *s, Tokenizer *tok, ThreadPool *tp);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_OUTPUT_INJECT_H */
