#ifndef TN_GEN_SESSION_H
#define TN_GEN_SESSION_H

/*
 * transformer/gen_session — one generation as a step-wise state machine
 * (Phase 23 stage 2b, 2026-10-22).
 *
 * generate_ex() is a thin loop over this API, and the API server's parallel
 * scheduler drives several sessions at once, so single and parallel serving
 * run exactly the same prompt handling, sampling, stop and stats code:
 *
 *     GenSession g;
 *     if (gen_session_begin(&g, ...) == 0) {
 *         int token, pos;
 *         while (gen_session_input(&g, &token, &pos))
 *             gen_session_consume(&g, transformer_forward(token, pos, ...));
 *     }
 *     gen_session_end(&g);
 *
 * Everything a generation owns lives in the session (its RunState is the
 * caller's); nothing is stored in shared objects such as the Tokenizer.
 */

#include "transformer/generate.h"
#include "transformer/stop_seq.h"
#include "sampling/fsm.h"
#include "kv_cache/prefix_cache.h"
#include "kv_cache/prompt_store.h"
#include <stdint.h>

typedef struct {
    const Config *cfg;
    const TransformerWeights *w;
    RunState *s;
    const MoEConfig *mc;
    Tokenizer *tok;
    ThreadPool *tp;
    const GenerationParams *gp;
    GenerationParams gp_default;
    TokenCallback callback;
    void *userdata;

    const SamplerSettings *ss;
    const WatermarkConfig *wm;
    int max_tokens;
    int json_mode;
    StopMatcher stop;
    FSMState json_fsm;
    unsigned char *chat_stops;     /* this generation's chat-mode turn ends */

    int *prompt_tokens;
    int n_prompt;
    int pos_offset;
    unsigned long long rng_state;
    int *gen_history;
    int n_history;
    float *raw_logits;
    double sum_logprob, step_lse;
    PrefixCache *pcache;
    PromptStore *pstore;
    int cache_eligible;

    int step, total_steps;
    int prev_token, next;
    int tokens_generated;
    GenFinishReason finish;
    int64_t gen_start_us;
    int started;                   /* begin succeeded (end must flush/save) */
    int done;
} GenSession;

/* Prepares a generation on RunState s: renders the prompt, tokenizes,
 * seeds sampling, reuses cached prefix K/V. Returns 0 when there is work to
 * do, -1 when nothing can be generated (stats say why); call
 * gen_session_end() either way. All pointers must stay valid until then. */
int gen_session_begin(GenSession *g, const Config *cfg, const TransformerWeights *w,
                      RunState *s, const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                      const char *prompt, const GenerationParams *gp,
                      TokenCallback callback, void *userdata);

/* 1 and the token/position to forward next, or 0 when the session is done. */
int gen_session_input(const GenSession *g, int *token, int *pos);

/* Consumes the logits of the forward pass for gen_session_input()'s token:
 * samples, checks end of generation and stop sequences, delivers text to the
 * callback. Returns 1 to continue, 0 when finished. */
int gen_session_consume(GenSession *g, float *logits);

/* Flushes, saves the prompt snapshot, fills stats and frees the session. */
void gen_session_end(GenSession *g);

#endif /* TN_GEN_SESSION_H */
