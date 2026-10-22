#ifndef TN_FSM_H
#define TN_FSM_H

/*
 * fsm.h — Phase 20: grammar-constrained token masking.
 *
 * Wraps a JsonGrammarState (grammar.h) with vocabulary-aware operations:
 * for every candidate token, decode what text it would actually emit
 * (reusing the same tokenizer_decode() path real generation uses -- byte-
 * level BPE, raw-byte <0xHH> tokens, the BOS-adjacent leading-space strip,
 * all included) and simulate it through the grammar to decide whether that
 * token could ever lead to valid JSON from here.
 */

#include "sampling/grammar.h"
#include "tokenizer/tokenizer.h"

typedef struct {
    JsonGrammarState json;
} FSMState;

/* Resets to "expecting the start of any JSON value". */
void fsm_init(FSMState *fsm);

/*
 * Sets logits[i] = -1e30f (the codebase's existing "impossible" sentinel,
 * see forward.c's lm_head_top8 dump) for every token whose decoded text
 * cannot possibly extend the in-progress JSON document. `prev_token` is the
 * last token already accepted (needed by tokenizer_decode's BOS-adjacent
 * leading-space-strip rule) -- pass -1 if there is no previous token.
 *
 * EOS-family tokens (t->eos_token_id and every entry in t->eos_list) are
 * handled specially: allowed only when the grammar could legally stop here
 * (json_grammar_is_complete() after a trial json_grammar_finalize() on a
 * scratch copy of the state -- never mutates fsm->json), regardless of
 * their literal piece text. This is what lets generation actually
 * terminate once the JSON is complete, and forces it to keep going
 * otherwise.
 */
void fsm_compute_token_mask(const FSMState *fsm, Tokenizer *t, int prev_token,
                             float *logits, int vocab_size);

/*
 * Advances the real grammar state by the token actually chosen. Call this
 * once per generated token, after sampling from the masked logits. Silently
 * no-ops on an EOS-family token (those terminate generation instead of
 * emitting grammar-visible text).
 */
void fsm_advance(FSMState *fsm, Tokenizer *t, int prev_token, int token_id);

#endif /* TN_FSM_H */
