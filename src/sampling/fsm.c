#include "sampling/fsm.h"

#define TN_GRAMMAR_MASK_SENTINEL (-1e30f)

void fsm_init(FSMState *fsm) {
    /* 2026-10-22: root mode -- output always starts with '{' or '[' and ends
     * at the matching bracket. The permissive any-value start let sampling at
     * high temperature open with a bare string ("{"...) and pad with
     * whitespace after the value until max_tokens. */
    json_grammar_init_root(&fsm->json);
}

static int is_eos_token(const Tokenizer *t, int token_id) {
    /* Includes chat-mode turn ends, so JSON mode cannot stop early on one. */
    return tokenizer_is_eog(t, token_id);
}

/* Feeds `piece` through a scratch copy of `base`, returning 1 if every
 * byte was accepted (never JSON_ST_INVALID). Does not mutate `base`. */
static int piece_is_acceptable(const JsonGrammarState *base, const char *piece) {
    /* A token that emits nothing never advances the document; allowing it
     * lets sampling idle until max_tokens (2026-10-22). */
    if (!piece || !*piece) return 0;
    JsonGrammarState g = *base; /* struct copy: plain fields + fixed array */
    for (const char *p = piece; *p; p++) {
        if (json_grammar_step(&g, *p) == JSON_ST_INVALID) return 0;
    }
    return 1;
}

static int eos_is_acceptable(const JsonGrammarState *base) {
    JsonGrammarState g = *base;
    json_grammar_finalize(&g);
    return json_grammar_is_complete(&g);
}

void fsm_compute_token_mask(const FSMState *fsm, Tokenizer *t, int prev_token,
                             float *logits, int vocab_size) {
    int eos_ok = eos_is_acceptable(&fsm->json);
    for (int token_id = 0; token_id < vocab_size; token_id++) {
        if (is_eos_token(t, token_id)) {
            if (!eos_ok) logits[token_id] = TN_GRAMMAR_MASK_SENTINEL;
            continue;
        }
        const char *piece = tokenizer_decode(t, prev_token, token_id);
        if (!piece_is_acceptable(&fsm->json, piece))
            logits[token_id] = TN_GRAMMAR_MASK_SENTINEL;
    }
}

void fsm_advance(FSMState *fsm, Tokenizer *t, int prev_token, int token_id) {
    if (is_eos_token(t, token_id)) return; /* terminates generation, not grammar-visible */
    const char *piece = tokenizer_decode(t, prev_token, token_id);
    for (const char *p = piece; *p; p++) {
        json_grammar_step(&fsm->json, *p);
    }
}
