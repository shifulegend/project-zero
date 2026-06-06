#include "agent/output_inject.h"
#include "transformer/forward.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int inject_text_into_kv(const char *text, const Config *cfg, const TransformerWeights *w, RunState *s, Tokenizer *tok, ThreadPool *tp) {
    if (!text || !cfg || !w || !s || !tok) return -1;

    /* Heuristic max tokens for the injected text */
    const int MAX_INJECT = 4096;
    int *tokens = (int *)calloc(MAX_INJECT, sizeof(int));
    if (!tokens) return -1;

    int n = tokenizer_encode(tok, text, strlen(text), tokens, MAX_INJECT);
    if (n < 0) {
        free(tokens);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        if (s->current_pos >= cfg->seq_len) {
            /* Can't inject beyond sequence length */
            free(tokens);
            return -2;
        }
        /* Run forward pass for the token in the current_pos to populate KV */
        (void)transformer_forward(tokens[i], s->current_pos, cfg, w, s, NULL, tp);
        s->current_pos++;
    }

    free(tokens);
    return n;
}
