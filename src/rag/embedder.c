#include "rag/embedder.h"
#include "transformer/forward.h"
#include "kv_cache/sliding_window.h"
#include "memory/aligned_alloc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

TernaryError embedder_init(Embedder *emb, const Config *cfg) {
    if (!emb || !cfg) return TN_ERR_INVALID_ARGS;
    memset(emb, 0, sizeof(*emb));

    emb->state = (RunState *)malloc(sizeof(RunState));
    if (!emb->state) return TN_ERR_OOM;

    TernaryError err = run_state_alloc(emb->state, cfg, EMBEDDER_MAX_SEQ);
    if (err != TN_OK) {
        free(emb->state);
        emb->state = NULL;
        return err;
    }

    emb->embed_dim   = cfg->dim;
    emb->initialised = 1;
    return TN_OK;
}

void embedder_free(Embedder *emb) {
    if (!emb) return;
    if (emb->state) {
        run_state_free(emb->state);
        free(emb->state);
        emb->state = NULL;
    }
    emb->initialised = 0;
}

int embedder_generate(Embedder *emb, float *out, const char *text,
                      const Config *cfg, const TransformerWeights *w,
                      Tokenizer *tok, ThreadPool *tp) {
    if (!emb || !emb->initialised || !out || !text || !cfg || !w || !tok)
        return -1;

    int dim = emb->embed_dim;

    /* Tokenise input text */
    int max_toks = EMBEDDER_MAX_SEQ;
    int *tokens = (int *)malloc((size_t)max_toks * sizeof(int));
    if (!tokens) return -1;

    int n_tokens = tokenizer_encode(tok, text, strlen(text), tokens, max_toks);
    if (n_tokens <= 0) {
        free(tokens);
        return -1;
    }

    /* Reset the dedicated RunState — clear KV cache, reset position */
    RunState *es = emb->state;
    memset(es->key_cache,   0, (size_t)cfg->n_layers * cfg->n_kv_heads *
                               EMBEDDER_MAX_SEQ * (cfg->dim / cfg->n_heads) * sizeof(float));
    memset(es->value_cache, 0, (size_t)cfg->n_layers * cfg->n_kv_heads *
                               EMBEDDER_MAX_SEQ * (cfg->dim / cfg->n_heads) * sizeof(float));
    es->current_pos = 0;
    sw_init(&es->sw, EMBEDDER_MAX_SEQ, n_tokens);

    /* Accumulator for mean pooling */
    float *accum = (float *)calloc((size_t)dim, sizeof(float));
    if (!accum) {
        free(tokens);
        return -1;
    }

    /* Forward pass: run all tokens, accumulate s->x after layernorm */
    for (int i = 0; i < n_tokens; i++) {
        /* transformer_forward updates es->x in-place through all layers
         * and applies the final RMSNorm — exactly what we want to pool. */
        (void)transformer_forward(tokens[i], i, cfg, w, es, NULL, tp);

        /* Accumulate the final hidden state (after final RMSNorm) */
        for (int j = 0; j < dim; j++) {
            accum[j] += es->x[j];
        }
        es->current_pos = i + 1;
    }

    /* Mean pool */
    float inv_n = 1.0f / (float)n_tokens;
    for (int j = 0; j < dim; j++) {
        out[j] = accum[j] * inv_n;
    }

    /* L2 normalise — unit vector so cos_sim == dot_product */
    float norm_sq = 0.0f;
    for (int j = 0; j < dim; j++) norm_sq += out[j] * out[j];
    if (norm_sq > 1e-12f) {
        float inv_norm = 1.0f / sqrtf(norm_sq);
        for (int j = 0; j < dim; j++) out[j] *= inv_norm;
    }

    free(accum);
    free(tokens);
    return n_tokens;
}
