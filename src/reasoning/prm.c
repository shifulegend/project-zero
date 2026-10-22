/*
 * reasoning/prm.c — Phase 30.1 process reward model (see reasoning/prm.h).
 */
#include "reasoning/prm.h"
#include "tokenizer/tokenizer_gguf.h"
#include "transformer/forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *prm_format(const char *fmt, const char *question, const char *steps) {
    if (!fmt) fmt = PRM_DEFAULT_FORMAT;
    if (!question) question = "";
    if (!steps) steps = "";
    size_t lq = strlen(question), ls = strlen(steps), n = 0;
    /* Upper bound: every byte of fmt plus one copy per placeholder. */
    size_t cap = strlen(fmt) + 1;
    for (const char *p = fmt; (p = strchr(p, '{')) != NULL; p++) cap += lq + ls;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    for (const char *p = fmt; *p;) {
        if (strncmp(p, "{question}", 10) == 0) { memcpy(out + n, question, lq); n += lq; p += 10; }
        else if (strncmp(p, "{steps}", 7) == 0) { memcpy(out + n, steps, ls); n += ls; p += 7; }
        else out[n++] = *p++;
    }
    out[n] = '\0';
    return out;
}

float prm_score_from_logits(const float *logits, int vocab, int good_id, int bad_id) {
    if (!logits || good_id < 0 || bad_id < 0 || good_id >= vocab || bad_id >= vocab ||
        good_id == bad_id)
        return -1.0f;
    double g = logits[good_id], b = logits[bad_id], m = g > b ? g : b;
    double eg = exp(g - m), eb = exp(b - m);
    return (float)(eg / (eg + eb));
}

static int first_token(Tokenizer *t, const char *s) {
    int ids[64];
    int n = tokenizer_encode(t, s, strlen(s), ids, 64);
    return n > 0 ? ids[0] : -1;
}

TernaryError prm_load(PrmModel *pm, const char *path, const char *format,
                      const char *good, const char *bad, int max_seq_len, ThreadPool *tp) {
    if (!pm || !path) return TN_ERR_INVALID_ARGS;
    memset(pm, 0, sizeof(*pm));
    TernaryError e = load_gguf_model(&pm->model, path, /*is_primary=*/false, max_seq_len, tp,
                                     /*stdout_is_tty=*/false);
    if (e != TN_OK) {
        fprintf(stderr, "[prm] cannot load '%s': %s\n", path, tn_error_str(e));
        return e;
    }
    pm->loaded = 1;
    pm->tp = tp;
    e = tokenizer_load_from_gguf(&pm->tok, &pm->model.gguf_hdr);
    if (e != TN_OK) { prm_free(pm); return e; }
    pm->format = strdup(format && format[0] ? format : PRM_DEFAULT_FORMAT);
    pm->good_id = first_token(&pm->tok, good && good[0] ? good : PRM_DEFAULT_GOOD);
    pm->bad_id = first_token(&pm->tok, bad && bad[0] ? bad : PRM_DEFAULT_BAD);
    pm->prev = (int *)malloc((size_t)pm->model.state->max_seq_len * sizeof(int));
    if (!pm->format || !pm->prev || pm->good_id < 0 || pm->bad_id < 0 || pm->good_id == pm->bad_id) {
        fprintf(stderr, "[prm] good/bad strings must map to two different tokens\n");
        prm_free(pm);
        return TN_ERR_INVALID_ARGS;
    }
    fprintf(stderr, "[prm] judge '%s' loaded (good token %d, bad token %d)\n",
            path, pm->good_id, pm->bad_id);
    return TN_OK;
}

void prm_free(PrmModel *pm) {
    if (!pm) return;
    if (pm->loaded) {
        tokenizer_free(&pm->tok);
        loaded_model_free(&pm->model);
    }
    free(pm->format);
    free(pm->prev);
    memset(pm, 0, sizeof(*pm));
}

float prm_score(PrmModel *pm, const char *question, const char *steps) {
    if (!pm || !pm->loaded) return -1.0f;
    char *text = prm_format(pm->format, question, steps);
    if (!text) return -1.0f;
    RunState *s = pm->model.state;
    int cap = s->max_seq_len;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { free(text); return -1.0f; }
    int n = 0;
    if (pm->tok.add_bos_token && pm->tok.bos_token_id >= 0) ids[n++] = pm->tok.bos_token_id;
    int k = tokenizer_encode(&pm->tok, text, strlen(text), ids + n, cap - n);
    free(text);
    if (k <= 0) { free(ids); return -1.0f; }
    n += k;

    /* Reuse the judge's KV for the prefix shared with the previous call
     * (question + earlier steps); recurrent models always restart. */
    int start = 0;
    if (!moe_is_recurrent(&pm->model.moe_config)) {
        while (start < n - 1 && start < pm->n_prev && pm->prev[start] == ids[start]) start++;
    }
    if (start == 0) sw_init(&s->sw, s->max_seq_len, 0);
    const float *logits = NULL;
    for (int i = start; i < n; i++) {
        logits = transformer_forward(ids[i], i, &pm->model.config, &pm->model.weights, s,
                                     &pm->model.moe_config, pm->tp);
        if (!logits) { pm->n_prev = 0; free(ids); return -1.0f; }
    }
    memcpy(pm->prev, ids, (size_t)n * sizeof(int));
    pm->n_prev = n;
    free(ids);
    return prm_score_from_logits(logits, pm->model.config.vocab_size, pm->good_id, pm->bad_id);
}
