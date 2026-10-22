#include "eval/perplexity.h"
#include "sampling/logprob.h"
#include "transformer/forward.h"
#include "kv_cache/sliding_window.h"
#include <math.h>
#include <string.h>
#include <time.h>

double ppl_token_nll(const float *logits, int vocab, int target) {
    if (!logits || target < 0 || target >= vocab) return NAN;
    return logprob_logsumexp(logits, vocab) - (double)logits[target];
}

void ppl_finish(PplResult *r, double sum_nll, double sum_nll2, int n) {
    r->n_scored = n;
    if (n <= 0) { r->mean_nll = r->ppl = r->ppl_err = NAN; return; }
    double mean = sum_nll / n;
    double var = sum_nll2 / n - mean * mean;
    if (var < 0.0) var = 0.0;
    r->mean_nll = mean;
    r->ppl = exp(mean);
    r->ppl_err = n > 1 ? r->ppl * sqrt(var / (n - 1)) : 0.0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

TernaryError perplexity_eval(const int *tokens, int n_tokens, const PplOptions *opt,
                             const Config *cfg, const TransformerWeights *w, RunState *s,
                             const MoEConfig *mc, ThreadPool *tp,
                             PplProgressFn progress, void *user, PplResult *out) {
    if (!tokens || !opt || !cfg || !w || !s || !out) return TN_ERR_INVALID_ARGS;
    memset(out, 0, sizeof(*out));
    int n_ctx = opt->n_ctx > 0 ? opt->n_ctx : 512;
    if (n_ctx > s->max_seq_len) n_ctx = s->max_seq_len;
    if (n_ctx < 4) return TN_ERR_INVALID_ARGS;
    int n_chunks = n_tokens / n_ctx;
    if (opt->max_chunks > 0 && n_chunks > opt->max_chunks) n_chunks = opt->max_chunks;
    if (n_chunks <= 0) return TN_ERR_INVALID_ARGS;   /* fewer tokens than one chunk */

    double t0 = now_s(), sum = 0.0, sum2 = 0.0;
    int n = 0;
    int first = n_ctx / 2;
    for (int c = 0; c < n_chunks; c++) {
        const int *chunk = tokens + (size_t)c * (size_t)n_ctx;
        sw_init(&s->sw, s->max_seq_len, 0);   /* fresh window per chunk, from position 0 */
        for (int j = 0; j < n_ctx - 1; j++) {
            int tok = (j == 0 && opt->bos_token >= 0) ? opt->bos_token : chunk[j];
            const float *logits = transformer_forward(tok, j, cfg, w, s, mc, tp);
            if (!logits) return TN_ERR_INVALID_ARGS;
            if (j >= first) {
                double nll = ppl_token_nll(logits, cfg->vocab_size, chunk[j + 1]);
                if (!isfinite(nll)) return TN_ERR_INVALID_ARGS;
                sum += nll;
                sum2 += nll * nll;
                n++;
            }
        }
        out->n_chunks = c + 1;
        out->seconds = now_s() - t0;
        ppl_finish(out, sum, sum2, n);
        if (progress) progress(c, out, user);
    }
    return TN_OK;
}
