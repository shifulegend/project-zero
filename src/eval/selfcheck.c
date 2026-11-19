#include "eval/selfcheck.h"
#include "core/hardware_profile.h"
#include "eval/perplexity.h"
#include "math/simd_dispatch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Plain English prose on a general subject; long enough for the default probe length. */
static const char SC_TEXT[] =
    "The history of navigation is a story of people learning to find their way across open "
    "water. Early sailors stayed within sight of the coast and steered by landmarks, the "
    "position of the sun and the direction of the wind. As voyages grew longer they learned "
    "to read the stars, and instruments such as the compass and the astrolabe made it "
    "possible to hold a course far from land. Maps improved slowly, each expedition adding "
    "a few more coastlines and correcting the errors of the last. The invention of an "
    "accurate marine clock finally let a ship's crew calculate its longitude, and the "
    "great distances of the oceans became routes that merchants could plan with confidence. "
    "Today satellites do the work that once took a lifetime of experience, yet many "
    "sailors still carry a paper chart and a sextant, because a good navigator never "
    "depends on a single tool.";

const char *selfcheck_text(void) {
    return SC_TEXT;
}

double selfcheck_delta_pct(double ppl_ref, double ppl_cand) {
    if (!isfinite(ppl_ref) || !isfinite(ppl_cand) || ppl_ref <= 0.0 || ppl_cand <= 0.0) return NAN;
    return (ppl_cand / ppl_ref - 1.0) * 100.0;
}

SelfCheckVerdict selfcheck_judge(double delta_pct, double tolerance_pct) {
    if (!isfinite(delta_pct)) return SC_FAIL;
    if (tolerance_pct <= 0.0) return SC_PASS;
    return delta_pct <= tolerance_pct ? SC_PASS : SC_FAIL;
}

const char *selfcheck_verdict_name(SelfCheckVerdict v) {
    if (v == SC_PASS) return "PASS";
    if (v == SC_FAIL) return "FAIL";
    return "SKIPPED";
}

void selfcheck_format(const SelfCheckResult *r, char *out, size_t cap) {
    if (r->verdict == SC_SKIPPED) {
        snprintf(out, cap, "self-check %s: %s", selfcheck_verdict_name(r->verdict), r->note);
        return;
    }
    snprintf(out, cap,
             "self-check %s: act-quant %s, head %s: perplexity %.3f vs %.3f reference (%+.2f%%, "
             "tolerance %.2f%%, %d tokens, %.1f s)",
             selfcheck_verdict_name(r->verdict), r->act_quant, r->classifier, r->ppl_cand,
             r->ppl_ref, r->delta_pct, r->tolerance_pct, r->n_scored, r->seconds);
}

static const char *cls_name(TnClassifierFormat f) {
    if (f == TN_CLS_INT4) return "int4";
    if (f == TN_CLS_INT8) return "int8";
    return "bf16";
}

static TernaryError score(const int *toks, int n, int bos, const Config *cfg,
                          const TransformerWeights *w, RunState *s, const MoEConfig *mc,
                          ThreadPool *tp, PplResult *r) {
    PplOptions o = {n, 1, bos};
    return perplexity_eval(toks, n, &o, cfg, w, s, mc, tp, NULL, NULL, r);
}

TernaryError selfcheck_run(const SelfCheckOptions *opt, const Config *cfg,
                           const TransformerWeights *w, RunState *s, const MoEConfig *mc,
                           Tokenizer *tok, ThreadPool *tp, SelfCheckResult *out) {
    if (!opt || !cfg || !w || !s || !tok || !out) return TN_ERR_INVALID_ARGS;
    memset(out, 0, sizeof(*out));
    out->tolerance_pct = opt->tolerance_pct;
    out->verdict       = SC_SKIPPED;

    const TnHardwareProfile *hp = tn_hardware_profile_get();
    TnClassifierFormat cand_cls = hp ? hp->classifier_fmt : TN_CLS_BF16;
    snprintf(out->act_quant, sizeof(out->act_quant), "%s", tn_act_quant_effective());
    snprintf(out->classifier, sizeof(out->classifier), "%s", cls_name(cand_cls));

    /* Tokens: BOS (when the model has one) + the passage, cut to the requested length. */
    size_t len = strlen(SC_TEXT);
    int cap    = (int)len + 2;
    int *toks  = (int *)malloc((size_t)cap * sizeof(int));
    if (!toks) return TN_ERR_OOM;
    int bos = tokenizer_bos_to_prepend(tok, cfg->bos_token_id);
    int n   = 0;
    if (bos >= 0) toks[n++] = bos;
    int k = tokenizer_encode(tok, SC_TEXT, len, toks + n, cap - n);
    if (k < 0) {
        free(toks);
        snprintf(out->note, sizeof(out->note), "the passage could not be tokenized");
        return TN_OK;
    }
    n += k;
    int want = opt->n_tokens > 0 ? opt->n_tokens : 96;
    if (n > want) n = want;
    if (n > s->max_seq_len) n = s->max_seq_len;
    if (n < 16) {
        free(toks);
        snprintf(out->note, sizeof(out->note), "context too short for the probe (%d tokens)", n);
        return TN_OK;
    }

    /* Candidate and reference identical: nothing to compare. */
    int cand_is_float  = strcmp(out->act_quant, "float") == 0;
    int head_quantised = cand_cls != TN_CLS_BF16 && (w->wcls_i8 || w->wcls_i4);
    if (cand_is_float && !head_quantised) {
        free(toks);
        snprintf(out->note, sizeof(out->note),
                 "already the reference (float activations, BF16 head)");
        return TN_OK;
    }

    PplResult cand, ref;
    TernaryError e = score(toks, n, bos, cfg, w, s, mc, tp, &cand);
    if (e == TN_OK) {
        TnActQuant old_act         = tn_act_quant_get();
        TnClassifierFormat old_cls = tn_hardware_profile_swap_classifier(TN_CLS_BF16);
        tn_act_quant_set(TN_ACT_FLOAT);
        e = score(toks, n, bos, cfg, w, s, mc, tp, &ref);
        tn_act_quant_set(old_act);
        tn_hardware_profile_swap_classifier(old_cls);
    }
    free(toks);
    if (e != TN_OK) return e;

    out->ppl_cand  = cand.ppl;
    out->ppl_ref   = ref.ppl;
    out->n_scored  = cand.n_scored;
    out->seconds   = cand.seconds + ref.seconds;
    out->delta_pct = selfcheck_delta_pct(ref.ppl, cand.ppl);
    out->verdict   = selfcheck_judge(out->delta_pct, opt->tolerance_pct);
    snprintf(out->note, sizeof(out->note),
             "as configured against float activations and a BF16 head");
    return TN_OK;
}

/* ---- first-run probe cache ---- */

void selfcheck_cache_key(char *out, size_t cap, const char *model_path, long long size,
                         long long mtime, const char *act_quant, const char *classifier,
                         double tolerance_pct, int n_tokens) {
    const char *base = model_path ? strrchr(model_path, '/') : NULL;
    base             = base ? base + 1 : (model_path ? model_path : "");
    snprintf(out, cap, "v%d|%s|%lld|%lld|act=%s|head=%s|tol=%.2f|tok=%d",
             TN_SELFCHECK_CACHE_VERSION, base, size, mtime, act_quant ? act_quant : "",
             classifier ? classifier : "", tolerance_pct, n_tokens);
}

int selfcheck_cache_find(const char *text, const char *key, SelfCheckVerdict *v) {
    if (!text || !key || !*key) return 0;
    size_t kl = strlen(key);
    for (const char *p = text; *p;) {
        const char *eol = strchr(p, '\n');
        size_t n        = eol ? (size_t)(eol - p) : strlen(p);
        if (n > kl && strncmp(p, key, kl) == 0 && p[kl] == '\t') {
            const char *w = p + kl + 1;
            if (strncmp(w, "PASS", 4) == 0) {
                if (v) *v = SC_PASS;
                return 1;
            }
            if (strncmp(w, "FAIL", 4) == 0) {
                if (v) *v = SC_FAIL;
                return 1;
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    return 0;
}

static int cache_path(char *buf, size_t cap) {
    const char *home = getenv("HOME");
    if (!home || !*home) return -1;
    snprintf(buf, cap, "%s/.project-zero/selfcheck.txt", home);
    return 0;
}

int selfcheck_cache_load(const char *key, SelfCheckVerdict *v) {
    char path[512];
    if (cache_path(path, sizeof(path)) != 0) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char *buf = (char *)malloc(65536);
    if (!buf) {
        fclose(f);
        return 0;
    }
    size_t n = fread(buf, 1, 65535, f);
    buf[n]   = '\0';
    fclose(f);
    int found = selfcheck_cache_find(buf, key, v);
    free(buf);
    return found;
}

void selfcheck_cache_store(const char *key, SelfCheckVerdict v) {
    char path[512], dir[512];
    if (v == SC_SKIPPED || cache_path(path, sizeof(path)) != 0) return;
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        mkdir(dir, 0755);
    }
    FILE *f = fopen(path, "ab");
    if (!f) return;
    fprintf(f, "%s\t%s\n", key, selfcheck_verdict_name(v));
    fclose(f);
}

int selfcheck_auto_applies(const SelfCheckAutoCtx *c) {
    return c && c->act_quant_requested == 0 && c->ternary_model && !c->no_calibrate &&
           !c->sanitized_build && c->tolerance_pct > 0.0 && !c->other_mode && c->effective_is_int8;
}

int selfcheck_auto_numerics(SelfCheckVerdict v) {
    return v == SC_FAIL ? 2 : 0;
}
