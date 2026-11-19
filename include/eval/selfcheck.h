#ifndef TN_SELFCHECK_H
#define TN_SELFCHECK_H

/*
 * selfcheck.h — the engine's own accuracy probe (2026-11-18).
 *
 * A fast configuration is only worth using if it still computes the model. The probe
 * scores a short built-in passage twice with the model that is loaded: once as configured
 * (activation numerics from --act-quant, LM head from --classifier) and once as the
 * reference (float activations, BF16 head), and compares the perplexities. The relative
 * increase must stay within --quality-tolerance percent (default 1.0, the same bar the
 * benchmark correctness gate uses against another engine, tools/bench/gate.py).
 *
 * Used by `pz --self-check`, by startup (an explicitly requested lossy LM head that fails
 * falls back to BF16) and by POST /v1/selfcheck. The judging and formatting are pure
 * functions (tests/test_selfcheck.c); selfcheck_run() needs a loaded model.
 */

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "runtime/run_state.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"

typedef enum { SC_PASS = 0, SC_FAIL, SC_SKIPPED } SelfCheckVerdict;

typedef struct {
    double tolerance_pct; /* allowed perplexity increase over the reference, percent */
    int n_tokens;         /* probe length including BOS (clamped to the passage and context) */
} SelfCheckOptions;

typedef struct {
    SelfCheckVerdict verdict;
    char act_quant[8];   /* "int8" / "float": the ternary matmul as configured */
    char classifier[10]; /* "bf16" / "int8" / "int4" */
    double ppl_ref, ppl_cand;
    double delta_pct; /* (ppl_cand / ppl_ref - 1) * 100 */
    double tolerance_pct;
    int n_scored;
    double seconds;
    char note[120]; /* why SKIPPED, or what the verdict compares */
} SelfCheckResult;

/* Pure: percent increase of the candidate over the reference (NAN when either is invalid). */
double selfcheck_delta_pct(double ppl_ref, double ppl_cand);
/* Pure: PASS when delta <= tolerance (a candidate that is better than the reference passes);
 * FAIL on a non-finite value. tolerance <= 0 means "no bar": always PASS when finite. */
SelfCheckVerdict selfcheck_judge(double delta_pct, double tolerance_pct);
const char *selfcheck_verdict_name(SelfCheckVerdict v);
/* The built-in passage the probe scores. */
const char *selfcheck_text(void);
/* One-line summary for the terminal and logs. */
void selfcheck_format(const SelfCheckResult *r, char *out, size_t cap);

/* Runs the probe on the model's own state. Not thread-safe against generation: the caller
 * owns the engine (CLI before generating, API with every generation unit held). The
 * activation mode and LM-head format are restored before returning. */
TernaryError selfcheck_run(const SelfCheckOptions *opt, const Config *cfg,
                           const TransformerWeights *w, RunState *s, const MoEConfig *mc,
                           Tokenizer *tok, ThreadPool *tp, SelfCheckResult *out);


/* ---- `--act-quant auto` decisions (pure, unit-tested; main.c applies them) ---- */
typedef struct {
    int act_quant_requested; /* 0 = auto; an explicit mode skips the check */
    int ternary_model;
    int no_calibrate;
    int sanitized_build;
    double tolerance_pct; /* 0 = no bar */
    int other_mode;       /* --self-check, --perplexity or --watermark-detect */
    int effective_is_int8;
} SelfCheckAutoCtx;
/* 1 when a start should check int8 against the accuracy bar (once per model, cached). */
int selfcheck_auto_applies(const SelfCheckAutoCtx *c);
/* The activation mode a verdict leaves in force: FAIL puts float activations in force; else AUTO. */
int selfcheck_auto_numerics(SelfCheckVerdict v); /* TnActQuant value: 0 auto, 2 float */

/* ---- First-run probe cache (2026-11-18) -------------------------------------------------------
 * --act-quant auto on a ternary model is accuracy-checked once per (model file, numerics, head,
 * tolerance, probe length): the verdict is stored in ~/.project-zero/selfcheck.txt and read on the
 * next start, so only the first run pays for the two short passes. Pure helpers, unit-tested. */
#define TN_SELFCHECK_CACHE_VERSION 1

/* Cache key text for a model file (size and mtime identify it) and the probed configuration. */
void selfcheck_cache_key(char *out, size_t cap, const char *model_path, long long size,
                         long long mtime, const char *act_quant, const char *classifier,
                         double tolerance_pct, int n_tokens);
/* Looks `key` up in the cache text (lines "key<TAB>VERDICT"). Returns 1 and sets *v when found. */
int selfcheck_cache_find(const char *cache_text, const char *key, SelfCheckVerdict *v);
/* File-backed wrappers; never fail the run (a cache that cannot be read or written is a miss). */
int selfcheck_cache_load(const char *key, SelfCheckVerdict *v);
void selfcheck_cache_store(const char *key, SelfCheckVerdict v);

#endif /* TN_SELFCHECK_H */
