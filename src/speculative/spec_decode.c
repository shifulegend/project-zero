#include "speculative/spec_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/log.h"
#include "core/step_timing.h"
#include "core/timer.h"
#include "distributed/dist_plan.h"
#include "math/simd_core_ops.h"
#include "runtime/spec_scratch.h"
#include "sampling/rng.h"
#include "sampling/sampling.h"
#include "speculative/accept_reject.h"
#include "tokenizer/chat_template.h"
#include "transformer/forward.h"
#include "transformer/moe_ffn.h"

/* ── stdout callback, matching generate.c's convention ─────────────────── */
static int stdout_token_callback(const char *piece, void *userdata) {
    (void)userdata;
    printf("%s", piece);
    fflush(stdout);
    return 0;
}

/* Sampling matching generate_with_callback()'s inline logic exactly (greedy
 * argmax / top-p / plain CDF walk), extracted here since the draft model's
 * per-token sampling needs the identical policy. Mutates `logits` in place
 * (temperature scaling + softmax) -- callers that need the pre-sampling
 * distribution (accept_reject_round()'s draft_logits) must copy it out
 * first. */
static int sample_next(float *logits, int vocab_size, float temperature, float top_p,
                       unsigned long long *rng_state) {
    if (temperature <= 0.0f || temperature < 1e-6f) {
        return sample_argmax(logits, vocab_size);
    }
    apply_temperature(logits, vocab_size, temperature);
    if (top_p < 1.0f && top_p > 0.0f) {
        return sample_top_p(logits, vocab_size, top_p, rng_state);
    }
    tn_softmax(logits, vocab_size);
    float r   = rng_float(rng_state);
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return vocab_size - 1;
}

static int is_eos_token(int token, const Config *cfg, const Tokenizer *tok,
                        const unsigned char *chat_stops) {
    if (cfg->eos_token_id > 0 && token == cfg->eos_token_id) return 1;
    return tokenizer_is_eog(tok, token, chat_stops); /* incl. chat-mode turn ends */
}

/* ── Core implementation ─────────────────────────────────────────────────
 *
 * Position bookkeeping: transformer_forward(token, pos, ...) writes `token`
 * into the KV cache at `pos` and returns logits predicting whatever comes
 * at `pos+1`. transformer_forward_batch(tokens, pos, n, ...)'s row i
 * predicts `pos+i+1` (see transformer/forward.h). Standard speculative
 * decoding verification (Leviathan/Chen et al.) checks draft_tokens[i]
 * (occupying position pos+i) against the verifier's prediction made BEFORE
 * tokens[i] was fed -- i.e. row i-1 of the batch output (or the
 * carried-over pre-round prediction for i=0), NOT row i. This file builds
 * that shifted "check" array explicitly (verify_check_logits) rather than
 * indexing the batch output directly, to keep that off-by-one from ever
 * being implicit.
 *
 * KV commit: once accept_reject_round() decides n_accept and an
 * emit_token (a resampled token on rejection, or a bonus token sampled
 * fresh after full acceptance), position pos+n_accept needs exactly ONE
 * more real transformer_forward(emit_token, pos+n_accept, ...) call on
 * EACH model -- on the verifier, this overwrites whatever (possibly wrong,
 * on rejection) K/V the batch call wrote there, which is the literal
 * mechanism behind "any KV slots written for rejected draft positions are
 * naturally overwritten the next time that logical position is actually
 * reached" (this call IS that next time); on the draft model, this is the
 * corrective commit the project's Phase 18 plan calls out explicitly,
 * since the draft's own speculative guess at that position (if any) is
 * stale once the verifier's real token is known. Neither call is a special
 * rollback path -- it's the same "commit one token" mechanism
 * generate_with_callback() already uses for every token, just made
 * explicit at the round boundary instead of implicit in a per-token loop. */
/* Everything one generation carries between the steps below. */
typedef struct {
    const Config *cfg;
    const TransformerWeights *w;
    RunState *s;
    const MoEConfig *mc;
    Tokenizer *tok;
    ThreadPool *tp;
    DraftModel *draft;
    int vocab_size, spec_length, max_tokens;
    float temperature, top_p;
    TokenCallback callback;
    void *userdata;
    unsigned char *chat_stops; /* chat-mode turn ends for this generation (owned, never shared) */
    unsigned long long rng_state;
    SpecBatchScratch sb;
    /* Round buffers */
    int *draft_tokens;
    float *draft_logits_buf;     /* [spec_length x vocab] draft distributions */
    float *verify_check_logits;  /* [spec_length x vocab] verifier prediction before token i */
    float *verifier_logits_out;  /* [spec_length x vocab] batch output */
    float *last_verifier_logits; /* [vocab] carried over between rounds */
    float *last_draft_logits;    /* [vocab] */
    /* Progress */
    int pos, prev_token, tokens_generated, stop;
    int64_t gen_start_us;
} SpecRun;

/* The cases speculative decoding refuses or degrades, in one place. Returns 0 to refuse. */
static int spec_supported(const MoEConfig *mc, const RunState *s, bool *json_mode) {
    if (dist_plan_active()) {
        TN_LOG_ERR(
            "spec",
            "[spec] Speculative decoding is not supported with --remote-layers -- refusing.\n");
        return 0;
    }
    if (moe_is_recurrent(mc)) {
        TN_LOG_ERR("spec", "[spec] Speculative decoding is not supported for recurrent-state "
                           "models (Qwen3.5/3.6 linear attention, Mamba SSM) -- refusing.\n");
        return 0;
    }
    if (s->current_pos != 0) {
        TN_LOG_ERR("spec",
                   "[spec] Speculative decoding does not support resuming from a "
                   "nonzero position (e.g. vision prefill) -- the draft model has no matching "
                   "prefix in its own KV cache. Use plain generation for this request.\n");
        return 0;
    }
    if (*json_mode) {
        TN_LOG_WRN("spec", "[spec] Warning: --json is not wired into speculative decoding's "
                           "accept/reject loop yet -- proceeding with plain sampling.\n");
        *json_mode = false;
    }
    return 1;
}

/* ── Prompt encoding (verbatim from generate_with_callback()) ────────────
 * Fills `prompt_tokens` (capacity seq_len + 1); returns the token count, or -1 on failure. */
static int encode_prompt(const Config *cfg, Tokenizer *tok, const char *prompt,
                         int *prompt_tokens) {
    int n_prompt = 0;
    if (tok->chat_template) {
        const char *bos_str = (tok->bos_token_id >= 0 && tok->bos_token_id < tok->vocab_size)
                                  ? tok->vocab[tok->bos_token_id]
                                  : "";
        const char *eos_str = (tok->eos_token_id >= 0 && tok->eos_token_id < tok->vocab_size)
                                  ? tok->vocab[tok->eos_token_id]
                                  : "";
        /* Content, never chat markup (prompt-injection defence, see tokenizer.h). */
        char *safe = tokenizer_neutralize_control(tok, prompt);
        if (!safe) return -1;
        const char *roles[1]    = {"user"};
        const char *contents[1] = {safe};
        char *formatted = chat_template_apply(tok->chat_template, roles, contents, 1, bos_str,
                                              eos_str, /*add_generation_prompt=*/1);
        free(safe);
        if (!formatted) return -1;
        n_prompt = tokenizer_encode(tok, formatted, strlen(formatted), prompt_tokens, cfg->seq_len);
        free(formatted);
        if (n_prompt < 0) return -1;
    } else {
        int bos = (cfg->bos_token_id > 0)    ? cfg->bos_token_id
                  : (tok->bos_token_id >= 0) ? tok->bos_token_id
                                             : -1;
        if (bos >= 0) prompt_tokens[n_prompt++] = bos;
        int n_encoded = tokenizer_encode(tok, prompt, strlen(prompt), &prompt_tokens[n_prompt],
                                         cfg->seq_len - n_prompt);
        if (n_encoded < 0) return -1;
        n_prompt += n_encoded;
    }
    return n_prompt == 0 ? -1 : n_prompt;
}

static void round_buffers_free(SpecRun *r) {
    free(r->draft_tokens);
    free(r->draft_logits_buf);
    free(r->verify_check_logits);
    free(r->verifier_logits_out);
    free(r->last_verifier_logits);
    free(r->last_draft_logits);
}

/* Returns 0 on success; frees whatever it allocated and returns -1 on failure. */
static int round_buffers_alloc(SpecRun *r) {
    const size_t rows       = (size_t)r->spec_length * r->vocab_size;
    r->draft_tokens         = (int *)malloc((size_t)r->spec_length * sizeof(int));
    r->draft_logits_buf     = (float *)malloc(rows * sizeof(float));
    r->verify_check_logits  = (float *)malloc(rows * sizeof(float));
    r->verifier_logits_out  = (float *)malloc(rows * sizeof(float));
    r->last_verifier_logits = (float *)malloc((size_t)r->vocab_size * sizeof(float));
    r->last_draft_logits    = (float *)malloc((size_t)r->vocab_size * sizeof(float));
    if (r->draft_tokens && r->draft_logits_buf && r->verify_check_logits &&
        r->verifier_logits_out && r->last_verifier_logits && r->last_draft_logits)
        return 0;
    TN_LOG_ERR("spec", "[spec] Failed to allocate round buffers.\n");
    round_buffers_free(r);
    return -1;
}

/* ── Prefill: both models process the full prompt sequentially ───────────── */
static void prefill(SpecRun *r, const int *prompt_tokens, int n_prompt) {
    float *v_logits = NULL, *d_logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        v_logits = transformer_forward(prompt_tokens[i], i, r->cfg, r->w, r->s, r->mc, r->tp);
        d_logits = transformer_forward(prompt_tokens[i], i, &r->draft->config, &r->draft->weights,
                                       r->draft->state, &r->draft->moe_config, r->tp);
    }
    memcpy(r->last_verifier_logits, v_logits, (size_t)r->vocab_size * sizeof(float));
    memcpy(r->last_draft_logits, d_logits, (size_t)r->vocab_size * sizeof(float));
}

/* ── Draft phase: spec_length sequential draft-model forward calls ── */
static void draft_phase(SpecRun *r) {
    const int vocab_size = r->vocab_size;
    int64_t t_step       = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    memcpy(r->draft_logits_buf, r->last_draft_logits, (size_t)vocab_size * sizeof(float));
    int dt = sample_next(r->draft_logits_buf, vocab_size, r->temperature, r->top_p, &r->rng_state);
    r->draft_tokens[0] = dt;
    for (int i = 1; i < r->spec_length; i++) {
        float *lg =
            transformer_forward(r->draft_tokens[i - 1], r->pos + i - 1, &r->draft->config,
                                &r->draft->weights, r->draft->state, &r->draft->moe_config, r->tp);
        memcpy(r->draft_logits_buf + (size_t)i * vocab_size, lg,
               (size_t)vocab_size * sizeof(float));
        r->draft_tokens[i] = sample_next(r->draft_logits_buf + (size_t)i * vocab_size, vocab_size,
                                         r->temperature, r->top_p, &r->rng_state);
    }
    if (t_step) {
        tn_step_timing_add(TN_STEP_24_SPEC_DRAFT_PHASE, tn_step_timing_now_ns() - t_step);
    }
}

/* ── Verify phase: one batched forward pass on the verifier ───────── */
static TernaryError verify_phase(SpecRun *r) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    TernaryError verr =
        transformer_forward_batch(r->draft_tokens, r->pos, r->spec_length, r->cfg, r->w, r->s,
                                  &r->sb, r->mc, r->tp, r->verifier_logits_out);
    if (t_step) {
        tn_step_timing_add(TN_STEP_25_SPEC_VERIFY_BATCH, tn_step_timing_now_ns() - t_step);
    }
    return verr;
}

/* ── Accept/reject: build the shifted "what would the verifier have predicted before seeing
 * tokens[i]" array -- see this function's header comment -- then decide n_accept/emit_token. */
static void accept_phase(SpecRun *r, int *n_accept, int *emit_token) {
    const int vocab_size = r->vocab_size;
    int64_t t_step       = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    memcpy(r->verify_check_logits, r->last_verifier_logits, (size_t)vocab_size * sizeof(float));
    if (r->spec_length > 1) {
        memcpy(r->verify_check_logits + vocab_size, r->verifier_logits_out,
               (size_t)(r->spec_length - 1) * vocab_size * sizeof(float));
    }
    const float *bonus_logits = r->verifier_logits_out + (size_t)(r->spec_length - 1) * vocab_size;

    *n_accept   = 0;
    *emit_token = 0;
    accept_reject_round(r->draft_tokens, r->draft_logits_buf, r->verify_check_logits, bonus_logits,
                        r->spec_length, vocab_size, r->temperature, &r->rng_state, n_accept,
                        emit_token);
    if (t_step) {
        tn_step_timing_add(TN_STEP_26_SPEC_ACCEPT_REJECT, tn_step_timing_now_ns() - t_step);
    }
}

/* ── Emit accepted tokens + the resampled/bonus token; sets r->stop ──────── */
static void emit_round(SpecRun *r, int n_accept, int emit_token) {
    for (int i = 0; i <= n_accept; i++) {
        int emitted = (i < n_accept) ? r->draft_tokens[i] : emit_token;
        if (is_eos_token(emitted, r->cfg, r->tok, r->chat_stops)) {
            r->stop = 1;
            break;
        }
        if (r->tokens_generated == 0) r->gen_start_us = timer_now_us();
        const char *piece  = tokenizer_decode(r->tok, r->prev_token, emitted);
        int stop_requested = 0;
        if (piece && r->callback) stop_requested = r->callback(piece, r->userdata);
        r->prev_token = emitted;
        r->tokens_generated++;
        if (stop_requested || r->tokens_generated >= r->max_tokens) {
            r->stop = 1;
            break;
        }
    }
}

/* ── Commit emit_token: one corrective forward call per model ────── */
static void commit_round(SpecRun *r, int n_accept, int emit_token) {
    int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
    int commit_pos = r->pos + n_accept;
    float *v_next  = transformer_forward(emit_token, commit_pos, r->cfg, r->w, r->s, r->mc, r->tp);
    float *d_next =
        transformer_forward(emit_token, commit_pos, &r->draft->config, &r->draft->weights,
                            r->draft->state, &r->draft->moe_config, r->tp);
    memcpy(r->last_verifier_logits, v_next, (size_t)r->vocab_size * sizeof(float));
    memcpy(r->last_draft_logits, d_next, (size_t)r->vocab_size * sizeof(float));
    r->pos = commit_pos + 1;
    if (t_step) {
        tn_step_timing_add(TN_STEP_27_SPEC_COMMIT, tn_step_timing_now_ns() - t_step);
    }
}

void speculative_generate_with_callback(const Config *cfg, const TransformerWeights *w, RunState *s,
                                        const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                                        const char *prompt, int max_tokens, float temperature,
                                        float top_p, bool json_mode, DraftModel *draft,
                                        int spec_length, long long seed, TokenCallback callback,
                                        void *userdata) {
    if (!prompt) prompt = "";
    tn_step_timing_reset();
    if (!spec_supported(mc, s, &json_mode)) return;
    if (mc && mc->is_moe) moe_expert_tracking_reset(cfg->n_layers, mc->num_experts);

    int *prompt_tokens = (int *)malloc((cfg->seq_len + 1) * sizeof(int));
    if (!prompt_tokens) return;
    int n_prompt = encode_prompt(cfg, tok, prompt, prompt_tokens);
    if (n_prompt < 0) {
        free(prompt_tokens);
        return;
    }

    SpecRun r;
    memset(&r, 0, sizeof(r));
    r.cfg         = cfg;
    r.w           = w;
    r.s           = s;
    r.mc          = mc;
    r.tok         = tok;
    r.tp          = tp;
    r.draft       = draft;
    r.vocab_size  = cfg->vocab_size;
    r.spec_length = spec_length;
    r.max_tokens  = max_tokens;
    r.temperature = temperature;
    r.top_p       = top_p;
    r.callback    = callback;
    r.userdata    = userdata;

    /* ── Speculative scratch ──────────────────────────────────────────── */
    if (spec_batch_scratch_alloc(&r.sb, cfg, spec_length) != TN_OK) {
        TN_LOG_ERR("spec", "[spec] Failed to allocate batch scratch.\n");
        free(prompt_tokens);
        return;
    }
    if (round_buffers_alloc(&r) != 0) {
        spec_batch_scratch_free(&r.sb);
        free(prompt_tokens);
        return;
    }

    rng_seed(&r.rng_state, seed >= 0 ? (unsigned long long)seed : (unsigned long long)time(NULL));
    r.chat_stops = tokenizer_chat_stops_new(tok, tok->chat_template);

    sw_init(&s->sw, s->max_seq_len, n_prompt);
    sw_init(&draft->state->sw, draft->state->max_seq_len, n_prompt);

    prefill(&r, prompt_tokens, n_prompt);
    r.pos        = n_prompt;
    r.prev_token = prompt_tokens[n_prompt - 1];

    while (!r.stop && r.tokens_generated < max_tokens && r.pos + spec_length < cfg->seq_len &&
           r.pos + spec_length < draft->state->max_seq_len) {
        draft_phase(&r);
        TernaryError verr = verify_phase(&r);
        if (verr != TN_OK) {
            TN_LOG_ERR("spec", "[spec] transformer_forward_batch failed (err=%d) -- stopping.\n",
                       (int)verr);
            break;
        }
        int n_accept, emit_token;
        accept_phase(&r, &n_accept, &emit_token);
        emit_round(&r, n_accept, emit_token);
        if (r.stop) break;
        commit_round(&r, n_accept, emit_token);
    }

    round_buffers_free(&r);
    spec_batch_scratch_free(&r.sb);
    free(prompt_tokens);
    free(r.chat_stops);

    if (r.tokens_generated > 1 && r.gen_start_us > 0) {
        int64_t end_us     = timer_now_us();
        double tok_per_sec = timer_tokens_per_sec(r.gen_start_us, end_us, r.tokens_generated);
        TN_LOG_INF("spec_gen", "\n[spec-gen] %.2f tok/s (%d tokens)\n", tok_per_sec,
                   r.tokens_generated);
    }
    if (mc && mc->is_moe) moe_expert_tracking_print(cfg->n_layers, mc->num_experts);
    tn_step_timing_report(stderr);
}

void speculative_generate(const Config *cfg, const TransformerWeights *w, RunState *s,
                          const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp, const char *prompt,
                          int max_tokens, float temperature, float top_p, bool json_mode,
                          DraftModel *draft, int spec_length, long long seed) {
    speculative_generate_with_callback(cfg, w, s, mc, tok, tp, prompt, max_tokens, temperature,
                                       top_p, json_mode, draft, spec_length, seed,
                                       stdout_token_callback, NULL);
    printf("\n");
}
