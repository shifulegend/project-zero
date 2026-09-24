#include "speculative/spec_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cli/timer.h"
#include "core/step_timing.h"
#include "math/simd_dispatch.h"
#include "sampling/rng.h"
#include "sampling/sampling.h"
#include "speculative/accept_reject.h"
#include "speculative/spec_scratch.h"
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
    float r = rng_float(rng_state);
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return vocab_size - 1;
}

static int is_eos_token(int token, const Config *cfg, const Tokenizer *tok) {
    if (cfg->eos_token_id > 0 && token == cfg->eos_token_id) return 1;
    for (int i = 0; i < tok->n_eos; i++) {
        if (token == tok->eos_list[i]) return 1;
    }
    return 0;
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
void speculative_generate_with_callback(const Config *cfg, const TransformerWeights *w,
                                         RunState *s, const MoEConfig *mc,
                                         Tokenizer *tok, ThreadPool *tp, const char *prompt,
                                         int max_tokens, float temperature, float top_p,
                                         bool json_mode, DraftModel *draft, int spec_length,
                                         TokenCallback callback, void *userdata) {
    if (!prompt) prompt = "";
    tn_step_timing_reset();

    if (mc && mc->has_linear_attn) {
        fprintf(stderr, "[spec] Speculative decoding is not supported for linear-attention "
                "models (Qwen3.5/3.6 hybrid) -- refusing.\n");
        return;
    }
    if (s->current_pos != 0) {
        fprintf(stderr, "[spec] Speculative decoding does not support resuming from a "
                "nonzero position (e.g. vision prefill) -- the draft model has no matching "
                "prefix in its own KV cache. Use plain generation for this request.\n");
        return;
    }
    if (json_mode) {
        fprintf(stderr, "[spec] Warning: --json is not wired into speculative decoding's "
                "accept/reject loop yet -- proceeding with plain sampling.\n");
        json_mode = false;
    }

    if (mc && mc->is_moe)
        moe_expert_tracking_reset(cfg->n_layers, mc->num_experts);

    int vocab_size = cfg->vocab_size;

    /* ── Prompt encoding (verbatim from generate_with_callback()) ────────── */
    int *prompt_tokens = (int *)malloc((cfg->seq_len + 1) * sizeof(int));
    if (!prompt_tokens) return;
    int n_prompt = 0;

    if (tok->chat_template) {
        const char *bos_str = (tok->bos_token_id >= 0 && tok->bos_token_id < tok->vocab_size)
                              ? tok->vocab[tok->bos_token_id] : "";
        const char *eos_str = (tok->eos_token_id >= 0 && tok->eos_token_id < tok->vocab_size)
                              ? tok->vocab[tok->eos_token_id] : "";
        const char *roles[1]    = { "user" };
        const char *contents[1] = { prompt };
        char *formatted = chat_template_apply(tok->chat_template, roles, contents, 1,
                                              bos_str, eos_str, /*add_generation_prompt=*/1);
        if (!formatted) { free(prompt_tokens); return; }
        n_prompt = tokenizer_encode(tok, formatted, strlen(formatted), prompt_tokens, cfg->seq_len);
        free(formatted);
        if (n_prompt < 0) { free(prompt_tokens); return; }
    } else {
        int bos = (cfg->bos_token_id > 0)   ? cfg->bos_token_id
                : (tok->bos_token_id >= 0)  ? tok->bos_token_id
                : -1;
        if (bos >= 0) prompt_tokens[n_prompt++] = bos;
        int n_encoded = tokenizer_encode(tok, prompt, strlen(prompt),
                                         &prompt_tokens[n_prompt], cfg->seq_len - n_prompt);
        if (n_encoded < 0) { free(prompt_tokens); return; }
        n_prompt += n_encoded;
    }
    if (n_prompt == 0) { free(prompt_tokens); return; }

    /* ── Speculative scratch ──────────────────────────────────────────── */
    SpecBatchScratch sb;
    if (spec_batch_scratch_alloc(&sb, cfg, spec_length) != TN_OK) {
        fprintf(stderr, "[spec] Failed to allocate batch scratch.\n");
        free(prompt_tokens);
        return;
    }
    int *draft_tokens = (int *)malloc((size_t)spec_length * sizeof(int));
    float *draft_logits_buf = (float *)malloc((size_t)spec_length * vocab_size * sizeof(float));
    float *verify_check_logits = (float *)malloc((size_t)spec_length * vocab_size * sizeof(float));
    float *verifier_logits_out = (float *)malloc((size_t)spec_length * vocab_size * sizeof(float));
    float *last_verifier_logits = (float *)malloc((size_t)vocab_size * sizeof(float));
    float *last_draft_logits = (float *)malloc((size_t)vocab_size * sizeof(float));
    if (!draft_tokens || !draft_logits_buf || !verify_check_logits || !verifier_logits_out ||
        !last_verifier_logits || !last_draft_logits) {
        fprintf(stderr, "[spec] Failed to allocate round buffers.\n");
        free(draft_tokens); free(draft_logits_buf); free(verify_check_logits);
        free(verifier_logits_out); free(last_verifier_logits); free(last_draft_logits);
        spec_batch_scratch_free(&sb);
        free(prompt_tokens);
        return;
    }

    unsigned long long rng_state;
    rng_seed(&rng_state, (unsigned long long)time(NULL));

    sw_init(&s->sw, s->max_seq_len, n_prompt);
    sw_init(&draft->state->sw, draft->state->max_seq_len, n_prompt);

    /* ── Prefill: both models process the full prompt sequentially ───────── */
    float *v_logits = NULL, *d_logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        v_logits = transformer_forward(prompt_tokens[i], i, cfg, w, s, mc, tp);
        d_logits = transformer_forward(prompt_tokens[i], i, &draft->config, &draft->weights,
                                        draft->state, &draft->moe_config, tp);
    }
    memcpy(last_verifier_logits, v_logits, (size_t)vocab_size * sizeof(float));
    memcpy(last_draft_logits, d_logits, (size_t)vocab_size * sizeof(float));

    int pos = n_prompt;
    int prev_token = prompt_tokens[n_prompt - 1];
    int tokens_generated = 0;
    int64_t gen_start_us = 0;
    int stop = 0;

    while (!stop && tokens_generated < max_tokens && pos + spec_length < cfg->seq_len
           && pos + spec_length < draft->state->max_seq_len) {

        /* ── Draft phase: spec_length sequential draft-model forward calls ── */
        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        memcpy(draft_logits_buf, last_draft_logits, (size_t)vocab_size * sizeof(float));
        int dt = sample_next(draft_logits_buf, vocab_size, temperature, top_p, &rng_state);
        draft_tokens[0] = dt;
        for (int i = 1; i < spec_length; i++) {
            float *lg = transformer_forward(draft_tokens[i - 1], pos + i - 1, &draft->config,
                                             &draft->weights, draft->state, &draft->moe_config, tp);
            memcpy(draft_logits_buf + (size_t)i * vocab_size, lg, (size_t)vocab_size * sizeof(float));
            draft_tokens[i] = sample_next(draft_logits_buf + (size_t)i * vocab_size, vocab_size,
                                           temperature, top_p, &rng_state);
        }
        if (t_step) {
            tn_step_timing_add(TN_STEP_24_SPEC_DRAFT_PHASE, tn_step_timing_now_ns() - t_step);
        }

        /* ── Verify phase: one batched forward pass on the verifier ─────── */
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        TernaryError verr = transformer_forward_batch(draft_tokens, pos, spec_length, cfg, w, s,
                                                        &sb, mc, tp, verifier_logits_out);
        if (t_step) {
            tn_step_timing_add(TN_STEP_25_SPEC_VERIFY_BATCH, tn_step_timing_now_ns() - t_step);
        }
        if (verr != TN_OK) {
            fprintf(stderr, "[spec] transformer_forward_batch failed (err=%d) -- stopping.\n",
                    (int)verr);
            break;
        }

        /* ── Accept/reject: build the shifted "what would the verifier have
         * predicted before seeing tokens[i]" array -- see this function's
         * header comment -- then decide n_accept/emit_token. ─────────────── */
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        memcpy(verify_check_logits, last_verifier_logits, (size_t)vocab_size * sizeof(float));
        if (spec_length > 1) {
            memcpy(verify_check_logits + vocab_size, verifier_logits_out,
                   (size_t)(spec_length - 1) * vocab_size * sizeof(float));
        }
        const float *bonus_logits = verifier_logits_out + (size_t)(spec_length - 1) * vocab_size;

        int n_accept = 0, emit_token = 0;
        accept_reject_round(draft_tokens, draft_logits_buf, verify_check_logits, bonus_logits,
                             spec_length, vocab_size, temperature, &rng_state,
                             &n_accept, &emit_token);
        if (t_step) {
            tn_step_timing_add(TN_STEP_26_SPEC_ACCEPT_REJECT, tn_step_timing_now_ns() - t_step);
        }

        /* ── Emit accepted tokens + the resampled/bonus token ────────────── */
        for (int i = 0; i <= n_accept; i++) {
            int emitted = (i < n_accept) ? draft_tokens[i] : emit_token;
            if (is_eos_token(emitted, cfg, tok)) { stop = 1; break; }
            if (tokens_generated == 0) gen_start_us = timer_now_us();
            const char *piece = tokenizer_decode(tok, prev_token, emitted);
            int stop_requested = 0;
            if (piece && callback) stop_requested = callback(piece, userdata);
            prev_token = emitted;
            tokens_generated++;
            if (stop_requested || tokens_generated >= max_tokens) { stop = 1; break; }
        }
        if (stop) break;

        /* ── Commit emit_token: one corrective forward call per model ────── */
        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        int commit_pos = pos + n_accept;
        float *v_next = transformer_forward(emit_token, commit_pos, cfg, w, s, mc, tp);
        float *d_next = transformer_forward(emit_token, commit_pos, &draft->config, &draft->weights,
                                             draft->state, &draft->moe_config, tp);
        memcpy(last_verifier_logits, v_next, (size_t)vocab_size * sizeof(float));
        memcpy(last_draft_logits, d_next, (size_t)vocab_size * sizeof(float));
        pos = commit_pos + 1;
        if (t_step) {
            tn_step_timing_add(TN_STEP_27_SPEC_COMMIT, tn_step_timing_now_ns() - t_step);
        }
    }

    free(draft_tokens);
    free(draft_logits_buf);
    free(verify_check_logits);
    free(verifier_logits_out);
    free(last_verifier_logits);
    free(last_draft_logits);
    spec_batch_scratch_free(&sb);
    free(prompt_tokens);

    if (tokens_generated > 1 && gen_start_us > 0) {
        int64_t end_us = timer_now_us();
        double tok_per_sec = timer_tokens_per_sec(gen_start_us, end_us, tokens_generated);
        fprintf(stderr, "\n[spec-gen] %.2f tok/s (%d tokens)\n", tok_per_sec, tokens_generated);
    }
    if (mc && mc->is_moe)
        moe_expert_tracking_print(cfg->n_layers, mc->num_experts);
    tn_step_timing_report(stderr);
}

void speculative_generate(const Config *cfg, const TransformerWeights *w, RunState *s,
                           const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                           const char *prompt, int max_tokens, float temperature,
                           float top_p, bool json_mode,
                           DraftModel *draft, int spec_length) {
    speculative_generate_with_callback(cfg, w, s, mc, tok, tp, prompt, max_tokens, temperature,
                                        top_p, json_mode, draft, spec_length,
                                        stdout_token_callback, NULL);
    printf("\n");
}
