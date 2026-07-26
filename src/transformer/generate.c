#include "transformer/generate.h"
#include "transformer/forward.h"
#include "transformer/moe_ffn.h"
#include "math/simd_dispatch.h"
#include "sampling/sampling.h"
#include "sampling/rng.h"
#include "cli/timer.h"
#include "core/step_timing.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/chat_template.h"
#include "core/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── stdout callback used by the public generate() wrapper ─────────────── */
static int stdout_token_callback(const char *piece, void *userdata) {
    (void)userdata;
    printf("%s", piece);
    fflush(stdout);
    return 0;
}

/* ── Core implementation: all output goes through callback ──────────────── */
void generate_with_callback(const Config *cfg, const TransformerWeights *w,
                             RunState *s, const MoEConfig *mc,
                             Tokenizer *tok, ThreadPool *tp, const char *prompt,
                             int max_tokens, float temperature, float top_p,
                             TokenCallback callback, void *userdata) {
    if (!prompt) prompt = "";
    tn_step_timing_reset();

    /* Reset expert hit tracking for MoE models */
    if (mc && mc->is_moe)
        moe_expert_tracking_reset(cfg->n_layers, mc->num_experts);

    /* Allocate token buffer */
    int *prompt_tokens = (int *)malloc((cfg->seq_len + 1) * sizeof(int));
    if (!prompt_tokens) return;

    int n_prompt = 0;

    if (tok->chat_template) {
        /* ── GGUF path: apply the model's own Jinja2 chat template ─────────
         * The template itself renders BOS ({{ bos_token }}) so we must NOT
         * prepend it manually — that would produce a double-BOS. */
        const char *bos_str = (tok->bos_token_id >= 0 && tok->bos_token_id < tok->vocab_size)
                              ? tok->vocab[tok->bos_token_id] : "";
        const char *eos_str = (tok->eos_token_id >= 0 && tok->eos_token_id < tok->vocab_size)
                              ? tok->vocab[tok->eos_token_id] : "";

        const char *roles[1]    = { "user" };
        const char *contents[1] = { prompt };

        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        char *formatted = chat_template_apply(tok->chat_template,
                                              roles, contents, 1,
                                              bos_str, eos_str,
                                              /*add_generation_prompt=*/1);
        if (t_step) {
            tn_step_timing_add(TN_STEP_2_BOS_INJECTION,
                               tn_step_timing_now_ns() - t_step);
        }
        if (!formatted) {
            free(prompt_tokens);
            return;
        }

        if (g_tn_verbose)
            fprintf(stderr, "[DBG] formatted prompt: %s\n", formatted);

        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        n_prompt = tokenizer_encode(tok, formatted, strlen(formatted),
                                    prompt_tokens, cfg->seq_len);
        if (t_step) {
            tn_step_timing_add(TN_STEP_1_TOKENIZATION,
                               tn_step_timing_now_ns() - t_step);
        }
        free(formatted);

        if (n_prompt < 0) {
            free(prompt_tokens);
            return;
        }
    } else {
        /* ── Legacy path: raw prompt + manual BOS ───────────────────────── */
        int bos = (cfg->bos_token_id > 0)   ? cfg->bos_token_id
                : (tok->bos_token_id >= 0)  ? tok->bos_token_id
                : -1;
        if (bos >= 0) {
            int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
            prompt_tokens[n_prompt++] = bos;
            if (t_step) {
                tn_step_timing_add(TN_STEP_2_BOS_INJECTION,
                                   tn_step_timing_now_ns() - t_step);
            }
        }

        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        int n_encoded = tokenizer_encode(tok, prompt, strlen(prompt),
                                         &prompt_tokens[n_prompt], cfg->seq_len - n_prompt);
        if (t_step) {
            tn_step_timing_add(TN_STEP_1_TOKENIZATION,
                               tn_step_timing_now_ns() - t_step);
        }
        if (n_encoded < 0) {
            free(prompt_tokens);
            return;
        }
        n_prompt += n_encoded;
    }

    /* Debug: print prompt token IDs when verbose */
    if (g_tn_verbose) {
        fprintf(stderr, "[DBG] n_prompt=%d  tokens:", n_prompt);
        for (int i = 0; i < n_prompt && i < 32; i++) fprintf(stderr, " %d", prompt_tokens[i]);
        fprintf(stderr, "\n");
    }

    /* pos_offset: skip over any vision tokens already pre-filled in the KV cache */
    int pos_offset = s->current_pos;

    /* Initialize RNG */
    unsigned long long rng_state;
    rng_seed(&rng_state, (unsigned long long)time(NULL));

    /* Initialize Sliding Window (pins the entire prompt) */
    sw_init(&s->sw, s->max_seq_len, n_prompt + pos_offset);

    int prev_token = -1;
    int next = 0;
    int tokens_generated = 0;
    int64_t gen_start_us = 0; /* wall-clock start of generation phase */

    int total_steps = n_prompt + max_tokens;
    for (int step = 0; step < total_steps && (step + pos_offset) < cfg->seq_len; step++) {
        int abs_pos = step + pos_offset;

        /* Step 1: Forward pass */
        int token = (step < n_prompt) ? prompt_tokens[step] : next;

        float *logits = transformer_forward(token, abs_pos, cfg, w, s, mc, tp);
        s->current_pos = abs_pos + 1;

        /* Step 2: Sampling */
        if (step < n_prompt - 1) {
            next = prompt_tokens[step + 1];
        } else {
            /* Full prompt processed, sample from logits */
            int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
            if (temperature <= 0.0f || temperature < 1e-6f) {
                next = sample_argmax(logits, cfg->vocab_size);
            } else {
                apply_temperature(logits, cfg->vocab_size, temperature);
                if (top_p < 1.0f && top_p > 0.0f) {
                    next = sample_top_p(logits, cfg->vocab_size, top_p, &rng_state);
                } else {
                    tn_softmax(logits, cfg->vocab_size);
                    float r = rng_float(&rng_state);
                    float cdf = 0.0f;
                    next = cfg->vocab_size - 1; /* fallback */
                    for (int i = 0; i < cfg->vocab_size; i++) {
                        cdf += logits[i];
                        if (r < cdf) {
                            next = i;
                            break;
                        }
                    }
                }
            }
            if (t_step) {
                tn_step_timing_add(TN_STEP_20_GREEDY_SAMPLING,
                                   tn_step_timing_now_ns() - t_step);
            }
            if (g_tn_verbose) {
                /* Print top-5 logit values to diagnose garbage output */
                float top5v[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
                int   top5i[5] = {0,0,0,0,0};
                for (int i = 0; i < cfg->vocab_size; i++) {
                    if (logits[i] > top5v[4]) {
                        top5v[4] = logits[i]; top5i[4] = i;
                        for (int j = 3; j >= 0 && top5v[j] < top5v[j+1]; j--) {
                            float tv = top5v[j]; top5v[j] = top5v[j+1]; top5v[j+1] = tv;
                            int   ti = top5i[j]; top5i[j] = top5i[j+1]; top5i[j+1] = ti;
                        }
                    }
                }
                fprintf(stderr, "[DBG] step=%d sampled=%d  top5:[%d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f) %d(%.3f)]\n",
                        step, next,
                        top5i[0], top5v[0], top5i[1], top5v[1], top5i[2], top5v[2],
                        top5i[3], top5v[3], top5i[4], top5v[4]);
            }
        }

        /* Step 3+4: EOS check first — do NOT print the EOS token itself.
         * All EOS IDs come from cfg (GGUF loader) or tok->eos_list (vocab scan).
         * No hardcoded token IDs here. */
        if (step >= n_prompt - 1) {
            int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
            int is_eos = 0;
            if (cfg->eos_token_id > 0 && next == cfg->eos_token_id) is_eos = 1;
            for (int ei = 0; ei < tok->n_eos && !is_eos; ei++) {
                if (next == tok->eos_list[ei]) is_eos = 1;
            }
            if (t_step) {
                tn_step_timing_add(TN_STEP_21_EOS_CHECK,
                                   tn_step_timing_now_ns() - t_step);
            }
            if (is_eos) {
                break;
            }
            /* Non-EOS token: decode and deliver to callback */
            if (tokens_generated == 0) {
                gen_start_us = timer_now_us();
            }
            const char *piece = tokenizer_decode(tok, prev_token, next);
            int stop_requested = 0;
            if (piece && callback) {
                stop_requested = callback(piece, userdata);
            }
            tokens_generated++;
            if (stop_requested) break;
        }

        prev_token = next;
        if (step < n_prompt - 1) {
            next = prompt_tokens[step + 1];
        }
    }

    free(prompt_tokens);

    /* Report throughput to stderr (does not interfere with callback output) */
    if (tokens_generated > 1 && gen_start_us > 0) {
        int64_t end_us = timer_now_us();
        double tok_per_sec = timer_tokens_per_sec(gen_start_us, end_us, tokens_generated);
        fprintf(stderr, "\n[gen] %.2f tok/s (%d tokens)\n", tok_per_sec, tokens_generated);
    }

    /* Print expert utilisation for MoE models */
    if (mc && mc->is_moe)
        moe_expert_tracking_print(cfg->n_layers, mc->num_experts);
    tn_step_timing_report(stderr);
}

/* ── Public wrapper: generate() prints to stdout ────────────────────────── */
void generate(const Config *cfg, const TransformerWeights *w, RunState *s,
              const MoEConfig *mc,
              Tokenizer *tok, ThreadPool *tp, const char *prompt,
              int max_tokens, float temperature, float top_p) {
    generate_with_callback(cfg, w, s, mc, tok, tp, prompt,
                           max_tokens, temperature, top_p,
                           stdout_token_callback, NULL);
    /* generate_with_callback already reports timing to stderr */
    printf("\n");
}
