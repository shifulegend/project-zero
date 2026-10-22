#include "transformer/generate.h"
#include "transformer/forward.h"
#include "transformer/moe_ffn.h"
#include "math/simd_dispatch.h"
#include "sampling/sampling.h"
#include "sampling/rng.h"
#include "sampling/constrained_sample.h"
#include "sampling/sampler_chain.h"
#include "sampling/logprob.h"
#include "transformer/stop_seq.h"
#include "distributed/dist_plan.h"
#include "cli/timer.h"
#include "core/step_timing.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/chat_template.h"
#include "core/debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── stdout callback used by the public generate() wrapper ─────────────── */
static int stdout_token_callback(const char *piece, void *userdata) {
    (void)userdata;
    printf("%s", piece);
    fflush(stdout);
    return 0;
}

/* Max messages rendered through the chat template in one call (system
 * prompt + history). Bounded so the role/content pointer arrays live on the
 * stack; 64 matches the API's CHAT_MAX_MESSAGES. */
#define GEN_MAX_TEMPLATE_MESSAGES 65

/* ── Core implementation: all output goes through callback ──────────────── */
/* 2026-10-21: a prompt the tokenizer rejects used to end generation with no
 * output and no message (seen with Falcon3, whose template writes <|user|> as
 * plain text). The API already answers 400; tell CLI users too. */
static void report_encode_failure(const char *what) {
    fprintf(stderr, "[error] the %s could not be tokenized for this model: it likely contains "
                    "a \"<|...|>\" control sequence the tokenizer does not accept. Nothing was "
                    "generated.\n", what);
}

void generate_ex(const Config *cfg, const TransformerWeights *w,
                 RunState *s, const MoEConfig *mc,
                 Tokenizer *tok, ThreadPool *tp, const char *prompt,
                 const GenerationParams *gp,
                 TokenCallback callback, void *userdata) {
    if (!prompt) prompt = "";
    GenerationParams gp_default;
    if (!gp) { gen_params_default(&gp_default); gp = &gp_default; }
    const int  max_tokens = gp->max_tokens;
    const bool json_mode  = gp->json_mode;
    const SamplerSettings *ss = &gp->sampler;
    const WatermarkConfig *wm = gp->watermark.enabled ? &gp->watermark : NULL;
    tn_step_timing_reset();
    if (gp->stats) {
        gp->stats->prompt_tokens = 0;
        gp->stats->completion_tokens = 0;
        gp->stats->cached_tokens = 0;
        gp->stats->sum_logprob = 0.0;
        gp->stats->finish_reason = GEN_FINISH_ERROR; /* until generation runs */
        gp->stats->tokens_per_sec = 0.0;
        gp->stats->hit_eos = false;
    }

    StopMatcher stop;
    stop_matcher_init(&stop, gp->stop, gp->n_stop);

    /* Phase 20: grammar-constrained decoding state (only touched when
     * json_mode is set). Fresh per call -- constrains this one generation,
     * starting from "expecting any JSON value". */
    FSMState json_fsm;
    if (json_mode) fsm_init(&json_fsm);

    /* Reset expert hit tracking for MoE models */
    if (mc && mc->is_moe)
        moe_expert_tracking_reset(cfg->n_layers, mc->num_experts);

    /* Allocate token buffer */
    int *prompt_tokens = (int *)malloc((cfg->seq_len + 1) * sizeof(int));
    if (!prompt_tokens) return;

    int n_prompt = 0;

    /* A request's own template (API "chat_template") wins over the model's. */
    const char *chat_tmpl = (gp->chat_template && gp->chat_template[0]) ? gp->chat_template
                                                                       : tok->chat_template;
    /* Chat mode ends a turn at any CONTROL token the template never writes
     * (Falcon3's ">>UNUSED_1<<"); raw prompts keep the declared EOG set. */
    tokenizer_set_chat_stops(tok, (chat_tmpl && !gp->raw_prompt) ? chat_tmpl : NULL);
    if (chat_tmpl && !gp->raw_prompt) {
        /* ── GGUF path: apply the model's own Jinja2 chat template ─────────
         * The template itself renders BOS ({{ bos_token }}) so we must NOT
         * prepend it manually — that would produce a double-BOS. */
        const char *bos_str = (tok->bos_token_id >= 0 && tok->bos_token_id < tok->vocab_size)
                              ? tok->vocab[tok->bos_token_id] : "";
        const char *eos_str = (tok->eos_token_id >= 0 && tok->eos_token_id < tok->vocab_size)
                              ? tok->vocab[tok->eos_token_id] : "";

        /* Conversation to render: the caller's history when supplied,
         * else the single prompt as one user turn; an optional system
         * prompt goes first unless the history already opens with one. */
        const char *roles[GEN_MAX_TEMPLATE_MESSAGES];
        const char *contents[GEN_MAX_TEMPLATE_MESSAGES];
        int n_msg = 0;
        int history_has_system = gp->n_messages > 0 && gp->roles && gp->roles[0] &&
                                 strcmp(gp->roles[0], "system") == 0;
        if (gp->system_prompt && gp->system_prompt[0] && !history_has_system) {
            roles[n_msg] = "system";
            contents[n_msg++] = gp->system_prompt;
        }
        if (gp->n_messages > 0 && gp->roles && gp->contents) {
            for (int i = 0; i < gp->n_messages && n_msg < GEN_MAX_TEMPLATE_MESSAGES; i++) {
                roles[n_msg] = gp->roles[i] ? gp->roles[i] : "user";
                contents[n_msg++] = gp->contents[i] ? gp->contents[i] : "";
            }
        } else {
            roles[n_msg] = "user";
            contents[n_msg++] = prompt;
        }

        /* Message text is content, never chat markup: a "<|im_end|>" typed by
         * a user (or pasted from a document) must not close their turn
         * (2026-10-22, prompt-injection defence; see tokenizer.h). */
        char *safe[GEN_MAX_TEMPLATE_MESSAGES];
        int n_safe = 0;
        for (; n_safe < n_msg; n_safe++) {
            safe[n_safe] = tokenizer_neutralize_control(tok, contents[n_safe]);
            if (!safe[n_safe]) break;
            contents[n_safe] = safe[n_safe];
        }
        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        char *formatted = n_safe < n_msg ? NULL
                        : chat_template_apply(chat_tmpl,
                                              roles, contents, n_msg,
                                              bos_str, eos_str,
                                              /*add_generation_prompt=*/1);
        for (int i = 0; i < n_safe; i++) free(safe[i]);
        if (t_step) {
            tn_step_timing_add(TN_STEP_2_BOS_INJECTION,
                               tn_step_timing_now_ns() - t_step);
        }
        if (!formatted) {
            free(prompt_tokens);
            stop_matcher_free(&stop);
            return;
        }

        if (g_tn_verbose)
            fprintf(stderr, "[DBG] formatted prompt: %s\n", formatted);

        if (gp->assistant_prefix && gp->assistant_prefix[0]) {
            size_t fl = strlen(formatted), al = strlen(gp->assistant_prefix);
            char *joined = (char *)realloc(formatted, fl + al + 1);
            if (!joined) {
                free(formatted);
                free(prompt_tokens);
                stop_matcher_free(&stop);
                return;
            }
            memcpy(joined + fl, gp->assistant_prefix, al + 1);
            formatted = joined;
        }

        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        n_prompt = tokenizer_encode(tok, formatted, strlen(formatted),
                                    prompt_tokens, cfg->seq_len);
        if (t_step) {
            tn_step_timing_add(TN_STEP_1_TOKENIZATION,
                               tn_step_timing_now_ns() - t_step);
        }
        free(formatted);

        if (n_prompt < 0) {
            report_encode_failure("chat-formatted prompt");
            free(prompt_tokens);
            stop_matcher_free(&stop);
            return;
        }
    } else {
        /* ── Legacy path / --raw-prompt: raw prompt + manual BOS ─────────── */
        /* 2026-10-19: honour tokenizer.ggml.add_bos_token — Qwen/SmolLM GGUFs
         * say false, and llama.cpp feeds them no BOS. */
        int bos = !tok->add_bos_token          ? -1
                : (cfg->bos_token_id > 0)      ? cfg->bos_token_id
                : (tok->bos_token_id >= 0)     ? tok->bos_token_id
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
            report_encode_failure("prompt");
            free(prompt_tokens);
            stop_matcher_free(&stop);
            return;
        }
        n_prompt += n_encoded;
        if (gp->assistant_prefix && gp->assistant_prefix[0]) {
            int n_more = tokenizer_encode(tok, gp->assistant_prefix, strlen(gp->assistant_prefix),
                                          &prompt_tokens[n_prompt], cfg->seq_len - n_prompt);
            if (n_more < 0) {
                report_encode_failure("assistant prefix");
                free(prompt_tokens);
                stop_matcher_free(&stop);
                return;
            }
            n_prompt += n_more;
        }
    }

    /* Debug: print prompt token IDs when verbose */
    if (g_tn_verbose) {
        fprintf(stderr, "[DBG] n_prompt=%d  tokens:", n_prompt);
        for (int i = 0; i < n_prompt && i < 32; i++) fprintf(stderr, " %d", prompt_tokens[i]);
        fprintf(stderr, "\n");
    }

    /* pos_offset: skip over any vision tokens already pre-filled in the KV cache */
    int pos_offset = s->current_pos;

    /* Initialize RNG: explicit seed (--seed / API "seed") for reproducible
     * sampling, else time-based. (2026-10-18: the seed used to be parsed by
     * the CLI and then ignored — every run was time-seeded.) */
    unsigned long long rng_state;
    rng_seed(&rng_state, gp->seed >= 0 ? (unsigned long long)gp->seed
                                       : (unsigned long long)time(NULL));

    /* Generated-token history for repetition/presence/frequency penalties. */
    int *gen_history = NULL;
    int  n_history = 0;
    /* Phase 30: raw-logit copy for log-probabilities of sampled tokens. */
    float *raw_logits = (gp->compute_logprobs && gp->stats)
                        ? (float *)malloc((size_t)cfg->vocab_size * sizeof(float)) : NULL;
    double sum_logprob = 0.0;
    double step_lse = 0.0;
    if (sampler_penalties_active(ss) && max_tokens > 0)
        gen_history = (int *)malloc((size_t)max_tokens * sizeof(int));

    /* Initialize Sliding Window (pins the entire prompt) */
    sw_init(&s->sw, s->max_seq_len, n_prompt + pos_offset);

    /* Commit the KV rows this call can write, up front (no-op once resident). */
    run_state_kv_reserve(s, cfg, pos_offset + n_prompt + (max_tokens > 0 ? max_tokens : 0));

    /* Phase 21.5: skip prompt positions whose K/V are already cached from a
     * previous call with the same leading tokens. The last prompt token is
     * always recomputed — its logits seed the first sample. Not valid after
     * a vision prefill (pos_offset > 0) or for recurrent linear-attention
     * layers, whose state cannot be rewound to a partial prefix. */
    PrefixCache *pcache = gp->prefix_cache;
    /* Snapshots copy F32 KV rows; with --kv-type i8 only the live prefix cache applies.
     * 2026-10-18: also off under --remote-layers — a snapshot restores only the local
     * layers' KV, while each worker still holds the KV of the last prompt it saw. The
     * live prefix cache stays valid there: workers mirror the coordinator's last sequence. */
    PromptStore *pstore = (pcache && prompt_store_supported(mc) && s->key_cache &&
                           !dist_plan_active())
                          ? gp->prompt_store : NULL;
    int start_step = 0;
    int cache_eligible = pcache && pos_offset == 0 && !moe_is_recurrent(mc);
    if (pcache) {
        if (cache_eligible && n_prompt > 1) {
            int lcp = prefix_cache_match(pcache, prompt_tokens, n_prompt);
            /* Phase 26: a stored snapshot may share more than the live KV. */
            if (pstore)
                lcp = prompt_store_restore(pstore, cfg, s, pcache, prompt_tokens, n_prompt, lcp);
            start_step = lcp < n_prompt - 1 ? lcp : n_prompt - 1;
            prefix_cache_truncate(pcache, start_step);
        } else {
            prefix_cache_reset(pcache);
        }
    }
    if (gp->stats) gp->stats->cached_tokens = start_step;

    /* prev_token at the start of step k is the token being forwarded at k
     * (for k >= 1) — see the end-of-loop update below. */
    int prev_token = start_step > 0 ? prompt_tokens[start_step] : -1;
    int next = (start_step < n_prompt - 1) ? prompt_tokens[start_step + 1] : 0;
    int tokens_generated = 0;
    GenFinishReason finish = GEN_FINISH_LENGTH;
    int64_t gen_start_us = 0; /* wall-clock start of generation phase */

    /* max_tokens generation-eligible steps: the first generated token comes
     * "free" from the last prompt-processing step (step == n_prompt - 1
     * below), so the loop only needs max_tokens - 1 MORE steps after that to
     * reach exactly max_tokens emitted tokens -- i.e. total_steps = n_prompt
     * + max_tokens - 1, not n_prompt + max_tokens (2026-09-24: the off-by-one
     * here silently generated max_tokens + 1 tokens for every caller;
     * invisible in practice because EOS almost always fires first on real
     * prompts, only surfaced by a synthetic test with EOS deliberately
     * disabled — see docs/ai/mistakes.md). */
    int total_steps = n_prompt + max_tokens - 1;
    for (int step = start_step; step < total_steps && (step + pos_offset) < cfg->seq_len; step++) {
        int abs_pos = step + pos_offset;

        /* Step 1: Forward pass */
        int token = (step < n_prompt) ? prompt_tokens[step] : next;

        float *logits = transformer_forward(token, abs_pos, cfg, w, s, mc, tp);
        if (dist_plan_active() && dist_plan_active()->failed) {
            finish = GEN_FINISH_ERROR;   /* a remote layer range is unreachable */
            break;
        }
        if (pcache && pos_offset == 0) prefix_cache_set(pcache, abs_pos, token);

        /* Step 2: Sampling */
        if (step < n_prompt - 1) {
            next = prompt_tokens[step + 1];
        } else {
            /* Full prompt processed, sample from logits */
            int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
            if (raw_logits) {
                memcpy(raw_logits, logits, (size_t)cfg->vocab_size * sizeof(float));
                step_lse = logprob_logsumexp(raw_logits, cfg->vocab_size);
            }
            /* ignore_eos: never sample an end-of-sequence token (llama.cpp's
             * --ignore-eos), so runs last exactly max_tokens. */
            if (gp->ignore_eos) {
                if (cfg->eos_token_id >= 0 && cfg->eos_token_id < cfg->vocab_size)
                    logits[cfg->eos_token_id] = -INFINITY;
                for (int ei = 0; ei < tok->n_eos; ei++)
                    if (tok->eos_list[ei] >= 0 && tok->eos_list[ei] < cfg->vocab_size)
                        logits[tok->eos_list[ei]] = -INFINITY;
                if (tok->chat_stop)
                    for (int vi = 0; vi < cfg->vocab_size && vi < tok->vocab_size; vi++)
                        if (tok->chat_stop[vi]) logits[vi] = -INFINITY;
            }
            if (json_mode) {
                next = sample_constrained_chain(logits, cfg->vocab_size, &json_fsm,
                                                tok, prev_token, ss,
                                                gen_history, n_history, wm,
                                                &rng_state);
            } else {
                next = sampler_chain_sample(logits, cfg->vocab_size, ss,
                                            gen_history, n_history, wm,
                                            prev_token, &rng_state);
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
            /* tokenizer_is_eog adds the chat-mode turn ends set below. */
            int is_eos = (cfg->eos_token_id > 0 && next == cfg->eos_token_id) ||
                         tokenizer_is_eog(tok, next);
            if (t_step) {
                tn_step_timing_add(TN_STEP_21_EOS_CHECK,
                                   tn_step_timing_now_ns() - t_step);
            }
            if (is_eos) {
                finish = GEN_FINISH_STOP;
                if (gp->stats) gp->stats->hit_eos = true;
                break;
            }
            /* Non-EOS token: decode and deliver to callback */
            if (tokens_generated == 0) {
                gen_start_us = timer_now_us();
            }
            const char *piece = tokenizer_decode(tok, prev_token, next);
            int stop_requested = 0;
            if (piece) {
                stop_requested = stop_matcher_feed(&stop, piece, callback, userdata);
            }
            if (gen_history && n_history < max_tokens) gen_history[n_history++] = next;
            if (raw_logits) sum_logprob += (double)raw_logits[next] - step_lse;
            tokens_generated++;
            if (stop_requested) {
                finish = stop.stopped ? GEN_FINISH_STOP : GEN_FINISH_CANCELLED;
                break;
            }
        }

        prev_token = next;
        if (step < n_prompt - 1) {
            next = prompt_tokens[step + 1];
        }
    }

    stop_matcher_flush(&stop, callback, userdata);
    stop_matcher_free(&stop);
    if (pstore && cache_eligible) prompt_store_save(pstore, cfg, s, pcache);
    free(gen_history);
    free(raw_logits);
    free(prompt_tokens);

    /* Report throughput to stderr (does not interfere with callback output) */
    double tok_per_sec = 0.0;
    if (tokens_generated > 1 && gen_start_us > 0) {
        int64_t end_us = timer_now_us();
        tok_per_sec = timer_tokens_per_sec(gen_start_us, end_us, tokens_generated);
        fprintf(stderr, "\n[gen] %.2f tok/s (%d tokens)\n", tok_per_sec, tokens_generated);
    }
    if (gp->stats) {
        gp->stats->prompt_tokens = n_prompt;
        gp->stats->completion_tokens = tokens_generated;
        gp->stats->finish_reason = finish;
        gp->stats->tokens_per_sec = tok_per_sec;
        gp->stats->sum_logprob = sum_logprob;
    }

    /* Print expert utilisation for MoE models */
    if (mc && mc->is_moe)
        moe_expert_tracking_print(cfg->n_layers, mc->num_experts);
    tn_step_timing_report(stderr);
}

/* ── Legacy positional wrappers ─────────────────────────────────────────── */
static void legacy_params(GenerationParams *gp, int max_tokens, float temperature,
                          float top_p, bool json_mode) {
    gen_params_default(gp);
    gp->max_tokens = max_tokens;
    gp->sampler.temperature = temperature;
    gp->sampler.top_p = top_p;
    gp->json_mode = json_mode;
}

void generate_with_callback(const Config *cfg, const TransformerWeights *w,
                             RunState *s, const MoEConfig *mc,
                             Tokenizer *tok, ThreadPool *tp, const char *prompt,
                             int max_tokens, float temperature, float top_p,
                             bool json_mode,
                             TokenCallback callback, void *userdata) {
    GenerationParams gp;
    legacy_params(&gp, max_tokens, temperature, top_p, json_mode);
    generate_ex(cfg, w, s, mc, tok, tp, prompt, &gp, callback, userdata);
}

void generate_params(const Config *cfg, const TransformerWeights *w, RunState *s,
                     const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                     const char *prompt, const GenerationParams *gp) {
    generate_ex(cfg, w, s, mc, tok, tp, prompt, gp, stdout_token_callback, NULL);
    printf("\n");
}

/* ── Public wrapper: generate() prints to stdout ────────────────────────── */
void generate(const Config *cfg, const TransformerWeights *w, RunState *s,
              const MoEConfig *mc,
              Tokenizer *tok, ThreadPool *tp, const char *prompt,
              int max_tokens, float temperature, float top_p, bool json_mode) {
    GenerationParams gp;
    legacy_params(&gp, max_tokens, temperature, top_p, json_mode);
    generate_params(cfg, w, s, mc, tok, tp, prompt, &gp);
}
