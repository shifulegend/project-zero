/*
 * transformer/gen_session.c — see transformer/gen_session.h.
 *
 * 2026-10-22: the body of generate_ex() split into begin / input / consume /
 * end so the API server can interleave several generations (Phase 23 stage
 * 2b) on the same code path the single-sequence loop uses. Logic is
 * unchanged from generate_ex; comments carried over.
 */
#include "transformer/gen_session.h"
#include "transformer/moe_ffn.h"
#include "sampling/sampling.h"
#include "sampling/rng.h"
#include "sampling/constrained_sample.h"
#include "sampling/sampler_chain.h"
#include "sampling/logprob.h"
#include "distributed/dist_plan.h"
#include "cli/timer.h"
#include "core/step_timing.h"
#include "tokenizer/chat_template.h"
#include "memory/safe_alloc.h"
#include "core/debug.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Max messages rendered through the chat template in one call (system
 * prompt + history). Bounded so the role/content pointer arrays live on the
 * stack; 64 matches the API's CHAT_MAX_MESSAGES. */
#define GEN_MAX_TEMPLATE_MESSAGES 65

/* 2026-10-22: since untrusted text is neutralised instead of rejected, the
 * tokenizer only fails when the prompt does not fit in the context. */
static void report_encode_failure(const char *what, int seq_len) {
    fprintf(stderr, "[error] the %s does not fit in the model's context of %d tokens. "
                    "Nothing was generated; shorten it or raise --ctx-size.\n", what, seq_len);
}

/* Renders and tokenizes the prompt into g->prompt_tokens / g->n_prompt.
 * Returns 0, or -1 when there is nothing to generate. */
static int build_prompt(GenSession *g, const char *prompt) {
    const Config *cfg = g->cfg;
    const GenerationParams *gp = g->gp;
    Tokenizer *tok = g->tok;
    int n_prompt = 0;

    /* A request's own template (API "chat_template") wins over the model's. */
    const char *chat_tmpl = (gp->chat_template && gp->chat_template[0]) ? gp->chat_template
                                                                       : tok->chat_template;
    /* Chat mode ends a turn at any CONTROL token the template never writes
     * (Falcon3's ">>UNUSED_1<<"); raw prompts keep the declared EOG set. The
     * set belongs to this session, so concurrent sessions never share it. */
    g->chat_stops = tokenizer_chat_stops_new(tok, (chat_tmpl && !gp->raw_prompt) ? chat_tmpl : NULL);
    g->json_fsm.chat_stops = g->chat_stops;
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
        if (!formatted) return -1;

        if (g_tn_verbose)
            fprintf(stderr, "[DBG] formatted prompt: %s\n", formatted);

        if (gp->assistant_prefix && gp->assistant_prefix[0]) {
            size_t fl = strlen(formatted), al = strlen(gp->assistant_prefix);
            char *joined = (char *)realloc(formatted, fl + al + 1);
            if (!joined) { free(formatted); return -1; }
            memcpy(joined + fl, gp->assistant_prefix, al + 1);
            formatted = joined;
        }

        t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        n_prompt = tokenizer_encode(tok, formatted, strlen(formatted),
                                    g->prompt_tokens, cfg->seq_len);
        if (t_step) {
            tn_step_timing_add(TN_STEP_1_TOKENIZATION,
                               tn_step_timing_now_ns() - t_step);
        }
        free(formatted);

        if (n_prompt < 0) {
            report_encode_failure("chat-formatted prompt", cfg->seq_len);
            return -1;
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
            g->prompt_tokens[n_prompt++] = bos;
            if (t_step) {
                tn_step_timing_add(TN_STEP_2_BOS_INJECTION,
                                   tn_step_timing_now_ns() - t_step);
            }
        }

        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        int n_encoded = tokenizer_encode(tok, prompt, strlen(prompt),
                                         &g->prompt_tokens[n_prompt], cfg->seq_len - n_prompt);
        if (t_step) {
            tn_step_timing_add(TN_STEP_1_TOKENIZATION,
                               tn_step_timing_now_ns() - t_step);
        }
        if (n_encoded < 0) {
            report_encode_failure("prompt", cfg->seq_len);
            return -1;
        }
        n_prompt += n_encoded;
        if (gp->assistant_prefix && gp->assistant_prefix[0]) {
            int n_more = tokenizer_encode(tok, gp->assistant_prefix, strlen(gp->assistant_prefix),
                                          &g->prompt_tokens[n_prompt], cfg->seq_len - n_prompt);
            if (n_more < 0) {
                report_encode_failure("assistant prefix", cfg->seq_len);
                return -1;
            }
            n_prompt += n_more;
        }
    }
    g->n_prompt = n_prompt;
    return 0;
}

int gen_session_begin(GenSession *g, const Config *cfg, const TransformerWeights *w,
                      RunState *s, const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                      const char *prompt, const GenerationParams *gp,
                      TokenCallback callback, void *userdata) {
    memset(g, 0, sizeof(*g));
    if (!prompt) prompt = "";
    if (!gp) { gen_params_default(&g->gp_default); gp = &g->gp_default; }
    g->cfg = cfg; g->w = w; g->s = s; g->mc = mc; g->tok = tok; g->tp = tp;
    g->gp = gp; g->callback = callback; g->userdata = userdata;
    g->max_tokens = gp->max_tokens;
    g->json_mode = gp->json_mode;
    g->ss = &gp->sampler;
    g->wm = gp->watermark.enabled ? &gp->watermark : NULL;
    g->finish = GEN_FINISH_LENGTH;
    g->done = 1;   /* until begin succeeds */
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

    stop_matcher_init(&g->stop, gp->stop, gp->n_stop);

    /* Phase 20: grammar-constrained decoding state (only touched when
     * json_mode is set). Fresh per call -- constrains this one generation. */
    if (g->json_mode) fsm_init(&g->json_fsm);

    /* Reset expert hit tracking for MoE models */
    if (mc && mc->is_moe)
        moe_expert_tracking_reset(cfg->n_layers, mc->num_experts);

    /* Token buffer (required): checked sizing, traps on absurd size rather
     * than letting a required buffer come back NULL. */
    size_t tok_bytes;
    if (tn_size_mul_overflow((size_t)cfg->seq_len + 1, sizeof(int), &tok_bytes)) return -1;
    g->prompt_tokens = (int *)tn_safe_malloc(tok_bytes, "generate prompt tokens");

    if (build_prompt(g, prompt) != 0) return -1;
    int n_prompt = g->n_prompt;

    /* Debug: print prompt token IDs when verbose */
    if (g_tn_verbose) {
        fprintf(stderr, "[DBG] n_prompt=%d  tokens:", n_prompt);
        for (int i = 0; i < n_prompt && i < 32; i++) fprintf(stderr, " %d", g->prompt_tokens[i]);
        fprintf(stderr, "\n");
    }

    /* pos_offset: skip over any vision tokens already pre-filled in the KV cache */
    g->pos_offset = s->current_pos;

    /* RNG: explicit seed (--seed / API "seed") for reproducible sampling,
     * else time-based. */
    rng_seed(&g->rng_state, gp->seed >= 0 ? (unsigned long long)gp->seed
                                          : (unsigned long long)time(NULL));

    /* Phase 30: raw-logit copy for log-probabilities of sampled tokens. */
    g->raw_logits = (gp->compute_logprobs && gp->stats)
                    ? (float *)malloc((size_t)cfg->vocab_size * sizeof(float)) : NULL;
    /* Generated-token history for repetition/presence/frequency penalties. */
    if (sampler_penalties_active(g->ss) && g->max_tokens > 0)
        g->gen_history = (int *)malloc((size_t)g->max_tokens * sizeof(int));

    /* Initialize Sliding Window (pins the entire prompt) */
    sw_init(&s->sw, s->max_seq_len, n_prompt + g->pos_offset);

    /* Commit the KV rows this call can write, up front (no-op once resident). */
    run_state_kv_reserve(s, cfg, g->pos_offset + n_prompt + (g->max_tokens > 0 ? g->max_tokens : 0));

    /* Phase 21.5: skip prompt positions whose K/V are already cached from a
     * previous call with the same leading tokens. The last prompt token is
     * always recomputed — its logits seed the first sample. Not valid after
     * a vision prefill (pos_offset > 0) or for recurrent linear-attention
     * layers, whose state cannot be rewound to a partial prefix. */
    g->pcache = gp->prefix_cache;
    /* Snapshots copy F32 KV rows; with --kv-type i8 only the live prefix cache applies.
     * 2026-10-18: also off under --remote-layers — a snapshot restores only the local
     * layers' KV, while each worker still holds the KV of the last prompt it saw. */
    g->pstore = (g->pcache && prompt_store_supported(mc) && s->key_cache && !dist_plan_active())
                ? gp->prompt_store : NULL;
    int start_step = 0;
    g->cache_eligible = g->pcache && g->pos_offset == 0 && !moe_is_recurrent(mc);
    if (g->pcache) {
        if (g->cache_eligible && n_prompt > 1) {
            int lcp = prefix_cache_match(g->pcache, g->prompt_tokens, n_prompt);
            /* Phase 26: a stored snapshot may share more than the live KV. */
            if (g->pstore)
                lcp = prompt_store_restore(g->pstore, cfg, s, g->pcache, g->prompt_tokens, n_prompt, lcp);
            start_step = lcp < n_prompt - 1 ? lcp : n_prompt - 1;
            prefix_cache_truncate(g->pcache, start_step);
        } else {
            prefix_cache_reset(g->pcache);
        }
    }
    if (gp->stats) gp->stats->cached_tokens = start_step;

    /* prev_token at the start of step k is the token being forwarded at k
     * (for k >= 1) — see the end-of-step update in gen_session_consume. */
    g->step = start_step;
    g->prev_token = start_step > 0 ? g->prompt_tokens[start_step] : -1;
    g->next = (start_step < n_prompt - 1) ? g->prompt_tokens[start_step + 1] : 0;

    /* max_tokens generation-eligible steps: the first generated token comes
     * "free" from the last prompt-processing step (step == n_prompt - 1), so
     * only max_tokens - 1 MORE steps are needed after that -- total_steps =
     * n_prompt + max_tokens - 1 (2026-09-24 off-by-one, see mistakes.md). */
    g->total_steps = n_prompt + g->max_tokens - 1;
    g->started = 1;
    g->done = !(g->step < g->total_steps && g->step + g->pos_offset < cfg->seq_len);
    return 0;
}

int gen_session_input(const GenSession *g, int *token, int *pos) {
    if (g->done) return 0;
    *token = (g->step < g->n_prompt) ? g->prompt_tokens[g->step] : g->next;
    *pos = g->step + g->pos_offset;
    return 1;
}

int gen_session_consume(GenSession *g, float *logits) {
    if (g->done) return 0;
    const Config *cfg = g->cfg;
    const GenerationParams *gp = g->gp;
    Tokenizer *tok = g->tok;
    int step = g->step, n_prompt = g->n_prompt;
    int abs_pos = step + g->pos_offset;
    int token = (step < n_prompt) ? g->prompt_tokens[step] : g->next;

    if (!logits || (dist_plan_active() && dist_plan_active()->failed)) {
        g->finish = GEN_FINISH_ERROR;   /* a remote layer range is unreachable */
        g->done = 1;
        return 0;
    }
    if (g->pcache && g->pos_offset == 0) prefix_cache_set(g->pcache, abs_pos, token);

    /* Sampling */
    if (step < n_prompt - 1) {
        g->next = g->prompt_tokens[step + 1];
    } else {
        /* Full prompt processed, sample from logits */
        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        if (g->raw_logits) {
            memcpy(g->raw_logits, logits, (size_t)cfg->vocab_size * sizeof(float));
            g->step_lse = logprob_logsumexp(g->raw_logits, cfg->vocab_size);
        }
        /* ignore_eos: never sample an end-of-sequence token (llama.cpp's
         * --ignore-eos), so runs last exactly max_tokens. */
        if (gp->ignore_eos) {
            if (cfg->eos_token_id >= 0 && cfg->eos_token_id < cfg->vocab_size)
                logits[cfg->eos_token_id] = -INFINITY;
            for (int ei = 0; ei < tok->n_eos; ei++)
                if (tok->eos_list[ei] >= 0 && tok->eos_list[ei] < cfg->vocab_size)
                    logits[tok->eos_list[ei]] = -INFINITY;
            if (g->chat_stops)
                for (int vi = 0; vi < cfg->vocab_size && vi < tok->vocab_size; vi++)
                    if (g->chat_stops[vi]) logits[vi] = -INFINITY;
        }
        if (g->json_mode) {
            g->next = sample_constrained_chain(logits, cfg->vocab_size, &g->json_fsm,
                                               tok, g->prev_token, g->ss,
                                               g->gen_history, g->n_history, g->wm,
                                               &g->rng_state);
        } else {
            g->next = sampler_chain_sample(logits, cfg->vocab_size, g->ss,
                                           g->gen_history, g->n_history, g->wm,
                                           g->prev_token, &g->rng_state);
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
                    step, g->next,
                    top5i[0], top5v[0], top5i[1], top5v[1], top5i[2], top5v[2],
                    top5i[3], top5v[3], top5i[4], top5v[4]);
        }
    }

    /* EOS check first — do NOT print the EOS token itself. All EOS IDs come
     * from cfg (GGUF loader), tok->eos_list and this session's chat stops. */
    if (step >= n_prompt - 1) {
        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        int is_eos = (cfg->eos_token_id > 0 && g->next == cfg->eos_token_id) ||
                     tokenizer_is_eog(tok, g->next, g->chat_stops);
        if (t_step) {
            tn_step_timing_add(TN_STEP_21_EOS_CHECK,
                               tn_step_timing_now_ns() - t_step);
        }
        if (is_eos) {
            g->finish = GEN_FINISH_STOP;
            if (gp->stats) gp->stats->hit_eos = true;
            g->done = 1;
            return 0;
        }
        /* Non-EOS token: decode and deliver to callback */
        if (g->tokens_generated == 0) g->gen_start_us = timer_now_us();
        const char *piece = tokenizer_decode(tok, g->prev_token, g->next);
        int stop_requested = 0;
        if (piece) stop_requested = stop_matcher_feed(&g->stop, piece, g->callback, g->userdata);
        if (g->gen_history && g->n_history < g->max_tokens) g->gen_history[g->n_history++] = g->next;
        if (g->raw_logits) g->sum_logprob += (double)g->raw_logits[g->next] - g->step_lse;
        g->tokens_generated++;
        if (stop_requested) {
            g->finish = g->stop.stopped ? GEN_FINISH_STOP : GEN_FINISH_CANCELLED;
            g->done = 1;
            return 0;
        }
    }

    g->prev_token = g->next;
    if (step < n_prompt - 1) g->next = g->prompt_tokens[step + 1];
    g->step = step + 1;
    if (!(g->step < g->total_steps && g->step + g->pos_offset < cfg->seq_len)) {
        g->done = 1;
        return 0;
    }
    return 1;
}

void gen_session_end(GenSession *g) {
    const GenerationParams *gp = g->gp;
    if (g->started) {
        stop_matcher_flush(&g->stop, g->callback, g->userdata);
        if (g->pstore && g->cache_eligible) prompt_store_save(g->pstore, g->cfg, g->s, g->pcache);
    }
    stop_matcher_free(&g->stop);
    free(g->gen_history);
    free(g->raw_logits);
    free(g->prompt_tokens);
    free(g->chat_stops);
    g->gen_history = NULL; g->raw_logits = NULL; g->prompt_tokens = NULL; g->chat_stops = NULL;
    if (!g->started) return;

    /* Report throughput to stderr (does not interfere with callback output) */
    double tok_per_sec = 0.0;
    if (g->tokens_generated > 1 && g->gen_start_us > 0) {
        int64_t end_us = timer_now_us();
        tok_per_sec = timer_tokens_per_sec(g->gen_start_us, end_us, g->tokens_generated);
        fprintf(stderr, "\n[gen] %.2f tok/s (%d tokens)\n", tok_per_sec, g->tokens_generated);
    }
    if (gp && gp->stats) {
        gp->stats->prompt_tokens = g->n_prompt;
        gp->stats->completion_tokens = g->tokens_generated;
        gp->stats->finish_reason = g->finish;
        gp->stats->tokens_per_sec = tok_per_sec;
        gp->stats->sum_logprob = g->sum_logprob;
    }

    /* Print expert utilisation for MoE models */
    if (g->mc && g->mc->is_moe)
        moe_expert_tracking_print(g->cfg->n_layers, g->mc->num_experts);
    tn_step_timing_report(stderr);
    g->started = 0;
}
