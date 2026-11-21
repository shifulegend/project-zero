/*
 * transformer/gen_session.c — see transformer/gen_session.h.
 *
 * 2026-10-22: the body of generate_ex() split into begin / input / consume /
 * end so the API server can interleave several generations (Phase 23 stage
 * 2b) on the same code path the single-sequence loop uses. Logic is
 * unchanged from generate_ex; comments carried over.
 */
#include "transformer/gen_session.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/debug.h"
#include "core/log.h"
#include "core/step_timing.h"
#include "core/timer.h"
#include "distributed/dist_plan.h"
#include "memory/safe_alloc.h"
#include "sampling/constrained_sample.h"
#include "sampling/logprob.h"
#include "sampling/rng.h"
#include "sampling/sampler_chain.h"
#include "sampling/sampling.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/template_guard.h"
#include "transformer/gen_prompt.h"
#include "transformer/moe_ffn.h"

static const char *token_text(const Tokenizer *tok, int id) {
    return (tok && tok->vocab && id >= 0 && id < tok->vocab_size && tok->vocab[id]) ? tok->vocab[id]
                                                                                    : "?";
}

/* 2026-11-16: the first token the model chose ended the turn, so nothing was
 * generated. Leading servers (llama.cpp, Ollama) answer with an empty message
 * and leave users to guess; name the evidence and the usual cause instead. */
static void report_empty_generation(const Tokenizer *tok, int first_token, int last_prompt_token) {
    TN_LOG_WRN("warn",
               "[warn] no tokens were generated: the first token the model chose was %d (\"%s\"), "
               "a stop token; the prompt ended with token %d (\"%s\"). For a chat model this "
               "usually means a chat-template problem: check the template "
               "(--chat-template-file replaces it) or run with --verbose.\n",
               first_token, token_text(tok, first_token), last_prompt_token,
               token_text(tok, last_prompt_token));
}

int gen_session_begin(GenSession *g, const Config *cfg, const TransformerWeights *w, RunState *s,
                      const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp, const char *prompt,
                      const GenerationParams *gp, TokenCallback callback, void *userdata) {
    if (gen_session_prepare(g, cfg, w, mc, tok, tp, prompt, gp, callback, userdata) != 0) return -1;
    return gen_session_start(g, s, g->req.gp->prefix_cache, g->req.gp->prompt_store);
}

int gen_session_prepare(GenSession *g, const Config *cfg, const TransformerWeights *w,
                        const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp, const char *prompt,
                        const GenerationParams *gp, TokenCallback callback, void *userdata) {
    memset(g, 0, sizeof(*g));
    if (!prompt) prompt = "";
    if (!gp) {
        gen_params_default(&g->req.gp_default);
        gp = &g->req.gp_default;
    }
    g->model.cfg       = cfg;
    g->model.w         = w;
    g->model.mc        = mc;
    g->model.tok       = tok;
    g->model.tp        = tp;
    g->req.gp          = gp;
    g->req.callback    = callback;
    g->req.userdata    = userdata;
    g->req.max_tokens  = gp->max_tokens;
    g->req.json_mode   = gp->json_mode;
    g->req.ss          = &gp->sampler;
    g->req.wm          = gp->watermark.enabled ? &gp->watermark : NULL;
    g->progress.finish = GEN_FINISH_LENGTH;
    g->progress.done   = 1; /* until begin succeeds */
    tn_step_timing_reset();
    if (gp->stats) {
        gp->stats->prompt_tokens     = 0;
        gp->stats->completion_tokens = 0;
        gp->stats->cached_tokens     = 0;
        gp->stats->sum_logprob       = 0.0;
        gp->stats->finish_reason     = GEN_FINISH_ERROR; /* until generation runs */
        gp->stats->tokens_per_sec    = 0.0;
        gp->stats->hit_eos           = false;
    }

    stop_matcher_init(&g->sampling.stop, gp->stop, gp->n_stop);

    /* Phase 20: grammar-constrained decoding state (only touched when
     * json_mode is set). Fresh per call -- constrains this one generation. */
    if (g->req.json_mode) fsm_init(&g->sampling.json_fsm);

    /* Reset expert hit tracking for MoE models */
    if (mc && mc->is_moe) moe_expert_tracking_reset(cfg->n_layers, mc->num_experts);

    /* Token buffer (required): checked sizing, traps on absurd size rather
     * than letting a required buffer come back NULL. */
    size_t tok_bytes;
    if (tn_size_mul_overflow((size_t)cfg->seq_len + 1, sizeof(int), &tok_bytes)) return -1;
    g->prompt.tokens = (int *)tn_safe_malloc(tok_bytes, "generate prompt tokens");

    if (gen_session_build_prompt(g, prompt) != 0) return -1;
    int n_prompt = g->prompt.n_tokens;

    /* Debug: print prompt token IDs when verbose */
    if (g_tn_verbose) {
        char ids[32 * 12 + 1];
        size_t used = 0;
        ids[0]      = '\0';
        for (int i = 0; i < n_prompt && i < 32; i++)
            used += (size_t)snprintf(ids + used, sizeof(ids) - used, " %d", g->prompt.tokens[i]);
        TN_LOG_DBG("gen_session", "[DBG] n_prompt=%d  tokens:%s\n", n_prompt, ids);
    }

    /* RNG: explicit seed (--seed / API "seed") for reproducible sampling,
     * else time-based. */
    rng_seed(&g->sampling.rng_state,
             gp->seed >= 0 ? (unsigned long long)gp->seed : (unsigned long long)time(NULL));

    /* Phase 30: raw-logit copy for log-probabilities of sampled tokens. */
    g->sampling.raw_logits = (gp->compute_logprobs && gp->stats)
                                 ? (float *)malloc((size_t)cfg->vocab_size * sizeof(float))
                                 : NULL;
    /* Generated-token history for repetition/presence/frequency penalties. */
    if (sampler_penalties_active(g->req.ss) && g->req.max_tokens > 0)
        g->sampling.gen_history = (int *)malloc((size_t)g->req.max_tokens * sizeof(int));

    return 0;
}

/* Snapshots copy F32 KV rows, so with --kv-type i8 only the live prefix cache applies; they are
 * also off under --remote-layers (a snapshot restores only the local layers' KV, while each
 * worker still holds the KV of the last prompt it saw). */
static PromptStore *snapshot_store_for(PrefixCache *pcache, const MoEConfig *mc, const RunState *s,
                                       PromptStore *pstore) {
    return (pcache && prompt_store_supported(mc) && s->key_cache && !s->dist) ? pstore : NULL;
}

int gen_session_start(GenSession *g, RunState *s, PrefixCache *pcache, PromptStore *pstore) {
    GenPrompt *pr   = &g->prompt;
    GenProgress *pg = &g->progress;
    if (!pr->tokens || pg->started) return -1;
    const Config *cfg          = g->model.cfg;
    const MoEConfig *mc        = g->model.mc;
    const GenerationParams *gp = g->req.gp;
    int n_prompt               = pr->n_tokens;
    g->model.s                 = s;

    /* pos_offset: skip over any vision tokens already pre-filled in the KV cache */
    pr->pos_offset = s->current_pos;

    /* Initialize Sliding Window (pins the entire prompt) */
    sw_init(&s->sw, s->max_seq_len, n_prompt + pr->pos_offset);

    /* Commit the KV rows this call can write, up front (no-op once resident). */
    run_state_kv_reserve(
        s, cfg, pr->pos_offset + n_prompt + (g->req.max_tokens > 0 ? g->req.max_tokens : 0));

    /* Phase 21.5: skip prompt positions whose K/V are already cached from a
     * previous call with the same leading tokens. The last prompt token is
     * always recomputed — its logits seed the first sample. Not valid after
     * a vision prefill (pos_offset > 0) or for recurrent linear-attention
     * layers, whose state cannot be rewound to a partial prefix. */
    pr->pcache         = pcache;
    pr->pstore         = snapshot_store_for(pcache, mc, s, pstore);
    int start_step     = 0;
    pr->cache_eligible = pr->pcache && pr->pos_offset == 0 && !moe_is_recurrent(mc);
    if (pr->pcache) {
        if (pr->cache_eligible && n_prompt > 1) {
            int lcp = prefix_cache_match(pr->pcache, pr->tokens, n_prompt);
            /* Phase 26: a stored snapshot may share more than the live KV. */
            if (pr->pstore)
                lcp =
                    prompt_store_restore(pr->pstore, cfg, s, pr->pcache, pr->tokens, n_prompt, lcp);
            start_step = lcp < n_prompt - 1 ? lcp : n_prompt - 1;
            prefix_cache_truncate(pr->pcache, start_step);
        } else {
            prefix_cache_reset(pr->pcache);
        }
    }
    if (gp->stats) gp->stats->cached_tokens = start_step;

    /* prev_token at the start of step k is the token being forwarded at k
     * (for k >= 1) — see the end-of-step update in gen_session_consume. */
    pg->step       = start_step;
    pg->prev_token = start_step > 0 ? pr->tokens[start_step] : -1;
    pg->next       = (start_step < n_prompt - 1) ? pr->tokens[start_step + 1] : 0;

    /* max_tokens generation-eligible steps: the first generated token comes
     * "free" from the last prompt-processing step (step == n_prompt - 1), so
     * only max_tokens - 1 MORE steps are needed after that -- total_steps =
     * n_prompt + max_tokens - 1 (2026-09-24 off-by-one, see mistakes.md). */
    pg->total_steps = n_prompt + g->req.max_tokens - 1;
    pg->started     = 1;
    pg->done        = !(pg->step < pg->total_steps && pg->step + pr->pos_offset < cfg->seq_len);
    return 0;
}

int gen_session_input(const GenSession *g, int *token, int *pos) {
    if (g->progress.done) return 0;
    *token = (g->progress.step < g->prompt.n_tokens) ? g->prompt.tokens[g->progress.step]
                                                     : g->progress.next;
    *pos   = g->progress.step + g->prompt.pos_offset;
    return 1;
}

int gen_session_input_rows(const GenSession *g, int *tokens, int *pos, int max_rows) {
    if (g->progress.done || max_rows <= 0) return 0;
    /* Prompt rows up to and including the last one (whose logits seed the
     * first sample) can go in one pass; after that, one row per step. */
    int n = 1;
    if (g->progress.step < g->prompt.n_tokens - 1) {
        n        = g->prompt.n_tokens - g->progress.step;
        int left = g->progress.total_steps - g->progress.step;
        int room = g->model.cfg->seq_len - (g->progress.step + g->prompt.pos_offset);
        if (n > left) n = left;
        if (n > room) n = room;
        if (n > max_rows) n = max_rows;
        if (n < 1) n = 1;
    }
    for (int k = 0; k < n; k++) {
        int st    = g->progress.step + k;
        tokens[k] = (st < g->prompt.n_tokens) ? g->prompt.tokens[st] : g->progress.next;
        pos[k]    = st + g->prompt.pos_offset;
    }
    return n;
}

int gen_session_needs_logits(const GenSession *g) {
    return !g->progress.done && g->progress.step >= g->prompt.n_tokens - 1;
}

int gen_session_consume(GenSession *g, float *logits) {
    if (g->progress.done) return 0;
    GenSampling *sm            = &g->sampling;
    const Config *cfg          = g->model.cfg;
    const GenerationParams *gp = g->req.gp;
    Tokenizer *tok             = g->model.tok;
    int step = g->progress.step, n_prompt = g->prompt.n_tokens;
    int abs_pos = step + g->prompt.pos_offset;
    int token   = (step < n_prompt) ? g->prompt.tokens[step] : g->progress.next;

    if ((!logits && step >= n_prompt - 1) || (g->model.s->dist && g->model.s->dist->failed)) {
        g->progress.finish = GEN_FINISH_ERROR; /* a remote layer range is unreachable */
        g->progress.done   = 1;
        return 0;
    }
    if (g->prompt.pcache && g->prompt.pos_offset == 0)
        prefix_cache_set(g->prompt.pcache, abs_pos, token);

    /* Sampling */
    if (step < n_prompt - 1) {
        g->progress.next = g->prompt.tokens[step + 1];
    } else {
        /* Full prompt processed, sample from logits */
        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        if (sm->raw_logits) {
            memcpy(sm->raw_logits, logits, (size_t)cfg->vocab_size * sizeof(float));
            sm->step_lse = logprob_logsumexp(sm->raw_logits, cfg->vocab_size);
        }
        /* ignore_eos: never sample an end-of-sequence token (llama.cpp's
         * --ignore-eos), so runs last exactly max_tokens. */
        if (gp->ignore_eos) {
            if (cfg->eos_token_id >= 0 && cfg->eos_token_id < cfg->vocab_size)
                logits[cfg->eos_token_id] = -INFINITY;
            for (int ei = 0; ei < tok->n_eos; ei++)
                if (tok->eos_list[ei] >= 0 && tok->eos_list[ei] < cfg->vocab_size)
                    logits[tok->eos_list[ei]] = -INFINITY;
            if (g->req.chat_stops)
                for (int vi = 0; vi < cfg->vocab_size && vi < tok->vocab_size; vi++)
                    if (g->req.chat_stops[vi]) logits[vi] = -INFINITY;
        }
        if (g->req.json_mode) {
            g->progress.next = sample_constrained_chain(
                logits, cfg->vocab_size, &sm->json_fsm, tok, g->progress.prev_token, g->req.ss,
                sm->gen_history, sm->n_history, g->req.wm, &sm->rng_state);
        } else {
            g->progress.next = sampler_chain_sample(logits, cfg->vocab_size, g->req.ss,
                                                    sm->gen_history, sm->n_history, g->req.wm,
                                                    g->progress.prev_token, &sm->rng_state);
        }
        if (t_step) {
            tn_step_timing_add(TN_STEP_20_GREEDY_SAMPLING, tn_step_timing_now_ns() - t_step);
        }
        if (g_tn_verbose) gen_session_log_top5(step, g->progress.next, logits, cfg->vocab_size);
    }

    /* EOS check first — do NOT print the EOS token itself. All EOS IDs come
     * from cfg (GGUF loader), tok->eos_list and this session's chat stops. */
    if (step >= n_prompt - 1) {
        int64_t t_step = tn_step_timing_enabled() ? tn_step_timing_now_ns() : 0;
        int is_eos     = (cfg->eos_token_id > 0 && g->progress.next == cfg->eos_token_id) ||
                     tokenizer_is_eog(tok, g->progress.next, g->req.chat_stops);
        if (t_step) {
            tn_step_timing_add(TN_STEP_21_EOS_CHECK, tn_step_timing_now_ns() - t_step);
        }
        if (is_eos) {
            g->progress.finish = GEN_FINISH_STOP;
            if (g->progress.tokens_generated == 0 && n_prompt > 0) {
                g->progress.empty_stop        = 1;
                g->progress.first_stop_token  = g->progress.next;
                g->progress.last_prompt_token = g->prompt.tokens[n_prompt - 1];
            }
            if (gp->stats) gp->stats->hit_eos = true;
            g->progress.done = 1;
            return 0;
        }
        /* Non-EOS token: decode and deliver to callback */
        if (g->progress.tokens_generated == 0) g->progress.gen_start_us = timer_now_us();
        const char *piece  = tokenizer_decode(tok, g->progress.prev_token, g->progress.next);
        int stop_requested = 0;
        if (piece)
            stop_requested = stop_matcher_feed(&sm->stop, piece, g->req.callback, g->req.userdata);
        if (sm->gen_history && sm->n_history < g->req.max_tokens)
            sm->gen_history[sm->n_history++] = g->progress.next;
        if (sm->raw_logits)
            sm->sum_logprob += (double)sm->raw_logits[g->progress.next] - sm->step_lse;
        g->progress.tokens_generated++;
        if (stop_requested) {
            g->progress.finish = sm->stop.stopped ? GEN_FINISH_STOP : GEN_FINISH_CANCELLED;
            g->progress.done   = 1;
            return 0;
        }
    }

    g->progress.prev_token = g->progress.next;
    if (step < n_prompt - 1) g->progress.next = g->prompt.tokens[step + 1];
    g->progress.step = step + 1;
    if (!(g->progress.step < g->progress.total_steps &&
          g->progress.step + g->prompt.pos_offset < cfg->seq_len)) {
        g->progress.done = 1;
        return 0;
    }
    return 1;
}

/* Frees what the session allocated (safe to call twice). */
static void release_buffers(GenSession *g) {
    stop_matcher_free(&g->sampling.stop);
    free(g->sampling.gen_history);
    free(g->sampling.raw_logits);
    free(g->prompt.tokens);
    free(g->req.chat_stops);
    g->sampling.gen_history = NULL;
    g->sampling.raw_logits  = NULL;
    g->prompt.tokens        = NULL;
    g->req.chat_stops       = NULL;
}

void gen_session_end(GenSession *g) {
    const GenerationParams *gp = g->req.gp;
    if (g->progress.started) {
        stop_matcher_flush(&g->sampling.stop, g->req.callback, g->req.userdata);
        if (g->prompt.pstore && g->prompt.cache_eligible)
            prompt_store_save(g->prompt.pstore, g->model.cfg, g->model.s, g->prompt.pcache);
    }
    release_buffers(g);
    if (!g->progress.started) return;

    /* 0 tokens in chat mode is almost always a chat-template problem, and the
     * API answer is an empty 200: say what happened (same style as
     * gen_prompt.c's report_encode_failure). */
    if (g->progress.empty_stop && g->progress.tokens_generated == 0 && !(gp && gp->raw_prompt))
        report_empty_generation(g->model.tok, g->progress.first_stop_token,
                                g->progress.last_prompt_token);

    /* Report throughput to stderr (does not interfere with callback output) */
    double tok_per_sec = 0.0;
    if (g->progress.tokens_generated > 1 && g->progress.gen_start_us > 0) {
        int64_t end_us = timer_now_us();
        tok_per_sec =
            timer_tokens_per_sec(g->progress.gen_start_us, end_us, g->progress.tokens_generated);
        TN_LOG_INF("gen", "\n[gen] %.2f tok/s (%d tokens)\n", tok_per_sec,
                   g->progress.tokens_generated);
    }
    if (gp && gp->stats) {
        gp->stats->prompt_tokens     = g->prompt.n_tokens;
        gp->stats->completion_tokens = g->progress.tokens_generated;
        gp->stats->finish_reason     = g->progress.finish;
        gp->stats->tokens_per_sec    = tok_per_sec;
        gp->stats->sum_logprob       = g->sampling.sum_logprob;
    }

    /* Print expert utilisation for MoE models */
    if (g->model.mc && g->model.mc->is_moe)
        moe_expert_tracking_print(g->model.cfg->n_layers, g->model.mc->num_experts);
    tn_step_timing_report(stderr);
    g->progress.started = 0;
}
