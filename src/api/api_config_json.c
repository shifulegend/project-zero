/*
 * src/api/api_config_json.c
 *
 * GET /v1/config: the server-side generation defaults, valid ranges and feature flags the web UI
 * builds every control from. One function per top-level section of the document. Split out of
 * api_info.c; the output is unchanged.
 */

#include <stdio.h>
#include <string.h>

#include "api/api_info.h"
#include "api/chat_compile.h"
#include "api/chat_request.h"
#include "api/config_endpoint.h"
#include "api/json_write.h"
#include "brand/block_font.h"
#include "distributed/dist_plan.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "reasoning/best_of.h"
#include "runtime/hardware_profile.h"
#include "sampling/watermark_text.h"
#include "threading/cache_probe.h"
#include "transformer/attn_tiled.h"

static void range(JsonBuf *jb, const char *key, double lo, double hi, double step) {
    jb_key(jb, key);
    jb_raw(jb, "{\"min\":");
    jb_num(jb, lo);
    jb_raw(jb, ",\"max\":");
    jb_num(jb, hi);
    jb_raw(jb, ",\"step\":");
    jb_num(jb, step);
    jb_raw(jb, "}");
}

/* Activation numerics and the quality bar a fast setting must clear (eval/selfcheck.h). */
static void runtime_act_quant(JsonBuf *jb, const ApiContext *ctx) {
    const TnHardwareProfile *hp = tn_hardware_profile_get();
    jb_raw(jb, ",");
    jb_key(jb, "act_quant");
    jb_raw(jb, "{");
    jb_key(jb, "requested");
    jb_str(jb, tn_act_quant_name(ctx ? (TnActQuant)ctx->server_config.act_quant : TN_ACT_AUTO));
    jb_raw(jb, ",");
    jb_key(jb, "effective");
    jb_str(jb, tn_act_quant_effective());
    jb_raw(jb, "}");
    jb_raw(jb, ",");
    jb_key(jb, "classifier");
    jb_str(jb, !hp                                 ? "bf16"
               : hp->classifier_fmt == TN_CLS_INT4 ? "int4"
               : hp->classifier_fmt == TN_CLS_INT8 ? "int8"
                                                   : "bf16");
    jb_raw(jb, ",");
    jb_key(jb, "quality_tolerance");
    jb_num(jb, ctx ? ctx->server_config.quality_tolerance : 1.0);
    jb_raw(jb, ",");
    jb_key(jb, "self_check_tokens");
    jb_int(jb, ctx ? ctx->server_config.self_check_tokens : 96);
}

/* The layer plan of a --remote-layers coordinator (an empty array otherwise). */
static void runtime_distributed(JsonBuf *jb) {
    jb_key(jb, "distributed");
    jb_raw(jb, "[");
    DistPlan *dp = dist_plan_active();
    for (int i = 0; dp && i < dp->n; i++) {
        if (i) jb_raw(jb, ",");
        jb_raw(jb, "{");
        jb_key(jb, "host");
        jb_str(jb, dp->nodes[i].host);
        jb_raw(jb, ",");
        jb_key(jb, "port");
        jb_int(jb, dp->nodes[i].port);
        jb_raw(jb, ",");
        jb_key(jb, "layers");
        char lr[32];
        snprintf(lr, sizeof(lr), "%d-%d", dp->nodes[i].start, dp->nodes[i].end);
        jb_str(jb, lr);
        jb_raw(jb, "}");
    }
    jb_raw(jb, "]");
}

/* The detected cache sizes and where they came from. */
static void runtime_cache(JsonBuf *jb, const CacheInfo *ci) {
    jb_key(jb, "l1d_bytes");
    jb_int(jb, ci->l1d_bytes);
    jb_raw(jb, ",");
    jb_key(jb, "l2_bytes");
    jb_int(jb, ci->l2_bytes);
    jb_raw(jb, ",");
    jb_key(jb, "l3_bytes");
    jb_int(jb, ci->l3_bytes);
    jb_raw(jb, ",");
    jb_key(jb, "source");
    jb_str(jb, tn_cache_source_name(ci->source));
}

/* Phase 23 stage 2b: requests served at once, what was asked for, and why fewer when the model or memory cannot do it. */
static void features_parallel(JsonBuf *jb, const ApiContext *ctx) {
    jb_raw(jb, ",");
    jb_key(jb, "parallel");
    jb_raw(jb, "{");
    jb_key(jb, "slots");
    jb_int(jb, ctx && ctx->parallel_slots > 0 ? ctx->parallel_slots : 1);
    jb_raw(jb, ",");
    jb_key(jb, "requested");
    jb_int(jb, ctx ? ctx->server_config.parallel : 1);
    jb_raw(jb, ",");
    jb_key(jb, "batch_tokens");
    jb_int(jb, ctx ? ctx->server_config.parallel_batch : PZ_PARALLEL_BATCH_DEFAULT);
    jb_raw(jb, ",");
    jb_key(jb, "kv_budget_tokens");
    jb_int(jb, ctx ? ctx->parallel_kv_budget : 0);
    jb_raw(jb, ",");
    jb_key(jb, "max");
    jb_int(jb, PZ_PARALLEL_MAX);
    jb_raw(jb, ",");
    jb_key(jb, "batch_tokens_max");
    jb_int(jb, PZ_PARALLEL_BATCH_MAX);
    jb_raw(jb, ",");
    jb_key(jb, "note");
    jb_str(jb, ctx ? ctx->parallel_note : NULL);
    jb_raw(jb, "}");
}

/* The web UI's title and logo. The block wordmark is rendered from the (customisable) title with
 * the same font as the CLI banner; titles the font can't spell fall back to text, as does
 * --ui-logo text. */
static void config_ui(JsonBuf *jb, const ApiContext *ctx) {
    const char *title = ctx && ctx->server_config.ui_title && ctx->server_config.ui_title[0]
                            ? ctx->server_config.ui_title
                            : "project-zero";
    char logo[TN_BLOCK_FONT_ROWS * 160];
    int ascii = !(ctx && ctx->server_config.ui_logo == 1) &&
                tn_block_font_render(title, logo, sizeof(logo), 0) > 0;
    jb_key(jb, "ui");
    jb_raw(jb, "{");
    jb_key(jb, "title");
    jb_str(jb, title);
    jb_raw(jb, ",");
    jb_key(jb, "logo");
    jb_str(jb, ascii ? "ascii" : "text");
    jb_raw(jb, ",");
    jb_key(jb, "logo_ascii");
    jb_str(jb, ascii ? logo : NULL);
    jb_raw(jb, "},");
}

/* Model metadata (name, architecture, context, vocabulary). */
static void config_model(JsonBuf *jb, const ApiContext *ctx) {
    jb_key(jb, "model");
    jb_raw(jb, "{");
    api_model_meta(jb, ctx);
    jb_raw(jb, "},");
}

/* Sampler defaults and the request-level switches. */
static void config_defaults_sampler(JsonBuf *jb, const ApiContext *ctx,
                                    const GenerationParams *defp, int lim) {
    jb_key(jb, "defaults");
    jb_raw(jb, "{");
    jb_key(jb, "max_tokens");
    jb_int(jb, defp->max_tokens < lim ? defp->max_tokens : lim);
    jb_raw(jb, ",");
    jb_key(jb, "temperature");
    jb_num(jb, defp->sampler.temperature);
    jb_raw(jb, ",");
    jb_key(jb, "top_p");
    jb_num(jb, defp->sampler.top_p);
    jb_raw(jb, ",");
    jb_key(jb, "top_k");
    jb_int(jb, defp->sampler.top_k);
    jb_raw(jb, ",");
    jb_key(jb, "min_p");
    jb_num(jb, defp->sampler.min_p);
    jb_raw(jb, ",");
    jb_key(jb, "repetition_penalty");
    jb_num(jb, defp->sampler.repetition_penalty);
    jb_raw(jb, ",");
    jb_key(jb, "presence_penalty");
    jb_num(jb, defp->sampler.presence_penalty);
    jb_raw(jb, ",");
    jb_key(jb, "frequency_penalty");
    jb_num(jb, defp->sampler.frequency_penalty);
    jb_raw(jb, ",");
    jb_key(jb, "penalty_last_n");
    jb_int(jb, defp->sampler.penalty_last_n);
    jb_raw(jb, ",");
    jb_key(jb, "seed");
    jb_int(jb, defp->seed);
    jb_raw(jb, ",");
    jb_key(jb, "json_mode");
    jb_bool(jb, defp->json_mode);
    jb_raw(jb, ",");
    jb_key(jb, "ignore_eos");
    jb_bool(jb, defp->ignore_eos);
    jb_raw(jb, ",");
    jb_key(jb, "cache_prompt");
    jb_bool(jb, ctx && ctx->server_config.prompt_cache);
    jb_raw(jb, ",");
    jb_key(jb, "best_of");
    jb_int(jb, ctx ? ctx->server_config.best_of : 1);
}

/* Tree-search (mcts) defaults. */
static void config_defaults_search(JsonBuf *jb, const ApiContext *ctx) {
    {
        MctsOptions m;
        if (ctx)
            m = ctx->server_config.mcts;
        else
            memset(&m, 0, sizeof(m));
        mcts_options_defaults(&m);
        jb_raw(jb, ",");
        jb_key(jb, "mcts");
        jb_raw(jb, "{");
        jb_key(jb, "simulations");
        jb_int(jb, m.simulations);
        jb_raw(jb, ",");
        jb_key(jb, "expand");
        jb_int(jb, m.expand);
        jb_raw(jb, ",");
        jb_key(jb, "max_depth");
        jb_int(jb, m.max_depth);
        jb_raw(jb, ",");
        jb_key(jb, "step_tokens");
        jb_int(jb, m.step_tokens);
        jb_raw(jb, ",");
        jb_key(jb, "c");
        jb_num(jb, m.c_ucb);
        jb_raw(jb, ",");
        jb_key(jb, "temperature");
        jb_num(jb, m.temperature);
        jb_raw(jb, "}");
    }
}

/* System prompt, stop strings and watermark defaults. */
static void config_defaults_tail(JsonBuf *jb, const GenerationParams *defp) {
    jb_raw(jb, ",");
    jb_key(jb, "system_prompt");
    jb_str(jb, defp->system_prompt ? defp->system_prompt : "");
    jb_raw(jb, ",");
    jb_key(jb, "stop");
    jb_raw(jb, "[");
    for (int i = 0; i < defp->n_stop; i++) {
        if (i) jb_raw(jb, ",");
        jb_str(jb, defp->stop[i]);
    }
    jb_raw(jb, "]");
    jb_raw(jb, ",");
    jb_key(jb, "watermark");
    jb_raw(jb, "{");
    jb_key(jb, "enabled");
    jb_bool(jb, defp->watermark.enabled);
    jb_raw(jb, ",");
    jb_key(jb, "gamma");
    jb_num(jb, defp->watermark.gamma);
    jb_raw(jb, ",");
    jb_key(jb, "delta");
    jb_num(jb, defp->watermark.delta);
    jb_raw(jb, ",");
    jb_key(jb, "server_key");
    jb_bool(jb, defp->watermark.key != 0);
    jb_raw(jb, ",");
    jb_key(jb, "z_threshold");
    jb_num(jb, WATERMARK_DETECT_Z_THRESHOLD);
    jb_raw(jb, "}");
    jb_raw(jb, "},");
}

/* Valid ranges, mirroring sampler_settings_sanitize() / watermark_config_sanitize(). */
static void config_limits(JsonBuf *jb, const ApiContext *ctx, int lim, int vocab) {
    jb_key(jb, "limits");
    jb_raw(jb, "{");
    range(jb, "max_tokens", 1, lim, 1);
    jb_raw(jb, ",");
    range(jb, "temperature", 0, 5, 0.05);
    jb_raw(jb, ",");
    range(jb, "top_p", 0.01, 1, 0.01);
    jb_raw(jb, ",");
    range(jb, "top_k", 0, vocab > 0 ? vocab : 1000, 1);
    jb_raw(jb, ",");
    range(jb, "min_p", 0, 0.99, 0.01);
    jb_raw(jb, ",");
    range(jb, "repetition_penalty", 0.5, 10, 0.05);
    jb_raw(jb, ",");
    range(jb, "presence_penalty", -2, 2, 0.05);
    jb_raw(jb, ",");
    range(jb, "frequency_penalty", -2, 2, 0.05);
    jb_raw(jb, ",");
    range(jb, "penalty_last_n", -1, lim, 1);
    jb_raw(jb, ",");
    range(jb, "watermark_gamma", 0.01, 0.99, 0.01);
    jb_raw(jb, ",");
    range(jb, "watermark_delta", 0, 20, 0.1);
    jb_raw(jb, ",");
    range(jb, "best_of", 1, ctx ? ctx->server_config.best_of_max : BEST_OF_MAX_DEFAULT, 1);
    jb_raw(jb, ",");
    range(jb, "mcts_simulations", 0, ctx ? ctx->server_config.mcts_max : 0, 1);
    jb_raw(jb, ",");
    range(jb, "mcts_expand", 1, MCTS_MAX_EXPAND, 1);
    jb_raw(jb, ",");
    range(jb, "mcts_max_depth", 1, MCTS_MAX_DEPTH, 1);
    jb_raw(jb, ",");
    range(jb, "mcts_step_tokens", 1, MCTS_MAX_STEP_TOKENS, 1);
    jb_raw(jb, ",");
    range(jb, "mcts_c", 0.1, 10, 0.1);
    jb_raw(jb, ",");
    range(jb, "mcts_temperature", 0.05, 2, 0.05);
    jb_raw(jb, ",");
    jb_key(jb, "max_stop");
    jb_int(jb, CHAT_MAX_STOP);
    jb_raw(jb, ",");
    jb_key(jb, "max_messages");
    jb_int(jb, CHAT_MAX_MESSAGES);
    jb_raw(jb, "},");
}

/* Read-only runtime facts (startup flags, not per-request settings). */
static void config_runtime(JsonBuf *jb, const ApiContext *ctx) {
    {
        const CacheInfo *ci = tn_cache_info();
        jb_key(jb, "runtime");
        jb_raw(jb, "{");
        jb_key(jb, "simd");
        jb_str(jb, tn_simd_backend_name());
        jb_raw(jb, ",");
        jb_key(jb, "threads");
        jb_int(jb, ctx && ctx->tp ? ctx->tp->num_threads : 1);
        runtime_act_quant(jb, ctx);
        jb_raw(jb, ",");
        jb_key(jb, "kv_type");
        jb_str(jb, !(ctx && ctx->run_state)                ? "f32"
                   : ctx->run_state->kv_type == KV_TYPE_I8 ? "i8"
                   : ctx->run_state->kv_type == KV_TYPE_I4 ? "i4"
                                                           : "f32");
        /* Phase 31: whether the KV cache is disk-backed (the path stays server-side). */
        jb_raw(jb, ",");
        jb_key(jb, "kv_offload");
        jb_raw(jb, ctx && ctx->run_state && ctx->run_state->kv_offload_base ? "true" : "false");
        jb_raw(jb, ",");
        jb_key(jb, "attention_kernel");
        jb_str(jb, attn_kernel_name(attn_kernel_get()));
        jb_raw(jb, ",");
        jb_key(jb, "rope_layout");
        jb_str(jb, rope_layout_name(ctx && ctx->cfg ? ctx->cfg->rope_layout : 0));
        jb_raw(jb, ",");
        jb_key(jb, "attention_tile");
        jb_int(jb, attn_tile_get());
        jb_raw(jb, ",");
        jb_key(jb, "attention_parallel_min_ctx");
        jb_int(jb, attn_parallel_min_ctx_get());
        jb_raw(jb, ",");
        runtime_distributed(jb);
        jb_raw(jb, ",");
        jb_key(jb, "cache");
        jb_raw(jb, "{");
        runtime_cache(jb, ci);
        jb_raw(jb, "}}");
        jb_raw(jb, ",");
    }
}

/* Feature switches the web UI renders its controls from. */
static void config_features(JsonBuf *jb, const ApiContext *ctx) {
    jb_key(jb, "features");
    jb_raw(jb, "{");
    jb_key(jb, "json_mode");
    jb_bool(jb, 1);
    jb_raw(jb, ",");
    jb_key(jb, "watermark");
    jb_bool(jb, 1);
    jb_raw(jb, ",");
    jb_key(jb, "vision");
    jb_bool(jb, api_vision_enabled(ctx));
    jb_raw(jb, ",");
    jb_key(jb, "cancel");
    jb_bool(jb, 1);
    jb_raw(jb, ",");
    jb_key(jb, "best_of");
    jb_bool(jb, ctx && ctx->server_config.best_of_max > 1);
    jb_raw(jb, ",");
    jb_key(jb, "mcts");
    jb_bool(jb, ctx && ctx->server_config.mcts_max > 0);
    jb_raw(jb, ",");
    jb_key(jb, "prm");
    if (ctx && ctx->prm && ctx->server_config.prm_model)
        jb_str(jb, ctx->server_config.prm_model);
    else
        jb_raw(jb, "null");
    jb_raw(jb, ",");
    jb_key(jb, "queue_size");
    jb_int(jb, ctx ? ctx->server_config.queue_size : 0);
    jb_raw(jb, ",");
    jb_key(jb, "rate_limit");
    jb_int(jb, ctx ? ctx->server_config.rate_limit : 0);
    jb_raw(jb, ",");
    jb_key(jb, "client_timeout_s");
    jb_int(jb, ctx ? ctx->server_config.client_timeout_s : 0);
    jb_raw(jb, ",");
    jb_key(jb, "max_connections");
    jb_int(jb, ctx ? ctx->server_config.max_connections : 0);
    features_parallel(jb, ctx);
    /* Recurrent linear-attention state cannot resume from a partial prefix. */
    jb_raw(jb, ",");
    jb_key(jb, "prompt_cache");
    jb_bool(jb, !(ctx && moe_is_recurrent(ctx->moe_cfg)));
    jb_raw(jb, ",");
    jb_key(jb, "prompt_store");
    jb_bool(jb, ctx && ctx->server_config.prompt_cache_mb > 0 &&
                    (!ctx->moe_cfg || prompt_store_supported(ctx->moe_cfg)) &&
                    !(ctx->run_state && ctx->run_state->kv_type != KV_TYPE_F32));
    jb_raw(jb, ",");
    jb_key(jb, "metrics");
    jb_bool(jb, ctx && ctx->server_config.metrics.enabled);
    jb_raw(jb, ",");
    jb_key(jb, "system_prompt");
    jb_bool(jb, ctx && ctx->tok && ctx->tok->chat_template != NULL);
    jb_raw(jb, ",");
    jb_key(jb, "request_chat_template");
    jb_bool(jb, ctx && ctx->server_config.allow_request_template);
    jb_raw(jb, "}");
}

/* The active chat template, so the web UI can show it and offer an edited copy as a per-request "chat_template" override. */
static void config_chat_template(JsonBuf *jb, const ApiContext *ctx) {
    jb_raw(jb, ",");
    jb_key(jb, "chat_template");
    jb_raw(jb, "{");
    jb_key(jb, "source");
    jb_str(jb, ctx && ctx->server_config.chat_template_source
                   ? ctx->server_config.chat_template_source
                   : (ctx && ctx->tok && ctx->tok->chat_template ? "gguf" : "none"));
    jb_raw(jb, ",");
    jb_key(jb, "text");
    jb_str(jb, ctx && ctx->tok && ctx->tok->chat_template ? ctx->tok->chat_template : "");
    jb_raw(jb, ",");
    jb_key(jb, "request_override");
    jb_bool(jb, ctx && ctx->server_config.allow_request_template);
    jb_raw(jb, ",");
    jb_key(jb, "max_bytes");
    jb_int(jb, CHAT_MAX_TEMPLATE);
    jb_raw(jb, "}");
}

char *api_config_json(const ApiContext *ctx) {
    GenerationParams def;
    if (ctx)
        def = ctx->server_config.gen_defaults;
    else
        gen_params_default(&def);
    int lim   = api_max_tokens_limit(ctx);
    int vocab = (ctx && ctx->cfg) ? ctx->cfg->vocab_size : 0;

    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{");
    config_ui(&jb, ctx);
    config_model(&jb, ctx);
    config_defaults_sampler(&jb, ctx, &def, lim);
    config_defaults_search(&jb, ctx);
    config_defaults_tail(&jb, &def);
    config_limits(&jb, ctx, lim, vocab);
    config_runtime(&jb, ctx);
    config_features(&jb, ctx);
    config_chat_template(&jb, ctx);
    /* Phase 40.3: where each default came from, and whether the UI may save. */
    jb_raw(&jb, ",");
    api_config_sources_json(&jb, ctx);
    jb_raw(&jb, "}");
    return jb_finish(&jb);
}
