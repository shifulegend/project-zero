#include "api/api_info.h"
#include <string.h>
#include "api/json_write.h"
#include "api/chat_compile.h"
#include "reasoning/best_of.h"
#include "api/chat_request.h"
#include "sampling/watermark_text.h"
#include "threading/cache_probe.h"
#include "transformer/attn_tiled.h"
#include "kv_cache/kv_quant_ops.h"
#include "brand/block_font.h"
#include "math/rope.h"
#include "distributed/dist_plan.h"
#include <stdio.h>
#include "math/simd_dispatch.h"

int api_max_tokens_limit(const ApiContext *ctx) {
    int ctx_len = (ctx && ctx->cfg) ? ctx->cfg->seq_len : 0;
    int lim = ctx ? ctx->server_config.max_tokens_limit : 0;
    if (lim > 0 && (ctx_len <= 0 || lim < ctx_len)) return lim;
    return ctx_len > 0 ? ctx_len : 4096;
}

static const char *model_id(const ApiContext *ctx) {
    const char *id = ctx ? ctx->server_config.model_id : NULL;
    return (id && id[0]) ? id : "local-model";
}

static int vision_enabled(const ApiContext *ctx) {
    return ctx && ctx->server_config.vision_path && ctx->server_config.proj_path;
}

static void model_meta(JsonBuf *jb, const ApiContext *ctx) {
    jb_key(jb, "id");             jb_str(jb, model_id(ctx));
    jb_raw(jb, ",");  jb_key(jb, "object");   jb_str(jb, "model");
    jb_raw(jb, ",");  jb_key(jb, "owned_by"); jb_str(jb, "project-zero");
    jb_raw(jb, ",");  jb_key(jb, "vision");   jb_bool(jb, vision_enabled(ctx));
    if (ctx && ctx->cfg) {
        const Config *c = ctx->cfg;
        jb_raw(jb, ","); jb_key(jb, "context_length"); jb_int(jb, c->seq_len);
        jb_raw(jb, ","); jb_key(jb, "vocab_size");     jb_int(jb, c->vocab_size);
        jb_raw(jb, ","); jb_key(jb, "n_layers");       jb_int(jb, c->n_layers);
        jb_raw(jb, ","); jb_key(jb, "dim");            jb_int(jb, c->dim);
        jb_raw(jb, ","); jb_key(jb, "n_heads");        jb_int(jb, c->n_heads);
        jb_raw(jb, ","); jb_key(jb, "n_kv_heads");     jb_int(jb, c->n_kv_heads);
    }
    if (ctx && ctx->moe_cfg) {
        jb_raw(jb, ","); jb_key(jb, "moe"); jb_bool(jb, ctx->moe_cfg->is_moe);
        jb_raw(jb, ","); jb_key(jb, "mla"); jb_bool(jb, ctx->moe_cfg->has_mla);
    }
    jb_raw(jb, ","); jb_key(jb, "chat_template");
    jb_bool(jb, ctx && ctx->tok && ctx->tok->chat_template != NULL);
    jb_raw(jb, ","); jb_key(jb, "chat_format");
    if (ctx && ctx->tok && ctx->tok->chat_template) jb_str(jb, "embedded");
    else if (ctx && ctx->server_config.chat_format >= 0)
        jb_str(jb, chat_template_name((ChatTemplateType)ctx->server_config.chat_format));
    else jb_str(jb, chat_template_name(chat_template_detect_vocab(ctx ? ctx->tok : NULL)));
    jb_raw(jb, ","); jb_key(jb, "recurrent"); jb_bool(jb, ctx && moe_is_recurrent(ctx->moe_cfg));
}

char *api_models_json(const ApiContext *ctx) {
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{\"object\":\"list\",\"data\":[{");
    model_meta(&jb, ctx);
    jb_raw(&jb, "}]}");
    return jb_finish(&jb);
}

static void range(JsonBuf *jb, const char *key, double lo, double hi, double step) {
    jb_key(jb, key);
    jb_raw(jb, "{\"min\":"); jb_num(jb, lo);
    jb_raw(jb, ",\"max\":"); jb_num(jb, hi);
    jb_raw(jb, ",\"step\":"); jb_num(jb, step);
    jb_raw(jb, "}");
}

char *api_config_json(const ApiContext *ctx) {
    GenerationParams def;
    if (ctx) def = ctx->server_config.gen_defaults;
    else gen_params_default(&def);
    const SamplerSettings *ss = &def.sampler;
    int lim = api_max_tokens_limit(ctx);
    int vocab = (ctx && ctx->cfg) ? ctx->cfg->vocab_size : 0;
    const char *title = ctx && ctx->server_config.ui_title && ctx->server_config.ui_title[0]
                        ? ctx->server_config.ui_title : "project-zero";

    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{");
    /* The block wordmark is rendered from the (customisable) title with the
     * same font as the CLI banner; titles the font can't spell fall back to
     * text, as does --ui-logo text. */
    char logo[TN_BLOCK_FONT_ROWS * 160];
    int ascii = !(ctx && ctx->server_config.ui_logo == 1) &&
                tn_block_font_render(title, logo, sizeof(logo), 0) > 0;
    jb_key(&jb, "ui"); jb_raw(&jb, "{"); jb_key(&jb, "title"); jb_str(&jb, title);
    jb_raw(&jb, ","); jb_key(&jb, "logo"); jb_str(&jb, ascii ? "ascii" : "text");
    jb_raw(&jb, ","); jb_key(&jb, "logo_ascii"); jb_str(&jb, ascii ? logo : NULL);
    jb_raw(&jb, "},");

    jb_key(&jb, "model"); jb_raw(&jb, "{"); model_meta(&jb, ctx); jb_raw(&jb, "},");

    jb_key(&jb, "defaults"); jb_raw(&jb, "{");
    jb_key(&jb, "max_tokens");         jb_int(&jb, def.max_tokens < lim ? def.max_tokens : lim);
    jb_raw(&jb, ","); jb_key(&jb, "temperature");        jb_num(&jb, ss->temperature);
    jb_raw(&jb, ","); jb_key(&jb, "top_p");              jb_num(&jb, ss->top_p);
    jb_raw(&jb, ","); jb_key(&jb, "top_k");              jb_int(&jb, ss->top_k);
    jb_raw(&jb, ","); jb_key(&jb, "min_p");              jb_num(&jb, ss->min_p);
    jb_raw(&jb, ","); jb_key(&jb, "repetition_penalty"); jb_num(&jb, ss->repetition_penalty);
    jb_raw(&jb, ","); jb_key(&jb, "presence_penalty");   jb_num(&jb, ss->presence_penalty);
    jb_raw(&jb, ","); jb_key(&jb, "frequency_penalty");  jb_num(&jb, ss->frequency_penalty);
    jb_raw(&jb, ","); jb_key(&jb, "penalty_last_n");     jb_int(&jb, ss->penalty_last_n);
    jb_raw(&jb, ","); jb_key(&jb, "seed");               jb_int(&jb, def.seed);
    jb_raw(&jb, ","); jb_key(&jb, "json_mode");          jb_bool(&jb, def.json_mode);
    jb_raw(&jb, ","); jb_key(&jb, "ignore_eos");         jb_bool(&jb, def.ignore_eos);
    jb_raw(&jb, ","); jb_key(&jb, "cache_prompt");       jb_bool(&jb, ctx && ctx->server_config.prompt_cache);
    jb_raw(&jb, ","); jb_key(&jb, "best_of");            jb_int(&jb, ctx ? ctx->server_config.best_of : 1);
    {
        MctsOptions m;
        if (ctx) m = ctx->server_config.mcts; else memset(&m, 0, sizeof(m));
        mcts_options_defaults(&m);
        jb_raw(&jb, ","); jb_key(&jb, "mcts"); jb_raw(&jb, "{");
        jb_key(&jb, "simulations"); jb_int(&jb, m.simulations);
        jb_raw(&jb, ","); jb_key(&jb, "expand");      jb_int(&jb, m.expand);
        jb_raw(&jb, ","); jb_key(&jb, "max_depth");   jb_int(&jb, m.max_depth);
        jb_raw(&jb, ","); jb_key(&jb, "step_tokens"); jb_int(&jb, m.step_tokens);
        jb_raw(&jb, ","); jb_key(&jb, "c");           jb_num(&jb, m.c_ucb);
        jb_raw(&jb, ","); jb_key(&jb, "temperature"); jb_num(&jb, m.temperature);
        jb_raw(&jb, "}");
    }
    jb_raw(&jb, ","); jb_key(&jb, "system_prompt");      jb_str(&jb, def.system_prompt ? def.system_prompt : "");
    jb_raw(&jb, ","); jb_key(&jb, "stop"); jb_raw(&jb, "[");
    for (int i = 0; i < def.n_stop; i++) { if (i) jb_raw(&jb, ","); jb_str(&jb, def.stop[i]); }
    jb_raw(&jb, "]");
    jb_raw(&jb, ","); jb_key(&jb, "watermark"); jb_raw(&jb, "{");
    jb_key(&jb, "enabled"); jb_bool(&jb, def.watermark.enabled);
    jb_raw(&jb, ","); jb_key(&jb, "gamma"); jb_num(&jb, def.watermark.gamma);
    jb_raw(&jb, ","); jb_key(&jb, "delta"); jb_num(&jb, def.watermark.delta);
    jb_raw(&jb, ","); jb_key(&jb, "server_key"); jb_bool(&jb, def.watermark.key != 0);
    jb_raw(&jb, ","); jb_key(&jb, "z_threshold"); jb_num(&jb, WATERMARK_DETECT_Z_THRESHOLD);
    jb_raw(&jb, "}");
    jb_raw(&jb, "},");

    /* Valid ranges — mirror sampler_settings_sanitize()/watermark_config_sanitize(). */
    jb_key(&jb, "limits"); jb_raw(&jb, "{");
    range(&jb, "max_tokens", 1, lim, 1);                         jb_raw(&jb, ",");
    range(&jb, "temperature", 0, 5, 0.05);                       jb_raw(&jb, ",");
    range(&jb, "top_p", 0.01, 1, 0.01);                          jb_raw(&jb, ",");
    range(&jb, "top_k", 0, vocab > 0 ? vocab : 1000, 1);         jb_raw(&jb, ",");
    range(&jb, "min_p", 0, 0.99, 0.01);                          jb_raw(&jb, ",");
    range(&jb, "repetition_penalty", 0.5, 10, 0.05);             jb_raw(&jb, ",");
    range(&jb, "presence_penalty", -2, 2, 0.05);                 jb_raw(&jb, ",");
    range(&jb, "frequency_penalty", -2, 2, 0.05);                jb_raw(&jb, ",");
    range(&jb, "penalty_last_n", -1, lim, 1);                    jb_raw(&jb, ",");
    range(&jb, "watermark_gamma", 0.01, 0.99, 0.01);             jb_raw(&jb, ",");
    range(&jb, "watermark_delta", 0, 20, 0.1);                   jb_raw(&jb, ",");
    range(&jb, "best_of", 1, ctx ? ctx->server_config.best_of_max : BEST_OF_MAX_DEFAULT, 1); jb_raw(&jb, ",");
    range(&jb, "mcts_simulations", 0, ctx ? ctx->server_config.mcts_max : 0, 1); jb_raw(&jb, ",");
    range(&jb, "mcts_expand", 1, MCTS_MAX_EXPAND, 1); jb_raw(&jb, ",");
    range(&jb, "mcts_max_depth", 1, MCTS_MAX_DEPTH, 1); jb_raw(&jb, ",");
    range(&jb, "mcts_step_tokens", 1, MCTS_MAX_STEP_TOKENS, 1); jb_raw(&jb, ",");
    range(&jb, "mcts_c", 0.1, 10, 0.1); jb_raw(&jb, ",");
    range(&jb, "mcts_temperature", 0.05, 2, 0.05); jb_raw(&jb, ",");
    jb_key(&jb, "max_stop");     jb_int(&jb, CHAT_MAX_STOP);     jb_raw(&jb, ",");
    jb_key(&jb, "max_messages"); jb_int(&jb, CHAT_MAX_MESSAGES);
    jb_raw(&jb, "},");

    /* Read-only runtime facts (startup flags, not per-request settings). */
    {
        const CacheInfo *ci = tn_cache_info();
        jb_key(&jb, "runtime"); jb_raw(&jb, "{");
        jb_key(&jb, "simd"); jb_str(&jb, tn_simd_backend_name());
        jb_raw(&jb, ","); jb_key(&jb, "threads"); jb_int(&jb, ctx && ctx->tp ? ctx->tp->num_threads : 1);
        jb_raw(&jb, ","); jb_key(&jb, "kv_type");
        jb_str(&jb, !(ctx && ctx->run_state) ? "f32" : ctx->run_state->kv_type == KV_TYPE_I8 ? "i8"
               : ctx->run_state->kv_type == KV_TYPE_I4 ? "i4" : "f32");
        /* Phase 31: whether the KV cache is disk-backed (the path stays server-side). */
        jb_raw(&jb, ","); jb_key(&jb, "kv_offload");
        jb_raw(&jb, ctx && ctx->run_state && ctx->run_state->kv_offload_base ? "true" : "false");
        jb_raw(&jb, ","); jb_key(&jb, "attention_kernel"); jb_str(&jb, attn_kernel_name(attn_kernel_get()));
        jb_raw(&jb, ","); jb_key(&jb, "rope_layout");
        jb_str(&jb, rope_layout_name(ctx && ctx->cfg ? ctx->cfg->rope_layout : 0));
        jb_raw(&jb, ","); jb_key(&jb, "attention_tile"); jb_int(&jb, attn_tile_get());
        jb_raw(&jb, ","); jb_key(&jb, "attention_parallel_min_ctx"); jb_int(&jb, attn_parallel_min_ctx_get());
        jb_raw(&jb, ","); jb_key(&jb, "distributed"); jb_raw(&jb, "[");
        DistPlan *dp = dist_plan_active();
        for (int i = 0; dp && i < dp->n; i++) {
            if (i) jb_raw(&jb, ",");
            jb_raw(&jb, "{"); jb_key(&jb, "host"); jb_str(&jb, dp->nodes[i].host);
            jb_raw(&jb, ","); jb_key(&jb, "port"); jb_int(&jb, dp->nodes[i].port);
            jb_raw(&jb, ","); jb_key(&jb, "layers");
            char lr[32]; snprintf(lr, sizeof(lr), "%d-%d", dp->nodes[i].start, dp->nodes[i].end);
            jb_str(&jb, lr); jb_raw(&jb, "}");
        }
        jb_raw(&jb, "]");
        jb_raw(&jb, ","); jb_key(&jb, "cache"); jb_raw(&jb, "{");
        jb_key(&jb, "l1d_bytes"); jb_int(&jb, ci->l1d_bytes);
        jb_raw(&jb, ","); jb_key(&jb, "l2_bytes"); jb_int(&jb, ci->l2_bytes);
        jb_raw(&jb, ","); jb_key(&jb, "l3_bytes"); jb_int(&jb, ci->l3_bytes);
        jb_raw(&jb, ","); jb_key(&jb, "source"); jb_str(&jb, tn_cache_source_name(ci->source));
        jb_raw(&jb, "}}");
        jb_raw(&jb, ",");
    }

    jb_key(&jb, "features"); jb_raw(&jb, "{");
    jb_key(&jb, "json_mode"); jb_bool(&jb, 1);
    jb_raw(&jb, ","); jb_key(&jb, "watermark"); jb_bool(&jb, 1);
    jb_raw(&jb, ","); jb_key(&jb, "vision"); jb_bool(&jb, vision_enabled(ctx));
    jb_raw(&jb, ","); jb_key(&jb, "cancel"); jb_bool(&jb, 1);
    jb_raw(&jb, ","); jb_key(&jb, "best_of"); jb_bool(&jb, ctx && ctx->server_config.best_of_max > 1);
    jb_raw(&jb, ","); jb_key(&jb, "mcts"); jb_bool(&jb, ctx && ctx->server_config.mcts_max > 0);
    jb_raw(&jb, ","); jb_key(&jb, "prm");
    if (ctx && ctx->prm && ctx->server_config.prm_model) jb_str(&jb, ctx->server_config.prm_model);
    else jb_raw(&jb, "null");
    jb_raw(&jb, ","); jb_key(&jb, "queue_size"); jb_int(&jb, ctx ? ctx->server_config.queue_size : 0);
    /* Recurrent linear-attention state cannot resume from a partial prefix. */
    jb_raw(&jb, ","); jb_key(&jb, "prompt_cache");
    jb_bool(&jb, !(ctx && moe_is_recurrent(ctx->moe_cfg)));
    jb_raw(&jb, ","); jb_key(&jb, "prompt_store");
    jb_bool(&jb, ctx && ctx->server_config.prompt_cache_mb > 0 &&
                 (!ctx->moe_cfg || prompt_store_supported(ctx->moe_cfg)) &&
                 !(ctx->run_state && ctx->run_state->kv_type != KV_TYPE_F32));
    jb_raw(&jb, ","); jb_key(&jb, "metrics"); jb_bool(&jb, ctx && ctx->server_config.metrics.enabled);
    jb_raw(&jb, ","); jb_key(&jb, "system_prompt"); jb_bool(&jb, ctx && ctx->tok && ctx->tok->chat_template != NULL);
    jb_raw(&jb, ","); jb_key(&jb, "request_chat_template");
    jb_bool(&jb, ctx && ctx->server_config.allow_request_template);
    jb_raw(&jb, "}");
    /* The active chat template, so the web UI can show it and offer an
     * edited copy as a per-request "chat_template" override. */
    jb_raw(&jb, ","); jb_key(&jb, "chat_template"); jb_raw(&jb, "{");
    jb_key(&jb, "source");
    jb_str(&jb, ctx && ctx->server_config.chat_template_source ? ctx->server_config.chat_template_source
                : (ctx && ctx->tok && ctx->tok->chat_template ? "gguf" : "none"));
    jb_raw(&jb, ","); jb_key(&jb, "text");
    jb_str(&jb, ctx && ctx->tok && ctx->tok->chat_template ? ctx->tok->chat_template : "");
    jb_raw(&jb, ","); jb_key(&jb, "request_override");
    jb_bool(&jb, ctx && ctx->server_config.allow_request_template);
    jb_raw(&jb, ","); jb_key(&jb, "max_bytes"); jb_int(&jb, CHAT_MAX_TEMPLATE);
    jb_raw(&jb, "}");
    jb_raw(&jb, "}");
    return jb_finish(&jb);
}

char *api_prompt_cache_json(const ApiContext *ctx) {
    const PromptStore *ps = ctx ? &ctx->prompt_store : NULL;
    /* Snapshots copy F32 KV rows: unavailable for SSM/MLA/hybrid models and
     * with --kv-type i8 (the live prefix cache still applies). */
    int supported = (!(ctx && ctx->moe_cfg) || prompt_store_supported(ctx->moe_cfg)) &&
                    !(ctx && ctx->run_state && ctx->run_state->kv_type != KV_TYPE_F32);
    size_t budget = ctx ? (size_t)(ctx->server_config.prompt_cache_mb > 0
                                   ? ctx->server_config.prompt_cache_mb : 0) * 1024u * 1024u : 0;
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{"); jb_key(&jb, "enabled");   jb_bool(&jb, budget > 0 && supported);
    jb_raw(&jb, ","); jb_key(&jb, "supported"); jb_bool(&jb, supported);
    jb_raw(&jb, ","); jb_key(&jb, "entries");   jb_int(&jb, ps ? ps->n : 0);
    jb_raw(&jb, ","); jb_key(&jb, "bytes_used"); jb_int(&jb, ps ? (long long)ps->bytes_used : 0);
    jb_raw(&jb, ","); jb_key(&jb, "budget_bytes"); jb_int(&jb, (long long)budget);
    jb_raw(&jb, ","); jb_key(&jb, "hits");      jb_int(&jb, ps ? (long long)ps->hits : 0);
    jb_raw(&jb, ","); jb_key(&jb, "misses");    jb_int(&jb, ps ? (long long)ps->misses : 0);
    jb_raw(&jb, ","); jb_key(&jb, "evictions"); jb_int(&jb, ps ? (long long)ps->evictions : 0);
    jb_raw(&jb, ","); jb_key(&jb, "live_tokens"); jb_int(&jb, ctx ? ctx->prefix_cache.n : 0);
    jb_raw(&jb, "}");
    return jb_finish(&jb);
}
