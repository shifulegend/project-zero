#include "api/api_info.h"
#include "api/json_write.h"
#include "api/chat_request.h"
#include "sampling/watermark_text.h"
#include "threading/cache_probe.h"
#include "transformer/attn_tiled.h"
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
    jb_key(&jb, "ui"); jb_raw(&jb, "{"); jb_key(&jb, "title"); jb_str(&jb, title); jb_raw(&jb, "},");

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
    jb_raw(&jb, ","); jb_key(&jb, "cache_prompt");       jb_bool(&jb, ctx && ctx->server_config.prompt_cache);
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
    jb_key(&jb, "max_stop");     jb_int(&jb, CHAT_MAX_STOP);     jb_raw(&jb, ",");
    jb_key(&jb, "max_messages"); jb_int(&jb, CHAT_MAX_MESSAGES);
    jb_raw(&jb, "},");

    /* Read-only runtime facts (startup flags, not per-request settings). */
    {
        const CacheInfo *ci = tn_cache_info();
        jb_key(&jb, "runtime"); jb_raw(&jb, "{");
        jb_key(&jb, "simd"); jb_str(&jb, tn_simd_backend_name());
        jb_raw(&jb, ","); jb_key(&jb, "threads"); jb_int(&jb, ctx && ctx->tp ? ctx->tp->num_threads : 1);
        jb_raw(&jb, ","); jb_key(&jb, "attention_kernel"); jb_str(&jb, attn_kernel_name(attn_kernel_get()));
        jb_raw(&jb, ","); jb_key(&jb, "attention_tile"); jb_int(&jb, attn_tile_get());
        jb_raw(&jb, ","); jb_key(&jb, "attention_parallel_min_ctx"); jb_int(&jb, attn_parallel_min_ctx_get());
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
    /* Recurrent linear-attention state cannot resume from a partial prefix. */
    jb_raw(&jb, ","); jb_key(&jb, "prompt_cache");
    jb_bool(&jb, !(ctx && ctx->moe_cfg && ctx->moe_cfg->has_linear_attn));
    jb_raw(&jb, ","); jb_key(&jb, "prompt_store");
    jb_bool(&jb, ctx && ctx->server_config.prompt_cache_mb > 0 &&
                 (!ctx->moe_cfg || prompt_store_supported(ctx->moe_cfg)));
    jb_raw(&jb, ","); jb_key(&jb, "metrics"); jb_bool(&jb, ctx && ctx->server_config.metrics.enabled);
    jb_raw(&jb, ","); jb_key(&jb, "system_prompt"); jb_bool(&jb, ctx && ctx->tok && ctx->tok->chat_template != NULL);
    jb_raw(&jb, "}");
    jb_raw(&jb, "}");
    return jb_finish(&jb);
}

char *api_prompt_cache_json(const ApiContext *ctx) {
    const PromptStore *ps = ctx ? &ctx->prompt_store : NULL;
    int supported = !(ctx && ctx->moe_cfg) || prompt_store_supported(ctx->moe_cfg);
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
