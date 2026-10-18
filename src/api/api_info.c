#include "api/api_info.h"
#include "api/json_write.h"
#include "api/chat_request.h"
#include "sampling/watermark_text.h"

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

    jb_key(&jb, "features"); jb_raw(&jb, "{");
    jb_key(&jb, "json_mode"); jb_bool(&jb, 1);
    jb_raw(&jb, ","); jb_key(&jb, "watermark"); jb_bool(&jb, 1);
    jb_raw(&jb, ","); jb_key(&jb, "vision"); jb_bool(&jb, vision_enabled(ctx));
    jb_raw(&jb, ","); jb_key(&jb, "cancel"); jb_bool(&jb, 1);
    jb_raw(&jb, ","); jb_key(&jb, "metrics"); jb_bool(&jb, ctx && ctx->server_config.metrics.enabled);
    jb_raw(&jb, ","); jb_key(&jb, "system_prompt"); jb_bool(&jb, ctx && ctx->tok && ctx->tok->chat_template != NULL);
    jb_raw(&jb, "}");
    jb_raw(&jb, "}");
    return jb_finish(&jb);
}
