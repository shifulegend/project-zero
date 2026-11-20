#include "api/api_info.h"

#include <stdio.h>
#include <string.h>

#include "api/chat_compile.h"
#include "api/chat_request.h"
#include "api/config_endpoint.h"
#include "api/json_write.h"
#include "brand/block_font.h"
#include "distributed/dist_plan.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "reasoning/best_of.h"
#include "runtime/hardware_profile.h"
#include "sampling/watermark_text.h"
#include "threading/cache_probe.h"
#include "transformer/attn_tiled.h"

int api_max_tokens_limit(const ApiContext *ctx) {
    int ctx_len = (ctx && ctx->cfg) ? ctx->cfg->seq_len : 0;
    int lim     = ctx ? ctx->server_config.limits.max_tokens_limit : 0;
    if (lim > 0 && (ctx_len <= 0 || lim < ctx_len)) return lim;
    return ctx_len > 0 ? ctx_len : 4096;
}

static const char *model_id(const ApiContext *ctx) {
    const char *id = ctx ? ctx->server_config.model_id : NULL;
    return (id && id[0]) ? id : "local-model";
}

int api_vision_enabled(const ApiContext *ctx) {
    return ctx && ctx->server_config.vision.vision_path && ctx->server_config.vision.proj_path;
}

void api_model_meta(JsonBuf *jb, const ApiContext *ctx) {
    jb_key(jb, "id");
    jb_str(jb, model_id(ctx));
    jb_raw(jb, ",");
    jb_key(jb, "object");
    jb_str(jb, "model");
    jb_raw(jb, ",");
    jb_key(jb, "owned_by");
    jb_str(jb, "project-zero");
    jb_raw(jb, ",");
    jb_key(jb, "vision");
    jb_bool(jb, api_vision_enabled(ctx));
    if (ctx && ctx->cfg) {
        const Config *c = ctx->cfg;
        jb_raw(jb, ",");
        jb_key(jb, "context_length");
        jb_int(jb, c->seq_len);
        jb_raw(jb, ",");
        jb_key(jb, "vocab_size");
        jb_int(jb, c->vocab_size);
        jb_raw(jb, ",");
        jb_key(jb, "n_layers");
        jb_int(jb, c->n_layers);
        jb_raw(jb, ",");
        jb_key(jb, "dim");
        jb_int(jb, c->dim);
        jb_raw(jb, ",");
        jb_key(jb, "n_heads");
        jb_int(jb, c->n_heads);
        jb_raw(jb, ",");
        jb_key(jb, "n_kv_heads");
        jb_int(jb, c->n_kv_heads);
    }
    if (ctx && ctx->moe_cfg) {
        jb_raw(jb, ",");
        jb_key(jb, "moe");
        jb_bool(jb, ctx->moe_cfg->is_moe);
        jb_raw(jb, ",");
        jb_key(jb, "mla");
        jb_bool(jb, ctx->moe_cfg->mla.enabled);
    }
    jb_raw(jb, ",");
    jb_key(jb, "chat_template");
    jb_bool(jb, ctx && ctx->tok && ctx->tok->chat_template != NULL);
    jb_raw(jb, ",");
    jb_key(jb, "chat_format");
    if (ctx && ctx->tok && ctx->tok->chat_template)
        jb_str(jb, "embedded");
    else if (ctx && ctx->server_config.chat.chat_format >= 0)
        jb_str(jb, chat_template_name((ChatTemplateType)ctx->server_config.chat.chat_format));
    else
        jb_str(jb, chat_template_name(chat_template_detect_vocab(ctx ? ctx->tok : NULL)));
    jb_raw(jb, ",");
    jb_key(jb, "recurrent");
    jb_bool(jb, ctx && moe_is_recurrent(ctx->moe_cfg));
}

char *api_models_json(const ApiContext *ctx) {
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{\"object\":\"list\",\"data\":[{");
    api_model_meta(&jb, ctx);
    jb_raw(&jb, "}]}");
    return jb_finish(&jb);
}

char *api_prompt_cache_json(const ApiContext *ctx) {
    const PromptStore *ps = ctx ? &ctx->prompt_store : NULL;
    /* Snapshots copy F32 KV rows: unavailable for SSM/MLA/hybrid models and
     * with --kv-type i8 (the live prefix cache still applies). */
    int supported = (!(ctx && ctx->moe_cfg) || prompt_store_supported(ctx->moe_cfg)) &&
                    !(ctx && ctx->run_state && ctx->run_state->kvs.type != KV_TYPE_F32);
    size_t budget = ctx ? (size_t)(ctx->server_config.cache.prompt_cache_mb > 0
                                       ? ctx->server_config.cache.prompt_cache_mb
                                       : 0) *
                              1024u * 1024u
                        : 0;
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{");
    jb_key(&jb, "enabled");
    jb_bool(&jb, budget > 0 && supported);
    jb_raw(&jb, ",");
    jb_key(&jb, "supported");
    jb_bool(&jb, supported);
    jb_raw(&jb, ",");
    jb_key(&jb, "entries");
    jb_int(&jb, ps ? ps->n : 0);
    jb_raw(&jb, ",");
    jb_key(&jb, "bytes_used");
    jb_int(&jb, ps ? (long long)ps->bytes_used : 0);
    jb_raw(&jb, ",");
    jb_key(&jb, "budget_bytes");
    jb_int(&jb, (long long)budget);
    jb_raw(&jb, ",");
    jb_key(&jb, "hits");
    jb_int(&jb, ps ? (long long)ps->hits : 0);
    jb_raw(&jb, ",");
    jb_key(&jb, "misses");
    jb_int(&jb, ps ? (long long)ps->misses : 0);
    jb_raw(&jb, ",");
    jb_key(&jb, "evictions");
    jb_int(&jb, ps ? (long long)ps->evictions : 0);
    jb_raw(&jb, ",");
    jb_key(&jb, "live_tokens");
    jb_int(&jb, ctx ? ctx->prefix_cache.n : 0);
    jb_raw(&jb, "}");
    return jb_finish(&jb);
}
