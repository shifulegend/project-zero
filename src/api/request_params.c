#include "api/request_params.h"
#include <string.h>

void request_params_resolve(const ChatRequest *req, const GenerationParams *defaults,
                            int max_tokens_limit, GenerationParams *out) {
    if (!out) return;
    if (defaults) *out = *defaults;
    else gen_params_default(out);
    /* Conversation pointers are per-request; never inherit them. */
    out->roles = NULL;
    out->contents = NULL;
    out->n_messages = 0;

    if (req) {
        unsigned m = req->set_mask;
        SamplerSettings *ss = &out->sampler;
        if (m & CHAT_SET_TEMPERATURE)  ss->temperature        = req->temperature;
        if (m & CHAT_SET_TOP_P)        ss->top_p              = req->top_p;
        if (m & CHAT_SET_TOP_K)        ss->top_k              = req->top_k;
        if (m & CHAT_SET_MIN_P)        ss->min_p              = req->min_p;
        if (m & CHAT_SET_REP_PENALTY)  ss->repetition_penalty = req->repetition_penalty;
        if (m & CHAT_SET_PRES_PENALTY) ss->presence_penalty   = req->presence_penalty;
        if (m & CHAT_SET_FREQ_PENALTY) ss->frequency_penalty  = req->frequency_penalty;
        if (m & CHAT_SET_PENALTY_N)    ss->penalty_last_n     = req->penalty_last_n;
        if (m & CHAT_SET_MAX_TOKENS)   out->max_tokens        = req->max_tokens;
        if (m & CHAT_SET_SEED)         out->seed              = req->seed;
        if (m & CHAT_SET_IGNORE_EOS)   out->ignore_eos        = req->ignore_eos;
        if (m & CHAT_SET_CHAT_TEMPLATE) out->chat_template    = req->chat_template;
        if (m & CHAT_SET_JSON_MODE)    out->json_mode         = req->json_mode;
        else if (req->json_mode)       out->json_mode         = true;
        if (m & CHAT_SET_STOP) {
            out->n_stop = 0;
            for (int i = 0; i < req->n_stop && out->n_stop < GEN_MAX_STOP; i++)
                out->stop[out->n_stop++] = req->stop[i];
        }
        if (m & CHAT_SET_WATERMARK) out->watermark.enabled = req->watermark_enabled;
        if (req->watermark_key)     out->watermark.key = watermark_key_from_string(req->watermark_key);
        if (m & CHAT_SET_WM_GAMMA)  out->watermark.gamma = req->watermark_gamma;
        if (m & CHAT_SET_WM_DELTA)  out->watermark.delta = req->watermark_delta;
    }

    sampler_settings_sanitize(&out->sampler);
    watermark_config_sanitize(&out->watermark);
    if (max_tokens_limit <= 0) max_tokens_limit = out->max_tokens > 0 ? out->max_tokens : 1;
    if (out->max_tokens <= 0) out->max_tokens = defaults && defaults->max_tokens > 0
                                                ? defaults->max_tokens : max_tokens_limit;
    if (out->max_tokens > max_tokens_limit) out->max_tokens = max_tokens_limit;
}
