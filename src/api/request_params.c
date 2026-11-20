#include "api/request_params.h"

#include <string.h>

void request_params_resolve(const ChatRequest *req, const GenerationParams *defaults,
                            int max_tokens_limit, GenerationParams *out) {
    if (!out) return;
    if (defaults)
        *out = *defaults;
    else
        gen_params_default(out);
    /* Conversation pointers are per-request; never inherit them. */
    out->roles      = NULL;
    out->contents   = NULL;
    out->n_messages = 0;

    if (req) {
        unsigned m          = req->set_mask;
        SamplerSettings *ss = &out->sampler;
        if (m & CHAT_SET_TEMPERATURE) ss->temperature = req->sampling.temperature;
        if (m & CHAT_SET_TOP_P) ss->top_p = req->sampling.top_p;
        if (m & CHAT_SET_TOP_K) ss->top_k = req->sampling.top_k;
        if (m & CHAT_SET_MIN_P) ss->min_p = req->sampling.min_p;
        if (m & CHAT_SET_REP_PENALTY) ss->repetition_penalty = req->sampling.repetition_penalty;
        if (m & CHAT_SET_PRES_PENALTY) ss->presence_penalty = req->sampling.presence_penalty;
        if (m & CHAT_SET_FREQ_PENALTY) ss->frequency_penalty = req->sampling.frequency_penalty;
        if (m & CHAT_SET_PENALTY_N) ss->penalty_last_n = req->sampling.penalty_last_n;
        if (m & CHAT_SET_MAX_TOKENS) out->max_tokens = req->output.max_tokens;
        if (m & CHAT_SET_SEED) out->seed = req->sampling.seed;
        if (m & CHAT_SET_IGNORE_EOS) out->ignore_eos = req->output.ignore_eos;
        if (m & CHAT_SET_CHAT_TEMPLATE) out->chat_template = req->chat_template;
        if (m & CHAT_SET_JSON_MODE)
            out->json_mode = req->output.json_mode;
        else if (req->output.json_mode)
            out->json_mode = true;
        if (m & CHAT_SET_STOP) {
            out->n_stop = 0;
            for (int i = 0; i < req->output.n_stop && out->n_stop < GEN_MAX_STOP; i++)
                out->stop[out->n_stop++] = req->output.stop[i];
        }
        if (m & CHAT_SET_WATERMARK) out->watermark.enabled = req->watermark.enabled;
        if (req->watermark.key) out->watermark.key = watermark_key_from_string(req->watermark.key);
        if (m & CHAT_SET_WM_GAMMA) out->watermark.gamma = req->watermark.gamma;
        if (m & CHAT_SET_WM_DELTA) out->watermark.delta = req->watermark.delta;
    }

    sampler_settings_sanitize(&out->sampler);
    watermark_config_sanitize(&out->watermark);
    if (max_tokens_limit <= 0) max_tokens_limit = out->max_tokens > 0 ? out->max_tokens : 1;
    if (out->max_tokens <= 0)
        out->max_tokens =
            defaults && defaults->max_tokens > 0 ? defaults->max_tokens : max_tokens_limit;
    if (out->max_tokens > max_tokens_limit) out->max_tokens = max_tokens_limit;
}

void request_mcts_resolve(const ChatRequest *req, const MctsOptions *defaults, int sim_cap,
                          MctsOptions *out) {
    if (!out) return;
    if (defaults)
        *out = *defaults;
    else
        memset(out, 0, sizeof(*out));
    if (req && (req->set_mask & CHAT_SET_MCTS)) {
        const MctsOptions *r = &req->search.mcts;
        out->simulations     = r->simulations; /* explicit 0 turns search off */
        if (r->expand > 0) out->expand = r->expand;
        if (r->max_depth > 0) out->max_depth = r->max_depth;
        if (r->step_tokens > 0) out->step_tokens = r->step_tokens;
        if (r->c_ucb > 0.0f) out->c_ucb = r->c_ucb;
        if (r->temperature > 0.0f) out->temperature = r->temperature;
    }
    mcts_options_clamp(out, sim_cap < 0 ? 0 : sim_cap);
}

const char *chat_request_error_message(TernaryError err) {
    if (err == TN_ERR_INVALID_ARGS)
        return "Unsupported value in request (response_format.type must be 'text' or "
               "'json_object'; chat_template is capped at 64 KiB)";
    return "Invalid JSON in request body";
}
