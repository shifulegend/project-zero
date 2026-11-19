/*
 * api/selfcheck_endpoint.c — see api/selfcheck_endpoint.h.
 */
#include "api/selfcheck_endpoint.h"
#include "api/json_fields.h"
#include "api/json_write.h"
#include "eval/selfcheck.h"
#include <stdlib.h>

static double number_or(const char *body, const char *key, double dflt) {
    char *t = body ? json_field_number_text(body, key) : NULL;
    if (!t) return dflt;
    double v = atof(t);
    free(t);
    return v;
}

static char *error_body(const char *msg, int code, int *status) {
    *status = code;
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{\"error\":{\"message\":");
    jb_str(&jb, msg);
    jb_raw(&jb, ",\"type\":");
    jb_str(&jb, code >= 500 ? "server_error" : "invalid_request_error");
    jb_raw(&jb, "}}");
    return jb_finish(&jb);
}

char *api_selfcheck(ApiContext *ctx, const char *json_body, int *status) {
    if (!ctx || !ctx->cfg || !ctx->weights || !ctx->run_state || !ctx->tok)
        return error_body("no model loaded", 500, status);
    SelfCheckOptions o;
    o.tolerance_pct = number_or(json_body, "tolerance", ctx->server_config.quality_tolerance);
    o.n_tokens = (int)number_or(json_body, "tokens", ctx->server_config.self_check_tokens);
    if (o.tolerance_pct < 0 || o.tolerance_pct > 100)
        return error_body("\"tolerance\" must be 0..100 (percent)", 400, status);
    if (o.n_tokens < 16 || o.n_tokens > 4096)
        return error_body("\"tokens\" must be 16..4096", 400, status);

    SelfCheckResult r;
    TernaryError e = selfcheck_run(&o, ctx->cfg, ctx->weights, ctx->run_state, ctx->moe_cfg,
                                   ctx->tok, ctx->tp, &r);
    if (e != TN_OK) return error_body("the accuracy probe failed to run", 500, status);

    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{"); jb_key(&jb, "verdict"); jb_str(&jb, selfcheck_verdict_name(r.verdict));
    jb_raw(&jb, ","); jb_key(&jb, "act_quant"); jb_str(&jb, r.act_quant);
    jb_raw(&jb, ","); jb_key(&jb, "classifier"); jb_str(&jb, r.classifier);
    jb_raw(&jb, ","); jb_key(&jb, "perplexity"); jb_raw(&jb, "{");
    jb_key(&jb, "reference"); jb_num(&jb, r.ppl_ref);
    jb_raw(&jb, ","); jb_key(&jb, "configured"); jb_num(&jb, r.ppl_cand);
    jb_raw(&jb, "}");
    jb_raw(&jb, ","); jb_key(&jb, "delta_pct"); jb_num(&jb, r.delta_pct);
    jb_raw(&jb, ","); jb_key(&jb, "tolerance_pct"); jb_num(&jb, r.tolerance_pct);
    jb_raw(&jb, ","); jb_key(&jb, "tokens_scored"); jb_int(&jb, r.n_scored);
    jb_raw(&jb, ","); jb_key(&jb, "seconds"); jb_num(&jb, r.seconds);
    jb_raw(&jb, ","); jb_key(&jb, "note"); jb_str(&jb, r.note);
    jb_raw(&jb, "}");
    *status = 200;
    return jb_finish(&jb);
}
