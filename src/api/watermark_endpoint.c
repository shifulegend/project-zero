#include "api/watermark_endpoint.h"
#include "api/json_fields.h"
#include "api/json_write.h"
#include "sampling/watermark_text.h"
#include <stdlib.h>
#include <string.h>

static char *error_body(const char *msg) {
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{\"error\":{\"message\":");
    jb_str(&jb, msg);
    jb_raw(&jb, ",\"type\":\"invalid_request_error\"}}");
    return jb_finish(&jb);
}

char *api_watermark_detect(const ApiContext *ctx, const char *json_body, int *status) {
    int st_dummy;
    if (!status) status = &st_dummy;
    if (!ctx || !ctx->tok) { *status = 500; return error_body("No tokenizer loaded"); }

    char *text = json_field_string(json_body, "text");
    if (!text) { *status = 400; return error_body("Missing 'text' string field"); }

    WatermarkConfig wm = ctx->server_config.gen_defaults.watermark;
    char *key = json_field_string(json_body, "key");
    if (key) { wm.key = watermark_key_from_string(key); free(key); }
    double v;
    if (json_field_number(json_body, "gamma", &v)) wm.gamma = (float)v;
    watermark_config_sanitize(&wm);
    double z_thr = WATERMARK_DETECT_Z_THRESHOLD;
    if (json_field_number(json_body, "z_threshold", &v) && v > 0.0) z_thr = v;

    WatermarkResult wr;
    TernaryError err = watermark_detect_text(&wm, ctx->tok, text, &wr);
    free(text);
    if (err != TN_OK) { *status = 500; return error_body("Tokenization failed"); }

    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{"); jb_key(&jb, "tokens_scored"); jb_int(&jb, wr.n_scored);
    jb_raw(&jb, ","); jb_key(&jb, "green_tokens");   jb_int(&jb, wr.n_green);
    jb_raw(&jb, ","); jb_key(&jb, "green_fraction"); jb_num(&jb, wr.green_fraction);
    jb_raw(&jb, ","); jb_key(&jb, "gamma");          jb_num(&jb, wm.gamma);
    jb_raw(&jb, ","); jb_key(&jb, "z_score");        jb_num(&jb, wr.z_score);
    jb_raw(&jb, ","); jb_key(&jb, "z_threshold");    jb_num(&jb, z_thr);
    jb_raw(&jb, ","); jb_key(&jb, "watermarked");    jb_bool(&jb, wr.n_scored > 0 && wr.z_score >= z_thr);
    jb_raw(&jb, "}");
    char *out = jb_finish(&jb);
    *status = out ? 200 : 500;
    return out;
}
