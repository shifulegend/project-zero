/*
 * tests/test_api_info.c — GET /v1/models + GET /v1/config bodies and the
 * watermark detect endpoint, built from an ApiContext without sockets.
 */
#include "test_harness.h"
#include "api/api_info.h"
#include "api/watermark_endpoint.h"
#include "api/server_config.h"

static void make_ctx(ApiContext *ctx, Config *cfg) {
    memset(ctx, 0, sizeof(*ctx));
    memset(cfg, 0, sizeof(*cfg));
    cfg->seq_len = 2048;
    cfg->vocab_size = 32000;
    cfg->n_layers = 4;
    ctx->cfg = cfg;
    server_config_init(&ctx->server_config);
}

static void test_models_reports_metadata(void) {
    ApiContext ctx; Config cfg;
    make_ctx(&ctx, &cfg);
    ctx.server_config.model_id = strdup("tiny.gguf");
    char *j = api_models_json(&ctx);
    TEST_ASSERT(j != NULL, "json built");
    TEST_ASSERT(strstr(j, "\"id\":\"tiny.gguf\"") != NULL, "model id from server config");
    TEST_ASSERT(strstr(j, "\"vision\":false") != NULL, "vision flag present");
    TEST_ASSERT(strstr(j, "\"context_length\":2048") != NULL, "context length");
    TEST_ASSERT(strstr(j, "\"chat_format\":\"plain\"") != NULL, "no tokenizer -> plain format");
    TEST_ASSERT(strstr(j, "\"recurrent\":false") != NULL, "dense is not recurrent");
    free(j);
    ctx.server_config.vision_path = strdup("v.bin");
    ctx.server_config.proj_path = strdup("p.bin");
    j = api_models_json(&ctx);
    TEST_ASSERT(strstr(j, "\"vision\":true") != NULL, "vision true when configured");
    free(j);
    server_config_free(&ctx.server_config);
}

static void test_config_reflects_defaults_and_limits(void) {
    ApiContext ctx; Config cfg;
    make_ctx(&ctx, &cfg);
    ctx.server_config.gen_defaults.max_tokens = 4096;           /* > limit: clamped */
    ctx.server_config.gen_defaults.sampler.temperature = 0.25f;
    ctx.server_config.gen_defaults.sampler.top_k = 33;
    ctx.server_config.gen_defaults.system_prompt = "Say \"hi\"";
    ctx.server_config.gen_defaults.stop[0] = "###";
    ctx.server_config.gen_defaults.n_stop = 1;
    ctx.server_config.max_tokens_limit = 1000;
    ctx.server_config.ui_title = strdup("acme-zero");
    char *j = api_config_json(&ctx);
    TEST_ASSERT(j != NULL, "json built");
    TEST_ASSERT(strstr(j, "\"title\":\"acme-zero\"") != NULL, "ui title");
    /* "acme" has letters the block font lacks -> text fallback, no ascii art. */
    TEST_ASSERT(strstr(j, "\"logo\":\"text\"") != NULL && strstr(j, "\"logo_ascii\":null") != NULL,
                "unspellable title falls back to text logo");
    TEST_ASSERT(strstr(j, "\"max_tokens\":1000") != NULL, "default clamped to limit");
    TEST_ASSERT(strstr(j, "\"temperature\":0.25") != NULL, "temperature default");
    TEST_ASSERT(strstr(j, "\"top_k\":33") != NULL, "top_k default");
    TEST_ASSERT(strstr(j, "\"system_prompt\":\"Say \\\"hi\\\"\"") != NULL, "escaped system prompt");
    TEST_ASSERT(strstr(j, "\"stop\":[\"###\"]") != NULL, "stop list");
    TEST_ASSERT(strstr(j, "\"cache_prompt\":true") != NULL, "prompt cache on by default");
    TEST_ASSERT(strstr(j, "\"max_tokens\":{\"min\":1,\"max\":1000") != NULL, "max_tokens range");
    TEST_ASSERT(strstr(j, "\"top_k\":{\"min\":0,\"max\":32000") != NULL, "top_k range = vocab");
    TEST_ASSERT_EQ(api_max_tokens_limit(&ctx), 1000, "explicit limit");
    free(ctx.server_config.ui_title);
    ctx.server_config.ui_title = strdup("project-zero");
    char *jl = api_config_json(&ctx);
    TEST_ASSERT(jl && strstr(jl, "\"logo\":\"ascii\"") && strstr(jl, "\"logo_ascii\":\"####  ####"),
                "default title -> ascii wordmark");
    free(jl);
    ctx.server_config.ui_logo = 1;
    jl = api_config_json(&ctx);
    TEST_ASSERT(jl && strstr(jl, "\"logo\":\"text\""), "--ui-logo text");
    free(jl);
    ctx.server_config.ui_logo = 0;
    ctx.server_config.max_tokens_limit = 0;
    TEST_ASSERT_EQ(api_max_tokens_limit(&ctx), 2048, "falls back to context length");
    free(j);
    server_config_free(&ctx.server_config);
}

static void test_prompt_cache_json(void) {
    ApiContext ctx; Config cfg;
    make_ctx(&ctx, &cfg);
    ctx.server_config.prompt_cache_mb = 8;
    char *j = api_prompt_cache_json(&ctx);
    TEST_ASSERT(j && strstr(j, "\"enabled\":true") && strstr(j, "\"budget_bytes\":8388608"), "budget reported");
    TEST_ASSERT(strstr(j, "\"entries\":0") != NULL, "empty store");
    free(j);
    ctx.server_config.prompt_cache_mb = 0;
    j = api_prompt_cache_json(&ctx);
    TEST_ASSERT(strstr(j, "\"enabled\":false") != NULL, "0 MB = disabled");
    free(j);
    server_config_free(&ctx.server_config);
}

static void test_config_chat_template(void) {
    ApiContext ctx; Config cfg;
    make_ctx(&ctx, &cfg);
    Tokenizer tok; memset(&tok, 0, sizeof(tok));
    tok.chat_template = (char *)"{{ bos_token }}User: \"q\"";
    ctx.tok = &tok;
    ctx.server_config.chat_template_source = "file";
    char *j = api_config_json(&ctx);
    TEST_ASSERT(j && strstr(j, "\"request_chat_template\":true") != NULL, "override allowed by default");
    TEST_ASSERT(strstr(j, "\"source\":\"file\"") != NULL, "template source published");
    TEST_ASSERT(strstr(j, "\"text\":\"{{ bos_token }}User: \\\"q\\\"\"") != NULL, "template text JSON-escaped");
    TEST_ASSERT(strstr(j, "\"max_bytes\":65536") != NULL, "size cap published");
    free(j);
    ctx.server_config.allow_request_template = 0;
    j = api_config_json(&ctx);
    TEST_ASSERT(j && strstr(j, "\"request_override\":false") != NULL, "--no-request-template published");
    free(j);
    server_config_free(&ctx.server_config);
}

static void test_watermark_endpoint_errors(void) {
    ApiContext ctx; Config cfg;
    make_ctx(&ctx, &cfg);
    int status = 0;
    char *j = api_watermark_detect(&ctx, "{\"text\":\"x\"}", &status);
    TEST_ASSERT_EQ(status, 500, "no tokenizer -> 500");
    free(j);
    Tokenizer tok; memset(&tok, 0, sizeof(tok));
    ctx.tok = &tok;
    j = api_watermark_detect(&ctx, "{\"nope\":1}", &status);
    TEST_ASSERT_EQ(status, 400, "missing text -> 400");
    TEST_ASSERT(j && strstr(j, "error") != NULL, "error body");
    free(j);
    server_config_free(&ctx.server_config);
}

int main(void) {
    printf("=== API Info Tests ===\n");
    RUN_TEST(test_models_reports_metadata);
    RUN_TEST(test_config_reflects_defaults_and_limits);
    RUN_TEST(test_prompt_cache_json);
    RUN_TEST(test_watermark_endpoint_errors);
    RUN_TEST(test_config_chat_template);
    TEST_SUMMARY();
}
