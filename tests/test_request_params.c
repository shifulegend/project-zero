/*
 * tests/test_request_params.c — extended ChatRequest fields, \u decoding,
 * and merging a request over server defaults.
 */
#include "test_harness.h"
#include "api/chat_request.h"
#include "api/request_params.h"
#include "api/json_fields.h"

static void test_parse_extended_fields(void) {
    ChatRequest r; chat_request_init(&r);
    const char *body =
        "{\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},"
        "{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":null}],"
        "\"top_k\":40,\"min_p\":0.05,\"repetition_penalty\":1.1,\"presence_penalty\":0.5,"
        "\"frequency_penalty\":0.25,\"repeat_last_n\":32,\"seed\":1234,"
        "\"stop\":[\"\\nUser:\",\"###\"],\"temperature\":null,"
        "\"watermark\":{\"enabled\":true,\"key\":\"k\",\"gamma\":0.5,\"delta\":3}}";
    TEST_ASSERT_EQ(chat_request_parse(body, &r), TN_OK, "parses");
    TEST_ASSERT_EQ(r.num_messages, 3, "3 messages incl. null content");
    TEST_ASSERT(r.messages[2].content && r.messages[2].content[0] == '\0', "null content -> empty");
    TEST_ASSERT_EQ(r.top_k, 40, "top_k");
    TEST_ASSERT_FLOAT_EQ(r.min_p, 0.05f, 1e-6, "min_p");
    TEST_ASSERT_EQ(r.penalty_last_n, 32, "repeat_last_n alias");
    TEST_ASSERT_EQ(r.seed, 1234, "seed");
    TEST_ASSERT_EQ(r.n_stop, 2, "two stops");
    TEST_ASSERT(strcmp(r.stop[0], "\nUser:") == 0, "stop escape decoded");
    TEST_ASSERT(!(r.set_mask & CHAT_SET_TEMPERATURE), "null temperature = unset");
    TEST_ASSERT(r.watermark_enabled && r.watermark_key && strcmp(r.watermark_key, "k") == 0, "watermark");
    TEST_ASSERT(r.set_mask & CHAT_SET_WM_GAMMA, "gamma set");
    chat_request_free(&r);
}

static void test_stop_string_form(void) {
    ChatRequest r; chat_request_init(&r);
    TEST_ASSERT_EQ(chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stop\":\"END\"}", &r),
                   TN_OK, "parses");
    TEST_ASSERT(r.n_stop == 1 && strcmp(r.stop[0], "END") == 0, "string stop");
    chat_request_free(&r);
}

static void test_unicode_escape_decoding(void) {
    ChatRequest r; chat_request_init(&r);
    /* é (U+00E9), नम (U+0928 U+092E), 😀 (surrogate pair) */
    const char *body = "{\"messages\":[{\"role\":\"user\",\"content\":\"caf\\u00e9 \\u0928\\u092e \\ud83d\\ude00\"}]}";
    TEST_ASSERT_EQ(chat_request_parse(body, &r), TN_OK, "parses");
    TEST_ASSERT(strcmp(r.messages[0].content, "caf\xc3\xa9 \xe0\xa4\xa8\xe0\xa4\xae \xf0\x9f\x98\x80") == 0,
                "\\u escapes decode to UTF-8");
    chat_request_free(&r);
}

static void test_resolve_defaults_and_overrides(void) {
    GenerationParams def;
    gen_params_default(&def);
    def.max_tokens = 300;
    def.sampler.temperature = 0.3f;
    def.sampler.top_k = 20;
    def.system_prompt = "server prompt";
    def.watermark.key = 77;

    ChatRequest r; chat_request_init(&r);
    chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                       "\"top_k\":5,\"max_tokens\":99999,\"watermark\":true}", &r);
    GenerationParams gp;
    request_params_resolve(&r, &def, 1024, &gp);
    TEST_ASSERT_FLOAT_EQ(gp.sampler.temperature, 0.3f, 1e-6, "absent field -> server default");
    TEST_ASSERT_EQ(gp.sampler.top_k, 5, "client override wins");
    TEST_ASSERT_EQ(gp.max_tokens, 1024, "max_tokens capped by limit");
    TEST_ASSERT(gp.watermark.enabled && gp.watermark.key == 77, "watermark uses server key");
    TEST_ASSERT(strcmp(gp.system_prompt, "server prompt") == 0, "server system prompt inherited");
    chat_request_free(&r);

    chat_request_init(&r);
    chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":99}", &r);
    request_params_resolve(&r, &def, 1024, &gp);
    TEST_ASSERT_FLOAT_EQ(gp.sampler.temperature, 5.0f, 1e-6, "out-of-range clamped");
    TEST_ASSERT_EQ(gp.max_tokens, 300, "default max_tokens kept");
    chat_request_free(&r);
}

static void test_cache_prompt_field(void) {
    ChatRequest r; chat_request_init(&r);
    TEST_ASSERT_EQ(chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"cache_prompt\":false}", &r),
                   TN_OK, "parses");
    TEST_ASSERT((r.set_mask & CHAT_SET_CACHE_PROMPT) && !r.cache_prompt, "cache_prompt=false recorded");
    chat_request_free(&r);
    chat_request_init(&r);
    chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", &r);
    TEST_ASSERT(!(r.set_mask & CHAT_SET_CACHE_PROMPT), "absent -> server default");
    chat_request_free(&r);
}

static void test_json_fields(void) {
    const char *b = "{\"a\":{\"x\":1},\"text\":\"hi\\nthere\",\"gamma\":0.3,\"flag\":true}";
    char *t = json_field_string(b, "text");
    TEST_ASSERT(t && strcmp(t, "hi\nthere") == 0, "string field");
    free(t);
    double v = 0;
    TEST_ASSERT(json_field_number(b, "gamma", &v) && fabs(v - 0.3) < 1e-9, "number field");
    int f = 0;
    TEST_ASSERT(json_field_bool(b, "flag", &f) && f == 1, "bool field");
    TEST_ASSERT(json_field_string(b, "missing") == NULL, "missing -> NULL");
    TEST_ASSERT(!json_field_number(b, "text", &v), "type mismatch -> 0");
}

int main(void) {
    printf("=== Request Params Tests ===\n");
    RUN_TEST(test_parse_extended_fields);
    RUN_TEST(test_stop_string_form);
    RUN_TEST(test_unicode_escape_decoding);
    RUN_TEST(test_resolve_defaults_and_overrides);
    RUN_TEST(test_cache_prompt_field);
    RUN_TEST(test_json_fields);
    TEST_SUMMARY();
}
