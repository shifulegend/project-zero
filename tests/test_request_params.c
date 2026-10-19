/*
 * tests/test_request_params.c — extended ChatRequest fields, \u decoding,
 * and merging a request over server defaults.
 */
#include "test_harness.h"
#include "api/chat_request.h"
#include "api/request_params.h"
#include "api/json_fields.h"
#include "api/chat_compile.h"
#include "reasoning/best_of.h"

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

static void test_chat_template_field(void) {
    const char *M = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]";
    char body[256];
    ChatRequest r; chat_request_init(&r);
    snprintf(body, sizeof(body), "%s,\"chat_template\":\"{{ messages[0].content }}\",\"top_k\":3}", M);
    TEST_ASSERT_EQ(chat_request_parse(body, &r), TN_OK, "template parses");
    TEST_ASSERT((r.set_mask & CHAT_SET_CHAT_TEMPLATE) && r.chat_template &&
                strcmp(r.chat_template, "{{ messages[0].content }}") == 0, "template recorded");
    TEST_ASSERT((r.set_mask & CHAT_SET_TOP_K) && r.top_k == 3, "fields after it still parse");
    GenerationParams def, out;
    gen_params_default(&def);
    request_params_resolve(&r, &def, 4096, &out);
    TEST_ASSERT(out.chat_template == r.chat_template, "resolved into GenerationParams");
    chat_request_free(&r);
    TEST_ASSERT(r.chat_template == NULL, "freed");

    const char *empties[] = { "\"\"", "null" };
    for (int i = 0; i < 2; i++) {
        chat_request_init(&r);
        snprintf(body, sizeof(body), "%s,\"chat_template\":%s,\"top_k\":4}", M, empties[i]);
        TEST_ASSERT_EQ(chat_request_parse(body, &r), TN_OK, "empty/null parses");
        TEST_ASSERT(!(r.set_mask & CHAT_SET_CHAT_TEMPLATE) && r.chat_template == NULL && r.top_k == 4,
                    "empty/null = server template");
        request_params_resolve(&r, &def, 4096, &out);
        TEST_ASSERT(out.chat_template == NULL, "no override resolved");
        chat_request_free(&r);
    }

    size_t n = CHAT_MAX_TEMPLATE + 64;
    char *big = (char *)malloc(n + 128);
    int k = snprintf(big, n + 128, "%s,\"chat_template\":\"", M);
    memset(big + k, 'a', CHAT_MAX_TEMPLATE + 1);
    strcpy(big + k + CHAT_MAX_TEMPLATE + 1, "\"}");
    chat_request_init(&r);
    TEST_ASSERT_EQ(chat_request_parse(big, &r), TN_ERR_INVALID_ARGS, "over CHAT_MAX_TEMPLATE refused");
    chat_request_free(&r);
    free(big);
}

static void test_ignore_eos_field(void) {
    ChatRequest r; chat_request_init(&r);
    TEST_ASSERT_EQ(chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"ignore_eos\":true}", &r),
                   TN_OK, "parses");
    TEST_ASSERT((r.set_mask & CHAT_SET_IGNORE_EOS) && r.ignore_eos, "ignore_eos=true recorded");
    GenerationParams def, out;
    gen_params_default(&def);
    request_params_resolve(&r, &def, 4096, &out);
    TEST_ASSERT(out.ignore_eos, "request overrides the server default");
    chat_request_free(&r);
    chat_request_init(&r);
    chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", &r);
    def.ignore_eos = true;
    request_params_resolve(&r, &def, 4096, &out);
    TEST_ASSERT(out.ignore_eos, "absent -> server default (--ignore-eos)");
    chat_request_free(&r);
}

static void test_chat_format_choice(void) {
    TEST_ASSERT_EQ(chat_template_parse("auto"), -1, "auto");
    TEST_ASSERT_EQ(chat_template_parse("plain"), CHAT_TMPL_RAW, "plain");
    TEST_ASSERT_EQ(chat_template_parse("llama3"), CHAT_TMPL_LLAMA3, "llama3");
    TEST_ASSERT_EQ(chat_template_parse("bogus"), -2, "unknown rejected");
    TEST_ASSERT_EQ(chat_template_detect_vocab(NULL), CHAT_TMPL_RAW, "no tokenizer -> plain");
    TEST_ASSERT(strcmp(chat_template_name(CHAT_TMPL_RAW), "plain") == 0, "name");
    ChatRequest r; chat_request_init(&r);
    chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &r);
    char out[256];
    TEST_ASSERT_EQ(chat_compile_prompt(&r, CHAT_TMPL_RAW, out, sizeof(out)), TN_OK, "compile plain");
    TEST_ASSERT(strcmp(out, "user: hi\nassistant:") == 0, "plain format ends with the assistant cue");
    chat_request_free(&r);
}

static void test_best_of_parse_and_score(void) {
    ChatRequest r; chat_request_init(&r);
    chat_request_parse("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"best_of\":3}", &r);
    TEST_ASSERT((r.set_mask & CHAT_SET_BEST_OF) && r.best_of == 3, "best_of parsed");
    chat_request_free(&r);
    GenerationStats a = {0}, b = {0};
    a.completion_tokens = 10; a.sum_logprob = -10.0;   /* mean -1.0 */
    b.completion_tokens = 2;  b.sum_logprob = -3.0;    /* mean -1.5, sum -3 */
    TEST_ASSERT(best_of_score(&a, BEST_OF_SCORE_MEAN_LOGPROB) > best_of_score(&b, BEST_OF_SCORE_MEAN_LOGPROB),
                "mean log-prob prefers the more likely-per-token candidate");
    TEST_ASSERT(best_of_score(&b, BEST_OF_SCORE_SUM_LOGPROB) > best_of_score(&a, BEST_OF_SCORE_SUM_LOGPROB),
                "sum log-prob prefers the shorter candidate");
    GenerationStats empty = {0};
    TEST_ASSERT(best_of_score(&empty, BEST_OF_SCORE_MEAN_LOGPROB) < -1e299, "empty candidate never wins");
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
    RUN_TEST(test_ignore_eos_field);
    RUN_TEST(test_chat_template_field);
    RUN_TEST(test_chat_format_choice);
    RUN_TEST(test_best_of_parse_and_score);
    RUN_TEST(test_json_fields);
    TEST_SUMMARY();
}
