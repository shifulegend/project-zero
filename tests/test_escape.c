/* The one-character string escapes shared by the JSON and TOML readers. */
#include <string.h>

#include "api/chat_request.h"
#include "config/pz_config.h"
#include "core/escape.h"
#include "test_harness.h"

static void test_simple_escapes(void) {
    TEST_ASSERT_EQ(tn_escape_simple('n'), '\n', "\\n");
    TEST_ASSERT_EQ(tn_escape_simple('t'), '\t', "\\t");
    TEST_ASSERT_EQ(tn_escape_simple('r'), '\r', "\\r");
    TEST_ASSERT_EQ(tn_escape_simple('b'), '\b', "\\b");
    TEST_ASSERT_EQ(tn_escape_simple('f'), '\f', "\\f");
    TEST_ASSERT_EQ(tn_escape_simple('"'), '"', "\\\"");
    TEST_ASSERT_EQ(tn_escape_simple('\\'), '\\', "\\\\");
    TEST_ASSERT_EQ(tn_escape_simple('x'), -1, "an unknown letter is not an escape");
    TEST_ASSERT_EQ(tn_escape_simple('/'), -1, "\\/ is JSON-only, handled by the JSON parser");
}

static void test_json_uses_them(void) {
    ChatRequest req;
    chat_request_init(&req);
    const char *body =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\\nb\\t\\\"c\\\"\\\\\\/\\u00e9\"}]}";
    TEST_ASSERT_EQ(chat_request_parse(body, &req), TN_OK, "parses");
    TEST_ASSERT(req.num_messages == 1 &&
                    strcmp(req.messages[0].content, "a\nb\t\"c\"\\/\xc3\xa9") == 0,
                "JSON escapes decode as before");
    chat_request_free(&req);
}

static void test_toml_uses_them(void) {
    PzConfig c;
    pz_config_init(&c);
    int n = pz_config_parse(&c, "system-prompt = \"a\\nb\\t\\\"q\\\"\\\\ \\r\"\n", NULL);
    const PzCfgEntry *e = NULL;
    for (int i = 0; i < c.n; i++)
        if (c.entries[i].set) e = &c.entries[i];
    TEST_ASSERT(n >= 0 && e && e->n == 1, "the line parses");
    TEST_ASSERT(e && strcmp(e->v[0], "a\nb\t\"q\"\\ \r") == 0, "TOML escapes decode");
    pz_config_free(&c);
    pz_config_init(&c);
    pz_config_parse(&c, "system-prompt = \"bad \\x escape\"\n", NULL);
    int any = 0;
    for (int i = 0; i < c.n; i++) any |= c.entries[i].set;
    TEST_ASSERT(!any, "an unknown escape is still refused");
    pz_config_free(&c);
}

int main(void) {
    RUN_TEST(test_simple_escapes);
    RUN_TEST(test_json_uses_them);
    RUN_TEST(test_toml_uses_them);
    TEST_SUMMARY();
}
