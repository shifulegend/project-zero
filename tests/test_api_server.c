/*
 * tests/test_api_server.c
 *
 * Phase 21 — OpenAI-Compatible API Layer
 * Unit tests for: JSON parser, chat template compiler, SSE formatter.
 *
 * Does NOT test the HTTP server socket code (that requires a running process);
 * tests the pure-logic components that can be validated without a model.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>   /* pipe(), read() */

#include "api/chat_request.h"
#include "api/chat_compile.h"
#include "api/sse_stream.h"

/* ── Test helpers ────────────────────────────────────────────────────────── */
#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

/* ── 1. chat_request_init / chat_request_free ────────────────────────────── */
static void test_chat_request_init(void) {
    ChatRequest req;
    chat_request_init(&req);
    assert(req.num_messages == 0);
    assert(req.temperature  == 1.0f);
    assert(req.top_p        == 1.0f);
    assert(req.max_tokens   == 512);
    assert(req.stream       == 0);
    chat_request_free(&req);
    PASS("chat_request_init");
}

/* ── 2. JSON parse: minimal request ─────────────────────────────────────── */
static void test_json_parse_basic(void) {
    const char *json =
        "{\"model\":\"gpt-test\","
         "\"messages\":[{\"role\":\"user\",\"content\":\"Hello world\"}],"
         "\"temperature\":0.7,"
         "\"max_tokens\":64,"
         "\"stream\":true}";

    ChatRequest req;
    chat_request_init(&req);
    TernaryError err = chat_request_parse(json, &req);

    if (err != TN_OK)              { FAIL("json_parse_basic", "parse returned error"); goto done; }
    if (strcmp(req.model, "gpt-test") != 0) { FAIL("json_parse_basic", "model mismatch"); goto done; }
    if (req.num_messages != 1)     { FAIL("json_parse_basic", "wrong message count"); goto done; }
    if (strcmp(req.messages[0].role, "user") != 0)
                                   { FAIL("json_parse_basic", "role mismatch"); goto done; }
    if (strcmp(req.messages[0].content, "Hello world") != 0)
                                   { FAIL("json_parse_basic", "content mismatch"); goto done; }
    if (req.temperature != 0.7f)   { FAIL("json_parse_basic", "temperature mismatch"); goto done; }
    if (req.max_tokens  != 64)     { FAIL("json_parse_basic", "max_tokens mismatch"); goto done; }
    if (!req.stream)               { FAIL("json_parse_basic", "stream flag mismatch"); goto done; }
    PASS("json_parse_basic");
done:
    chat_request_free(&req);
}

/* ── 3. JSON parse: multi-turn conversation ─────────────────────────────── */
static void test_json_parse_multiturn(void) {
    const char *json =
        "{"
          "\"messages\":["
            "{\"role\":\"system\",\"content\":\"You are helpful.\"},"
            "{\"role\":\"user\",\"content\":\"What is 2+2?\"},"
            "{\"role\":\"assistant\",\"content\":\"4.\"},"
            "{\"role\":\"user\",\"content\":\"And 3+3?\"}"
          "]"
        "}";

    ChatRequest req;
    chat_request_init(&req);
    TernaryError err = chat_request_parse(json, &req);

    if (err != TN_OK)            { FAIL("json_parse_multiturn", "parse error"); goto done; }
    if (req.num_messages != 4)   { FAIL("json_parse_multiturn", "wrong count"); goto done; }
    if (strcmp(req.messages[0].role, "system") != 0)
                                 { FAIL("json_parse_multiturn", "msg0 role"); goto done; }
    if (strcmp(req.messages[3].content, "And 3+3?") != 0)
                                 { FAIL("json_parse_multiturn", "msg3 content"); goto done; }
    PASS("json_parse_multiturn");
done:
    chat_request_free(&req);
}

/* ── 4. JSON parse: escape sequences in content ─────────────────────────── */
static void test_json_parse_escapes(void) {
    const char *json =
        "{\"messages\":[{\"role\":\"user\","
                        "\"content\":\"line1\\nline2\\ttabbed\"}]}";

    ChatRequest req;
    chat_request_init(&req);
    TernaryError err = chat_request_parse(json, &req);

    if (err != TN_OK) { FAIL("json_parse_escapes", "parse error"); goto done; }
    if (req.num_messages != 1) { FAIL("json_parse_escapes", "count"); goto done; }
    const char *c = req.messages[0].content;
    if (!c) { FAIL("json_parse_escapes", "null content"); goto done; }
    if (c[5] != '\n') { FAIL("json_parse_escapes", "\\n not decoded"); goto done; }
    if (c[11] != '\t') { FAIL("json_parse_escapes", "\\t not decoded"); goto done; }
    PASS("json_parse_escapes");
done:
    chat_request_free(&req);
}

/* ── 5. JSON parse: unknown keys are silently skipped ────────────────────── */
static void test_json_parse_unknown_keys(void) {
    const char *json =
        "{\"future_field\":\"ignored\","
         "\"messages\":[{\"role\":\"user\",\"content\":\"hi\","
                         "\"extra\":42}],"
         "\"some_list\":[1,2,3]}";

    ChatRequest req;
    chat_request_init(&req);
    TernaryError err = chat_request_parse(json, &req);

    if (err != TN_OK)          { FAIL("json_parse_unknown", "parse error"); goto done; }
    if (req.num_messages != 1) { FAIL("json_parse_unknown", "count"); goto done; }
    if (strcmp(req.messages[0].content, "hi") != 0)
                               { FAIL("json_parse_unknown", "content"); goto done; }
    PASS("json_parse_unknown_keys");
done:
    chat_request_free(&req);
}

/* ── 6. JSON parse: malformed input returns error ───────────────────────── */
static void test_json_parse_malformed(void) {
    ChatRequest req;
    chat_request_init(&req);
    TernaryError err = chat_request_parse("not json at all {{{", &req);
    if (err == TN_OK) { FAIL("json_parse_malformed", "expected error"); }
    else PASS("json_parse_malformed");
    chat_request_free(&req);
}

/* ── 6b. JSON parse: OpenAI "content parts" array (image upload, Phase 22.2) */
static void test_json_parse_content_parts(void) {
    const char *json =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
            "{\"type\":\"text\",\"text\":\"What is in this image?\"},"
            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,QUJD\"}}"
        "]}]}";

    ChatRequest req;
    chat_request_init(&req);
    TernaryError err = chat_request_parse(json, &req);

    if (err != TN_OK) { FAIL("json_parse_content_parts", "parse error"); goto done; }
    if (req.num_messages != 1) { FAIL("json_parse_content_parts", "wrong message count"); goto done; }
    if (!req.messages[0].content || strcmp(req.messages[0].content, "What is in this image?") != 0)
                                   { FAIL("json_parse_content_parts", "text part mismatch"); goto done; }
    if (!req.messages[0].image_data_url ||
        strcmp(req.messages[0].image_data_url, "data:image/png;base64,QUJD") != 0)
                                   { FAIL("json_parse_content_parts", "image_url mismatch"); goto done; }
    PASS("json_parse_content_parts");
done:
    chat_request_free(&req);
}

/* ── 6c. JSON parse: plain string content still works alongside content-parts */
static void test_json_parse_content_parts_text_only_message_still_plain_string(void) {
    const char *json =
        "{\"messages\":["
          "{\"role\":\"user\",\"content\":["
            "{\"type\":\"text\",\"text\":\"describe it\"},"
            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,Zm9v\"}}"
          "]},"
          "{\"role\":\"assistant\",\"content\":\"plain reply\"}"
        "]}";

    ChatRequest req;
    chat_request_init(&req);
    TernaryError err = chat_request_parse(json, &req);

    if (err != TN_OK) { FAIL("content_parts_mixed_with_plain_string", "parse error"); goto done; }
    if (req.num_messages != 2) { FAIL("content_parts_mixed_with_plain_string", "wrong count"); goto done; }
    if (!req.messages[0].image_data_url) { FAIL("content_parts_mixed_with_plain_string", "msg0 missing image"); goto done; }
    if (req.messages[1].image_data_url != NULL) { FAIL("content_parts_mixed_with_plain_string", "msg1 must have no image"); goto done; }
    if (strcmp(req.messages[1].content, "plain reply") != 0) { FAIL("content_parts_mixed_with_plain_string", "msg1 content mismatch"); goto done; }
    PASS("content_parts_mixed_with_plain_string");
done:
    chat_request_free(&req);
}

/* ── 7. chat_template_detect ─────────────────────────────────────────────── */
static void test_template_detect(void) {
    assert(chat_template_detect("llama-3-8b")    == CHAT_TMPL_LLAMA3);
    assert(chat_template_detect("Mistral-7B")    == CHAT_TMPL_MISTRAL);
    assert(chat_template_detect("deepseek-r1")   == CHAT_TMPL_DEEPSEEK);
    assert(chat_template_detect("SmolLM")        == CHAT_TMPL_CHATML);
    assert(chat_template_detect(NULL)            == CHAT_TMPL_CHATML);
    PASS("chat_template_detect");
}

/* ── 8. chat_compile_prompt: ChatML format ───────────────────────────────── */
static void test_compile_chatml(void) {
    ChatRequest req;
    chat_request_init(&req);
    const char *json =
        "{\"messages\":["
          "{\"role\":\"system\",\"content\":\"Be helpful.\"},"
          "{\"role\":\"user\",\"content\":\"Hello!\"}"
        "]}";
    chat_request_parse(json, &req);

    char out[4096];
    TernaryError err = chat_compile_prompt(&req, CHAT_TMPL_CHATML, out, sizeof(out));
    if (err != TN_OK) { FAIL("compile_chatml", "compile error"); goto done; }
    if (!strstr(out, "<|im_start|>system\nBe helpful."))
                       { FAIL("compile_chatml", "missing system block"); goto done; }
    if (!strstr(out, "<|im_start|>user\nHello!"))
                       { FAIL("compile_chatml", "missing user block"); goto done; }
    if (!strstr(out, "<|im_start|>assistant"))
                       { FAIL("compile_chatml", "missing assistant prompt"); goto done; }
    PASS("compile_chatml");
done:
    chat_request_free(&req);
}

/* ── 9. chat_compile_prompt: Llama-3 format ─────────────────────────────── */
static void test_compile_llama3(void) {
    ChatRequest req;
    chat_request_init(&req);
    const char *json =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"What is AI?\"}]}";
    chat_request_parse(json, &req);

    char out[4096];
    TernaryError err = chat_compile_prompt(&req, CHAT_TMPL_LLAMA3, out, sizeof(out));
    if (err != TN_OK) { FAIL("compile_llama3", "compile error"); goto done; }
    if (!strstr(out, "<|begin_of_text|>"))
                       { FAIL("compile_llama3", "missing begin_of_text"); goto done; }
    if (!strstr(out, "<|start_header_id|>user<|end_header_id|>"))
                       { FAIL("compile_llama3", "missing header"); goto done; }
    if (!strstr(out, "What is AI?"))
                       { FAIL("compile_llama3", "missing content"); goto done; }
    PASS("compile_llama3");
done:
    chat_request_free(&req);
}

/* ── 10. sse_write_token writes valid SSE data line ─────────────────────── */
static void test_sse_write_token(void) {
    /* Use a pipe to capture output without a real socket */
    int pipefd[2];
    if (pipe(pipefd) < 0) { FAIL("sse_write_token", "pipe failed"); return; }

    int written = sse_write_token(pipefd[1], "req001", "Hello");
    close(pipefd[1]);

    if (written <= 0) { close(pipefd[0]); FAIL("sse_write_token", "write failed"); return; }

    char buf[1024];
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    close(pipefd[0]);
    if (n <= 0) { FAIL("sse_write_token", "read failed"); return; }
    buf[n] = '\0';

    if (!strstr(buf, "data: "))         { FAIL("sse_write_token", "missing 'data: '"); return; }
    if (!strstr(buf, "chatcmpl-req001")) { FAIL("sse_write_token", "missing id"); return; }
    if (!strstr(buf, "Hello"))           { FAIL("sse_write_token", "missing token"); return; }
    if (!strstr(buf, "\n\n"))            { FAIL("sse_write_token", "missing \\n\\n"); return; }
    PASS("sse_write_token");
}

/* ── 11. sse_write_done writes [DONE] terminator ────────────────────────── */
static void test_sse_write_done(void) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { FAIL("sse_write_done", "pipe failed"); return; }

    sse_write_done(pipefd[1], "req002");
    close(pipefd[1]);

    char buf[1024];
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    close(pipefd[0]);
    if (n <= 0) { FAIL("sse_write_done", "read failed"); return; }
    buf[n] = '\0';

    if (!strstr(buf, "[DONE]")) { FAIL("sse_write_done", "missing [DONE]"); return; }
    if (!strstr(buf, "stop"))   { FAIL("sse_write_done", "missing stop reason"); return; }
    PASS("sse_write_done");
}

/* ── 12. sse JSON escaping of special chars ──────────────────────────────── */
static void test_sse_json_escaping(void) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { FAIL("sse_json_escaping", "pipe failed"); return; }

    sse_write_token(pipefd[1], "r", "say \"hello\"");
    close(pipefd[1]);

    char buf[2048];
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    close(pipefd[0]);
    if (n <= 0) { FAIL("sse_json_escaping", "read failed"); return; }
    buf[n] = '\0';

    /* The raw " must be escaped as \" in the JSON payload */
    if (!strstr(buf, "\\\"hello\\\"")) { FAIL("sse_json_escaping", "quotes not escaped"); return; }
    PASS("sse_json_escaping");
}

/* ── 13. chat_compile_prompt: buffer exactly full returns TN_OK ─────────── */
static void test_compile_tiny_buffer(void) {
    ChatRequest req;
    chat_request_init(&req);
    const char *json = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    chat_request_parse(json, &req);

    char out[1];
    /* Buffer of 1 byte is too small for any template — should return TN_ERR_OOM */
    TernaryError err = chat_compile_prompt(&req, CHAT_TMPL_CHATML, out, 1);
    if (err == TN_OK) {
        FAIL("compile_tiny_buffer_no_crash", "expected TN_ERR_OOM but got TN_OK");
    } else {
        PASS("compile_tiny_buffer_no_crash");
    }
    chat_request_free(&req);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    printf("=== Phase 21 API Server Tests ===\n");

    test_chat_request_init();
    test_json_parse_basic();
    test_json_parse_multiturn();
    test_json_parse_escapes();
    test_json_parse_unknown_keys();
    test_json_parse_malformed();
    test_json_parse_content_parts();
    test_json_parse_content_parts_text_only_message_still_plain_string();
    test_template_detect();
    test_compile_chatml();
    test_compile_llama3();
    test_sse_write_token();
    test_sse_write_done();
    test_sse_json_escaping();
    test_compile_tiny_buffer();

    printf("\n");
    if (g_failures == 0) {
        printf("=== All API tests passed ===\n");
        return 0;
    } else {
        printf("=== %d test(s) FAILED ===\n", g_failures);
        return 1;
    }
}
