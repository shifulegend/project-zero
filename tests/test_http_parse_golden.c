/*
 * tests/test_http_parse_golden.c — http_parse over 150,000 generated raw requests: methods and
 * paths of every length (cut at their buffers), the six headers it reads in random case and order
 * with short and over-long values, valid and invalid Content-Length and X-Request-ID, other
 * headers, CRLF and bare-LF line endings, a missing blank line, bodies, and random garbage bytes.
 * Each is parsed from an exactly sized NUL-terminated copy under ASan. A hash of the return value
 * and of every field (strings by content, the body by its offset and length), recorded from the
 * function before its header chain became a table.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/http_request.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x0476d802cc13fd29ull
#define N_REQUESTS 150000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}
static void mix_str(const char *s) {
    mix(s, strlen(s) + 1);
}

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}
#define PICK(t) ((t)[rnd(sizeof(t) / sizeof((t)[0]))])

static const char *const k_method[] = {
    "GET", "POST", "OPTIONS", "DELETE", "", "G", "VERYLONGMETHODNAME12345", "get"};
static const char *const k_path[] = {
    "/", "/health", "/v1/chat/completions", "/v1/models?x=1", "", "/a b", "/x\r"};
static const char *const k_name[] = {
    "Content-Length",       "content-length", "CONTENT-LENGTH", "Origin",       "ORIGIN",
    "Authorization",        "authorization",  "Content-Type",   "content-type", "X-Request-Id",
    "x-request-id",         "Host",           "HOST",           "X-Other",      "Accept",
    "Content-Length-Extra", "Hostx"};
static const char *const k_value[] = {"0",
                                      "12",
                                      "99999999999",
                                      "-3",
                                      "abc",
                                      " 7",
                                      "12 ",
                                      "",
                                      "http://127.0.0.1:8080",
                                      "Bearer sk-abc",
                                      "application/json",
                                      "req-123",
                                      "bad id!",
                                      "a.b_c:d-e",
                                      "127.0.0.1:8080",
                                      "x"};
static const char *const k_eol[]   = {"\r\n", "\r\n", "\r\n", "\n"};

static void append(char *out, size_t cap, const char *s) {
    size_t n = strlen(out);
    if (n + strlen(s) + 1 <= cap) strcpy(out + n, s);
}

static void append_long(char *out, size_t cap, unsigned n) {
    for (unsigned i = 0; i < n; i++) append(out, cap, "v");
}

static void gen_request(char *out, size_t cap) {
    out[0] = '\0';
    append(out, cap, PICK(k_method));
    append(out, cap, rnd(25) ? " " : "");
    append(out, cap, PICK(k_path));
    append(out, cap, rnd(25) ? " HTTP/1.1" : "");
    append(out, cap, PICK(k_eol));
    unsigned n = rnd(8);
    for (unsigned i = 0; i < n; i++) {
        append(out, cap, PICK(k_name));
        append(out, cap, rnd(10) ? ": " : ":");
        if (rnd(12) == 0)
            append_long(out, cap, 60 + rnd(500)); /* longer than the field */
        else
            append(out, cap, PICK(k_value));
        append(out, cap, PICK(k_eol));
    }
    if (rnd(8) != 0) append(out, cap, PICK(k_eol)); /* blank line, sometimes missing */
    if (rnd(2)) append(out, cap, "{\"messages\":[]}");
    if (rnd(40) == 0) out[rnd((unsigned)strlen(out) + 1)] = (char)(1 + rnd(255)); /* garbage byte */
}

static void test_parse_golden(void) {
    int ok = 0;
    for (int t = 0; t < N_REQUESTS; t++) {
        char text[2048];
        gen_request(text, sizeof text);
        size_t n   = strlen(text);
        char *copy = (char *)malloc(n + 1); /* exact size */
        memcpy(copy, text, n + 1);
        HttpRequest req;
        memset(&req, 0xA5, sizeof req);
        int rc = http_parse(copy, n, &req);
        mix(&rc, sizeof rc);
        ok += rc;
        if (rc) {
            mix_str(req.method);
            mix_str(req.path);
            mix(&req.content_length, sizeof req.content_length);
            mix_str(req.origin);
            mix_str(req.authorization);
            mix_str(req.content_type);
            mix_str(req.host);
            mix_str(req.request_id);
            mix(&req.has_origin, sizeof req.has_origin);
            mix(&req.has_authorization, sizeof req.has_authorization);
            size_t off = (size_t)(req.body - copy);
            mix(&off, sizeof off);
            mix(&req.body_len, sizeof req.body_len);
        }
        free(copy);
    }
    if (getenv("HP_GOLDEN_PRINT"))
        printf("ok %d of %d, hash 0x%016llxull\n", ok, N_REQUESTS, (unsigned long long)g_hash);
    TEST_ASSERT(ok > 50000, "the generator reaches the success path");
    TEST_ASSERT(g_hash == EXPECTED_HASH, "parses match the recorded hash");
}

int main(void) {
    printf("=== http_parse golden ===\n");
    RUN_TEST(test_parse_golden);
    TEST_SUMMARY();
}
