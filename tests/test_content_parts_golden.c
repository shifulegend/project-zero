/*
 * tests/test_content_parts_golden.c — chat_request_parse over 60,000 generated bodies whose
 * messages carry an OpenAI "content parts" array: several text parts (joined), image_url objects
 * with extra keys, repeated urls, unknown part keys and nested values, string and array contents
 * mixed in one request, and malformed or cut-off JSON. A hash of the result code and of every
 * message's role, text and image URL, recorded from the function before parse_content_parts was
 * split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/chat_request.h"
#include "test_harness.h"

#define EXPECTED_HASH 0xe08cb2985d477173ull

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}
static void mix_str(const char *s) {
    if (!s) {
        mix("\x01", 1);
        return;
    }
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

static const char *const k_text[] = {
    "\"hello\"",
    "\"\"",
    "\"a\\nb\"",
    "\"caf\\u00e9\"",
    "\"quote \\\" inside\"",
    "\"x\"",
    "\"a much longer piece of text that goes on for a while to force the buffer to grow\"",
    "null",
    "5",
    "\"unterminated"};
static const char *const k_url[]   = {"\"data:image/png;base64,AAAA\"",
                                      "\"http://x/y.png\"",
                                      "\"\"",
                                      "\"data:image/jpeg;base64,/9j/\"",
                                      "7",
                                      "null"};
static const char *const k_extra[] = {"\"detail\":\"high\"", "\"n\":[1,2,{\"a\":3}]",
                                      "\"z\":{\"q\":\"r\"}", "\"flag\":true", "\"s\":\"t\""};
static const char *const k_ws[]    = {"", " ", "\n", "\t "};

/* One content part object. */
static void gen_part(char *out, size_t cap) {
    const char *ws     = PICK(k_ws);
    unsigned kind      = rnd(10);
    const char *text   = PICK(k_text);
    const char *url    = PICK(k_url);
    const char *extra  = PICK(k_extra);
    const char *extra2 = PICK(k_extra);
    switch (kind) {
        case 0:
        case 1:
        case 2:
            snprintf(out, cap, "{%s\"type\":\"text\",%s\"text\":%s}", ws, ws, text);
            break;
        case 3:
        case 4:
            snprintf(out, cap, "{\"type\":\"image_url\",\"image_url\":{\"url\":%s}}", url);
            break;
        case 5:
            snprintf(out, cap, "{\"image_url\":{%s,\"url\":%s,%s}}", extra, url, extra2);
            break;
        case 6:
            snprintf(out, cap, "{%s,\"text\":%s,%s}", extra, text, extra2);
            break;
        case 7:
            snprintf(out, cap, "{\"image_url\":\"%s\"}", "not-an-object");
            break;
        case 8:
            snprintf(out, cap, "{\"type\":\"text\"%s,\"text\":%s,\"image_url\":{\"url\":%s}}", ws,
                     text, url);
            break;
        default:
            snprintf(out, cap, "{%s}", extra);
            break;
    }
}

static void gen_content(char *out, size_t cap) {
    unsigned shape = rnd(8);
    if (shape == 0) {
        snprintf(out, cap, "%s", PICK(k_text)); /* a plain string or scalar content */
        return;
    }
    out[0] = '\0';
    strncat(out, "[", cap - strlen(out) - 1);
    unsigned n = rnd(5);
    for (unsigned i = 0; i < n; i++) {
        char part[512];
        gen_part(part, sizeof part);
        if (i) strncat(out, rnd(8) == 0 ? "" : ",", cap - strlen(out) - 1);
        strncat(out, part, cap - strlen(out) - 1);
    }
    strncat(out, rnd(10) == 0 ? "" : "]", cap - strlen(out) - 1); /* sometimes unterminated */
}

static void gen_body(char *body, size_t cap) {
    static const char *const k_role[] = {"user", "system", "assistant"};
    snprintf(body, cap, "{\"messages\":[");
    unsigned nm = 1 + rnd(3);
    for (unsigned i = 0; i < nm; i++) {
        char content[3072];
        gen_content(content, sizeof content);
        char msg[3500];
        snprintf(msg, sizeof msg, "%s{\"role\":\"%s\",\"content\":%s}", i ? "," : "", PICK(k_role),
                 content);
        strncat(body, msg, cap - strlen(body) - 1);
    }
    strncat(body, "],\"max_tokens\":4}", cap - strlen(body) - 1);
    if (rnd(30) == 0) body[rnd((unsigned)strlen(body))] = '\0'; /* cut off at a random byte */
}

static int g_ok;

static void test_parts_golden(void) {
    for (int t = 0; t < 60000; t++) {
        char body[12000];
        gen_body(body, sizeof body);
        ChatRequest r;
        chat_request_init(&r);
        int rc = (int)chat_request_parse(body, &r);
        mix(&rc, sizeof rc);
        g_ok += rc == TN_OK;
        mix(&r.num_messages, sizeof r.num_messages);
        for (int i = 0; i < r.num_messages && i < CHAT_MAX_MESSAGES; i++) {
            mix_str(r.messages[i].role);
            mix_str(r.messages[i].content);
            mix_str(r.messages[i].image_data_url);
        }
        chat_request_free(&r);
    }
    if (getenv("CP_GOLDEN_PRINT"))
        printf("ok %d of 60000, hash 0x%016llxull\n", g_ok, (unsigned long long)g_hash);
    TEST_ASSERT(g_ok > 10000, "the generator reaches the success path");
    TEST_ASSERT(g_hash == EXPECTED_HASH, "parsed messages match the recorded hash");
}

int main(void) {
    printf("=== content parts golden ===\n");
    RUN_TEST(test_parts_golden);
    TEST_SUMMARY();
}
