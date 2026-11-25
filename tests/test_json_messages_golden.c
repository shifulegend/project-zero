/*
 * tests/test_json_messages_golden.c — json_parse_messages over 60,000 generated "messages" arrays:
 * one to seventy message objects with a role (plain, escaped, longer than the field), content as
 * a string, null, a content-parts array (text parts, image_url objects, unknown parts) or missing,
 * repeated and unknown keys with nested values, whitespace, and damage — truncation at any byte, a
 * dropped closing bracket, doubled commas. Each text sits in a heap block of exactly its length +
 * NUL, so a read past the end is an ASan failure, and the parsed messages are freed (leaks fail
 * too). The hash covers the result code, how far the cursor moved, the message count and every
 * message's role, content and image, recorded from the parser before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/chat_request.h"
#include "api/json_scan.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x254368c7fd6b3951ull
#define N_CASES 60000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}
static void mix_str(const char *s) {
    int has = s != NULL;
    mix(&has, sizeof has);
    if (s) mix(s, strlen(s) + 1);
}

static uint64_t g_rng = 0x3c6ef372fe94f82bull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const ROLES[] = {
    "\"user\"",
    "\"assistant\"",
    "\"system\"",
    "\"tool\"",
    "\"\"",
    "\"us\\u0065r\"",
    "\"abcdefghijklmnopqrstuvwxyz0123456789ABCDEF\"",
    "\"a\\nb\"",
    "null",
    "5",
    "[]",
};
static const char *const CONTENTS[] = {
    /* well formed (the first GOOD_CONTENTS) */
    "\"hello\"",
    "\"\"",
    "\"line\\nbreak \\\"q\\\" \\u00e9\\ud83d\\ude00\"",
    "null",
    "[]",
    "[{\"type\":\"text\",\"text\":\"a\"}]",
    "[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"text\",\"text\":\"b\"}]",
    "[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}]",
    "[{\"text\":\"x\"},{\"image_url\":{\"url\":\"data:x\"}}]",
    "[{\"type\":\"image_url\",\"image_url\":\"u\"}]",
    "[{\"type\":\"video\",\"x\":[1,{}]}]",
    /* malformed */
    "[{\"text\":5}]",
    "nul",
    "5",
    "{}",
    "[1]",
    "[{]",
    "\"unterminated",
};
static const char *const EXTRAS[] = {
    "\"name\":\"n\"", "\"tool_calls\":[{\"id\":1,\"f\":{\"a\":[1,2,{}]}}]",
    "\"x\":null",     "\"y\":{\"z\":\"}\"}",
    "\"n\":1.5e3",    "\"b\":true",
    "\"s\":\"a,b}\"",
};

/* An index below `good` nine times in ten, else any index: most values are well formed. */
static int g_clean; /* long arrays use only well-formed values so they reach the 64-message cap */
static size_t pick(size_t n, size_t good) {
    unsigned wide = rnd(10);
    if (g_clean) return rnd((unsigned)good);
    return wide == 0 ? rnd((unsigned)n) : rnd((unsigned)good);
}

static void append(char *out, size_t cap, const char *s) {
    size_t n = strlen(out);
    if (n + strlen(s) + 1 < cap) strcpy(out + n, s);
}

/* Whitespace in the places a JSON writer may put it. */
static void maybe_space(char *out, size_t cap) {
    if (rnd(6) == 0) append(out, cap, rnd(2) ? " " : "\n ");
}

static void add_member(char *out, size_t cap, int *first, const char *key, const char *value) {
    char buf[200];
    snprintf(buf, sizeof buf, "%s\"%s\":%s", *first ? "" : ",", key, value);
    maybe_space(out, cap);
    append(out, cap, buf);
    *first = 0;
}

static void add_message(char *out, size_t cap) {
    int first = 1;
    append(out, cap, "{");
    maybe_space(out, cap);
    if (rnd(10) != 0)
        add_member(out, cap, &first, "role", ROLES[pick(sizeof ROLES / sizeof ROLES[0], 6)]);
    unsigned extra = rnd(3);
    for (unsigned k = 0; k < extra; k++) {
        const char *e = EXTRAS[rnd(sizeof EXTRAS / sizeof EXTRAS[0])];
        append(out, cap, first ? "" : ",");
        append(out, cap, e);
        first = 0;
    }
    unsigned contents = rnd(8) == 0 ? 2 : (rnd(8) == 0 ? 0 : 1);
    for (unsigned k = 0; k < contents; k++)
        add_member(out, cap, &first, "content",
                   CONTENTS[pick(sizeof CONTENTS / sizeof CONTENTS[0], 11)]);
    maybe_space(out, cap);
    append(out, cap, "}");
}

static void build(char *out, size_t cap) {
    out[0]     = '\0';
    unsigned n = rnd(20) == 0 ? 60 + rnd(12) : 1 + rnd(4);
    g_clean    = n > 10;
    append(out, cap, "[");
    for (unsigned i = 0; i < n; i++) {
        if (i) append(out, cap, !g_clean && rnd(25) == 0 ? ",," : ",");
        add_message(out, cap);
    }
    if (rnd(10) != 0) append(out, cap, "]");
    size_t len = strlen(out);
    if (rnd(8) == 0 && len > 0) out[rnd((unsigned)len)] = '\0'; /* truncated */
}

static int g_ok, g_img, g_full;

static void one_case(ChatRequest *req, char *text, size_t cap) {
    build(text, cap);
    size_t len = strlen(text);
    char *in   = (char *)malloc(len + 1);
    memcpy(in, text, len + 1);
    memset(req, 0, sizeof *req);
    const char *p = in;
    int rc        = (int)json_parse_messages(&p, req);
    size_t moved  = (size_t)(p - in);
    mix(&rc, sizeof rc);
    g_ok += rc == 0;
    g_full += req->num_messages >= CHAT_MAX_MESSAGES;
    mix(&moved, sizeof moved);
    mix(&req->num_messages, sizeof req->num_messages);
    for (int i = 0; i < req->num_messages; i++) {
        ChatMessage *m = &req->messages[i];
        mix_str(m->role);
        mix_str(m->content);
        mix_str(m->image_data_url);
        g_img += m->image_data_url != NULL;
        free(m->content);
        free(m->image_data_url);
    }
    free(in);
}

static void test_json_messages_golden(void) {
    ChatRequest *req = (ChatRequest *)malloc(sizeof *req);
    size_t cap       = 40000;
    char *text       = (char *)malloc(cap);
    for (int t = 0; t < N_CASES; t++) one_case(req, text, cap);
    free(text);
    free(req);
    if (getenv("JSON_MESSAGES_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d ok, %d images, %d full arrays)\n", (unsigned long long)g_hash,
               g_ok, g_img, g_full);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "message parsing matches the recorded hash");
}

int main(void) {
    printf("=== json_parse_messages golden ===\n");
    RUN_TEST(test_json_messages_golden);
    TEST_SUMMARY();
}
