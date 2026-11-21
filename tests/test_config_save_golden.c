/*
 * tests/test_config_save_golden.c — api_config_save (POST /v1/config/defaults) over 6,000 generated
 * requests against one config file: every saveable key and some that are not, numbers (valid and
 * out of range), strings, lists of zero to ten items, nulls and missing values, "unset" true and
 * false, malformed JSON, a server started with --no-config-write or without a config location,
 * and flags marked as given on the command line (the response says they override the saved value).
 * A hash of every status, every response body (the temp path normalised) and the config file after
 * each call, recorded from the function before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "api/api_info.h"
#include "api/config_endpoint.h"
#include "config/pz_config.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x592a00100a104e55ull
#define N_REQUESTS 6000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}
static void mix_text(const char *s, const char *path) {
    if (!s) {
        mix("\x01", 1);
        return;
    }
    size_t pl = strlen(path);
    for (const char *p = s; *p;) {
        if (strncmp(p, path, pl) == 0) {
            mix("<P>", 3);
            p += pl;
        } else {
            mix(p, 1);
            p++;
        }
    }
    mix("\0", 1);
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}
#define PICK(t) ((t)[rnd(sizeof(t) / sizeof((t)[0]))])

static const char *const k_keys[]   = {"max_tokens",
                                       "temperature",
                                       "top_p",
                                       "top_k",
                                       "min_p",
                                       "repetition_penalty",
                                       "presence_penalty",
                                       "frequency_penalty",
                                       "penalty_last_n",
                                       "seed",
                                       "system_prompt",
                                       "stop",
                                       "ui_title",
                                       "parallel",
                                       "parallel_batch",
                                       "act_quant",
                                       "quality_tolerance",
                                       "self_check_tokens",
                                       "api_key",
                                       "model",
                                       "nope",
                                       ""};
static const char *const k_values[] = {
    "0.7",
    "0",
    "-1",
    "99",
    "1e3",
    "12",
    "\"Be brief.\"",
    "\"\"",
    "\"auto\"",
    "\"int8\"",
    "\"float\"",
    "\"x y\"",
    "null",
    "true",
    "[\"END\",\"###\"]",
    "[]",
    "[\"a\"]",
    "[\"a\",\"b\",\"c\",\"d\",\"e\",\"f\",\"g\",\"h\"]",
    "[\"a\",\"b\",\"c\",\"d\",\"e\",\"f\",\"g\",\"h\",\"i\",\"j\"]",
    "[1,2]",
    "\"END\"",
    "1234567890123",
    "\"quote \\\" and \\\\ slash\"",
    "{}"};

static void gen_body(char *out, size_t cap) {
    const char *key   = PICK(k_keys);
    const char *value = PICK(k_values);
    unsigned form     = rnd(14);
    switch (form) {
        case 0:
            snprintf(out, cap, "{\"key\":\"%s\",\"unset\":true}", key);
            break;
        case 1:
            snprintf(out, cap, "{\"key\":\"%s\",\"unset\":false,\"value\":%s}", key, value);
            break;
        case 2:
            snprintf(out, cap, "{\"key\":\"%s\"}", key);
            break;
        case 3:
            snprintf(out, cap, "{\"value\":%s}", value);
            break;
        case 4:
            snprintf(out, cap, "{\"key\":%s,\"value\":1}", value);
            break;
        case 5:
            snprintf(out, cap, "{\"key\":\"%s\",\"value\":%s", key, value);
            break; /* cut off */
        case 6:
            snprintf(out, cap, "not json");
            break;
        default:
            snprintf(out, cap, "{\"key\":\"%s\",\"value\":%s}", key, value);
            break;
    }
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    static char buf[16384];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

static void apply_context(ApiContext *ctx, PzConfig *c, const char *path) {
    unsigned mode                              = rnd(40);
    ctx->server_config.chat.allow_config_write = mode != 0; /* sometimes --no-config-write */
    ctx->server_config.provenance              = (PzProvenance){c, mode == 1 ? NULL : path, 1};
    if (mode == 2) {
        char *argv[] = {"pz", "--top-p", "0.9", "--seed", "3", NULL};
        pz_config_mark_cli(c, 5, argv);
    }
}

static void test_save_golden(void) {
    char path[] = "/tmp/pz_cfgsave_golden_XXXXXX";
    int fd      = mkstemp(path);
    TEST_ASSERT(fd >= 0, "temp file");
    close(fd);
    ApiContext *ctx = (ApiContext *)calloc(1, sizeof(ApiContext));
    PzConfig c;
    pz_config_init(&c);
    int ok = 0;
    for (int t = 0; t < N_REQUESTS; t++) {
        char body[512];
        gen_body(body, sizeof body);
        apply_context(ctx, &c, path);
        int status   = 0;
        char *answer = api_config_save(ctx, body, &status);
        mix(&status, sizeof status);
        mix_text(answer, path);
        mix_text(slurp(path), path);
        ok += status == 200;
        free(answer);
    }
    pz_config_free(&c);
    free(ctx);
    remove(path);
    if (getenv("CS_GOLDEN_PRINT"))
        printf("ok %d of %d, hash 0x%016llxull\n", ok, N_REQUESTS, (unsigned long long)g_hash);
    TEST_ASSERT(ok > 600, "the generator reaches the success path");
    TEST_ASSERT(g_hash == EXPECTED_HASH, "saves match the recorded hash");
}

int main(void) {
    printf("=== api_config_save golden ===\n");
    RUN_TEST(test_save_golden);
    TEST_SUMMARY();
}
