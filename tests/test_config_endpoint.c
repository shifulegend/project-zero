/*
 * tests/test_config_endpoint.c — Phase 40.3: per-setting sources (default /
 * file / env / cli) published in GET /v1/config, the cross-site guard and
 * POST /v1/config/defaults writing the `pz config` file.
 */
#include "test_harness.h"
#include "api/api_info.h"
#include "api/config_endpoint.h"
#include "api/json_fields.h"
#include "config/pz_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* mkdtemp on macOS */

static char g_dir[] = "/tmp/pz_cfgep_XXXXXX";
static char g_path[300];

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    static char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

static void test_sources(void) {
    PzConfig c;
    pz_config_init(&c);
    char err[256];
    pz_config_parse(&c, "temperature = 0.5\ntop-p = 0.8\n", NULL);
    setenv("PZ_TOP_K", "7", 1);
    pz_config_apply_env(&c, NULL);
    unsetenv("PZ_TOP_K");
    char *argv[] = { "pz", "--seed", "3", "--prompt", "x", "--top-p", "0.9", NULL };
    TEST_ASSERT_EQ(pz_config_mark_cli(&c, 7, argv), 2, "two schema flags marked (prompt is not a setting)");
    pz_config_publish(&c, g_path, 1);
    TEST_ASSERT(strcmp(pz_config_active_source("temperature"), "file") == 0, "file value");
    TEST_ASSERT(strcmp(pz_config_active_source("top-k"), "env") == 0, "env value");
    TEST_ASSERT(strcmp(pz_config_active_source("top-p"), "cli") == 0, "flag beats file");
    TEST_ASSERT(strcmp(pz_config_active_source("seed"), "cli") == 0, "flag-only key");
    TEST_ASSERT(strcmp(pz_config_active_source("min-p"), "default") == 0, "untouched key");
    TEST_ASSERT(strcmp(pz_config_active_source("nope"), "default") == 0, "unknown key");
    TEST_ASSERT(pz_config_validate(pz_config_find_key("repeat-last-n"), "-1", err, sizeof(err)) == 0,
                "repeat-last-n accepts -1 (all tokens), as --repeat-last-n does");

    char *json = api_config_json(NULL);
    TEST_ASSERT(json && strstr(json, "\"config\":{\"path\":") != NULL, "/v1/config has a config block");
    TEST_ASSERT(json && strstr(json, "\"top_p\":\"cli\"") != NULL, "sources keyed by JSON name");
    TEST_ASSERT(json && strstr(json, "\"penalty_last_n\":\"repeat-last-n\"") != NULL, "JSON name -> config key map");
    TEST_ASSERT(json && strstr(json, "\"writable\":false") != NULL, "no context: not writable");
    free(json);
    pz_config_publish(NULL, NULL, 0);
    pz_config_free(&c);
}

static void test_write_guard(void) {
    CorsConfig cors;
    memset(&cors, 0, sizeof(cors));
    TEST_ASSERT(api_config_write_allowed("application/json", NULL, "127.0.0.1:8080", &cors), "no Origin: allowed");
    TEST_ASSERT(api_config_write_allowed("application/json; charset=utf-8", "http://127.0.0.1:8080",
                                         "127.0.0.1:8080", &cors), "same origin");
    TEST_ASSERT(!api_config_write_allowed("text/plain", NULL, "127.0.0.1:8080", &cors),
                "a form/simple content type is refused");
    TEST_ASSERT(!api_config_write_allowed(NULL, NULL, NULL, &cors), "no content type refused");
    TEST_ASSERT(!api_config_write_allowed("application/json", "http://evil.example", "127.0.0.1:8080", &cors),
                "foreign origin refused");
    TEST_ASSERT(!api_config_write_allowed("application/json", "http://127.0.0.1:80800", "127.0.0.1:8080", &cors),
                "host prefix is not a match");
    char star[] = "*";
    cors.enabled = 1;
    cors.origins[0] = star;
    cors.num_origins = 1;
    TEST_ASSERT(api_config_write_allowed("application/json", "http://evil.example", "127.0.0.1:8080", &cors),
                "CORS-allowed origin");
}

static int save(ApiContext *ctx, const char *body, char **out) {
    int status = 0;
    *out = api_config_save(ctx, body, &status);
    return status;
}

static void test_save(void) {
    ApiContext *ctx = (ApiContext *)calloc(1, sizeof(ApiContext));
    TEST_ASSERT(ctx != NULL, "context");
    if (!ctx) return;
    PzConfig c;
    pz_config_init(&c);
    pz_config_publish(&c, g_path, 1);
    char *r = NULL;

    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"temperature\",\"value\":0.7}", &r), 403, "--no-config-write refuses");
    free(r);
    ctx->server_config.allow_config_write = 1;

    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"temperature\",\"value\":0.7}", &r), 200, "number saved");
    TEST_ASSERT(r && strstr(r, "\"applies\":\"restart\"") && strstr(r, "\"overridden_by\":null"), "response");
    free(r);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"system_prompt\",\"value\":\"Be brief.\"}", &r), 200, "string saved");
    free(r);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"stop\",\"value\":[\"END\",\"###\"]}", &r), 200, "list saved");
    free(r);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"seed\",\"value\":1234567890123}", &r), 200, "big integer kept exactly");
    free(r);
    const char *text = slurp(g_path);
    TEST_ASSERT(text && strstr(text, "temperature = 0.7"), "file has the exact number");
    TEST_ASSERT(text && strstr(text, "system-prompt = \"Be brief.\""), "file has the string");
    TEST_ASSERT(text && strstr(text, "stop = [\"END\", \"###\"]"), "file has the list");
    TEST_ASSERT(text && strstr(text, "seed = 1234567890123"), "file has the seed");

    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"temperature\",\"value\":99}", &r), 400, "out of range");
    TEST_ASSERT(r && strstr(r, "temperature"), "error names the setting");
    free(r);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"api_key\",\"value\":\"x\"}", &r), 400, "only defaults are saveable");
    free(r);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"top_p\"}", &r), 400, "value required");
    free(r);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"stop\",\"value\":\"END\"}", &r), 400, "stop needs an array");
    free(r);

    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"temperature\",\"unset\":true}", &r), 200, "unset");
    free(r);
    text = slurp(g_path);
    TEST_ASSERT(text && !strstr(text, "temperature") && strstr(text, "system-prompt"),
                "unset removes only that key");

    /* A flag given at startup still wins after the save. */
    char *argv[] = { "pz", "--top-p", "0.9", NULL };
    pz_config_mark_cli(&c, 3, argv);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"top_p\",\"value\":0.95}", &r), 200, "saved under a flag");
    TEST_ASSERT(r && strstr(r, "\"overridden_by\":\"cli\""), "response warns the flag overrides it");
    free(r);

    char *json = api_config_json(ctx);
    TEST_ASSERT(json && strstr(json, "\"writable\":true"), "writable with a path and no --no-config-write");
    free(json);

    pz_config_publish(NULL, NULL, 0);
    TEST_ASSERT_EQ(save(ctx, "{\"key\":\"top_p\",\"value\":0.5}", &r), 403, "no config location");
    free(r);
    pz_config_free(&c);
    free(ctx);
}

static void test_json_helpers(void) {
    char *t = json_field_number_text("{\"a\": -1.5e3, \"b\": \"x\"}", "a");
    TEST_ASSERT(t && strcmp(t, "-1.5e3") == 0, "number literal text");
    free(t);
    TEST_ASSERT(json_field_number_text("{\"b\": \"x\"}", "b") == NULL, "string is not a number");
    char *items[4];
    int n = json_field_string_list("{\"s\": [\"a\", \"b\"]}", "s", items, 4);
    TEST_ASSERT(n == 2 && strcmp(items[1], "b") == 0, "string list");
    for (int i = 0; i < n; i++) free(items[i]);
    TEST_ASSERT_EQ(json_field_string_list("{\"s\": [\"a\", 1]}", "s", items, 4), -1, "mixed list rejected");
    TEST_ASSERT_EQ(json_field_string_list("{\"s\": [\"a\",\"b\",\"c\"]}", "s", items, 2), -1, "too many rejected");
}

int main(void) {
    printf("=== /v1/config sources + POST /v1/config/defaults (Phase 40.3) ===\n");
    if (!mkdtemp(g_dir)) { printf("cannot create temp dir\n"); return 1; }
    snprintf(g_path, sizeof(g_path), "%s/config.toml", g_dir);
    RUN_TEST(test_sources);
    RUN_TEST(test_write_guard);
    RUN_TEST(test_save);
    RUN_TEST(test_json_helpers);
    remove(g_path);
    rmdir(g_dir);
    TEST_SUMMARY();
}
