/*
 * tests/test_pz_config.c — Phase 40.1 persistent configuration: schema,
 * validation, TOML-subset parsing, save/load round trip, environment overlay
 * and argv merging (config tokens before the user's own flags).
 */
#include "test_harness.h"
#include "config/pz_config.h"
#include "core/platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if TN_POSIX
#include <sys/stat.h>
#include <unistd.h>
#endif

static int g_warnings;
static void count_warn(const char *m) { (void)m; g_warnings++; }

static void test_schema_and_validate(void) {
    int n = 0;
    const PzCfgKey *s = pz_config_schema(&n);
    TEST_ASSERT(s && n > 20, "schema has entries");
    TEST_ASSERT(pz_config_find_key("threads") != NULL, "threads in schema");
    TEST_ASSERT(pz_config_find_key("prompt") == NULL, "per-run inputs are not settings");
    char err[256];
    const PzCfgKey *t = pz_config_find_key("threads");
    TEST_ASSERT(pz_config_validate(t, "8", err, sizeof(err)) == 0, "int ok");
    TEST_ASSERT(pz_config_validate(t, "8x", err, sizeof(err)) != 0, "trailing junk rejected");
    TEST_ASSERT(pz_config_validate(t, "-1", err, sizeof(err)) != 0, "below range rejected");
    const PzCfgKey *kv = pz_config_find_key("kv-type");
    TEST_ASSERT(pz_config_validate(kv, "i4", err, sizeof(err)) == 0, "choice ok");
    TEST_ASSERT(pz_config_validate(kv, "i3", err, sizeof(err)) != 0 && strstr(err, "f32|i8|i4"),
                "bad choice rejected with the allowed list");
    const PzCfgKey *tp = pz_config_find_key("top-p");
    TEST_ASSERT(pz_config_validate(tp, "0.95", err, sizeof(err)) == 0, "float ok");
    TEST_ASSERT(pz_config_validate(tp, "1.5", err, sizeof(err)) != 0, "float range");
    const PzCfgKey *cors = pz_config_find_key("cors");
    TEST_ASSERT(pz_config_validate(cors, "yes", err, sizeof(err)) != 0, "bool must be true/false");
}

static void test_set_add_unset(void) {
    PzConfig c;
    TEST_ASSERT(pz_config_init(&c) == 0, "init");
    char err[256];
    TEST_ASSERT(pz_config_set(&c, "threads", "4", err, sizeof(err)) == 0, "set");
    TEST_ASSERT(pz_config_set(&c, "nope", "1", err, sizeof(err)) != 0 && strstr(err, "unknown"), "unknown key");
    TEST_ASSERT(pz_config_add(&c, "stop", "</s>", err, sizeof(err)) == 0 &&
                pz_config_add(&c, "stop", ">>UNUSED_1<<", err, sizeof(err)) == 0, "list add");
    const PzCfgEntry *e = pz_config_get(&c, "stop");
    TEST_ASSERT(e && e->n == 2 && strcmp(e->v[1], ">>UNUSED_1<<") == 0, "list holds both");
    TEST_ASSERT(pz_config_set(&c, "stop", "x", err, sizeof(err)) == 0 && pz_config_get(&c, "stop")->n == 1,
                "set on a list replaces it");
    pz_config_unset(&c, "threads");
    TEST_ASSERT(pz_config_get(&c, "threads") == NULL, "unset");
    pz_config_free(&c);
}

static void test_parse(void) {
    PzConfig c;
    pz_config_init(&c);
    const char *text =
        "# comment\n"
        "threads = 6   # trailing comment\n"
        "kv_type = \"i8\"\n"                       /* underscore spelling accepted */
        "system-prompt = \"Say \\\"hi\\\"\\nthen stop\"\n"
        "cors = true\n"
        "stop = [\"</s>\", \"<|end|>\"]\n"
        "bogus = 1\n"                              /* unknown: warned, skipped */
        "temperature = hot\n"                      /* invalid: warned, skipped */
        "[server]\n";                              /* tables unsupported: warned */
    g_warnings = 0;
    int n = pz_config_parse(&c, text, count_warn);
    TEST_ASSERT_EQ(n, 5, "five values loaded");
    TEST_ASSERT_EQ(g_warnings, 3, "unknown key, bad value and table each warned");
    TEST_ASSERT(strcmp(pz_config_get(&c, "threads")->v[0], "6") == 0, "int");
    TEST_ASSERT(strcmp(pz_config_get(&c, "kv-type")->v[0], "i8") == 0, "underscore key");
    TEST_ASSERT(strcmp(pz_config_get(&c, "system-prompt")->v[0], "Say \"hi\"\nthen stop") == 0, "escapes");
    TEST_ASSERT(pz_config_get(&c, "stop")->n == 2, "list");
    TEST_ASSERT(pz_config_get(&c, "temperature") == NULL, "invalid value not stored");
    pz_config_free(&c);
}

static void test_round_trip_and_save(void) {
    PzConfig c, d;
    pz_config_init(&c);
    pz_config_init(&d);
    char err[256];
    pz_config_set(&c, "model", "/models/a b \"q\".gguf", err, sizeof(err));
    pz_config_set(&c, "port", "8081", err, sizeof(err));
    pz_config_set(&c, "api-key", "s3cret", err, sizeof(err));
    pz_config_add(&c, "cors-origin", "http://a", err, sizeof(err));
    pz_config_add(&c, "cors-origin", "http://b", err, sizeof(err));
    char dir[] = "/tmp/pz_cfg_test_XXXXXX";
    TEST_ASSERT(mkdtemp(dir) != NULL, "temp dir");
    char path[256];
    snprintf(path, sizeof(path), "%s/sub/config.toml", dir);
    TEST_ASSERT(pz_config_save(&c, path) == 0, "save creates parents");
#if TN_POSIX
    struct stat st;
    TEST_ASSERT(stat(path, &st) == 0 && (st.st_mode & 077) == 0, "file is private (0600)");
#endif
    TEST_ASSERT(pz_config_load(&d, path, NULL) == 4, "load back 4 values");
    TEST_ASSERT(strcmp(pz_config_get(&d, "model")->v[0], "/models/a b \"q\".gguf") == 0, "string round trip");
    TEST_ASSERT(pz_config_get(&d, "cors-origin")->n == 2, "list round trip");
    char missing[300];
    snprintf(missing, sizeof(missing), "%s/none.toml", dir);
    TEST_ASSERT(pz_config_load(&d, missing, NULL) == 0, "missing file is not an error");
    char cmd[300];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dir);
    if (system(cmd) != 0) printf("(cleanup failed)\n");
    pz_config_free(&c);
    pz_config_free(&d);
}

static void test_env_and_path(void) {
    PzConfig c;
    pz_config_init(&c);
    setenv("PZ_THREADS", "3", 1);
    setenv("PZ_KV_TYPE", "nonsense", 1);
    g_warnings = 0;
    TEST_ASSERT_EQ(pz_config_apply_env(&c, count_warn), 1, "valid env applied");
    TEST_ASSERT_EQ(g_warnings, 1, "invalid env warned, not applied");
    TEST_ASSERT(strcmp(pz_config_get(&c, "threads")->v[0], "3") == 0, "PZ_THREADS");
    unsetenv("PZ_THREADS");
    unsetenv("PZ_KV_TYPE");
    char p[512];
    setenv("PZ_CONFIG", "/x/y.toml", 1);
    TEST_ASSERT(pz_config_path(p, sizeof(p)) == 0 && strcmp(p, "/x/y.toml") == 0, "PZ_CONFIG wins");
    unsetenv("PZ_CONFIG");
    setenv("XDG_CONFIG_HOME", "/cfg", 1);
    TEST_ASSERT(pz_config_path(p, sizeof(p)) == 0 && strcmp(p, "/cfg/project-zero/config.toml") == 0, "XDG");
    unsetenv("XDG_CONFIG_HOME");
    pz_config_free(&c);
}

static void test_merge_argv(void) {
    PzConfig c;
    pz_config_init(&c);
    char err[256];
    pz_config_set(&c, "threads", "2", err, sizeof(err));
    pz_config_set(&c, "cors", "true", err, sizeof(err));
    pz_config_set(&c, "metrics", "false", err, sizeof(err));
    pz_config_add(&c, "stop", "a", err, sizeof(err));
    char *argv[] = { "pz", "--threads", "8", NULL };
    char **out = NULL;
    int n = pz_config_merge_argv(&c, 3, argv, &out);
    /* pz --threads 2 --stop a --cors --threads 8 */
    TEST_ASSERT_EQ(n, 8, "argc = 1 + 2 (threads) + 2 (stop) + 1 (cors) + 2 (user)");
    int ok = n == 8 && strcmp(out[0], "pz") == 0 && strcmp(out[1], "--threads") == 0 && strcmp(out[2], "2") == 0 &&
             strcmp(out[6], "--threads") == 0 && strcmp(out[7], "8") == 0 && out[8] == NULL;
    TEST_ASSERT(ok, "config first, user flags last (so they win)");
    int saw_cors = 0, saw_metrics = 0;
    for (int i = 1; i < n; i++) { saw_cors |= strcmp(out[i], "--cors") == 0; saw_metrics |= strcmp(out[i], "--metrics") == 0; }
    TEST_ASSERT(saw_cors && !saw_metrics, "true bool adds the flag, false omits it");
    free(out);
    pz_config_free(&c);
}

int main(void) {
    printf("=== pz config (Phase 40.1) ===\n");
    RUN_TEST(test_schema_and_validate);
    RUN_TEST(test_set_add_unset);
    RUN_TEST(test_parse);
    RUN_TEST(test_round_trip_and_save);
    RUN_TEST(test_env_and_path);
    RUN_TEST(test_merge_argv);
    TEST_SUMMARY();
}
