/*
 * tests/test_config_cmd.c — Phase 40.2 `pz config` wizard, driven by scripted
 * input (no terminal): model discovery and pick, keep-on-Enter, re-ask after
 * an invalid value, save vs cancel; and argv preparation with --no-config.
 */
#include "test_harness.h"
#include "cli/config_cmd.h"
#include "config/pz_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* mkdtemp on macOS */

static FILE *script(const char *text) {
    FILE *f = tmpfile();
    if (f) { fputs(text, f); rewind(f); }
    return f;
}

static void test_wizard_save(void) {
    char dir[] = "/tmp/pz_wiz_XXXXXX";
    TEST_ASSERT(mkdtemp(dir) != NULL, "temp dir");
    char gguf[300];
    snprintf(gguf, sizeof(gguf), "%s/tiny.gguf", dir);
    FILE *g = fopen(gguf, "w");
    if (g) { fputs("x", g); fclose(g); }
    const char *dirs[] = { dir, NULL };

    PzConfig c;
    pz_config_init(&c);
    char err[256];
    pz_config_set(&c, "threads", "6", err, sizeof(err));
    /* model: pick #1; threads: Enter keeps 6; kv-type: bad then i4;
     * port: 8090; update-channel: Enter; save: y */
    FILE *in = script("1\n\nq9\ni4\n8090\n\ny\n");
    FILE *out = tmpfile();
    int r = pz_config_wizard(&c, in, out, dirs);
    TEST_ASSERT_EQ(r, 1, "wizard asks to save");
    TEST_ASSERT(pz_config_get(&c, "model") && strcmp(pz_config_get(&c, "model")->v[0], gguf) == 0,
                "discovered model picked by number");
    TEST_ASSERT(strcmp(pz_config_get(&c, "threads")->v[0], "6") == 0, "Enter keeps the current value");
    TEST_ASSERT(strcmp(pz_config_get(&c, "kv-type")->v[0], "i4") == 0, "re-asked after an invalid value");
    TEST_ASSERT(strcmp(pz_config_get(&c, "port")->v[0], "8090") == 0, "port");
    TEST_ASSERT(pz_config_get(&c, "update-channel") == NULL, "Enter leaves an unset value unset");
    char outbuf[4096] = "";
    rewind(out);
    size_t got = fread(outbuf, 1, sizeof(outbuf) - 1, out);
    outbuf[got] = '\0';
    TEST_ASSERT(strstr(outbuf, "tiny.gguf") != NULL, "found models are listed");
    TEST_ASSERT(strstr(outbuf, "is not one of") != NULL, "invalid value explained");
    fclose(in);
    fclose(out);
    pz_config_free(&c);
    char cmd[300];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", dir);
    if (system(cmd) != 0) printf("(cleanup failed)\n");
}

static void test_wizard_cancel_and_eof(void) {
    PzConfig c;
    pz_config_init(&c);
    FILE *in = script("/m.gguf\n\n\n\n\nn\n");
    FILE *out = tmpfile();
    TEST_ASSERT_EQ(pz_config_wizard(&c, in, out, NULL), 0, "answering n cancels");
    fclose(in);
    in = script("");                       /* EOF at the first question */
    TEST_ASSERT_EQ(pz_config_wizard(&c, in, out, NULL), 0, "EOF cancels cleanly");
    fclose(in);
    fclose(out);
    pz_config_free(&c);
}

static void test_prepare_argv_no_config(void) {
    setenv("PZ_THREADS", "2", 1);
    char *argv[] = { "pz", "--no-config", "--prompt", "hi", NULL };
    char **out = NULL;
    int n = pz_config_prepare_argv(4, argv, &out);
    TEST_ASSERT_EQ(n, 3, "--no-config removed, env ignored");
    TEST_ASSERT(n == 3 && strcmp(out[1], "--prompt") == 0 && out[3] == NULL, "argv intact");
    char *argv2[] = { "pz", "--prompt", "hi", NULL };
    setenv("PZ_CONFIG", "/nonexistent/pz.toml", 1);
    n = pz_config_prepare_argv(3, argv2, &out);
    TEST_ASSERT(n == 5 && strcmp(out[1], "--threads") == 0 && strcmp(out[2], "2") == 0 &&
                strcmp(out[3], "--prompt") == 0, "env value placed before user flags");
    unsetenv("PZ_THREADS");
    unsetenv("PZ_CONFIG");
}

int main(void) {
    printf("=== pz config command (Phase 40.2) ===\n");
    RUN_TEST(test_wizard_save);
    RUN_TEST(test_wizard_cancel_and_eof);
    RUN_TEST(test_prepare_argv_no_config);
    TEST_SUMMARY();
}
