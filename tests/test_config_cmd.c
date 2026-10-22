/*
 * tests/test_config_cmd.c — Phase 40.2 `pz config` wizard, driven by scripted
 * input (no terminal): model discovery and pick, keep-on-Enter, re-ask after
 * an invalid value, save vs cancel; and argv preparation with --no-config.
 */
#include "test_harness.h"
#include "cli/config_cmd.h"
#include "config/pz_config.h"
#include "core/sha256.h"
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

static void test_sha256_vectors(void) {
    /* FIPS 180-4 / NIST examples. */
    const char *abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const char *empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const char *two = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
    unsigned char d[32];
    char hex[65];
    Sha256 s;
    sha256_init(&s); sha256_update(&s, "abc", 3); sha256_final(&s, d); sha256_to_hex(d, hex);
    TEST_ASSERT(strcmp(hex, abc) == 0, "sha256(abc)");
    sha256_init(&s); sha256_final(&s, d); sha256_to_hex(d, hex);
    TEST_ASSERT(strcmp(hex, empty) == 0, "sha256(empty)");
    const char *m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    sha256_init(&s);
    for (size_t i = 0; m[i]; i++) sha256_update(&s, m + i, 1);   /* byte-at-a-time path */
    sha256_final(&s, d); sha256_to_hex(d, hex);
    TEST_ASSERT(strcmp(hex, two) == 0, "sha256(two-block message, byte by byte)");
    FILE *f = fopen("/tmp/pz_sha_test.bin", "wb");
    if (f) { fputs("abc", f); fclose(f); }
    TEST_ASSERT(sha256_file_hex("/tmp/pz_sha_test.bin", hex) == 0 && strcmp(hex, abc) == 0, "sha256 of a file");
    remove("/tmp/pz_sha_test.bin");
    TEST_ASSERT(sha256_file_hex("/nonexistent/x", hex) != 0, "missing file reported");
}

static void test_catalog(void) {
    PzStarterModel cat[PZ_CATALOG_MAX];
    const char *src = NULL;
    unsetenv("PZ_MODEL_CATALOG");
    int n = pz_model_catalog(cat, PZ_CATALOG_MAX, &src);
    TEST_ASSERT(n >= 2 && strcmp(src, "built-in") == 0, "built-in starter list");
    for (int i = 0; i < n; i++)
        TEST_ASSERT(strlen(cat[i].sha256) == 64 && strncmp(cat[i].url, "https://", 8) == 0 && cat[i].file[0],
                    "each starter model has an https URL, a SHA-256 and a file name");
    TEST_ASSERT(pz_model_recommend(cat, n, 64.0) == n - 1, "plenty of memory: the largest");
    TEST_ASSERT(pz_model_recommend(cat, n, 0.1) == 0, "almost no memory: the smallest");
    FILE *f = fopen("/tmp/pz_cat_test.txt", "w");
    if (f) {
        fputs("# my catalog\n"
              "mine | My Model | 1234 | 1 | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA | https://example.com/m.gguf | Mine.\n"
              "bad | No hash | 1 | 1 | xyz | https://example.com/b.gguf | Bad.\n"
              "plain | Plain http | 1 | 1 | AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA | http://example.com/p.gguf | Bad.\n", f);
        fclose(f);
    }
    setenv("PZ_MODEL_CATALOG", "/tmp/pz_cat_test.txt", 1);
    n = pz_model_catalog(cat, PZ_CATALOG_MAX, &src);
    TEST_ASSERT(n == 1 && strcmp(cat[0].id, "mine") == 0 && cat[0].bytes == 1234, "file replaces the list; invalid rows skipped");
    TEST_ASSERT(n == 1 && strcmp(cat[0].sha256, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") == 0, "hash lower-cased");
    TEST_ASSERT(n == 1 && strcmp(cat[0].file, "mine-m.gguf") == 0, "file name prefixed with the id");
    unsetenv("PZ_MODEL_CATALOG");
    remove("/tmp/pz_cat_test.txt");
}

static int g_downloads;
static int fake_download(const PzStarterModel *m, const char *dir, char *path, size_t n, FILE *out) {
    (void)out;
    g_downloads++;
    snprintf(path, n, "%s/%s", dir, m->file);
    return 0;
}

static void test_guided_setup(void) {
    PzStarterModel cat[2];
    memset(cat, 0, sizeof(cat));
    snprintf(cat[0].id, sizeof(cat[0].id), "tiny");  snprintf(cat[0].name, sizeof(cat[0].name), "Tiny");
    cat[0].bytes = 100000000; cat[0].min_ram_gb = 0.5; snprintf(cat[0].file, sizeof(cat[0].file), "tiny-a.gguf");
    snprintf(cat[1].id, sizeof(cat[1].id), "big");   snprintf(cat[1].name, sizeof(cat[1].name), "Big");
    cat[1].bytes = 2000000000; cat[1].min_ram_gb = 3; snprintf(cat[1].file, sizeof(cat[1].file), "big-b.gguf");
    PzConfig c;
    pz_config_init(&c);
    PzWizardOpts o;
    memset(&o, 0, sizeof(o));
    o.catalog = cat; o.n_catalog = 2; o.download_dir = "/tmp/pzmodels";
    o.free_ram_gb = 1.0;          /* only the tiny one fits -> suggested */
    o.download = fake_download; o.offer_chat = 1;
    /* model: "?" (explain again), Enter (= suggested download), Enter (confirm);
     * threads/kv/port/updates: Enter; save: Enter; open chat page: Enter. */
    FILE *in = script("?\n\n\n\n\n\n\n\n\n");
    FILE *out = tmpfile();
    g_downloads = 0;
    int r = pz_config_wizard_ex(&c, in, out, &o);
    TEST_ASSERT_EQ(r, 1, "guided setup saves");
    TEST_ASSERT_EQ(g_downloads, 1, "Enter took the suggested starter model and downloaded it");
    const PzCfgEntry *e = pz_config_get(&c, "model");
    TEST_ASSERT(e && strcmp(e->v[0], "/tmp/pzmodels/tiny-a.gguf") == 0, "suggestion fits the free memory");
    TEST_ASSERT_EQ(o.want_chat, 1, "chat page offered after saving");
    char buf[16384] = "";
    rewind(out);
    size_t got = fread(buf, 1, sizeof(buf) - 1, out);
    buf[got] = '\0';
    TEST_ASSERT(strstr(buf, "Step 1 of 5") && strstr(buf, "Step 5 of 5"), "steps are numbered");
    const char *p = strstr(buf, "What this is");
    TEST_ASSERT(p && strstr(p + 1, "What this is") && strstr(buf, "Your options") &&
                strstr(buf, "Can you change it later?"), "each step explains itself; ? repeats it");
    TEST_ASSERT(strstr(buf, "suggested for this computer") != NULL, "the suggestion is marked");
    TEST_ASSERT(strstr(buf, "Nothing here is permanent") != NULL, "summary says what can change later");
    fclose(in);
    fclose(out);

    /* Declining the download keeps asking; then a typed path wins. */
    pz_config_free(&c);
    pz_config_init(&c);
    in = script("2\nn\n/my/own.gguf\n\n\n\n\nn\n");
    out = tmpfile();
    g_downloads = 0;
    r = pz_config_wizard_ex(&c, in, out, &o);
    TEST_ASSERT_EQ(r, 0, "answering n at save cancels");
    TEST_ASSERT_EQ(g_downloads, 0, "declined download not started");
    e = pz_config_get(&c, "model");
    TEST_ASSERT(e && strcmp(e->v[0], "/my/own.gguf") == 0, "typed path used");
    fclose(in);
    fclose(out);
    pz_config_free(&c);
}

int main(void) {
    printf("=== pz config command (Phase 40.2) ===\n");
    RUN_TEST(test_wizard_save);
    RUN_TEST(test_wizard_cancel_and_eof);
    RUN_TEST(test_prepare_argv_no_config);
    RUN_TEST(test_sha256_vectors);
    RUN_TEST(test_catalog);
    RUN_TEST(test_guided_setup);
    TEST_SUMMARY();
}
