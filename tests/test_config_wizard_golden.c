/*
 * tests/test_config_wizard_golden.c — the `pz config` wizard driven by 12,000 generated answer
 * scripts (blank, "?", "-", numbers in and out of range, paths, yes/no, junk) against a fake
 * catalog, a fake downloader and a directory of model files; the return value, the wanted-chat
 * flag, everything printed and the resulting settings are hashed. The hash was recorded from the
 * wizard before ask_model and pz_config_wizard_ex were split into smaller functions.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli/config_cmd.h"
#include "config/pz_config.h"
#include "test_harness.h"

#define EXPECTED_HASH 0xe80ded419f4e5cb6ull
#define N_SCRIPTS 12000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static int fake_download(const PzStarterModel *m, const char *dir, char *path, size_t pathn,
                         FILE *out) {
    if (m->id[0] == 'b') return -1; /* the "big" one fails */
    fprintf(out, "  (fake download of %s)\n", m->id);
    snprintf(path, pathn, "%s/%s", dir, m->file);
    return 0;
}

static const char *const k_answers[] = {
    "",   "",     "",      "",     "?",   "-",          "1",         "2",      "3",
    "4",  "5",    "0",     "-1",   "99",  "abc",        "/x/y.gguf", "y",      "n",
    "Y",  "N",    "yes",   "no",   "8",   "8080",       "70000",     "stable", "auto",
    "i4", "true", "maybe", "junk", "1 2", "prerelease", " ",         "\t",     "f32"};

static void fill_catalog(PzStarterModel *cat) {
    memset(cat, 0, 3 * sizeof(*cat));
    const char *ids[3]   = {"tiny", "big", "mid"};
    const char *names[3] = {"Tiny", "Big", "Mid"};
    for (int i = 0; i < 3; i++) {
        snprintf(cat[i].id, sizeof(cat[i].id), "%s", ids[i]);
        snprintf(cat[i].name, sizeof(cat[i].name), "%s", names[i]);
        cat[i].bytes      = 100000000LL * (1 + i * 9);
        cat[i].min_ram_gb = 0.5 + i * 2;
        snprintf(cat[i].file, sizeof(cat[i].file), "%s.gguf", ids[i]);
        snprintf(cat[i].about, sizeof(cat[i].about), "about %s", ids[i]);
    }
}

static void mix_replacing(const char *text, const char *from) {
    size_t fl = strlen(from);
    for (const char *p = text; *p;) {
        if (strncmp(p, from, fl) == 0) {
            mix("<D>", 3);
            p += fl;
        } else {
            mix(p, 1);
            p++;
        }
    }
}

/* One generated answer script, one per line. */
static void gen_script(char *text) {
    text[0]    = '\0';
    unsigned n = 6 + rnd(22);
    for (unsigned i = 0; i < n; i++) {
        strcat(text, k_answers[rnd(sizeof k_answers / sizeof *k_answers)]);
        strcat(text, "\n");
    }
}

/* Random wizard options over the fake catalog. */
static void gen_opts(PzWizardOpts *o, PzStarterModel *cat, const char *const *dirs) {
    memset(o, 0, sizeof(*o));
    o->model_dirs   = rnd(3) ? dirs : NULL;
    o->catalog      = rnd(4) ? cat : NULL;
    o->n_catalog    = 3;
    o->download_dir = rnd(5) ? "/dl" : NULL;
    o->free_ram_gb  = (double[]){0, 1.0, 3.0, 64.0}[rnd(4)];
    (void)rnd(6); /* the recorded run chose between a fake and the default downloader here */
    o->download   = fake_download; /* never reach the network */
    o->offer_chat = (int)rnd(2);
}

static void run_one(const char *dir, PzStarterModel *cat, const char *const *dirs) {
    char text[2048];
    gen_script(text);
    FILE *in = tmpfile();
    fputs(text, in);
    rewind(in);
    FILE *out = tmpfile();
    PzConfig c;
    pz_config_init(&c);
    PzWizardOpts o;
    gen_opts(&o, cat, dirs);
    int r = pz_config_wizard_ex(&c, in, out, &o);
    mix(&r, sizeof r);
    mix(&o.want_chat, sizeof o.want_chat);
    char buf[65536];
    rewind(out);
    size_t got = fread(buf, 1, sizeof(buf) - 1, out);
    buf[got]   = '\0';
    mix_replacing(buf, dir);
    char *dump = pz_config_format(&c);
    mix_replacing(dump ? dump : "(null)", dir);
    free(dump);
    pz_config_free(&c);
    fclose(in);
    fclose(out);
}

static void test_wizard_golden(void) {
    char dir[] = "/tmp/pz_wizg_XXXXXX";
    TEST_ASSERT(mkdtemp(dir) != NULL, "temp dir");
    const char *files[] = {"a.gguf", "b.gguf", "notes.txt"};
    for (int i = 0; i < 3; i++) {
        char f[300];
        snprintf(f, sizeof(f), "%s/%s", dir, files[i]);
        FILE *g = fopen(f, "w");
        if (g) {
            fputs("x", g);
            fclose(g);
        }
    }
    const char *dirs_with[] = {dir, NULL};
    PzStarterModel cat[3];
    fill_catalog(cat);
    for (int t = 0; t < N_SCRIPTS; t++) run_one(dir, cat, dirs_with);
    for (int i = 0; i < 3; i++) {
        char f[300];
        snprintf(f, sizeof(f), "%s/%s", dir, files[i]);
        remove(f);
    }
    rmdir(dir);
    if (getenv("PZ_WIZARD_GOLDEN_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "wizard behaviour matches the recorded hash");
}

int main(void) {
    printf("=== pz config wizard golden ===\n");
    RUN_TEST(test_wizard_golden);
    TEST_SUMMARY();
}
