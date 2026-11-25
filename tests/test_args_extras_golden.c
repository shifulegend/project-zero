/*
 * tests/test_args_extras_golden.c — the draft-model / LoRA / colour / update-check flags of
 * parse_args() over every flag x a spread of values (empty, 0, 1, -1, 24, 8760, 8761, overflow,
 * words, the three colour modes in two cases, the update channels, repositories and URLs, another
 * flag) alone, missing its value, and followed by a second flag. The hash covers the result code
 * and every field these flags set, recorded from the parser before it was split into a table plus
 * handlers; it complements tests/test_args_golden.c, whose sample values do not reach the exact
 * range limits or every colour mode.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/args.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x553a9fec784a2537ull

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

static const char *const FLAGS[] = {
    "--draft-model",     "--spec-length", "--lora",       "--color",           "--check-update",
    "--no-update-check", "--update-repo", "--update-api", "--update-interval", "--update-channel",
};
static const char *const VALUES[] = {
    "",
    "0",
    "1",
    "-1",
    "5",
    "24",
    "8760",
    "8761",
    "2147483647",
    "2147483648",
    "99999999999",
    "abc",
    "1x",
    "auto",
    "always",
    "never",
    "AUTO",
    "Never",
    "stable",
    "prerelease",
    "beta",
    "owner/name",
    "owner",
    "a/b/c",
    "/name",
    "https://api.example.com",
    "http://x",
    "https://",
    "--lora",
    "--color",
    "path/to/file.gguf",
};

static void run_case(const char *const *av, int ac) {
    char *argv[8];
    for (int k = 0; k < ac; k++) argv[k] = (char *)av[k];
    CliArgs a;
    memset(&a, 0, sizeof a);
    int rc = (int)parse_args(&a, ac, argv);
    mix(&rc, sizeof rc);
    mix(&a.output.color_mode, sizeof a.output.color_mode);
    mix(&a.search.spec_length, sizeof a.search.spec_length);
    mix(&a.update.check_update, sizeof a.update.check_update);
    mix(&a.update.no_update_check, sizeof a.update.no_update_check);
    mix(&a.update.update_interval_h, sizeof a.update.update_interval_h);
    mix(&a.update.update_channel, sizeof a.update.update_channel);
    mix_str(a.update.update_repo);
    mix_str(a.update.update_api);
    mix_str(a.input.draft_model_path);
    mix_str(a.input.lora_path);
}

static void test_args_extras_golden(void) {
    const char *av[8];
    size_t nf = sizeof FLAGS / sizeof FLAGS[0], nv = sizeof VALUES / sizeof VALUES[0];
    for (size_t f = 0; f < nf; f++) {
        av[0] = "pz";
        av[1] = FLAGS[f];
        run_case(av, 2); /* missing value */
        av[2] = "--lora";
        av[3] = "L";
        run_case(av, 4); /* the next flag taken as the value, or as a flag */
        for (size_t v = 0; v < nv; v++) {
            av[1] = FLAGS[f];
            av[2] = VALUES[v];
            run_case(av, 3);
            av[3] = "--update-channel";
            av[4] = "stable";
            run_case(av, 5); /* the parser moved on correctly */
        }
    }
    if (getenv("ARGS_EXTRAS_GOLDEN_PRINT"))
        printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "flag parsing matches the recorded hash");
}

int main(void) {
    freopen("/dev/null", "w", stderr); /* error messages are not what is pinned here */
    printf("=== args extras golden ===\n");
    RUN_TEST(test_args_extras_golden);
    TEST_SUMMARY();
}
