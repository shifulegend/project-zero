/*
 * tests/test_semver_parse_equiv.c — semver_parse over 400,000 generated strings: a hash of every
 * result code and the whole SemVer it fills, recorded from the single-function implementation that
 * was later split into parse_core and parse_prerelease. Covers versions with and without a v prefix,
 * numbers at and beyond the limit, pre-release tags, git-describe suffixes (alone and glued onto a
 * pre-release), -dirty, build metadata, over-long identifiers, junk and high-bit bytes.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "update/semver.h"

/* ── generator ─────────────────────────────────────────────────────────── */

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const k_prefix[] = {"", "", "v", "V", " ", "x"};
static const char *const k_num[]    = {"0",       "1",  "2", "10", "12", "999999", "1000000",
                                       "1000001", "01", "",  "-1", "a",  "7x"};
static const char *const k_dot[]    = {".", ".", ".", "..", ",", ""};
static const char *const k_suffix[] = {"",
                                       "",
                                       "",
                                       "-dirty",
                                       "-rc.1",
                                       "-rc.1-dirty",
                                       "-3-gabc1234",
                                       "-3-gabc1234-dirty",
                                       "-rc.1-3-gabc",
                                       "-rc.1-3-gabc-dirty",
                                       "-beta-3-g1f",
                                       "+build.5",
                                       "-rc.1+b",
                                       "-",
                                       "-+x",
                                       "--",
                                       "-g",
                                       "-3-g",
                                       "-3-gxyz",
                                       "-3-gabc-xyz",
                                       " ",
                                       "junk",
                                       "-dirty-dirty",
                                       "-a-dirty",
                                       "-0-g0",
                                       "-rc_1",
                                       "-\xc3\xa9",
                                       "-\x80\xff",
                                       "-rc.1.2.3.4.5.6.7.8.9.10.11.12.13.14.15",
                                       "-3-g-",
                                       "-rc.1-3",
                                       "-rc-g1-g2",
                                       "-rc.1-12-gdeadbeef+meta",
                                       "-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"};

/* One random pick from a table. Separate statements, not call arguments: the order C evaluates
 * arguments in is unspecified, and the recorded hash must not depend on the compiler. */
#define PICK(t) ((t)[rnd(sizeof(t) / sizeof((t)[0]))])

static void gen(char *out, size_t cap) {
    const char *prefix = PICK(k_prefix);
    const char *n1     = PICK(k_num);
    const char *d1     = PICK(k_dot);
    const char *n2     = PICK(k_num);
    const char *d2     = PICK(k_dot);
    const char *n3     = PICK(k_num);
    const char *suffix = PICK(k_suffix);
    snprintf(out, cap, "%s%s%s%s%s%s%s", prefix, n1, d1, n2, d2, n3, suffix);
}

#define EXPECTED_HASH 0x64d1bf05c1eda554ull

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static void test_matches_the_recorded_results(void) {
    long ok_count = 0;
    for (int i = 0; i < 400000; i++) {
        char text[128];
        gen(text, sizeof text);
        SemVer v;
        int rc = semver_parse(text, &v);
        ok_count += rc == 0;
        mix(&rc, sizeof rc);
        mix(&v, sizeof v);
    }
    if (getenv("SEMVER_EQ_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(ok_count > 3000, "the generator reaches the success paths");
    TEST_ASSERT(g_hash == EXPECTED_HASH, "results match the recorded hash");
    SemVer v;
    TEST_ASSERT(semver_parse(NULL, &v) == -1 && semver_parse("1.0.0", NULL) == -1, "null args");
}

int main(void) {
    printf("=== semver_parse equivalence ===\n");
    RUN_TEST(test_matches_the_recorded_results);
    TEST_SUMMARY();
}
