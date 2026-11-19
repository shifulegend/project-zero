/*
 * tests/test_selfcheck.c — the accuracy probe's pure parts (eval/selfcheck.c) and the
 * act-quant mode of the dispatcher: judging, formatting, name parsing, and that selecting
 * float activations really moves the ternary matmul off the int8 kernel and back.
 */
#include "test_harness.h"
#include "eval/selfcheck.h"
#include "math/simd_dispatch.h"
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void test_delta_and_judge(void) {
    TEST_ASSERT(fabs(selfcheck_delta_pct(10.0, 10.1) - 1.0) < 1e-9, "+1% perplexity");
    TEST_ASSERT(selfcheck_delta_pct(10.0, 9.0) < 0, "better than the reference is negative");
    TEST_ASSERT(isnan(selfcheck_delta_pct(0.0, 5.0)), "invalid reference");
    TEST_ASSERT(isnan(selfcheck_delta_pct(5.0, NAN)), "invalid candidate");
    TEST_ASSERT_EQ(selfcheck_judge(0.99, 1.0), SC_PASS, "inside the tolerance");
    TEST_ASSERT_EQ(selfcheck_judge(1.0, 1.0), SC_PASS, "exactly at the tolerance");
    TEST_ASSERT_EQ(selfcheck_judge(1.01, 1.0), SC_FAIL, "over the tolerance");
    TEST_ASSERT_EQ(selfcheck_judge(-5.0, 1.0), SC_PASS, "an improvement passes");
    TEST_ASSERT_EQ(selfcheck_judge(NAN, 1.0), SC_FAIL, "non-finite fails");
    TEST_ASSERT_EQ(selfcheck_judge(1500.0, 0.0), SC_PASS, "tolerance 0 = no bar");
    TEST_ASSERT_EQ(selfcheck_judge(NAN, 0.0), SC_FAIL, "no bar still rejects non-finite values");
}

static void test_format_and_text(void) {
    SelfCheckResult r;
    memset(&r, 0, sizeof(r));
    r.verdict = SC_FAIL;
    strcpy(r.act_quant, "int8");
    strcpy(r.classifier, "int4");
    r.ppl_ref       = 21.12;
    r.ppl_cand      = 377.06;
    r.delta_pct     = 1685.2;
    r.tolerance_pct = 1.0;
    r.n_scored      = 47;
    char line[300];
    selfcheck_format(&r, line, sizeof(line));
    TEST_ASSERT(strstr(line, "FAIL") && strstr(line, "head int4") && strstr(line, "+1685.20%"),
                "FAIL line");
    r.verdict = SC_SKIPPED;
    strcpy(r.note, "already the reference");
    selfcheck_format(&r, line, sizeof(line));
    TEST_ASSERT(strstr(line, "SKIPPED") && strstr(line, "already the reference"), "SKIPPED line");
    TEST_ASSERT(strlen(selfcheck_text()) > 400,
                "the built-in passage is long enough for a 96-token probe");
}

static void test_act_quant_modes(void) {
    TnActQuant m;
    TEST_ASSERT_EQ(tn_act_quant_parse("auto", &m), 0, "auto");
    TEST_ASSERT_EQ(m, TN_ACT_AUTO, "auto value");
    TEST_ASSERT_EQ(tn_act_quant_parse("int8", &m), 0, "int8");
    TEST_ASSERT_EQ(m, TN_ACT_INT8, "int8 value");
    TEST_ASSERT_EQ(tn_act_quant_parse("float", &m), 0, "float");
    TEST_ASSERT_EQ(m, TN_ACT_FLOAT, "float value");
    TEST_ASSERT_EQ(tn_act_quant_parse("bf16", &m), -1, "unknown name rejected");
    TEST_ASSERT_EQ(tn_act_quant_parse(NULL, &m), -1, "NULL rejected");

    tn_simd_init();
    tn_act_quant_set(TN_ACT_AUTO);
    const char *before         = tn_act_quant_effective();
    int preq_before            = tn_simd_ternary_preq();
    const char *backend_before = tn_simd_backend_name();

    tn_act_quant_set(TN_ACT_FLOAT);
    TEST_ASSERT(strcmp(tn_act_quant_effective(), "float") == 0,
                "float mode runs float activations");
    TEST_ASSERT_EQ(tn_simd_ternary_preq(), 0, "float mode has no pre-quantised int8 path");

    tn_act_quant_set(TN_ACT_AUTO);
    TEST_ASSERT(strcmp(tn_act_quant_effective(), before) == 0,
                "auto restores the original numerics");
    TEST_ASSERT_EQ(tn_simd_ternary_preq(), preq_before, "auto restores the pre-quantised path");
    TEST_ASSERT(strcmp(tn_simd_backend_name(), backend_before) == 0, "and the same tier");

    /* A forced tier survives a mode switch (calibration and --simd bind it through the environment
     * for one tn_simd_init only). */
    setenv("TN_FORCE_BACKEND", "scalar", 1);
    tn_simd_init();
    unsetenv("TN_FORCE_BACKEND");
    tn_act_quant_set(TN_ACT_FLOAT);
    tn_act_quant_set(TN_ACT_AUTO);
    TEST_ASSERT(strcmp(tn_simd_backend_name(), "Scalar") == 0,
                "forced tier kept across act-quant changes");
    tn_simd_init(); /* back to auto for the tests that follow */
}

static void test_cache(void) {
    char k1[300], k2[300], k3[300];
    selfcheck_cache_key(k1, sizeof(k1), "/models/bitnet.gguf", 1187801280LL, 1700000000LL, "int8",
                        "bf16", 1.0, 96);
    selfcheck_cache_key(k2, sizeof(k2), "/other/dir/bitnet.gguf", 1187801280LL, 1700000000LL,
                        "int8", "bf16", 1.0, 96);
    selfcheck_cache_key(k3, sizeof(k3), "/models/bitnet.gguf", 1187801280LL, 1700000001LL, "int8",
                        "bf16", 1.0, 96);
    TEST_ASSERT(strcmp(k1, k2) == 0, "same file name, size and mtime: same key wherever it lives");
    TEST_ASSERT(strcmp(k1, k3) != 0, "a changed mtime is another file");
    char text[1024];
    snprintf(text, sizeof(text), "%s\tFAIL\nother\tPASS\n%s\tPASS\n", k3, k1);
    SelfCheckVerdict v = SC_SKIPPED;
    TEST_ASSERT_EQ(selfcheck_cache_find(text, k1, &v), 1, "found");
    TEST_ASSERT_EQ(v, SC_PASS, "verdict of the matching line");
    TEST_ASSERT_EQ(selfcheck_cache_find(text, k3, &v), 1, "other key found");
    TEST_ASSERT_EQ(v, SC_FAIL, "FAIL read back");
    TEST_ASSERT_EQ(selfcheck_cache_find(text, "missing", &v), 0, "miss");
    TEST_ASSERT_EQ(selfcheck_cache_find("", k1, &v), 0, "empty cache");
    TEST_ASSERT_EQ(selfcheck_cache_find(k1, k1, &v), 0, "a key without a verdict is a miss");
}

/* File-backed cache: what main.c reads on the next start. A stored FAIL is what switches `auto` to
 * float activations; a cache that cannot be written or read is only a miss. */
static void test_cache_file(void) {
    char dir[64];
    snprintf(dir, sizeof(dir), "/tmp/pz_selfcheck_%ld",
             (long)getpid()); /* mkdtemp is not declared under strict POSIX on macOS */
    TEST_ASSERT(mkdir(dir, 0700) == 0 || errno == EEXIST, "temp HOME");
    const char *old = getenv("HOME");
    char *saved     = old ? strdup(old) : NULL;
    setenv("HOME", dir, 1);

    SelfCheckVerdict v = SC_SKIPPED;
    TEST_ASSERT_EQ(selfcheck_cache_load("k-pass", &v), 0, "empty: miss");
    selfcheck_cache_store("k-pass", SC_PASS);
    selfcheck_cache_store("k-fail", SC_FAIL);
    selfcheck_cache_store("k-skip", SC_SKIPPED);
    TEST_ASSERT_EQ(selfcheck_cache_load("k-pass", &v), 1, "stored PASS found");
    TEST_ASSERT_EQ(v, SC_PASS, "PASS");
    TEST_ASSERT_EQ(selfcheck_cache_load("k-fail", &v), 1, "stored FAIL found");
    TEST_ASSERT_EQ(v, SC_FAIL, "FAIL read back");
    TEST_ASSERT_EQ(selfcheck_cache_load("k-skip", &v), 0, "SKIPPED is never stored");

    setenv("HOME", "/nonexistent/dir/for/pz", 1);
    selfcheck_cache_store("k-x", SC_FAIL); /* must not crash */
    TEST_ASSERT_EQ(selfcheck_cache_load("k-fail", &v), 0, "unreadable cache is a miss");

    if (saved) {
        setenv("HOME", saved, 1);
        free(saved);
    } else
        unsetenv("HOME");
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
    TEST_ASSERT_EQ(system(cmd), 0, "cleanup");
}

static void test_auto_decisions(void) {
    SelfCheckAutoCtx c = {0, 1, 0, 0, 1.0, 0, 1};
    TEST_ASSERT_EQ(selfcheck_auto_applies(&c), 1, "auto + ternary + int8 + bar: checks");
    SelfCheckAutoCtx x    = c;
    x.act_quant_requested = 1;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0, "explicit int8: no check");
    x                     = c;
    x.act_quant_requested = 2;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0, "explicit float: no check");
    x               = c;
    x.ternary_model = 0;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0, "not a ternary model");
    x              = c;
    x.no_calibrate = 1;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0, "--no-calibrate");
    x                 = c;
    x.sanitized_build = 1;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0, "sanitizer build");
    x               = c;
    x.tolerance_pct = 0.0;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0, "tolerance 0 = no bar");
    x            = c;
    x.other_mode = 1;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0,
                   "--self-check / --perplexity / --watermark-detect");
    x                   = c;
    x.effective_is_int8 = 0;
    TEST_ASSERT_EQ(selfcheck_auto_applies(&x), 0, "already float activations");
    TEST_ASSERT(selfcheck_auto_applies(NULL) == 0, "NULL");
    TEST_ASSERT_EQ(selfcheck_auto_numerics(SC_FAIL), TN_ACT_FLOAT,
                   "a FAIL verdict puts float activations in force");
    TEST_ASSERT_EQ(selfcheck_auto_numerics(SC_PASS), TN_ACT_AUTO, "PASS keeps auto");
    TEST_ASSERT_EQ(selfcheck_auto_numerics(SC_SKIPPED), TN_ACT_AUTO, "SKIPPED keeps auto");
}

int main(void) {
    printf("=== Self-check / act-quant Tests ===\n");
    RUN_TEST(test_delta_and_judge);
    RUN_TEST(test_format_and_text);
    RUN_TEST(test_act_quant_modes);
    RUN_TEST(test_cache);
    RUN_TEST(test_cache_file);
    RUN_TEST(test_auto_decisions);
    TEST_SUMMARY();
}
