/*
 * tests/test_selfcheck.c — the accuracy probe's pure parts (eval/selfcheck.c) and the
 * act-quant mode of the dispatcher: judging, formatting, name parsing, and that selecting
 * float activations really moves the ternary matmul off the int8 kernel and back.
 */
#include "test_harness.h"
#include "eval/selfcheck.h"
#include "math/simd_dispatch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
    r.verdict = SC_FAIL; strcpy(r.act_quant, "int8"); strcpy(r.classifier, "int4");
    r.ppl_ref = 21.12; r.ppl_cand = 377.06; r.delta_pct = 1685.2; r.tolerance_pct = 1.0; r.n_scored = 47;
    char line[300];
    selfcheck_format(&r, line, sizeof(line));
    TEST_ASSERT(strstr(line, "FAIL") && strstr(line, "head int4") && strstr(line, "+1685.20%"), "FAIL line");
    r.verdict = SC_SKIPPED; strcpy(r.note, "already the reference");
    selfcheck_format(&r, line, sizeof(line));
    TEST_ASSERT(strstr(line, "SKIPPED") && strstr(line, "already the reference"), "SKIPPED line");
    TEST_ASSERT(strlen(selfcheck_text()) > 400, "the built-in passage is long enough for a 96-token probe");
}

static void test_act_quant_modes(void) {
    TnActQuant m;
    TEST_ASSERT_EQ(tn_act_quant_parse("auto", &m), 0, "auto"); TEST_ASSERT_EQ(m, TN_ACT_AUTO, "auto value");
    TEST_ASSERT_EQ(tn_act_quant_parse("int8", &m), 0, "int8"); TEST_ASSERT_EQ(m, TN_ACT_INT8, "int8 value");
    TEST_ASSERT_EQ(tn_act_quant_parse("float", &m), 0, "float"); TEST_ASSERT_EQ(m, TN_ACT_FLOAT, "float value");
    TEST_ASSERT_EQ(tn_act_quant_parse("bf16", &m), -1, "unknown name rejected");
    TEST_ASSERT_EQ(tn_act_quant_parse(NULL, &m), -1, "NULL rejected");

    tn_simd_init();
    tn_act_quant_set(TN_ACT_AUTO);
    const char *before = tn_act_quant_effective();
    int preq_before = tn_simd_ternary_preq();
    const char *backend_before = tn_simd_backend_name();

    tn_act_quant_set(TN_ACT_FLOAT);
    TEST_ASSERT(strcmp(tn_act_quant_effective(), "float") == 0, "float mode runs float activations");
    TEST_ASSERT_EQ(tn_simd_ternary_preq(), 0, "float mode has no pre-quantised int8 path");

    tn_act_quant_set(TN_ACT_AUTO);
    TEST_ASSERT(strcmp(tn_act_quant_effective(), before) == 0, "auto restores the original numerics");
    TEST_ASSERT_EQ(tn_simd_ternary_preq(), preq_before, "auto restores the pre-quantised path");
    TEST_ASSERT(strcmp(tn_simd_backend_name(), backend_before) == 0, "and the same tier");

    /* A forced tier survives a mode switch (calibration and --simd bind it through the environment
     * for one tn_simd_init only). */
    setenv("TN_FORCE_BACKEND", "scalar", 1);
    tn_simd_init();
    unsetenv("TN_FORCE_BACKEND");
    tn_act_quant_set(TN_ACT_FLOAT);
    tn_act_quant_set(TN_ACT_AUTO);
    TEST_ASSERT(strcmp(tn_simd_backend_name(), "Scalar") == 0, "forced tier kept across act-quant changes");
    tn_simd_init();   /* back to auto for the tests that follow */
}

int main(void) {
    printf("=== Self-check / act-quant Tests ===\n");
    RUN_TEST(test_delta_and_judge);
    RUN_TEST(test_format_and_text);
    RUN_TEST(test_act_quant_modes);
    TEST_SUMMARY();
}
