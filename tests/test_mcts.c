/*
 * tests/test_mcts.c — Phase 30 step search helpers: UCB1, self-score from
 * generation stats, option defaults, PRM input formatting and the two-token
 * judge score. (The search loop itself is exercised live against real GGUFs —
 * see IMPLEMENTATION_PLAN Phase 30.)
 */
#include "test_harness.h"
#include "reasoning/mcts.h"
#include "reasoning/prm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void test_ucb(void) {
    TEST_ASSERT(isinf(mcts_ucb(0.0, 0, 5, 1.4)), "unvisited child explored first");
    double a = mcts_ucb(3.0, 4, 10, 1.4);
    TEST_ASSERT_FLOAT_EQ(a, 0.75 + 1.4 * sqrt(log(10.0) / 4.0), 1e-9, "UCB1 formula");
    TEST_ASSERT(mcts_ucb(2.0, 4, 10, 1.4) < a, "higher value wins at equal visits");
    TEST_ASSERT(mcts_ucb(3.0, 4, 100, 1.4) > a, "exploration bonus grows with parent visits");
    TEST_ASSERT_FLOAT_EQ(mcts_ucb(1.0, 1, 1, 1.4), 1.0, 1e-9, "parent with one visit: no bonus");
}

static void test_self_score(void) {
    GenerationStats st;
    memset(&st, 0, sizeof(st));
    TEST_ASSERT_FLOAT_EQ(mcts_self_score(&st), 1.0f, 1e-7f, "no tokens (immediate end) -> 1");
    st.completion_tokens = 4;
    st.sum_logprob = 4.0 * log(0.5);
    TEST_ASSERT_FLOAT_EQ(mcts_self_score(&st), 0.5f, 1e-6f, "geometric mean probability");
    st.sum_logprob = 0.3;                      /* numerical noise above 0 */
    TEST_ASSERT_FLOAT_EQ(mcts_self_score(&st), 1.0f, 1e-7f, "clamped to 1");
    TEST_ASSERT_FLOAT_EQ(mcts_self_score(NULL), 1.0f, 1e-7f, "NULL safe");
}

static void test_defaults(void) {
    MctsOptions o;
    memset(&o, 0, sizeof(o));
    o.simulations = 5;
    o.expand = 2;
    mcts_options_defaults(&o);
    TEST_ASSERT(o.simulations == 5 && o.expand == 2, "set fields kept");
    TEST_ASSERT(o.max_depth == MCTS_DEFAULT_DEPTH && o.step_tokens == MCTS_DEFAULT_STEP_TOKENS,
                "unset fields defaulted");
    TEST_ASSERT(o.c_ucb > 0.0f && o.temperature > 0.0f && strcmp(o.step_delim, "\n") == 0,
                "c, temperature, delimiter defaulted");
}

static void test_prm_helpers(void) {
    char *s = prm_format("Q: {question} | S: {steps} | again {question}", "2+2?", "It is 4.");
    TEST_ASSERT(s && strcmp(s, "Q: 2+2? | S: It is 4. | again 2+2?") == 0, "placeholders substituted");
    free(s);
    s = prm_format(NULL, "q", "s");
    TEST_ASSERT(s && strstr(s, "Question: q") && strstr(s, "s\nIs the last step"), "default judge format");
    free(s);
    s = prm_format("{nope} {steps", "q", "s");
    TEST_ASSERT(s && strcmp(s, "{nope} {steps") == 0, "unknown/unclosed braces kept verbatim");
    free(s);

    float logits[4] = { 0.0f, 2.0f, 0.0f, 5.0f };
    float p = prm_score_from_logits(logits, 4, 1, 2);
    TEST_ASSERT_FLOAT_EQ(p, (float)(exp(2.0) / (exp(2.0) + 1.0)), 1e-6f, "P(good)/(P(good)+P(bad))");
    TEST_ASSERT_FLOAT_EQ(prm_score_from_logits(logits, 4, 2, 1) + p, 1.0f, 1e-6f, "complementary");
    float big[2] = { 1000.0f, 990.0f };
    TEST_ASSERT(prm_score_from_logits(big, 2, 0, 1) > 0.99f, "stable for large logits");
    TEST_ASSERT(prm_score_from_logits(logits, 4, 1, 1) < 0.0f, "same token refused");
    TEST_ASSERT(prm_score_from_logits(logits, 4, 1, 7) < 0.0f, "out-of-vocab refused");
}

int main(void) {
    printf("=== MCTS / PRM (Phase 30) Tests ===\n");
    RUN_TEST(test_ucb);
    RUN_TEST(test_self_score);
    RUN_TEST(test_defaults);
    RUN_TEST(test_prm_helpers);
    TEST_SUMMARY();
}
