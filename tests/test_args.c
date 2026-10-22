/* Phase 18 (speculative decoding) Stage 4: CLI-arg coverage for the new
 * --draft-model / --spec-length flags. No prior test_args.c existed for
 * parse_args() at all -- scoped here to the new flags this stage adds,
 * matching the plan's Stage 4 verification requirement, not a general
 * CLI-arg test sweep (a real but separate coverage gap, out of scope here). */
#include "test_harness.h"
#include "cli/args.h"
#include <string.h>

static void test_draft_model_flag_sets_path(void) {
    char *argv[] = {"prog", "--model", "m.gguf", "--draft-model", "draft.gguf"};
    CliArgs args;
    TernaryError err = parse_args(&args, 5, argv);
    TEST_ASSERT_EQ(err, TN_OK, "parse succeeds with --draft-model");
    TEST_ASSERT(args.draft_model_path != NULL, "draft_model_path is set");
    TEST_ASSERT(strcmp(args.draft_model_path, "draft.gguf") == 0, "draft_model_path value matches");
}

static void test_draft_model_defaults_to_null(void) {
    char *argv[] = {"prog", "--model", "m.gguf"};
    CliArgs args;
    TernaryError err = parse_args(&args, 3, argv);
    TEST_ASSERT_EQ(err, TN_OK, "parse succeeds without --draft-model");
    TEST_ASSERT(args.draft_model_path == NULL,
                "draft_model_path defaults to NULL -- speculative decoding is opt-in only");
}

static void test_spec_length_flag_sets_value(void) {
    char *argv[] = {"prog", "--model", "m.gguf", "--draft-model", "d.gguf", "--spec-length", "8"};
    CliArgs args;
    TernaryError err = parse_args(&args, 7, argv);
    TEST_ASSERT_EQ(err, TN_OK, "parse succeeds with --spec-length");
    TEST_ASSERT_EQ(args.spec_length, 8, "spec_length value matches");
}

static void test_spec_length_defaults_to_five(void) {
    char *argv[] = {"prog", "--model", "m.gguf"};
    CliArgs args;
    TernaryError err = parse_args(&args, 3, argv);
    TEST_ASSERT_EQ(err, TN_OK, "parse succeeds without --spec-length");
    TEST_ASSERT_EQ(args.spec_length, 5, "spec_length defaults to 5");
}

static void test_spec_length_zero_rejected(void) {
    char *argv[] = {"prog", "--model", "m.gguf", "--spec-length", "0"};
    CliArgs args;
    TernaryError err = parse_args(&args, 5, argv);
    TEST_ASSERT_EQ(err, TN_ERR_INVALID_CONFIG, "--spec-length 0 is rejected");
}

static void test_spec_length_negative_rejected(void) {
    char *argv[] = {"prog", "--model", "m.gguf", "--spec-length", "-3"};
    CliArgs args;
    TernaryError err = parse_args(&args, 5, argv);
    TEST_ASSERT_EQ(err, TN_ERR_INVALID_CONFIG, "--spec-length -3 is rejected");
}

static void test_spec_length_missing_value_rejected(void) {
    char *argv[] = {"prog", "--model", "m.gguf", "--spec-length"};
    CliArgs args;
    TernaryError err = parse_args(&args, 4, argv);
    TEST_ASSERT_EQ(err, TN_ERR_INVALID_CONFIG, "--spec-length with no value is rejected");
}

static void test_draft_model_missing_value_rejected(void) {
    char *argv[] = {"prog", "--model", "m.gguf", "--draft-model"};
    CliArgs args;
    TernaryError err = parse_args(&args, 4, argv);
    TEST_ASSERT_EQ(err, TN_ERR_INVALID_CONFIG, "--draft-model with no value is rejected");
}

int main(void) {
    RUN_TEST(test_draft_model_flag_sets_path);
    RUN_TEST(test_draft_model_defaults_to_null);
    RUN_TEST(test_spec_length_flag_sets_value);
    RUN_TEST(test_spec_length_defaults_to_five);
    RUN_TEST(test_spec_length_zero_rejected);
    RUN_TEST(test_spec_length_negative_rejected);
    RUN_TEST(test_spec_length_missing_value_rejected);
    RUN_TEST(test_draft_model_missing_value_rejected);
    TEST_SUMMARY();
}
