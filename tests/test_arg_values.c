/*
 * test_arg_values.c -- numeric flag values are parsed strictly. "--rate-limit abc" used to become
 * 0 (atoi), which turns the limiter off; the same went for --client-timeout and --max-connections.
 */
#include <string.h>

#include "cli/arg_values.h"
#include "cli/args.h"
#include "test_harness.h"

static TernaryError parse(const char *flag, const char *value, CliArgs *args) {
    char *argv[] = {"prog", "--model", "m.gguf", (char *)flag, (char *)value};
    return parse_args(args, 5, argv);
}

static void test_helpers(void) {
    int i;
    double d;
    TEST_ASSERT(arg_to_int("42", &i) && i == 42, "integer parses");
    TEST_ASSERT(arg_to_int("-7", &i) && i == -7, "negative integer parses");
    TEST_ASSERT(!arg_to_int("abc", &i), "text is refused");
    TEST_ASSERT(!arg_to_int("", &i), "empty is refused");
    TEST_ASSERT(!arg_to_int("12x", &i), "trailing junk is refused");
    TEST_ASSERT(!arg_to_int("99999999999", &i), "out-of-range integer is refused");
    TEST_ASSERT(!arg_to_int("1.5", &i), "a fraction is not an integer");
    TEST_ASSERT(arg_to_double("0.7", &d) && d > 0.69 && d < 0.71, "number parses");
    TEST_ASSERT(arg_to_double("-1e2", &d) && d == -100.0, "exponent form parses");
    TEST_ASSERT(!arg_to_double("nan", &d) && !arg_to_double("inf", &d), "NaN and inf are refused");
    TEST_ASSERT(!arg_to_double("0.7abc", &d) && !arg_to_double("", &d), "junk and empty refused");
}

static void test_flags(void) {
    CliArgs a;
    TEST_ASSERT_EQ(parse("--rate-limit", "abc", &a), TN_ERR_INVALID_CONFIG,
                   "--rate-limit abc is an error, not 0");
    TEST_ASSERT_EQ(parse("--client-timeout", "ten", &a), TN_ERR_INVALID_CONFIG,
                   "--client-timeout ten is an error");
    TEST_ASSERT_EQ(parse("--max-connections", "64x", &a), TN_ERR_INVALID_CONFIG,
                   "--max-connections 64x is an error");
    TEST_ASSERT_EQ(parse("--temperature", "hot", &a), TN_ERR_INVALID_CONFIG,
                   "--temperature hot is an error");
    TEST_ASSERT_EQ(parse("--threads", "four", &a), TN_ERR_INVALID_CONFIG,
                   "--threads four is an error");
    TEST_ASSERT_EQ(parse("--rate-limit", "5", &a), TN_OK, "a valid --rate-limit is accepted");
    TEST_ASSERT_EQ(a.server.rate_limit, 5, "and stored");
    TEST_ASSERT_EQ(parse("--temperature", "0.25", &a), TN_OK, "a valid --temperature is accepted");
    TEST_ASSERT(a.sampling.temperature > 0.249f && a.sampling.temperature < 0.251f, "and stored");
    TEST_ASSERT_EQ(parse("--rate-limit", "0", &a), TN_OK,
                   "0 is still allowed when typed on purpose");
}

int main(void) {
    RUN_TEST(test_helpers);
    RUN_TEST(test_flags);
    TEST_SUMMARY();
}
