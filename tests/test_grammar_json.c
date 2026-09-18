/*
 * test_grammar_json.c — Phase 20: JSON-mode pushdown automaton correctness.
 */
#include "sampling/grammar.h"
#include "test_harness.h"

#include <string.h>

/* Feeds an entire C string through the automaton from a fresh state.
 * Returns 1 if every character was accepted (no JSON_ST_INVALID reached)
 * AND (after json_grammar_finalize) the automaton ended in JSON_ST_DONE.
 * Returns 0 if JSON_ST_INVALID was reached at any point, or if the string
 * is a valid-but-incomplete prefix (didn't reach DONE). */
static int feed_full(const char *s) {
    JsonGrammarState g;
    json_grammar_init(&g);
    for (size_t i = 0; s[i]; i++) {
        if (json_grammar_step(&g, s[i]) == JSON_ST_INVALID) return 0;
    }
    json_grammar_finalize(&g);
    return json_grammar_is_complete(&g);
}

/* Feeds a string and returns 1 if it never hit JSON_ST_INVALID (regardless
 * of whether it reached DONE) -- for testing "valid prefix, not yet closed". */
static int feed_never_invalid(const char *s) {
    JsonGrammarState g;
    json_grammar_init(&g);
    for (size_t i = 0; s[i]; i++) {
        if (json_grammar_step(&g, s[i]) == JSON_ST_INVALID) return 0;
    }
    return 1;
}

static void test_valid_complete_documents(void) {
    TEST_ASSERT(feed_full("{}"), "empty object");
    TEST_ASSERT(feed_full("[]"), "empty array");
    TEST_ASSERT(feed_full("{\"a\":1}"), "simple object");
    TEST_ASSERT(feed_full("[1,2,3]"), "simple array");
    TEST_ASSERT(feed_full("\"simple string\""), "bare string");
    TEST_ASSERT(feed_full("42"), "bare positive int");
    TEST_ASSERT(feed_full("-3.14e10"), "negative float with exponent");
    TEST_ASSERT(feed_full("-3.14E+10"), "exponent with explicit + sign");
    TEST_ASSERT(feed_full("true"), "bare true");
    TEST_ASSERT(feed_full("false"), "bare false");
    TEST_ASSERT(feed_full("null"), "bare null");
    TEST_ASSERT(feed_full("  {  \"a\"  :  1  }  "), "object with generous whitespace");
    TEST_ASSERT(feed_full("[[1,2],[3,4]]"), "nested arrays");
    TEST_ASSERT(feed_full("{\"a\":[1,2,{\"b\":true,\"c\":null}],\"d\":\"hello\\nworld\"}"),
                "deeply nested mixed object/array with escaped string");
    TEST_ASSERT(feed_full("\"\\u00e9\""), "string with unicode escape");
    TEST_ASSERT(feed_full("0"), "bare zero");
    TEST_ASSERT(feed_full("0.5"), "zero with fraction");
    TEST_ASSERT(feed_full("{\"nested\":{\"deep\":{\"deeper\":[1,2,3]}}}"),
                "three levels of object nesting plus an array");
}

/* RFC 8259: int = "0" / ( digit1-9 *DIGIT ) -- a leading zero must be the
 * whole integer part. Found via TS-2.1 differential fuzzing against
 * json.loads (2026-09-18): this PDA previously treated '0' like any other
 * leading digit, silently accepting "01"/"09"/etc. as complete valid JSON. */
static void test_leading_zero_numbers_rejected(void) {
    TEST_ASSERT(!feed_never_invalid("01"), "'01' goes INVALID (leading zero + digit)");
    TEST_ASSERT(!feed_never_invalid("09"), "'09' goes INVALID (leading zero + digit)");
    TEST_ASSERT(!feed_never_invalid("00"), "'00' goes INVALID (leading zero + digit)");
    TEST_ASSERT(!feed_never_invalid("-01"), "'-01' goes INVALID (negative leading zero + digit)");
    TEST_ASSERT(!feed_never_invalid("[01]"), "leading-zero number inside an array goes INVALID");
    TEST_ASSERT(!feed_never_invalid("{\"a\":01}"), "leading-zero number as an object value goes INVALID");
    /* These must NOT regress: '0' alone, and '0' followed by a legal
     * non-digit continuation (fraction/exponent/terminator), all still work. */
    TEST_ASSERT(feed_full("0"), "bare '0' still complete");
    TEST_ASSERT(feed_full("-0"), "bare '-0' still complete");
    TEST_ASSERT(feed_full("0.5"), "'0.5' still complete");
    TEST_ASSERT(feed_full("0e10"), "'0e10' still complete");
    TEST_ASSERT(feed_full("[0,1,2]"), "'[0,1,2]' (zero as one of several elements) still complete");
    TEST_ASSERT(feed_full("10"), "'10' (non-leading-zero multi-digit) still complete");
}

static void test_invalid_documents(void) {
    JsonGrammarState g;

    /* Single-quoted key: never legal. */
    TEST_ASSERT(!feed_never_invalid("{'a':1}"), "single-quoted key rejected");
    /* Trailing comma before close: strict JSON forbids it. */
    TEST_ASSERT(!feed_never_invalid("[1,2,]"), "trailing comma in array rejected");
    TEST_ASSERT(!feed_never_invalid("{\"a\":1,}"), "trailing comma in object rejected");
    /* Unquoted key. */
    TEST_ASSERT(!feed_never_invalid("{a:1}"), "unquoted key rejected");
    /* Missing comma between array elements. */
    TEST_ASSERT(!feed_never_invalid("[1 2]"), "missing comma between array elements rejected");
    /* Malformed literal. */
    TEST_ASSERT(!feed_never_invalid("trux"), "malformed 'true' literal rejected");
    /* Raw control byte inside a string (must be escaped per RFC 8259). */
    {
        char bad[] = { '"', 'a', 0x01, 'b', '"', '\0' };
        TEST_ASSERT(!feed_never_invalid(bad), "raw control byte in string rejected");
    }
    /* Extra content after a complete top-level value. */
    json_grammar_init(&g);
    {
        const char *s = "{}x";
        int saw_invalid = 0;
        for (size_t i = 0; s[i]; i++)
            if (json_grammar_step(&g, s[i]) == JSON_ST_INVALID) { saw_invalid = 1; break; }
        TEST_ASSERT(saw_invalid, "trailing garbage after a complete value rejected");
    }
    /* Bad escape sequence. */
    TEST_ASSERT(!feed_never_invalid("\"\\x\""), "unknown escape sequence rejected");
    /* Colon missing between key and value. */
    TEST_ASSERT(!feed_never_invalid("{\"a\" 1}"), "missing colon rejected");
}

static void test_valid_but_incomplete_prefixes(void) {
    /* These are legal PREFIXES of valid JSON -- must never hit INVALID,
     * but must not be considered DONE either (still open). */
    JsonGrammarState g;
    const char *prefixes[] = {
        "{\"a\":\"unterminated string",
        "[1,2,",
        "{\"key\":",
        "-3.14e",
        "tru",
        "{\"a\":{\"b\":1",
    };
    for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
        json_grammar_init(&g);
        int ok = 1;
        for (size_t j = 0; prefixes[i][j]; j++) {
            if (json_grammar_step(&g, prefixes[i][j]) == JSON_ST_INVALID) { ok = 0; break; }
        }
        TEST_ASSERT(ok, "valid prefix must not be rejected mid-stream");
        TEST_ASSERT(!json_grammar_is_complete(&g), "valid-but-incomplete prefix must not read as DONE");
    }
}

static void test_finalize_closes_bare_number(void) {
    JsonGrammarState g;
    json_grammar_init(&g);
    const char *s = "123";
    for (size_t i = 0; s[i]; i++)
        TEST_ASSERT(json_grammar_step(&g, s[i]) != JSON_ST_INVALID, "digit accepted");
    TEST_ASSERT(!json_grammar_is_complete(&g), "bare number not DONE before finalize (no trailing delimiter yet)");
    json_grammar_finalize(&g);
    TEST_ASSERT(json_grammar_is_complete(&g), "bare number IS complete after finalize");
}

static void test_deep_nesting_within_limit(void) {
    /* JSON_GRAMMAR_MAX_DEPTH levels of array nesting must still work. */
    JsonGrammarState g;
    json_grammar_init(&g);
    int ok = 1;
    for (int i = 0; i < 50; i++) if (json_grammar_step(&g, '[') == JSON_ST_INVALID) { ok = 0; break; }
    if (ok) for (int i = 0; i < 50; i++) if (json_grammar_step(&g, ']') == JSON_ST_INVALID) { ok = 0; break; }
    TEST_ASSERT(ok, "50 levels of array nesting within JSON_GRAMMAR_MAX_DEPTH accepted");
    TEST_ASSERT(json_grammar_is_complete(&g), "deeply nested empty arrays close back to DONE");
}

int main(void) {
    RUN_TEST(test_valid_complete_documents);
    RUN_TEST(test_invalid_documents);
    RUN_TEST(test_valid_but_incomplete_prefixes);
    RUN_TEST(test_finalize_closes_bare_number);
    RUN_TEST(test_leading_zero_numbers_rejected);
    RUN_TEST(test_deep_nesting_within_limit);
    TEST_SUMMARY();
}
