/**
 * test_reasoning.c — Unit tests for Phase 9: Hidden Reasoning Engine
 *
 * Tests:
 *  1. ThoughtFilter state machine
 *  2. Prompt injector
 *  3. Edge cases and partial tag matches
 */

#include "reasoning/prompt_inject.h"
#include "reasoning/thought_filter.h"
#include "test_harness.h"

#include <string.h>

/* ================================================================
 * TEST: ThoughtFilter State Machine (Phase 9.1)
 * ================================================================ */

static void test_filter_init(void) {
  ThoughtFilter f;
  thought_filter_init(&f);
  TEST_ASSERT(f.state == FILTER_PASSTHROUGH, "initial state is passthrough");
  TEST_ASSERT_EQ(f.tag_pos, 0, "tag_pos is 0");
  TEST_ASSERT_EQ(f.think_token_count, 0, "think count is 0");
}

static void test_filter_passthrough(void) {
  ThoughtFilter f;
  thought_filter_init(&f);
  char out[256];

  bool show = thought_filter_process(&f,"hello world", out, sizeof(out));
  TEST_ASSERT(show, "passthrough returns true");
  TEST_ASSERT(strcmp(out, "hello world") == 0, "passthrough text preserved");
}

static void test_filter_full_think_block(void) {
  ThoughtFilter f;
  thought_filter_init(&f);
  char out[256];

  /* Opening tag */
  bool show = thought_filter_process(&f,"<think>", out, sizeof(out));
  TEST_ASSERT(!show, "opening tag not shown");
  TEST_ASSERT(f.state == FILTER_THINKING, "state is THINKING");

  /* Hidden thinking content */
  show = thought_filter_process(&f,"let me reason", out, sizeof(out));
  TEST_ASSERT(!show, "thinking content hidden");

  show = thought_filter_process(&f,"step 1 is...", out, sizeof(out));
  TEST_ASSERT(!show, "more thinking hidden");

  /* Closing tag */
  show = thought_filter_process(&f,"</think>", out, sizeof(out));
  TEST_ASSERT(!show, "closing tag not shown");
  TEST_ASSERT(f.state == FILTER_OUTPUT, "state is OUTPUT after close tag");

  /* Now output should pass through */
  show = thought_filter_process(&f,"The answer is 42.", out, sizeof(out));
  TEST_ASSERT(show, "post-think text shown");
  TEST_ASSERT(strcmp(out, "The answer is 42.") == 0, "output text correct");
}

static void test_filter_think_count(void) {
  ThoughtFilter f;
  thought_filter_init(&f);
  char out[256];

  thought_filter_process(&f,"<think>", out, sizeof(out));
  thought_filter_process(&f,"a", out, sizeof(out));
  thought_filter_process(&f,"b", out, sizeof(out));
  thought_filter_process(&f,"c", out, sizeof(out));
  thought_filter_process(&f,"</think>", out, sizeof(out));

  int count = thought_filter_think_count(&f);
  TEST_ASSERT(count > 0, "think count > 0 after thinking");
}

static void test_filter_no_tags_passthrough(void) {
  ThoughtFilter f;
  thought_filter_init(&f);
  char out[256];

  /* Text without any tags should all pass through */
  bool show = thought_filter_process(&f,"no tags here", out, sizeof(out));
  TEST_ASSERT(show, "no tags = passthrough");
  TEST_ASSERT(strcmp(out, "no tags here") == 0, "text preserved");

  show = thought_filter_process(&f," more text", out, sizeof(out));
  TEST_ASSERT(show, "continued passthrough");
}

static void test_filter_combined_token(void) {
  /* Full think block in a single token */
  ThoughtFilter f;
  thought_filter_init(&f);
  char out[256];

  bool show = thought_filter_process(&f,"<think>hidden</think>visible", out, sizeof(out));
  TEST_ASSERT(show, "combined token has visible output");
  TEST_ASSERT(strcmp(out, "visible") == 0, "only post-think text shown");
}

static void test_filter_multiple_think_blocks(void) {
  ThoughtFilter f;
  thought_filter_init(&f);
  char out[256];

  /* First think block */
  thought_filter_process(&f,"<think>hidden1</think>", out, sizeof(out));

  /* After first block, state is OUTPUT — everything passes through,
   * including another <think> tag (only first block is filtered) */
  bool show = thought_filter_process(&f,"answer1 ", out, sizeof(out));
  TEST_ASSERT(show, "text after first think passes through");
}

static void test_filter_empty_token(void) {
  ThoughtFilter f;
  thought_filter_init(&f);
  char out[256];

  bool show = thought_filter_process(&f,"", out, sizeof(out));
  TEST_ASSERT(!show, "empty token returns false");
}

/* ================================================================
 * TEST: Prompt Injector (Phase 9.2)
 * ================================================================ */

static void test_inject_basic(void) {
  char result[512];
  inject_reasoning_prompt(result, "What is 2+2?", 512);

  /* Should contain original prompt */
  TEST_ASSERT(strstr(result, "What is 2+2?") != NULL,
              "original prompt preserved");

  /* Should contain reasoning instruction */
  TEST_ASSERT(strstr(result, "<think>") != NULL,
              "reasoning instruction appended");
  TEST_ASSERT(strstr(result, "</think>") != NULL, "close tag in instruction");
  TEST_ASSERT(strstr(result, "step-by-step") != NULL,
              "step-by-step in instruction");
}

static void test_inject_empty_prompt(void) {
  char result[512];
  inject_reasoning_prompt(result, "", 512);

  /* Should still have the reasoning suffix */
  TEST_ASSERT(strlen(result) > 0, "non-empty output for empty prompt");
  TEST_ASSERT(strstr(result, "Think") != NULL, "reasoning instruction present");
}

static void test_inject_small_buffer(void) {
  char result[20];
  inject_reasoning_prompt(result, "Hello world", 20);

  /* Should be truncated but null-terminated */
  TEST_ASSERT(strlen(result) < 20, "result fits buffer");
  TEST_ASSERT(result[19] == '\0' || strlen(result) <= 19, "null terminated");
}

static void test_inject_exact_fit(void) {
  char result[256];
  const char *prompt = "test";
  inject_reasoning_prompt(result, prompt, 256);

  /* Result should start with the prompt */
  TEST_ASSERT(strncmp(result, prompt, strlen(prompt)) == 0,
              "starts with prompt");
}

static void test_inject_preserves_original(void) {
  const char *original = "Tell me a joke";
  char copy[256];
  strcpy(copy, original);

  char result[512];
  inject_reasoning_prompt(result, copy, 512);

  /* Original must not be modified */
  TEST_ASSERT(strcmp(copy, original) == 0, "original prompt unmodified");
}

static void test_inject_zero_buffer(void) {
  char result[4] = "abc";
  inject_reasoning_prompt(result, "test", 0);

  /* Buffer with max_len=0 should not crash, and result should be unchanged */
  TEST_ASSERT(result[0] == 'a', "zero buffer doesn't crash");
}

/* ================================================================ */

int main(void) {
  RUN_TEST(test_filter_init);
  RUN_TEST(test_filter_passthrough);
  RUN_TEST(test_filter_full_think_block);
  RUN_TEST(test_filter_think_count);
  RUN_TEST(test_filter_no_tags_passthrough);
  RUN_TEST(test_filter_combined_token);
  RUN_TEST(test_filter_multiple_think_blocks);
  RUN_TEST(test_filter_empty_token);

  RUN_TEST(test_inject_basic);
  RUN_TEST(test_inject_empty_prompt);
  RUN_TEST(test_inject_small_buffer);
  RUN_TEST(test_inject_exact_fit);
  RUN_TEST(test_inject_preserves_original);
  RUN_TEST(test_inject_zero_buffer);

  TEST_SUMMARY();
}
