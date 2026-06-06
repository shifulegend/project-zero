#include "reasoning/prompt_inject.h"
#include <string.h>

static const char REASONING_SUFFIX[] =
    "\nThink step-by-step inside <think> and </think> tags before answering.";

void inject_reasoning_prompt(char *output_prompt, const char *user_prompt,
                             int max_len) {
  if (max_len <= 0)
    return;

  int prompt_len = (int)strlen(user_prompt);
  int suffix_len = (int)sizeof(REASONING_SUFFIX) - 1; /* exclude null */

  /* Copy the user prompt first */
  int copy_len = prompt_len;
  if (copy_len >= max_len) {
    copy_len = max_len - 1;
  }
  memcpy(output_prompt, user_prompt, copy_len);

  /* Append the reasoning suffix if there's room */
  int remaining = max_len - copy_len - 1; /* -1 for null terminator */
  if (remaining > 0 && suffix_len > 0) {
    int append_len = suffix_len < remaining ? suffix_len : remaining;
    memcpy(output_prompt + copy_len, REASONING_SUFFIX, append_len);
    copy_len += append_len;
  }

  output_prompt[copy_len] = '\0';
}
