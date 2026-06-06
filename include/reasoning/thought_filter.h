#ifndef TN_THOUGHT_FILTER_H
#define TN_THOUGHT_FILTER_H

#include "core/platform.h"

/**
 * State machine for filtering hidden <think>...</think> reasoning tokens.
 *
 * Tokens within <think> tags are consumed (hidden from user output).
 * All other tokens pass through for display.
 */
typedef enum {
  FILTER_PASSTHROUGH,         /* Normal output mode */
  FILTER_BUFFERING_OPEN_TAG,  /* Accumulating partial "<think>" match */
  FILTER_THINKING,            /* Inside <think> block, consuming tokens */
  FILTER_BUFFERING_CLOSE_TAG, /* Accumulating partial "</think>" match */
  FILTER_OUTPUT               /* Past </think>, back to output mode */
} FilterState;

/**
 * Thought filter context.
 */
typedef struct {
  FilterState state;
  char tag_buffer[16];   /* Buffer for partial tag match */
  int tag_pos;           /* Current position in tag_buffer */
  int think_token_count; /* Number of tokens consumed during thinking */
} ThoughtFilter;

/**
 * Initialize the thought filter to passthrough mode.
 */
void thought_filter_init(ThoughtFilter *f);

/**
 * Process a single decoded token string through the filter.
 *
 * @param f               Filter state (modified in-place)
 * @param token_str       The decoded token string (null-terminated)
 * @param output_buf      Output buffer for text to display
 * @param output_buf_size Size of output_buf in bytes (must be >= 1)
 * @return                true if output_buf contains text to print,
 *                        false if the token was consumed (hidden thinking)
 */
bool thought_filter_process(ThoughtFilter *f, const char *token_str,
                            char *output_buf, size_t output_buf_size);

/**
 * Get the number of tokens consumed during <think> blocks.
 */
int thought_filter_think_count(const ThoughtFilter *f);

#endif /* TN_THOUGHT_FILTER_H */
