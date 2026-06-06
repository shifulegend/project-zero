#ifndef TN_SLIDING_WINDOW_H
#define TN_SLIDING_WINDOW_H

#include "core/platform.h"

/**
 * Sliding Window Attention Manager.
 *
 * Implements a circular buffer for KV cache positions, allowing
 * long-context generation without unbounded memory growth.
 *
 * System prompt tokens (0..system_prompt_len-1) are PINNED and
 * never evicted, ensuring the model always has access to its
 * instructions.
 */
typedef struct {
  int window_size;       /* Total physical slots available */
  int system_prompt_len; /* Number of pinned system prompt slots */
  int write_head;        /* Next write position in the circular region */
  bool wrapped;          /* Whether the circular region has wrapped around */
} SlidingWindow;

/**
 * Initialize the sliding window.
 *
 * @param sw                Output sliding window struct
 * @param window_size       Total KV cache slots (must be > system_prompt_len)
 * @param system_prompt_len Number of pinned slots at the start
 */
void sw_init(SlidingWindow *sw, int window_size, int system_prompt_len);

/**
 * Map a logical token position to a physical KV cache index.
 *
 * Positions < system_prompt_len map directly (pinned).
 * Positions >= system_prompt_len map into a circular buffer
 * in the range [system_prompt_len, window_size).
 *
 * @param sw           Sliding window state
 * @param logical_pos  Logical token position in the sequence
 * @return             Physical cache index [0, window_size)
 */
int sw_map_position(const SlidingWindow *sw, int logical_pos);

/**
 * Advance the write head after storing a new KV entry.
 * Must be called after each new token beyond the system prompt.
 *
 * @param sw  Sliding window state (modified in-place)
 */
void sw_advance(SlidingWindow *sw);

/**
 * Get the number of valid cached positions.
 * Includes pinned system prompt slots plus the filled circular region.
 *
 * @param sw           Sliding window state
 * @param logical_pos  Current logical position (total tokens seen)
 * @return             Number of valid cached KV entries
 */
int sw_valid_count(const SlidingWindow *sw, int logical_pos);

#endif /* TN_SLIDING_WINDOW_H */
