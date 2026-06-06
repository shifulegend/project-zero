#include "kv_cache/sliding_window.h"

void sw_init(SlidingWindow *sw, int window_size, int system_prompt_len) {
  sw->window_size = window_size;
  sw->system_prompt_len = system_prompt_len;
  /* Write head starts right after the pinned region */
  sw->write_head = system_prompt_len;
  sw->wrapped = false;
}

int sw_map_position(const SlidingWindow *sw, int logical_pos) {
  int spl = sw->system_prompt_len;

  /* System prompt positions are pinned — identity mapping */
  if (logical_pos < spl) {
    return logical_pos;
  }

  /* Non-prompt positions map into the circular region
   * [system_prompt_len, window_size) */
  int circular_size = sw->window_size - spl;
  if (circular_size <= 0)
    return spl; /* degenerate case */

  int offset = (logical_pos - spl) % circular_size;
  return spl + offset;
}

void sw_advance(SlidingWindow *sw) {
  int spl = sw->system_prompt_len;
  int circular_size = sw->window_size - spl;

  if (circular_size <= 0)
    return; /* degenerate case */

  sw->write_head++;

  /* Wrap around when we hit the end of the window */
  if (sw->write_head >= sw->window_size) {
    sw->write_head = spl;
    sw->wrapped = true;
  }
}

int sw_valid_count(const SlidingWindow *sw, int logical_pos) {
  int count = logical_pos + 1;
  return count > sw->window_size ? sw->window_size : count;
}
