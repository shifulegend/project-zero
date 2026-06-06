#include "reasoning/thought_filter.h"
#include <string.h>

/* The tags we're matching */
static const char OPEN_TAG[] = "<think>";
static const char CLOSE_TAG[] = "</think>";

void thought_filter_init(ThoughtFilter *f) {
  f->state = FILTER_PASSTHROUGH;
  f->tag_pos = 0;
  f->think_token_count = 0;
  memset(f->tag_buffer, 0, sizeof(f->tag_buffer));
}

int thought_filter_think_count(const ThoughtFilter *f) {
  return f->think_token_count;
}

bool thought_filter_process(ThoughtFilter *f, const char *token_str,
                            char *output_buf, size_t output_buf_size) {
  if (output_buf_size == 0)
    return false;

  output_buf[0] = '\0';

  int len = (int)strlen(token_str);
  if (len == 0)
    return false;

  for (int i = 0; i < len; i++) {
    char c = token_str[i];

    switch (f->state) {
    case FILTER_PASSTHROUGH:
      /* Look for start of <think> */
      if (c == '<') {
        f->state = FILTER_BUFFERING_OPEN_TAG;
        f->tag_buffer[0] = '<';
        f->tag_pos = 1;
      } else {
        /* Pass character through to output (with bounds check, QA-BUG-002) */
        int out_len = (int)strlen(output_buf);
        if (out_len + 1 < (int)output_buf_size) {
          output_buf[out_len] = c;
          output_buf[out_len + 1] = '\0';
        }
      }
      break;

    case FILTER_BUFFERING_OPEN_TAG:
      /* Accumulating characters to match "<think>" */
      if (f->tag_pos < 15) {
        f->tag_buffer[f->tag_pos] = c;
        f->tag_pos++;
        f->tag_buffer[f->tag_pos] = '\0';
      }

      if (f->tag_pos == 7 && memcmp(f->tag_buffer, OPEN_TAG, 7) == 0) {
        /* Full match: enter thinking mode */
        f->state = FILTER_THINKING;
        f->tag_pos = 0;
      } else if (f->tag_pos > 0 && OPEN_TAG[f->tag_pos - 1] != c) {
        /* Mismatch: flush buffered chars as output, return to passthrough */
        int out_len = (int)strlen(output_buf);
        /* Bounds check: ensure tag_buffer flush fits (QA-BUG-002) */
        if (out_len + f->tag_pos < (int)output_buf_size) {
          for (int j = 0; j < f->tag_pos; j++) {
            output_buf[out_len + j] = f->tag_buffer[j];
          }
          output_buf[out_len + f->tag_pos] = '\0';
        }
        f->state = FILTER_PASSTHROUGH;
        f->tag_pos = 0;
      }
      break;

    case FILTER_THINKING:
      /* Inside <think> block — consume tokens, watch for </think> */
      f->think_token_count++;
      if (c == '<') {
        f->state = FILTER_BUFFERING_CLOSE_TAG;
        f->tag_buffer[0] = '<';
        f->tag_pos = 1;
      }
      /* else: silently consume */
      break;

    case FILTER_BUFFERING_CLOSE_TAG:
      /* Accumulating characters to match "</think>" */
      if (f->tag_pos < 15) {
        f->tag_buffer[f->tag_pos] = c;
        f->tag_pos++;
        f->tag_buffer[f->tag_pos] = '\0';
      }

      if (f->tag_pos == 8 && memcmp(f->tag_buffer, CLOSE_TAG, 8) == 0) {
        /* Full match: exit thinking mode */
        f->state = FILTER_OUTPUT;
        f->tag_pos = 0;
      } else if (f->tag_pos > 0 && CLOSE_TAG[f->tag_pos - 1] != c) {
        /* Mismatch: stay in thinking mode */
        f->state = FILTER_THINKING;
        f->tag_pos = 0;
      }
      break;

    case FILTER_OUTPUT:
      /* Past </think>, output everything (with bounds check, QA-BUG-002) */
      {
        int out_len = (int)strlen(output_buf);
        if (out_len + 1 < (int)output_buf_size) {
          output_buf[out_len] = c;
          output_buf[out_len + 1] = '\0';
        }
      }
      break;
    }
  }

  return output_buf[0] != '\0';
}
