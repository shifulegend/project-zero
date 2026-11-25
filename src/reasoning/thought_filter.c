#include "reasoning/thought_filter.h"

#include <string.h>

/* The tags we're matching */
static const char OPEN_TAG[]  = "<think>";
static const char CLOSE_TAG[] = "</think>";

void thought_filter_init(ThoughtFilter *f) {
    f->state             = FILTER_PASSTHROUGH;
    f->tag_pos           = 0;
    f->think_token_count = 0;
    memset(f->tag_buffer, 0, sizeof(f->tag_buffer));
}

int thought_filter_think_count(const ThoughtFilter *f) {
    return f->think_token_count;
}

/* Appends c to the output, if it fits (with bounds check, QA-BUG-002). */
static void emit_char(char *output_buf, size_t output_buf_size, char c) {
    int out_len = (int)strlen(output_buf);
    if (out_len + 1 < (int)output_buf_size) {
        output_buf[out_len]     = c;
        output_buf[out_len + 1] = '\0';
    }
}

/* Starts buffering a tag at a '<'. */
static void begin_tag(ThoughtFilter *f, FilterState state) {
    f->state         = state;
    f->tag_buffer[0] = '<';
    f->tag_pos       = 1;
}

/* Adds c to the partial-tag buffer (it holds at most 15 characters). */
static void buffer_tag_char(ThoughtFilter *f, char c) {
    if (f->tag_pos >= 15) return;
    f->tag_buffer[f->tag_pos] = c;
    f->tag_pos++;
    f->tag_buffer[f->tag_pos] = '\0';
}

/* True when the buffered text cannot be a prefix of `tag` any more, c being its last character. */
static bool tag_mismatch(const ThoughtFilter *f, const char *tag, size_t tag_len, char c) {
    return f->tag_pos > 0 && (f->tag_pos > (int)tag_len || tag[f->tag_pos - 1] != c);
}

/* Writes the buffered characters back out, if they fit (bounds check, QA-BUG-002). */
static void flush_tag_buffer(const ThoughtFilter *f, char *output_buf, size_t output_buf_size) {
    int out_len = (int)strlen(output_buf);
    if (out_len + f->tag_pos >= (int)output_buf_size) return;
    for (int j = 0; j < f->tag_pos; j++) output_buf[out_len + j] = f->tag_buffer[j];
    output_buf[out_len + f->tag_pos] = '\0';
}

/* Look for the start of <think>. */
static void step_passthrough(ThoughtFilter *f, char c, char *output_buf, size_t output_buf_size) {
    if (c == '<')
        begin_tag(f, FILTER_BUFFERING_OPEN_TAG);
    else
        emit_char(output_buf, output_buf_size, c);
}

/* Accumulating characters to match "<think>". */
static void step_open_tag(ThoughtFilter *f, char c, char *output_buf, size_t output_buf_size) {
    buffer_tag_char(f, c);
    if (f->tag_pos == 7 && memcmp(f->tag_buffer, OPEN_TAG, 7) == 0) {
        /* Full match: enter thinking mode */
        f->state   = FILTER_THINKING;
        f->tag_pos = 0;
    } else if (tag_mismatch(f, OPEN_TAG, sizeof(OPEN_TAG) - 1, c)) {
        /* Mismatch: flush buffered chars as output, return to passthrough */
        flush_tag_buffer(f, output_buf, output_buf_size);
        f->state   = FILTER_PASSTHROUGH;
        f->tag_pos = 0;
    }
}

/* Inside <think> block: consume tokens, watch for </think>. */
static void step_thinking(ThoughtFilter *f, char c) {
    f->think_token_count++;
    if (c == '<') begin_tag(f, FILTER_BUFFERING_CLOSE_TAG);
}

/* Accumulating characters to match "</think>". */
static void step_close_tag(ThoughtFilter *f, char c) {
    buffer_tag_char(f, c);
    if (f->tag_pos == 8 && memcmp(f->tag_buffer, CLOSE_TAG, 8) == 0) {
        /* Full match: exit thinking mode */
        f->state   = FILTER_OUTPUT;
        f->tag_pos = 0;
    } else if (tag_mismatch(f, CLOSE_TAG, sizeof(CLOSE_TAG) - 1, c)) {
        /* Mismatch: stay in thinking mode */
        f->state   = FILTER_THINKING;
        f->tag_pos = 0;
    }
}

static void step_char(ThoughtFilter *f, char c, char *output_buf, size_t output_buf_size) {
    switch (f->state) {
        case FILTER_PASSTHROUGH:
            step_passthrough(f, c, output_buf, output_buf_size);
            break;
        case FILTER_BUFFERING_OPEN_TAG:
            step_open_tag(f, c, output_buf, output_buf_size);
            break;
        case FILTER_THINKING:
            step_thinking(f, c);
            break;
        case FILTER_BUFFERING_CLOSE_TAG:
            step_close_tag(f, c);
            break;
        case FILTER_OUTPUT:
            /* Past </think>: output everything. */
            emit_char(output_buf, output_buf_size, c);
            break;
    }
}

bool thought_filter_process(ThoughtFilter *f, const char *token_str, char *output_buf,
                            size_t output_buf_size) {
    if (output_buf_size == 0) return false;

    output_buf[0] = '\0';

    int len = (int)strlen(token_str);
    if (len == 0) return false;

    for (int i = 0; i < len; i++) step_char(f, token_str[i], output_buf, output_buf_size);

    return output_buf[0] != '\0';
}
