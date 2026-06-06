#include "agent/tool_interceptor.h"
#include <string.h>
#include <stdio.h>

/* Simple streaming tag parser. Buffers partial pieces between calls. */
static char buffer[8192];
static size_t buf_pos = 0;

void ti_init(void) {
    buf_pos = 0;
    buffer[0] = '\0';
}

void ti_reset(void) { ti_init(); }

static const struct { const char *open; const char *close; TI_Tag tag; } tags[] = {
    {"<exec>", "</exec>", TI_TAG_EXEC},
    {"<think>", "</think>", TI_TAG_THINK},
    {"<save_memory>", "</save_memory>", TI_TAG_SAVE_MEMORY},
    {"<search_memory>", "</search_memory>", TI_TAG_SEARCH_MEMORY},
    {"<result>", "</result>", TI_TAG_RESULT},
};

TI_Tag ti_process_piece(const char *piece, char *out_buf, size_t out_buf_size) {
    if (!piece || piece[0] == '\0') return TI_TAG_NONE;

    size_t piece_len = strlen(piece);
    /* Append piece into buffer (truncate if full) */
    size_t space = sizeof(buffer) - 1 - buf_pos;
    size_t to_copy = piece_len < space ? piece_len : space;
    if (to_copy > 0) {
        memcpy(buffer + buf_pos, piece, to_copy);
        buf_pos += to_copy;
        buffer[buf_pos] = '\0';
    }

    /* Look for any complete tag */
    for (size_t i = 0; i < sizeof(tags)/sizeof(tags[0]); i++) {
        char *open = strstr(buffer, tags[i].open);
        if (!open) continue;
        char *close = strstr(open + strlen(tags[i].open), tags[i].close);
        if (!close) continue; /* wait for closing tag */

        /* Extract inner content */
        char *start = open + strlen(tags[i].open);
        size_t inner_len = (size_t)(close - start);
        size_t copy_len = (out_buf_size > 0) ? (inner_len < (out_buf_size - 1) ? inner_len : (out_buf_size - 1)) : 0;
        if (out_buf && out_buf_size > 0) {
            if (copy_len > 0) memcpy(out_buf, start, copy_len);
            out_buf[copy_len] = '\0';
        }

        /* Remove consumed prefix up to end of closing tag */
        char *rest = close + strlen(tags[i].close);
        size_t rest_len = strlen(rest);
        memmove(buffer, rest, rest_len + 1);
        buf_pos = rest_len;
        return tags[i].tag;
    }

    return TI_TAG_NONE;
}
