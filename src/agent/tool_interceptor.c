#include "agent/tool_interceptor.h"

#include <stdio.h>
#include <string.h>

/* Simple streaming tag parser. Buffers partial pieces between calls (in the TiParser). */
void ti_init(TiParser *parser) {
    parser->pos       = 0;
    parser->buffer[0] = '\0';
}

void ti_reset(TiParser *parser) {
    ti_init(parser);
}

static const struct {
    const char *open;
    const char *close;
    TI_Tag tag;
} tags[] = {
    {"<exec>", "</exec>", TI_TAG_EXEC},
    {"<think>", "</think>", TI_TAG_THINK},
    {"<save_memory>", "</save_memory>", TI_TAG_SAVE_MEMORY},
    {"<search_memory>", "</search_memory>", TI_TAG_SEARCH_MEMORY},
    {"<result>", "</result>", TI_TAG_RESULT},
};

TI_Tag ti_process_piece(TiParser *parser, const char *piece, char *out_buf, size_t out_buf_size) {
    if (!piece || piece[0] == '\0') return TI_TAG_NONE;

    size_t piece_len = strlen(piece);
    /* Append piece into parser->buffer (truncate if full) */
    size_t space   = sizeof(parser->buffer) - 1 - parser->pos;
    size_t to_copy = piece_len < space ? piece_len : space;
    if (to_copy > 0) {
        memcpy(parser->buffer + parser->pos, piece, to_copy);
        parser->pos += to_copy;
        parser->buffer[parser->pos] = '\0';
    }

    /* Look for any complete tag */
    for (size_t i = 0; i < sizeof(tags) / sizeof(tags[0]); i++) {
        char *open = strstr(parser->buffer, tags[i].open);
        if (!open) continue;
        char *close = strstr(open + strlen(tags[i].open), tags[i].close);
        if (!close) continue; /* wait for closing tag */

        /* Extract inner content */
        char *start      = open + strlen(tags[i].open);
        size_t inner_len = (size_t)(close - start);
        size_t copy_len  = 0;
        if (out_buf_size > 0)
            copy_len = inner_len < out_buf_size - 1 ? inner_len : out_buf_size - 1;
        if (out_buf && out_buf_size > 0) {
            if (copy_len > 0) memcpy(out_buf, start, copy_len);
            out_buf[copy_len] = '\0';
        }

        /* Remove consumed prefix up to end of closing tag */
        char *rest      = close + strlen(tags[i].close);
        size_t rest_len = strlen(rest);
        memmove(parser->buffer, rest, rest_len + 1);
        parser->pos = rest_len;
        return tags[i].tag;
    }

    return TI_TAG_NONE;
}
