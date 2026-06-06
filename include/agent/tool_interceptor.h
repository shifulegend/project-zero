#ifndef AGENT_TOOL_INTERCEPTOR_H
#define AGENT_TOOL_INTERCEPTOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TI_TAG_NONE = 0,
    TI_TAG_EXEC,
    TI_TAG_THINK,
    TI_TAG_SAVE_MEMORY,
    TI_TAG_SEARCH_MEMORY,
    TI_TAG_RESULT
} TI_Tag;

/* Initialize / reset internal parser state */
void ti_init(void);

/*
 * Process a decoded text piece (one tokenizer_decode() output).
 * If a complete tag pair is detected (e.g. <exec>...</exec>), the
 * inner content is copied to out_buf (null-terminated) and the
 * corresponding TI_Tag is returned. Partial tags are buffered until
 * closed.
 */
TI_Tag ti_process_piece(const char *piece, char *out_buf, size_t out_buf_size);

/* Forcefully reset internal buffer */
void ti_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_TOOL_INTERCEPTOR_H */
