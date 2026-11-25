#ifndef TN_API_CHAT_STREAM_H
#define TN_API_CHAT_STREAM_H

/* Per-request output state of POST /v1/chat/completions and the callbacks that feed it
 * (src/api/chat_stream.c). */

#include <stddef.h>

#include "api/cancel.h"
#include "reasoning/best_of.h"
#include "reasoning/mcts.h"

typedef struct {
    int conn_fd;        /* accepted socket */
    char id[32];        /* raw request ID — passed to sse_write_* (which
                           * itself prepends "chatcmpl-") */
    char public_id[48]; /* "chatcmpl-"+id — what the client actually sees
                           * in the streamed JSON "id" field, and therefore
                           * what a client will send back to the cancel
                           * endpoint. Must match cancel_registry_begin(). */
    int is_streaming;   /* SSE mode flag   */
    /* For non-streaming mode, accumulate full text */
    char *accum_buf;
    size_t accum_len;
    size_t accum_cap;
    long long tokens_emitted;
    /* Cancellation */
    CancelState *cancel;
} StreamState;

/* Adds "best_of" / "mcts" result objects to a finished response (consumes and returns the heap
 * string). */
char *chat_append_best_of_json(char *json, const BestOfResult *r);
char *chat_append_mcts_json(char *json, const MctsResult *r);

/* Best-of-N abort check: non-zero once the client cancelled. */
int chat_best_of_abort_check(void *userdata);

/* Token callback: streams one SSE chunk or accumulates the text; non-zero stops generation. */
int chat_streaming_token_callback(const char *piece, void *userdata);

#endif
