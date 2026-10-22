#ifndef TN_STOP_SEQ_H
#define TN_STOP_SEQ_H

/*
 * stop_seq.h — user-defined stop sequences for streaming generation.
 *
 * OpenAI-style `stop`: generation ends as soon as the decoded output contains
 * any configured string, and the stop string itself is never emitted. Because
 * output streams token-by-token, text that *might* be the beginning of a stop
 * string is held back until it is disambiguated — so a streaming client never
 * sees a partial stop sequence. Standalone: depends only on libc.
 */

#include <stddef.h>

#define STOP_MAX_SEQS 8
#define STOP_MAX_LEN  128

/* Same shape as generate.h's TokenCallback: nonzero return = stop requested. */
typedef int (*StopEmitFn)(const char *text, void *userdata);

typedef struct {
    char   seqs[STOP_MAX_SEQS][STOP_MAX_LEN];
    int    n_seqs;
    char  *pending;   /* held-back tail that may begin a stop sequence */
    size_t len, cap;
    int    stopped;   /* 1 once a stop sequence matched */
} StopMatcher;

/* Copies up to STOP_MAX_SEQS non-empty strings (each truncated to
 * STOP_MAX_LEN-1 bytes). seqs may be NULL when n == 0. */
void stop_matcher_init(StopMatcher *m, const char *const *seqs, int n);

/* Feeds one decoded piece. Emits every byte that can no longer be part of a
 * stop sequence through emit(). Returns 1 when generation must end (a stop
 * sequence matched, or emit() asked to stop), else 0. With no sequences
 * configured the piece is forwarded unchanged. */
int stop_matcher_feed(StopMatcher *m, const char *piece, StopEmitFn emit, void *userdata);

/* End of generation without a match: emits whatever is still held back. */
void stop_matcher_flush(StopMatcher *m, StopEmitFn emit, void *userdata);

void stop_matcher_free(StopMatcher *m);

#endif /* TN_STOP_SEQ_H */
