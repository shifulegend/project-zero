#ifndef TN_CANCEL_H
#define TN_CANCEL_H

/*
 * Phase 22 — In-flight generation cancellation.
 *
 * Only one generate_with_callback() call runs at a time (serialized by
 * ApiContext.generation_mutex), so tracking a single active request id +
 * cancel flag is sufficient — no multi-request registry is needed.
 */

#include <pthread.h>

typedef struct {
    pthread_mutex_t lock;
    char id[32];
    volatile int active;            /* 1 while a generation is in flight */
    volatile int cancel_requested;  /* 1 once cancel_registry_request() matched */
} CancelState;

void cancel_state_init(CancelState *cs);
void cancel_state_destroy(CancelState *cs);

/* Called by the request handler right before starting generation. */
void cancel_registry_begin(CancelState *cs, const char *id);

/* Called by the request handler right after generation finishes/stops. */
void cancel_registry_end(CancelState *cs);

/* Polled from inside the generation loop. Returns 1 if this id's
 * cancellation was requested (generation should stop). */
int cancel_registry_check(CancelState *cs, const char *id);

/* Called by POST /v1/chat/completions/cancel. Returns 1 if id matched the
 * currently active generation and was marked for cancellation, 0 otherwise
 * (no generation in flight, or id mismatch). */
int cancel_registry_request(CancelState *cs, const char *id);

#endif /* TN_CANCEL_H */
