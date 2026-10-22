#ifndef TN_CANCEL_H
#define TN_CANCEL_H

/*
 * Phase 22 — In-flight generation cancellation.
 *
 * Each running generation registers its public id; POST
 * /v1/chat/completions/cancel marks a matching one, and the generation polls
 * its own flag. Phase 23 stage 2b (2026-10-22): with --parallel N several
 * generations run at once, so the registry holds up to CANCEL_MAX_ACTIVE
 * entries (the --parallel ceiling) instead of a single id.
 */

#include <pthread.h>

#define CANCEL_MAX_ACTIVE 64

typedef struct {
    char id[32];
    int  active;            /* 1 while that generation is in flight */
    int  cancel_requested;  /* 1 once cancel_registry_request() matched */
} CancelEntry;

typedef struct {
    pthread_mutex_t lock;
    CancelEntry     entries[CANCEL_MAX_ACTIVE];
    int             shutdown;       /* sticky: every check() reports cancelled */
} CancelState;

void cancel_state_init(CancelState *cs);
void cancel_state_destroy(CancelState *cs);

/* Called by the request handler right before starting generation. Returns
 * 0, or -1 when every entry is taken (the generation then simply cannot be
 * cancelled by id). */
int cancel_registry_begin(CancelState *cs, const char *id);

/* Called by the request handler right after generation finishes/stops. */
void cancel_registry_end(CancelState *cs, const char *id);

/* Polled from inside the generation loop. Returns 1 if this id's
 * cancellation was requested (generation should stop). */
int cancel_registry_check(CancelState *cs, const char *id);

/* Called by POST /v1/chat/completions/cancel. Returns 1 if id matched an
 * active generation and it was marked for cancellation, 0 otherwise (no
 * such generation in flight). */
int cancel_registry_request(CancelState *cs, const char *id);

/* Server shutdown: from now on every generation's check() reports
 * cancelled, including ones that register later. */
void cancel_registry_shutdown(CancelState *cs);

#endif /* TN_CANCEL_H */
