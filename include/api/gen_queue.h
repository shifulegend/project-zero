#ifndef TN_GEN_QUEUE_H
#define TN_GEN_QUEUE_H

/*
 * gen_queue.h — Phase 23 (stage 1): FIFO admission for the single generation
 * slot.
 *
 * The server runs one generation at a time. Without a queue a second chat
 * request is rejected with 429 (still the default: max_waiting = 0). With
 * --queue-size N, up to N requests wait in arrival order, each told its
 * position as it changes, and give up after --queue-timeout seconds. A
 * waiter that times out or whose client disconnects leaves the list without
 * disturbing the order of the others. Routes that must never block (static,
 * metrics, cancel, health) never touch this.
 *
 * Stage 2b (2026-10-22): the queue admits up to `capacity` units at once.
 * With --parallel N a batched chat request takes 1 unit (one scheduler
 * slot) and a request that needs the whole engine (image input, best-of,
 * tree search, cache reset) takes all N. Grants stay strictly FIFO: a
 * waiter at the head that needs more units than are free holds back the
 * ones behind it, so whole-engine requests are never starved. Capacity 1
 * (the default) is the original single-slot behaviour.
 */

#include <pthread.h>

typedef struct GenWaiter {
    struct GenWaiter *next;
    int units;
    int granted;
} GenWaiter;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int        capacity;     /* units that may be held at once (>= 1) */
    int        in_use;       /* units held now */
    GenWaiter *head, *tail;
    int        n_waiting;
    int        max_waiting;  /* 0 = no queue (reject when busy) */
} GenQueue;

typedef enum {
    GEN_QUEUE_OK = 0,
    GEN_QUEUE_FULL,      /* busy and the queue is full / disabled */
    GEN_QUEUE_TIMEOUT,   /* waited timeout_ms without reaching the front */
    GEN_QUEUE_ABORTED    /* the wait callback asked to stop (client gone) */
} GenQueueResult;

/* Called while waiting with the 1-based position (1 = next). Return nonzero
 * to abandon the wait. Invoked without the queue lock held. */
typedef int (*GenQueueWaitFn)(int position, void *userdata);

void gen_queue_init(GenQueue *q, int max_waiting);
void gen_queue_destroy(GenQueue *q);
void gen_queue_set_max_waiting(GenQueue *q, int max_waiting);

/* Units available at once (clamped to >= 1). Set before serving starts. */
void gen_queue_set_capacity(GenQueue *q, int capacity);
int  gen_queue_capacity(GenQueue *q);

/* Take the slot now if it is free and nobody is waiting; never blocks. */
GenQueueResult gen_queue_try_acquire(GenQueue *q);

/* Take the slot, waiting in FIFO order up to timeout_ms (<= 0: forever).
 * on_wait (may be NULL) is called when first queued, whenever the position
 * changes, and about once a second otherwise. */
GenQueueResult gen_queue_acquire(GenQueue *q, int timeout_ms,
                                 GenQueueWaitFn on_wait, void *userdata);

/* Hand the slot to the next waiter (or free it). */
void gen_queue_release(GenQueue *q);

/* The same for `units` units (clamped to [1, capacity]); the plain
 * functions above take one unit. */
GenQueueResult gen_queue_try_acquire_n(GenQueue *q, int units);
GenQueueResult gen_queue_acquire_n(GenQueue *q, int units, int timeout_ms,
                                   GenQueueWaitFn on_wait, void *userdata);
void gen_queue_release_n(GenQueue *q, int units);

/* Snapshot of how many units are held. */
int gen_queue_in_use(GenQueue *q);

/* Snapshot of how many requests are waiting. */
int gen_queue_waiting(GenQueue *q);

#endif /* TN_GEN_QUEUE_H */
