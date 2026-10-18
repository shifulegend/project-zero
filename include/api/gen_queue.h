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
 */

#include <pthread.h>

typedef struct GenWaiter {
    struct GenWaiter *next;
    int granted;
} GenWaiter;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int        busy;         /* the generation slot is held */
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

/* Take the slot now if it is free and nobody is waiting; never blocks. */
GenQueueResult gen_queue_try_acquire(GenQueue *q);

/* Take the slot, waiting in FIFO order up to timeout_ms (<= 0: forever).
 * on_wait (may be NULL) is called when first queued, whenever the position
 * changes, and about once a second otherwise. */
GenQueueResult gen_queue_acquire(GenQueue *q, int timeout_ms,
                                 GenQueueWaitFn on_wait, void *userdata);

/* Hand the slot to the next waiter (or free it). */
void gen_queue_release(GenQueue *q);

/* Snapshot of how many requests are waiting. */
int gen_queue_waiting(GenQueue *q);

#endif /* TN_GEN_QUEUE_H */
