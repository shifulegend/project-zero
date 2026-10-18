#include "api/gen_queue.h"
#include <errno.h>
#include <time.h>

void gen_queue_init(GenQueue *q, int max_waiting) {
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->cv, NULL);
    q->busy = 0;
    q->head = q->tail = NULL;
    q->n_waiting = 0;
    q->max_waiting = max_waiting > 0 ? max_waiting : 0;
}

void gen_queue_destroy(GenQueue *q) {
    pthread_cond_destroy(&q->cv);
    pthread_mutex_destroy(&q->mu);
}

void gen_queue_set_max_waiting(GenQueue *q, int max_waiting) {
    pthread_mutex_lock(&q->mu);
    q->max_waiting = max_waiting > 0 ? max_waiting : 0;
    pthread_mutex_unlock(&q->mu);
}

GenQueueResult gen_queue_try_acquire(GenQueue *q) {
    pthread_mutex_lock(&q->mu);
    GenQueueResult r = GEN_QUEUE_FULL;
    if (!q->busy && !q->head) { q->busy = 1; r = GEN_QUEUE_OK; }
    pthread_mutex_unlock(&q->mu);
    return r;
}

static int position_of(const GenQueue *q, const GenWaiter *w) {
    int pos = 1;
    for (const GenWaiter *it = q->head; it && it != w; it = it->next) pos++;
    return pos;
}

static void unlink_waiter(GenQueue *q, GenWaiter *w) {
    GenWaiter *prev = NULL;
    for (GenWaiter *it = q->head; it; prev = it, it = it->next) {
        if (it != w) continue;
        if (prev) prev->next = it->next; else q->head = it->next;
        if (q->tail == it) q->tail = prev;
        q->n_waiting--;
        return;
    }
}

static long long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000;
}

GenQueueResult gen_queue_acquire(GenQueue *q, int timeout_ms,
                                 GenQueueWaitFn on_wait, void *userdata) {
    pthread_mutex_lock(&q->mu);
    if (!q->busy && !q->head) {
        q->busy = 1;
        pthread_mutex_unlock(&q->mu);
        return GEN_QUEUE_OK;
    }
    if (q->max_waiting <= 0 || q->n_waiting >= q->max_waiting) {
        pthread_mutex_unlock(&q->mu);
        return GEN_QUEUE_FULL;
    }
    GenWaiter me = { NULL, 0 };
    if (q->tail) q->tail->next = &me; else q->head = &me;
    q->tail = &me;
    q->n_waiting++;

    long long deadline = timeout_ms > 0 ? now_ms() + timeout_ms : 0;
    GenQueueResult result = GEN_QUEUE_OK;
    while (!me.granted) {
        int pos = position_of(q, &me);
        if (on_wait) {
            /* Report on change, and at most once per wake-up otherwise. */
            pthread_mutex_unlock(&q->mu);
            int stop = on_wait(pos, userdata);
            pthread_mutex_lock(&q->mu);
            if (stop && !me.granted) { unlink_waiter(q, &me); result = GEN_QUEUE_ABORTED; break; }
            if (me.granted) break;
        }
        long long wake = now_ms() + 1000;
        if (deadline && wake > deadline) wake = deadline;
        struct timespec ts = { (time_t)(wake / 1000), (long)((wake % 1000) * 1000000L) };
        pthread_cond_timedwait(&q->cv, &q->mu, &ts);
        if (!me.granted && deadline && now_ms() >= deadline) {
            unlink_waiter(q, &me);
            result = GEN_QUEUE_TIMEOUT;
            break;
        }
    }
    pthread_mutex_unlock(&q->mu);
    return result;
}

void gen_queue_release(GenQueue *q) {
    pthread_mutex_lock(&q->mu);
    GenWaiter *next = q->head;
    if (next) {
        q->head = next->next;
        if (!q->head) q->tail = NULL;
        q->n_waiting--;
        next->granted = 1;         /* slot passes straight to it; busy stays 1 */
        pthread_cond_broadcast(&q->cv);
    } else {
        q->busy = 0;
    }
    pthread_mutex_unlock(&q->mu);
}

int gen_queue_waiting(GenQueue *q) {
    pthread_mutex_lock(&q->mu);
    int n = q->n_waiting;
    pthread_mutex_unlock(&q->mu);
    return n;
}
