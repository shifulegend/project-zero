/*
 * tests/test_gen_queue.c — Phase 23 stage 1 FIFO admission queue: reject
 * when disabled, FIFO hand-off, full queue, timeout, abort, try-acquire.
 * Multi-threaded; part of TSAN_TESTS.
 */
#include "test_harness.h"
#include "api/gen_queue.h"
#include <pthread.h>
#include <time.h>

/* usleep() is not in POSIX.1-2008 (the build's -D_POSIX_C_SOURCE level). */
static void sleep_ms(int ms) {
    struct timespec ts = { ms / 1000, (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static GenQueue Q;
static int g_order[8];
static int g_order_n;
static pthread_mutex_t g_order_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct { int id; int last_pos; } Waiter;

static int note_pos(int pos, void *ud) { ((Waiter *)ud)->last_pos = pos; return 0; }

static void *waiter_thread(void *arg) {
    Waiter *w = (Waiter *)arg;
    if (gen_queue_acquire(&Q, 5000, note_pos, w) == GEN_QUEUE_OK) {
        pthread_mutex_lock(&g_order_mu);
        g_order[g_order_n++] = w->id;
        pthread_mutex_unlock(&g_order_mu);
        sleep_ms(2);
        gen_queue_release(&Q);
    }
    return NULL;
}

static void wait_for_waiting(int n) {
    for (int i = 0; i < 2000 && gen_queue_waiting(&Q) < n; i++) sleep_ms(1);
}

static void test_disabled_rejects(void) {
    gen_queue_init(&Q, 0);
    TEST_ASSERT_EQ(gen_queue_acquire(&Q, 100, NULL, NULL), GEN_QUEUE_OK, "free slot taken");
    TEST_ASSERT_EQ(gen_queue_acquire(&Q, 100, NULL, NULL), GEN_QUEUE_FULL, "busy + no queue -> FULL (429)");
    TEST_ASSERT_EQ(gen_queue_try_acquire(&Q), GEN_QUEUE_FULL, "try while busy");
    gen_queue_release(&Q);
    TEST_ASSERT_EQ(gen_queue_try_acquire(&Q), GEN_QUEUE_OK, "try when free");
    gen_queue_release(&Q);
    gen_queue_destroy(&Q);
}

static void test_fifo_order(void) {
    gen_queue_init(&Q, 4);
    g_order_n = 0;
    TEST_ASSERT_EQ(gen_queue_acquire(&Q, 0, NULL, NULL), GEN_QUEUE_OK, "holder takes slot");
    pthread_t th[3];
    Waiter w[3] = {{1, 0}, {2, 0}, {3, 0}};
    for (int i = 0; i < 3; i++) {   /* start one at a time so arrival order is known */
        pthread_create(&th[i], NULL, waiter_thread, &w[i]);
        wait_for_waiting(i + 1);
    }
    TEST_ASSERT_EQ(gen_queue_waiting(&Q), 3, "three waiting");
    gen_queue_release(&Q);  /* holder done: hands the slot to waiter 1 */
    for (int i = 0; i < 3; i++) pthread_join(th[i], NULL);
    TEST_ASSERT_EQ(g_order_n, 3, "all three ran");
    TEST_ASSERT(g_order[0] == 1 && g_order[1] == 2 && g_order[2] == 3, "served in arrival order");
    TEST_ASSERT_EQ(gen_queue_try_acquire(&Q), GEN_QUEUE_OK, "slot free afterwards");
    gen_queue_release(&Q);
    gen_queue_destroy(&Q);
}

static int abort_now(int pos, void *ud) { (void)pos; (void)ud; return 1; }

static void test_full_timeout_abort(void) {
    gen_queue_init(&Q, 1);
    TEST_ASSERT_EQ(gen_queue_acquire(&Q, 0, NULL, NULL), GEN_QUEUE_OK, "holder");
    TEST_ASSERT_EQ(gen_queue_acquire(&Q, 150, NULL, NULL), GEN_QUEUE_TIMEOUT, "times out at the front");
    TEST_ASSERT_EQ(gen_queue_waiting(&Q), 0, "timed-out waiter left the list");
    TEST_ASSERT_EQ(gen_queue_acquire(&Q, 1000, abort_now, NULL), GEN_QUEUE_ABORTED, "callback abort");
    TEST_ASSERT_EQ(gen_queue_waiting(&Q), 0, "aborted waiter left the list");

    pthread_t th; Waiter w = {7, 0};
    pthread_create(&th, NULL, waiter_thread, &w);
    wait_for_waiting(1);
    TEST_ASSERT_EQ(gen_queue_acquire(&Q, 100, NULL, NULL), GEN_QUEUE_FULL, "queue of 1 is full");
    TEST_ASSERT_EQ(w.last_pos, 1, "waiter told it is next");
    gen_queue_release(&Q);
    pthread_join(th, NULL);
    gen_queue_destroy(&Q);
}

int main(void) {
    printf("=== Generation Queue (Phase 23 stage 1) Tests ===\n");
    RUN_TEST(test_disabled_rejects);
    RUN_TEST(test_fifo_order);
    RUN_TEST(test_full_timeout_abort);
    TEST_SUMMARY();
}
