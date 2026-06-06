#include "test_harness.h"
#include "threading/thread_pool.h"
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

/**
 * Audit Test: AUD-THR-01 (Thread Pool Stress & Re-dispatch)
 * Tests if rapid-fire dispatching causes any race conditions or deadlocks.
 */

static void stress_task(void *arg, int thread_id, int start, int end) {
  (void)thread_id;
  (void)start;
  (void)end;
  volatile int *counter = (int *)arg;
  /* Increment counter atomically */
  __sync_fetch_and_add(counter, 1);
}

static void aud_threadpool_stress(void) {
  int n_threads = 8;
  ThreadPool *tp = threadpool_create(n_threads);
  TEST_ASSERT(tp != NULL, "Threadpool created");

  int counter = 0;
  int iterations = 1000;

  printf("[Auditor] Running %d rapid-fire dispatches...\n", iterations);
  for (int i = 0; i < iterations; i++) {
    /* Total items = 16. Each of 8 threads should be called twice?
       No, the dispatcher logic divides the 16 items among 8 threads.
       In our implementation, each thread handles ONE slice. */
    threadpool_dispatch(tp, stress_task, &counter, 16);
  }

  /*
   * In each dispatch, precisely tp->num_threads threads are woken and report
   * back. Even if items < num_threads, all threads report back. So counter
   * should be iterations * num_threads.
   */
  printf("[Auditor] Final counter value: %d (Expected: %d)\n", counter,
         iterations * n_threads);
  TEST_ASSERT(counter == iterations * n_threads,
              "All worker iterations completed");

  threadpool_destroy(tp);
}

int main(void) {
  RUN_TEST(aud_threadpool_stress);
  TEST_SUMMARY();
}
