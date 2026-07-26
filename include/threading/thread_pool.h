#ifndef TN_THREAD_POOL_H
#define TN_THREAD_POOL_H

#include "core/platform.h"
#include <stdbool.h>
#include <stdatomic.h>

#if TN_POSIX
#include <pthread.h>
#elif TN_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

/**
 * Task function signature for thread pool dispatch.
 *
 * @param arg       Opaque user data passed through from dispatch
 * @param thread_id Which worker thread is executing (0-based)
 * @param start     Start index of this thread's work slice (inclusive)
 * @param end       End index of this thread's work slice (exclusive)
 */
typedef void (*tn_task_fn)(void *arg, int thread_id, int start, int end);

/* Alignment macro for cacheline padding */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  include <stdalign.h>
#  define TN_ALIGN64 alignas(64)
#elif defined(_MSC_VER)
#  define TN_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#  define TN_ALIGN64 __attribute__((aligned(64)))
#else
#  define TN_ALIGN64
#endif

/**
 * Thread pool for parallel work distribution.
 *
 * Threads are created once at initialization and reused across all
 * dispatches. No per-call pthread_create overhead.
 */
typedef struct {
#if TN_POSIX
    pthread_t *threads;
    pthread_mutex_t mutex;
    pthread_cond_t cond_work;   /* wake sleeping workers */
    pthread_cond_t cond_done;   /* wake sleeping dispatcher */
#elif TN_WIN32
    HANDLE *threads;
    CRITICAL_SECTION mutex;
    CONDITION_VARIABLE cond_work;   /* wake sleeping workers */
    CONDITION_VARIABLE cond_done;   /* wake sleeping dispatcher */
#endif
    int num_threads;

    /* Task parameters — written by dispatcher before epoch increment */
    tn_task_fn task_fn;
    void *task_arg;
    int task_total;

    /* Spinlock-based dispatch (lock-free hot path).
     *
     * spin_epoch: dispatcher increments this (release) to trigger workers.
     *   Workers spin until their cached epoch differs, then execute.
     * spin_claimed: atomic counter — each worker claims its slice index.
     * spin_remaining: decremented by each completing worker.
     *   Dispatcher spins until it reaches 0.
     *
     * Hybrid fallback: after SPIN_LIMIT idle spins, workers/dispatcher
     * fall back to condition variables so idle periods don't burn 100% CPU.
     *
     * Padded to 64-byte boundaries to avoid cacheline bouncing between cores.
     */
    TN_ALIGN64 atomic_uint spin_epoch;
    char _pad1[64];

    TN_ALIGN64 atomic_int  spin_claimed;
    char _pad2[64];

    TN_ALIGN64 atomic_int  spin_remaining;
    char _pad3[64];

    TN_ALIGN64 atomic_int  sleeping_workers;
    char _pad4[64];

    /* K-5: caller-participates design.
     * num_threads  = total parallelism (N slices of work, indices 0..N-1)
     * num_workers  = N-1 OS threads (indices 0..N-2)
     * Calling thread (main/dispatcher) executes slice N-1 directly.
     * Total OS threads on HW = N, not N+1 — oversubscription eliminated. */
    int num_workers;     /* = num_threads - 1; OS threads actually created */
    int physical_cores;  /* distinct (physical_id, core_id) pairs on this CPU */

    /*
     * When n_threads >= physical_cores * 2 (all HT siblings active),
     * spinning wastes occupied HW slots. Set to true to make both
     * workers and dispatcher go straight to pthread_cond_wait instead
     * of spinning. Eliminates T=8 collapse under background system load.
     */
    bool use_blocking_wait;

    /* Legacy fields — kept for shutdown signaling only */
    unsigned int dispatch_epoch; /* mirrors spin_epoch for compatibility */
    bool shutdown;
} ThreadPool;

/**
 * Create a thread pool with n worker threads.
 * Returns NULL on allocation or thread creation failure.
 */
ThreadPool *threadpool_create(int n);

/**
 * Dispatch work across the pool.
 *
 * Divides 'total' work items evenly among threads, calls fn for each slice,
 * and blocks until all threads complete.
 *
 * @param tp    Thread pool
 * @param fn    Task function called by each worker
 * @param arg   Opaque data forwarded to fn
 * @param total Number of work items to distribute
 */
void threadpool_dispatch(ThreadPool *tp, tn_task_fn fn, void *arg, int total);

/**
 * Destroy the thread pool. Signals all workers to exit and joins them.
 * Frees all resources. tp is invalid after this call.
 */
void threadpool_destroy(ThreadPool *tp);

#endif /* TN_THREAD_POOL_H */
