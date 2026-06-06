#include "threading/thread_pool.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>  /* sysconf(_SC_NPROCESSORS_ONLN) */

#if TN_POSIX

/*
 * How many times a WORKER spins with CPU_RELAX() before falling back to
 * pthread_cond_wait. During active inference, dispatches arrive every
 * ~170 µs (at 19 tok/s with ~300 matmuls/token). Workers spin briefly
 * between dispatches to avoid kernel wakeup latency.
 *
 * At 4 GHz, _mm_pause adds ~10–40 cycles → SPIN_LIMIT = 40 000 ≈ 160 µs.
 * Workers spin for up to 160 µs, then sleep until next cond_broadcast.
 *
 * When use_blocking_wait is true (n_threads >= physical_cores * 2),
 * SPIN_LIMIT is effectively 0 — workers and dispatcher go straight to
 * cond_var to avoid burning HW slots that are already fully subscribed.
 */
#define SPIN_LIMIT 40000

/*
 * Count distinct (physical_id, core_id) pairs from /proc/cpuinfo.
 * Returns logical core count as fallback if /proc/cpuinfo is unavailable.
 */
static int detect_physical_cores(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return (n > 0) ? (int)n : 1;
    }
    int phys_ids[512], core_ids[512], count = 0;
    int cur_phys = -1, cur_core = -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "physical id", 11) == 0) {
            char *p = strchr(line, ':');
            if (p) cur_phys = (int)strtol(p + 1, NULL, 10);
        } else if (strncmp(line, "core id", 7) == 0) {
            char *p = strchr(line, ':');
            if (p) cur_core = (int)strtol(p + 1, NULL, 10);
            if (cur_phys >= 0 && cur_core >= 0 && count < 512) {
                int found = 0;
                for (int i = 0; i < count; i++) {
                    if (phys_ids[i] == cur_phys && core_ids[i] == cur_core) {
                        found = 1; break;
                    }
                }
                if (!found) {
                    phys_ids[count] = cur_phys;
                    core_ids[count] = cur_core;
                    count++;
                }
            }
            cur_phys = -1; cur_core = -1;
        }
    }
    fclose(f);
    if (count <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return (n > 0) ? (int)n : 1;
    }
    return count;
}

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define CPU_RELAX() _mm_pause()
#else
#  define CPU_RELAX() do {} while(0)
#endif

/*
 * K-5: Caller-participates thread pool.
 *
 * Root cause of T=7/T=8 oversubscription cliff:
 *   Old design: N workers + 1 spinning dispatcher = N+1 threads on N HW slots.
 *   At T=7 (8 logical CPUs): 8 threads fighting for 8 HW slots → OS preempts.
 *   At T=8: 9 threads on 8 HW slots → severe context-switch storm.
 *
 * Fix: create only N-1 OS worker threads. The calling (main) thread executes
 * the Nth work slice directly in threadpool_dispatch(). Total = N threads,
 * exactly matching N hardware slots. No wasted spinning dispatcher.
 *
 *   T=1: 0 OS workers, caller does all work (no thread overhead)
 *   T=4: 3 OS workers + caller = 4 threads on 4 physical cores
 *   T=8: 7 OS workers + caller = 8 threads on 8 logical cores (perfect fit)
 *
 * Worker threads handle slices 0..N-2 via atomic claim.
 * Caller always handles slice N-1.
 */
static void *worker_entry(void *opaque) {
    ThreadPool *tp = (ThreadPool *)opaque;
    unsigned int last_epoch = 0;

    for (;;) {
        /* ── Phase 1: Wait for a new dispatch ───────────────────────── */
        if (tp->use_blocking_wait) {
            /*
             * Fully subscribed mode (n_threads >= physical_cores * 2):
             * Skip spin loop entirely — go straight to cond_wait.
             * Spinning when HW slots are already saturated wastes cycles
             * and causes priority inversion vs the caller's work slice.
             */
            pthread_mutex_lock(&tp->mutex);
            while (!tp->shutdown &&
                   atomic_load_explicit(&tp->spin_epoch,
                                        memory_order_acquire) == last_epoch) {
                pthread_cond_wait(&tp->cond_work, &tp->mutex);
            }
            if (!tp->shutdown) {
                last_epoch = atomic_load_explicit(&tp->spin_epoch,
                                                  memory_order_relaxed);
            }
            pthread_mutex_unlock(&tp->mutex);
        } else {
            int spins = 0;
            for (;;) {
                if (tp->shutdown) return NULL;
                unsigned int cur = atomic_load_explicit(&tp->spin_epoch,
                                                        memory_order_acquire);
                if (cur != last_epoch) {
                    last_epoch = cur;
                    break;
                }
                if (++spins < SPIN_LIMIT) {
                    CPU_RELAX();
                } else {
                    /* Fall back to OS sleep to avoid burning CPU when idle */
                    pthread_mutex_lock(&tp->mutex);
                    while (!tp->shutdown &&
                           atomic_load_explicit(&tp->spin_epoch,
                                                memory_order_acquire) == last_epoch) {
                        pthread_cond_wait(&tp->cond_work, &tp->mutex);
                    }
                    if (!tp->shutdown) {
                        last_epoch = atomic_load_explicit(&tp->spin_epoch,
                                                          memory_order_relaxed);
                    }
                    pthread_mutex_unlock(&tp->mutex);
                    break;
                }
            }
        }
        if (tp->shutdown) return NULL;

        /* ── Phase 2: Atomically claim a slice index (0..N-2) ────────── */
        int idx      = atomic_fetch_add_explicit(&tp->spin_claimed, 1,
                                                  memory_order_relaxed);
        int total    = tp->task_total;
        int nthreads = tp->num_threads;
        tn_task_fn fn = tp->task_fn;
        void *arg     = tp->task_arg;

        /* Compute row range [start, end) for this thread */
        int chunk     = total / nthreads;
        int remainder = total % nthreads;
        int start, end;
        if (idx < remainder) {
            start = idx * (chunk + 1);
            end   = start + chunk + 1;
        } else {
            start = remainder * (chunk + 1) + (idx - remainder) * chunk;
            end   = start + chunk;
        }

        /* ── Phase 3: Execute ───────────────────────────────────────── */
        if (start < total && fn) {
            fn(arg, idx, start, end);
        }

        /* ── Phase 4: Signal completion ─────────────────────────────── */
        int rem = atomic_fetch_sub_explicit(&tp->spin_remaining, 1,
                                             memory_order_acq_rel);
        if (rem == 1) {
            /*
             * Last worker: wake the caller in case it fell back to
             * cond_var sleep (rare — only when tasks are very slow).
             */
            pthread_mutex_lock(&tp->mutex);
            pthread_cond_signal(&tp->cond_done);
            pthread_mutex_unlock(&tp->mutex);
        }
    }
}

ThreadPool *threadpool_create(int n) {
    if (n <= 0) return NULL;

    ThreadPool *tp = calloc(1, sizeof(ThreadPool));
    if (!tp) return NULL;

    tp->num_threads = n;
    tp->num_workers = n - 1; /* OS threads; caller handles the Nth slice */
    tp->shutdown    = false;
    tp->task_fn     = NULL;

    /*
     * K-5 blocking-wait mode: when thread count >= physical_cores * 2
     * (i.e., using both HyperThreading siblings on every physical core),
     * spin-waiting burns occupied HW slots instead of yielding them.
     * Switch workers and dispatcher to immediate cond_wait in that case.
     */
    int phys = detect_physical_cores();
    tp->physical_cores    = phys;
    tp->use_blocking_wait = (n >= phys * 2);

    atomic_store(&tp->spin_epoch,     0u);
    atomic_store(&tp->spin_claimed,   0);
    atomic_store(&tp->spin_remaining, 0);

    if (pthread_mutex_init(&tp->mutex, NULL) != 0) {
        free(tp);
        return NULL;
    }
    if (pthread_cond_init(&tp->cond_work, NULL) != 0) {
        pthread_mutex_destroy(&tp->mutex);
        free(tp);
        return NULL;
    }
    if (pthread_cond_init(&tp->cond_done, NULL) != 0) {
        pthread_cond_destroy(&tp->cond_work);
        pthread_mutex_destroy(&tp->mutex);
        free(tp);
        return NULL;
    }

    /* Allocate thread array (may be 0 for n=1 — malloc(0) is valid in C99) */
    tp->threads = malloc(sizeof(pthread_t) * (size_t)(n > 1 ? n - 1 : 1));
    if (!tp->threads) {
        pthread_cond_destroy(&tp->cond_done);
        pthread_cond_destroy(&tp->cond_work);
        pthread_mutex_destroy(&tp->mutex);
        free(tp);
        return NULL;
    }

    for (int i = 0; i < tp->num_workers; i++) {
        if (pthread_create(&tp->threads[i], NULL, worker_entry, tp) != 0) {
            /* Shut down threads created so far */
            pthread_mutex_lock(&tp->mutex);
            tp->shutdown = true;
            pthread_cond_broadcast(&tp->cond_work);
            pthread_mutex_unlock(&tp->mutex);
            for (int j = 0; j < i; j++) {
                pthread_join(tp->threads[j], NULL);
            }
            free(tp->threads);
            pthread_cond_destroy(&tp->cond_done);
            pthread_cond_destroy(&tp->cond_work);
            pthread_mutex_destroy(&tp->mutex);
            free(tp);
            return NULL;
        }
    }

    return tp;
}

void threadpool_dispatch(ThreadPool *tp, tn_task_fn fn, void *arg, int total) {
    if (!tp || !fn || total <= 0) return;

    int n         = tp->num_threads;
    int n_workers = tp->num_workers; /* n - 1 */

    /*
     * Write task parameters BEFORE incrementing spin_epoch.
     * The release fence ensures workers see these writes after they
     * observe the new epoch via their acquire load.
     */
    tp->task_fn    = fn;
    tp->task_arg   = arg;
    tp->task_total = total;
    atomic_store_explicit(&tp->spin_claimed,   0,        memory_order_relaxed);
    atomic_store_explicit(&tp->spin_remaining, n_workers, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);

    if (n_workers > 0) {
        /* Increment epoch — workers spin-watching this will wake immediately */
        atomic_fetch_add_explicit(&tp->spin_epoch, 1u, memory_order_release);
        /* Also broadcast to wake any workers that fell back to cond_var sleep */
        pthread_mutex_lock(&tp->mutex);
        tp->dispatch_epoch++;
        pthread_cond_broadcast(&tp->cond_work);
        pthread_mutex_unlock(&tp->mutex);
    }

    /*
     * K-5: Caller executes slice N-1 in parallel with the N-1 workers.
     * No spinning waste — this thread does useful computation instead.
     */
    {
        int idx       = n - 1;
        int chunk     = total / n;
        int remainder = total % n;
        int start, end;
        if (idx < remainder) {
            start = idx * (chunk + 1);
            end   = start + chunk + 1;
        } else {
            start = remainder * (chunk + 1) + (idx - remainder) * chunk;
            end   = start + chunk;
        }
        if (start < total) {
            fn(arg, idx, start, end);
        }
    }

    /* Wait for N-1 workers to finish */
    if (n_workers > 0) {
        if (tp->use_blocking_wait) {
            /*
             * Fully-subscribed mode: go straight to cond_wait.
             * Workers signal cond_done when spin_remaining hits 0.
             */
            pthread_mutex_lock(&tp->mutex);
            while (atomic_load_explicit(&tp->spin_remaining,
                                        memory_order_acquire) > 0) {
                pthread_cond_wait(&tp->cond_done, &tp->mutex);
            }
            pthread_mutex_unlock(&tp->mutex);
        } else {
            int spins = 0;
            while (atomic_load_explicit(&tp->spin_remaining, memory_order_acquire) > 0) {
                if (++spins < SPIN_LIMIT) {
                    CPU_RELAX();
                } else {
                    pthread_mutex_lock(&tp->mutex);
                    while (atomic_load_explicit(&tp->spin_remaining,
                                                memory_order_acquire) > 0) {
                        pthread_cond_wait(&tp->cond_done, &tp->mutex);
                    }
                    pthread_mutex_unlock(&tp->mutex);
                    break;
                }
            }
        }
    }
}

void threadpool_destroy(ThreadPool *tp) {
    if (!tp) return;

    pthread_mutex_lock(&tp->mutex);
    tp->shutdown = true;
    pthread_cond_broadcast(&tp->cond_work);
    pthread_mutex_unlock(&tp->mutex);

    for (int i = 0; i < tp->num_workers; i++) {
        pthread_join(tp->threads[i], NULL);
    }

    free(tp->threads);
    pthread_cond_destroy(&tp->cond_done);
    pthread_cond_destroy(&tp->cond_work);
    pthread_mutex_destroy(&tp->mutex);
    free(tp);
}

#else /* Windows stub — Phase 4 Windows support deferred */

ThreadPool *threadpool_create(int n) {
    (void)n;
    return NULL;
}

void threadpool_dispatch(ThreadPool *tp, tn_task_fn fn, void *arg, int total) {
    (void)tp; (void)fn; (void)arg; (void)total;
}

void threadpool_destroy(ThreadPool *tp) {
    (void)tp;
}

#endif /* TN_POSIX */
