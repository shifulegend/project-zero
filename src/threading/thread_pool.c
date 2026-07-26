#include "threading/thread_pool.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#if TN_POSIX
#include <unistd.h>  /* sysconf(_SC_NPROCESSORS_ONLN) */
#endif

#if TN_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

/*
 * How many times a WORKER spins with CPU_RELAX() before falling back to
 * pthread_cond_wait / SleepConditionVariableCS.
 */
#define SPIN_LIMIT 40000

/* ── CPU_RELAX macro yielding on x86, ARM64, and MSVC ──────────────────────── */
#if defined(_MSC_VER)
#  include <intrin.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  if !defined(_MSC_VER)
#    include <immintrin.h>
#  endif
#  define CPU_RELAX() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
#  if defined(_MSC_VER)
#    define CPU_RELAX() __yield()
#  elif defined(__GNUC__) || defined(__clang__)
#    define CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#  else
#    define CPU_RELAX() do {} while(0)
#  endif
#else
#  define CPU_RELAX() do {} while(0)
#endif

/* ── Platform Synchronization Helpers ──────────────────────────────────────── */
#if TN_POSIX
#  define MUTEX_LOCK(m)         pthread_mutex_lock(m)
#  define MUTEX_UNLOCK(m)       pthread_mutex_unlock(m)
#  define COND_WAIT(c, m)       pthread_cond_wait(c, m)
#  define COND_SIGNAL(c)        pthread_cond_signal(c)
#  define COND_BROADCAST(c)     pthread_cond_broadcast(c)
#elif TN_WIN32
#  define MUTEX_LOCK(m)         EnterCriticalSection(m)
#  define MUTEX_UNLOCK(m)       LeaveCriticalSection(m)
#  define COND_WAIT(c, m)       SleepConditionVariableCS(c, m, INFINITE)
#  define COND_SIGNAL(c)        WakeConditionVariable(c)
#  define COND_BROADCAST(c)     WakeAllConditionVariable(c)
#endif

/* ── Topology Detection ────────────────────────────────────────────────────── */
static bool is_blank_line(const char *s) {
    while (*s) {
        if (!isspace((unsigned char)*s)) return false;
        s++;
    }
    return true;
}

static void commit_cpu_pair(int cur_phys, int cur_core, int *phys_ids, int *core_ids, int *count) {
    if (cur_phys >= 0 && cur_core >= 0 && *count < 512) {
        for (int i = 0; i < *count; i++) {
            if (phys_ids[i] == cur_phys && core_ids[i] == cur_core) {
                return;
            }
        }
        phys_ids[*count] = cur_phys;
        core_ids[*count] = cur_core;
        (*count)++;
    }
}

static int count_physical_cores_linux(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) {
#if TN_POSIX
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return (n > 0) ? (int)n : 1;
#else
        return 1;
#endif
    }
    int phys_ids[512], core_ids[512], count = 0;
    int cur_phys = -1, cur_core = -1;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (is_blank_line(line) || strncmp(line, "processor", 9) == 0) {
            commit_cpu_pair(cur_phys, cur_core, phys_ids, core_ids, &count);
            cur_phys = -1;
            cur_core = -1;
            continue;
        }

        char *colon = strchr(line, ':');
        if (!colon) continue;

        char key[128];
        size_t key_len = (size_t)(colon - line);
        if (key_len >= sizeof(key)) key_len = sizeof(key) - 1;
        memcpy(key, line, key_len);
        key[key_len] = '\0';

        long val = strtol(colon + 1, NULL, 10);

        if (strstr(key, "physical id") != NULL) {
            cur_phys = (int)val;
        } else if (strstr(key, "core id") != NULL) {
            cur_core = (int)val;
        }
    }
    commit_cpu_pair(cur_phys, cur_core, phys_ids, core_ids, &count);
    fclose(f);

    if (count <= 0) {
#if TN_POSIX
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return (n > 0) ? (int)n : 1;
#else
        return 1;
#endif
    }
    return count;
}

#if TN_WIN32
static int count_physical_cores_win32(void) {
    DWORD returnLength = 0;
    GetLogicalProcessorInformation(NULL, &returnLength);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || returnLength == 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (si.dwNumberOfProcessors > 0) ? (int)si.dwNumberOfProcessors : 1;
    }

    DWORD count = returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(returnLength);
    if (!buffer) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (si.dwNumberOfProcessors > 0) ? (int)si.dwNumberOfProcessors : 1;
    }

    if (!GetLogicalProcessorInformation(buffer, &returnLength)) {
        free(buffer);
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (si.dwNumberOfProcessors > 0) ? (int)si.dwNumberOfProcessors : 1;
    }

    int physical_cores = 0;
    for (DWORD i = 0; i < count; i++) {
        if (buffer[i].Relationship == RelationProcessorCore) {
            physical_cores++;
        }
    }
    free(buffer);
    return (physical_cores > 0) ? physical_cores : 1;
}
#endif

static int detect_physical_cores(void) {
#if TN_POSIX
    return count_physical_cores_linux();
#elif TN_WIN32
    return count_physical_cores_win32();
#else
    return 1;
#endif
}

/* ── Worker Thread Entry Point ─────────────────────────────────────────────── */
#if TN_POSIX
static void *worker_entry(void *opaque)
#elif TN_WIN32
static DWORD WINAPI worker_entry(LPVOID opaque)
#endif
{
    ThreadPool *tp = (ThreadPool *)opaque;
    unsigned int last_epoch = 0;

    for (;;) {
        /* ── Phase 1: Wait for a new dispatch ───────────────────────── */
        if (tp->use_blocking_wait) {
            atomic_fetch_add_explicit(&tp->sleeping_workers, 1, memory_order_acq_rel);
            MUTEX_LOCK(&tp->mutex);
            while (!tp->shutdown &&
                   atomic_load_explicit(&tp->spin_epoch,
                                        memory_order_acquire) == last_epoch) {
                COND_WAIT(&tp->cond_work, &tp->mutex);
            }
            if (!tp->shutdown) {
                last_epoch = atomic_load_explicit(&tp->spin_epoch,
                                                  memory_order_relaxed);
            }
            MUTEX_UNLOCK(&tp->mutex);
            atomic_fetch_sub_explicit(&tp->sleeping_workers, 1, memory_order_acq_rel);
        } else {
            int spins = 0;
            for (;;) {
                if (tp->shutdown) return 0;
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
                    atomic_fetch_add_explicit(&tp->sleeping_workers, 1, memory_order_acq_rel);
                    MUTEX_LOCK(&tp->mutex);
                    while (!tp->shutdown &&
                           atomic_load_explicit(&tp->spin_epoch,
                                                memory_order_acquire) == last_epoch) {
                        COND_WAIT(&tp->cond_work, &tp->mutex);
                    }
                    if (!tp->shutdown) {
                        last_epoch = atomic_load_explicit(&tp->spin_epoch,
                                                          memory_order_relaxed);
                    }
                    MUTEX_UNLOCK(&tp->mutex);
                    atomic_fetch_sub_explicit(&tp->sleeping_workers, 1, memory_order_acq_rel);
                    break;
                }
            }
        }
        if (tp->shutdown) return 0;

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
            MUTEX_LOCK(&tp->mutex);
            COND_SIGNAL(&tp->cond_done);
            MUTEX_UNLOCK(&tp->mutex);
        }
    }
}

/* ── ThreadPool Lifecycle ──────────────────────────────────────────────────── */
ThreadPool *threadpool_create(int n) {
    if (n <= 0) return NULL;

    ThreadPool *tp = calloc(1, sizeof(ThreadPool));
    if (!tp) return NULL;

    tp->num_threads = n;
    tp->num_workers = n - 1; /* OS threads; caller handles the Nth slice */
    tp->shutdown    = false;
    tp->task_fn     = NULL;

    int phys = detect_physical_cores();
    tp->physical_cores    = phys;
    tp->use_blocking_wait = (n >= phys * 2);

    atomic_store(&tp->spin_epoch,     0u);
    atomic_store(&tp->spin_claimed,   0);
    atomic_store(&tp->spin_remaining, 0);
    atomic_store(&tp->sleeping_workers, 0);

#if TN_POSIX
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
#elif TN_WIN32
    InitializeCriticalSection(&tp->mutex);
    InitializeConditionVariable(&tp->cond_work);
    InitializeConditionVariable(&tp->cond_done);

    tp->threads = malloc(sizeof(HANDLE) * (size_t)(n > 1 ? n - 1 : 1));
    if (!tp->threads) {
        DeleteCriticalSection(&tp->mutex);
        free(tp);
        return NULL;
    }

    for (int i = 0; i < tp->num_workers; i++) {
        tp->threads[i] = CreateThread(NULL, 0, worker_entry, tp, 0, NULL);
        if (!tp->threads[i]) {
            EnterCriticalSection(&tp->mutex);
            tp->shutdown = true;
            WakeAllConditionVariable(&tp->cond_work);
            LeaveCriticalSection(&tp->mutex);
            for (int j = 0; j < i; j++) {
                WaitForSingleObject(tp->threads[j], INFINITE);
                CloseHandle(tp->threads[j]);
            }
            free(tp->threads);
            DeleteCriticalSection(&tp->mutex);
            free(tp);
            return NULL;
        }
    }
#endif

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
        /* Increment epoch — workers spin-watching this will wake immediately in user-space */
        atomic_fetch_add_explicit(&tp->spin_epoch, 1u, memory_order_release);
        /* Broadcast ONLY if workers fell back to sleeping or in blocking wait mode */
        if (tp->use_blocking_wait || atomic_load_explicit(&tp->sleeping_workers, memory_order_acquire) > 0) {
            MUTEX_LOCK(&tp->mutex);
            tp->dispatch_epoch++;
            COND_BROADCAST(&tp->cond_work);
            MUTEX_UNLOCK(&tp->mutex);
        }
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
            MUTEX_LOCK(&tp->mutex);
            while (atomic_load_explicit(&tp->spin_remaining,
                                        memory_order_acquire) > 0) {
                COND_WAIT(&tp->cond_done, &tp->mutex);
            }
            MUTEX_UNLOCK(&tp->mutex);
        } else {
            int spins = 0;
            while (atomic_load_explicit(&tp->spin_remaining, memory_order_acquire) > 0) {
                if (++spins < SPIN_LIMIT) {
                    CPU_RELAX();
                } else {
                    MUTEX_LOCK(&tp->mutex);
                    while (atomic_load_explicit(&tp->spin_remaining,
                                                memory_order_acquire) > 0) {
                        COND_WAIT(&tp->cond_done, &tp->mutex);
                    }
                    MUTEX_UNLOCK(&tp->mutex);
                    break;
                }
            }
        }
    }
}

void threadpool_destroy(ThreadPool *tp) {
    if (!tp) return;

    MUTEX_LOCK(&tp->mutex);
    tp->shutdown = true;
    COND_BROADCAST(&tp->cond_work);
    MUTEX_UNLOCK(&tp->mutex);

#if TN_POSIX
    for (int i = 0; i < tp->num_workers; i++) {
        pthread_join(tp->threads[i], NULL);
    }
    free(tp->threads);
    pthread_cond_destroy(&tp->cond_done);
    pthread_cond_destroy(&tp->cond_work);
    pthread_mutex_destroy(&tp->mutex);
#elif TN_WIN32
    for (int i = 0; i < tp->num_workers; i++) {
        WaitForSingleObject(tp->threads[i], INFINITE);
        CloseHandle(tp->threads[i]);
    }
    free(tp->threads);
    DeleteCriticalSection(&tp->mutex);
#endif
    free(tp);
}
