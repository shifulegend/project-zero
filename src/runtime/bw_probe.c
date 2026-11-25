/* bw_probe.c — the multi-threaded DRAM bandwidth probe (hardware_profile.c decides how big a
 * buffer each thread gets and when to run it). */
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "core/platform.h"
#include "runtime/hardware_profile.h"

static int64_t bw_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/*
 * Multi-threaded DRAM bandwidth probe.
 *
 * Uses N threads each reading a buffer > L3 cache to measure true DRAM
 * bandwidth (not L3). This is critical for systems with large L3 caches
 * (e.g., Xeon Emerald Rapids with 260 MB L3) where a small single-threaded
 * probe would measure L3 speed instead of DRAM speed.
 *
 * Each thread reads its own buffer sequentially; total is aggregated.
 * 3 passes, best aggregate result returned.
 */

#if TN_POSIX
#include <pthread.h>
#endif

/* Shared by the POSIX multi-threaded probe and the single-threaded
 * fallback — no pthread dependency in the pass function itself. */
typedef struct {
    const unsigned char *buf;
    size_t size;
    double gbps;
    uint64_t sink; /* per-thread accumulator result, folded by the caller */
} BwThreadArg;

/* Defeats dead-code elimination of the read loops: each thread deposits its
 * accumulator sum into its own arg (no shared writes from workers — a
 * previous version had all threads += into this global directly, a benign
 * but real data race; independent-review finding, 2026-07-17); the caller
 * folds the per-thread sinks in here after join, single-threaded. */
static volatile uint64_t g_bw_probe_sink;

/*
 * One full-buffer read pass. TWO fixes on 2026-07-17 (both had been
 * under-stating measured bandwidth, compounding to ~3x total — see
 * docs/architecture/CEILING_CALCULATION.md §2 and docs/ai/mistakes.md):
 *
 * 1. ACCOUNTING: this function used to run its own 3 passes internally,
 *    while the caller timed the whole create/join round but divided only
 *    ONE pass worth of bytes by that wall time — a systematic ~3x
 *    under-measurement (the correctly-accounted per-thread bytes/elapsed
 *    figures were computed and then discarded). Now each thread runs
 *    exactly one pass per round; the caller's existing 3-round best-of
 *    loop provides the repetition.
 *
 * 2. ACCESS PATTERN: reads every byte via 8 independent 64-bit accumulator
 *    chains (one cache line per iteration) instead of one volatile char
 *    per line. The volatile-scalar loop serialized a 1-byte load + a
 *    volatile store per line and couldn't keep enough line fills in
 *    flight. Independent non-volatile accumulators let the compiler
 *    vectorize and the core overlap misses; the volatile sink store
 *    happens once per pass, outside the timed loop.
 */
#if TN_POSIX
/* First-touch of a probe buffer, run on the probing thread itself so the
 * page faults are taken in parallel (single-threaded touching of the old
 * 2 GB buffer cost up to 15 s on VMs; 2026-10-19). */
static void *bw_touch_fn(void *arg) {
    BwThreadArg *a   = (BwThreadArg *)arg;
    unsigned char *m = (unsigned char *)(uintptr_t)a->buf;
    for (size_t j = 0; j < a->size; j += 4096) m[j] = (unsigned char)j;
    return NULL;
}
#endif

static void *bw_thread_fn(void *arg) {
    BwThreadArg *a = (BwThreadArg *)arg;
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    const uint64_t *p = (const uint64_t *)(const void *)a->buf;
    size_t n_words    = a->size / 8;
    int64_t t0        = bw_now_ns();
    for (size_t i = 0; i + 8 <= n_words; i += 8) {
        s0 += p[i];
        s1 += p[i + 1];
        s2 += p[i + 2];
        s3 += p[i + 3];
        s4 += p[i + 4];
        s5 += p[i + 5];
        s6 += p[i + 6];
        s7 += p[i + 7];
    }
    int64_t elapsed = bw_now_ns() - t0;
    a->sink         = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    a->gbps         = (elapsed > 0) ? (double)a->size / (double)elapsed : 0.0;
    return NULL;
}


/* Free RAM seen by tn_hardware_profile_init() before the probe runs (caps
 * the probe buffer); 0 = unknown. */

#if TN_POSIX
/* Allocates one buffer per thread; returns how many succeeded (the rest are not used). */
static int bw_alloc_buffers(BwThreadArg *args, int n_threads, size_t per_thread) {
    for (int i = 0; i < n_threads; i++) {
        unsigned char *mem = (unsigned char *)malloc(per_thread);
        if (!mem) return i;
        args[i].buf  = mem;
        args[i].size = per_thread;
        args[i].gbps = 0.0;
    }
    return n_threads;
}

/* Runs `fn` on every thread's buffer and waits for all of them. */
static void bw_run_threads(BwThreadArg *args, int n_threads, void *(*fn)(void *)) {
    pthread_t threads[8];
    for (int i = 0; i < n_threads; i++) pthread_create(&threads[i], NULL, fn, &args[i]);
    for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
}

/* Measure WALL TIME for all threads to complete ONE pass each — true
 * aggregate bandwidth including bus contention (plus a small
 * thread-create/join overhead, which biases slightly conservative).
 * 5 rounds, best kept (the buffers are smaller than they used to be). */
static double bw_best_of_passes(BwThreadArg *args, int n_threads, size_t per_thread) {
    size_t total_bytes = per_thread * (size_t)n_threads;
    double best_gbps   = 0.0;
    for (int pass = 0; pass < 5; pass++) {
        int64_t wall_t0 = bw_now_ns();
        bw_run_threads(args, n_threads, bw_thread_fn);
        int64_t wall_elapsed = bw_now_ns() - wall_t0;
        for (int i = 0; i < n_threads; i++)
            g_bw_probe_sink += args[i].sink; /* single-threaded fold */

        if (wall_elapsed > 0) {
            double gbps = (double)total_bytes / (double)wall_elapsed;
            if (gbps > best_gbps) best_gbps = gbps;
        }
    }
    return best_gbps;
}

double tn_hw_probe_bandwidth(int n_threads, size_t per_thread) {
    BwThreadArg args[8];
    /* Allocate, then touch all pages from the probing threads */
    n_threads = bw_alloc_buffers(args, n_threads, per_thread);
    if (n_threads == 0) return 0.0;
    bw_run_threads(args, n_threads, bw_touch_fn);
    double best_gbps = bw_best_of_passes(args, n_threads, per_thread);
    for (int i = 0; i < n_threads; i++) free((void *)args[i].buf);
    return best_gbps;
}
#else
/* Fallback: single-threaded probe (same one-pass-per-round
 * multi-accumulator read as bw_thread_fn — see its comment). */
double tn_hw_probe_bandwidth(int n_threads, size_t per_thread) {
    (void)n_threads;
    unsigned char *buf = (unsigned char *)malloc(per_thread);
    if (!buf) return 0.0;
    for (size_t i = 0; i < per_thread; i += 4096) buf[i] = (unsigned char)i;

    double best = 0.0;
    for (int pass = 0; pass < 3; pass++) {
        BwThreadArg a = {.buf = buf, .size = per_thread, .gbps = 0.0, .sink = 0};
        bw_thread_fn(&a);
        g_bw_probe_sink += a.sink; /* keep the reads observable here too */
        if (a.gbps > best) best = a.gbps;
    }
    free(buf);
    return best;
}
#endif
