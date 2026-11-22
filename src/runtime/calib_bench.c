/* calib_bench.c — the timed ternary-matmul benchmark the calibration phases are built on, and the
 * background-load probe that tells whether a measurement can be trusted. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/clamp.h"
#include "core/sys_probe.h"
#include "math/parallel_matmul.h"
#include "runtime/calib_internal.h"
#include "runtime/calibration.h"
#include "threading/cpu_probe.h"

#if TN_POSIX
#include <unistd.h>
#endif

/* ── Monotonic clock ─────────────────────────────────────────────────── */

int64_t calib_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ── Background system load ──────────────────────────────────────────── */

/*
 * Read aggregate CPU counters from /proc/stat (Linux only).
 * Returns 1 on success, 0 on failure (non-Linux, permission error, etc.).
 */
#if TN_POSIX
typedef struct {
    long long user, nice, sys, idle, iowait, irq, softirq;
} CpuStat;

static int read_cpu_stat(CpuStat *s) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return 0;
    char line[256];
    memset(s, 0, sizeof(*s));
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu ", 4) == 0) {
            sscanf(line + 4, "%lld %lld %lld %lld %lld %lld %lld", &s->user, &s->nice, &s->sys,
                   &s->idle, &s->iowait, &s->irq, &s->softirq);
            found = 1;
            break;
        }
    }
    fclose(f);
    return found;
}
#endif

/*
 * Sample background CPU load over 500 ms BEFORE the benchmark starts.
 * Since this runs before our benchmark threads are active, the result
 * reflects load from OTHER processes only — a clean background signal.
 *
 * Returns 0-100 (percent), or -1 if /proc/stat is unavailable.
 */
int calib_sample_background_load(void) {
#if TN_POSIX
    CpuStat s0, s1;
    if (!read_cpu_stat(&s0)) return -1; /* /proc/stat absent — no sleep */
    {
        struct timespec ts = {0, 500000000L};
        nanosleep(&ts, NULL);
    } /* 500 ms */
    if (!read_cpu_stat(&s1)) return -1;

    long long busy = (s1.user - s0.user) + (s1.nice - s0.nice) + (s1.sys - s0.sys) +
                     (s1.irq - s0.irq) + (s1.softirq - s0.softirq);
    long long idle  = (s1.idle - s0.idle) + (s1.iowait - s0.iowait);
    long long total = busy + idle;
    if (total <= 0) return -1;
    int pct = (int)(busy * 100LL / total);
    return tn_clampi(pct, 0, 100);
#else
    return -1;
#endif
}

/* ── Robust parallel benchmark ───────────────────────────────────────── */

/*
 * Benchmark constants — simulate one full token worth of ternary matmuls.
 * 7 matmuls/layer × 30 layers = 210 per "token equivalent".
 * Dimension 2560×2560 matches BitNet b1.58 2B hidden size.
 */
#define CALIB_DIM 2560
#define CALIB_MATMULS_PER_TOK 210
#define CALIB_MAX_REPS 9

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

bool calib_bench_buf_alloc(BenchBuf *b) {
    const int d = CALIB_DIM, n = CALIB_DIM, packed_bytes = d * (n / 4);
    b->x    = (float *)malloc((size_t)n * sizeof(float));
    b->pw   = (tn_u8 *)malloc((size_t)packed_bytes * sizeof(tn_u8));
    b->outv = (float *)calloc((size_t)d, sizeof(float));
    if (!b->x || !b->pw || !b->outv) {
        free(b->x);
        free(b->pw);
        free(b->outv);
        b->x    = NULL;
        b->pw   = NULL;
        b->outv = NULL;
        return false;
    }
    /* Non-trivial data so SIMD kernels cannot short-circuit */
    for (int i = 0; i < n; i++) b->x[i] = 0.01f * (float)((i % 100) + 1);
    for (int i = 0; i < packed_bytes; i++) b->pw[i] = (tn_u8)((i & 0xFE) | 0x01);
    return true;
}

void calib_bench_buf_free(BenchBuf *b) {
    free(b->x);
    free(b->pw);
    free(b->outv);
    b->x    = NULL;
    b->pw   = NULL;
    b->outv = NULL;
}

/* Time-based warmup: continuous matmuls for warmup_secs (one dot per second). */
void calib_bench_warmup(ThreadPool *tp, const BenchBuf *b, double warmup_secs) {
    int64_t warmup_end = calib_now_ns() + (int64_t)(warmup_secs * 1.0e9);
    int64_t next_dot   = calib_now_ns() + 1000000000LL;
    while (calib_now_ns() < warmup_end) {
        parallel_ternary_matmul_packed(b->outv, b->x, b->pw, CALIB_DIM, CALIB_DIM, 1.0f, tp);
        if (calib_now_ns() >= next_dot) {
            putchar('.');
            fflush(stdout);
            next_dot += 1000000000LL;
        }
    }
}

/* One timed repetition: CALIB_MATMULS_PER_TOK matmuls, returned as tokens/s. */
double calib_bench_one_rep(ThreadPool *tp, const BenchBuf *b) {
    int64_t t0 = calib_now_ns();
    for (int i = 0; i < CALIB_MATMULS_PER_TOK; i++)
        parallel_ternary_matmul_packed(b->outv, b->x, b->pw, CALIB_DIM, CALIB_DIM, 1.0f, tp);
    int64_t elapsed = calib_now_ns() - t0;
    return (elapsed > 0) ? (1.0e9 / (double)elapsed) : 0.0;
}

/*
 * Run a robust multi-threaded ternary matmul benchmark.
 *
 * Key design decisions vs the old bench_matmul():
 *   1. Uses parallel_ternary_matmul_packed (NOT tn_ternary_matmul_packed)
 *      so the thread pool is genuinely exercised.  The old code created a
 *      ThreadPool but discarded it — thread count had zero effect on results.
 *   2. Time-based warmup (warmup_secs) instead of a fixed 3-iteration warmup.
 *      Three iterations at T=4/VNNI is ~135 ms — nowhere near long enough to
 *      exit Tiger Lake's 20-28 s PL2 turbo window.  Subsequent measurements
 *      would reflect burst frequency, not sustained throughput.
 *   3. Each of measure_reps reps is timed individually → min/median/max.
 *      The old code timed all reps together so variance was invisible.
 *   4. Progress dots are flushed to stdout once per second during warmup and
 *      once per rep during measurement so the terminal never looks frozen.
 *
 * @param tp            Pre-created thread pool (caller manages lifetime)
 * @param warmup_secs   Seconds of continuous matmuls before timing begins
 * @param measure_reps  Individual timed repetitions (must be <= CALIB_MAX_REPS)
 * @param out_min       Written with lowest observed tok/s across reps
 * @param out_max       Written with highest observed tok/s across reps
 * @return              Median tok/s (robust against single-rep outliers)
 */
double calib_bench_robust(ThreadPool *tp, double warmup_secs, int measure_reps, double *out_min,
                          double *out_max) {
    BenchBuf b;
    if (!calib_bench_buf_alloc(&b)) {
        *out_min = *out_max = 0.0;
        return 0.0;
    }

    printf("warmup[");
    fflush(stdout);
    calib_bench_warmup(tp, &b, warmup_secs);
    printf("] reps[");
    fflush(stdout);

    if (measure_reps < 1) measure_reps = 1;
    if (measure_reps > CALIB_MAX_REPS) measure_reps = CALIB_MAX_REPS;

    double samples[CALIB_MAX_REPS];
    for (int r = 0; r < measure_reps; r++) {
        samples[r] = calib_bench_one_rep(tp, &b);
        putchar('.');
        fflush(stdout);
    }
    printf("]");
    fflush(stdout);

    calib_bench_buf_free(&b);

    qsort(samples, (size_t)measure_reps, sizeof(double), cmp_double_asc);
    *out_min = samples[0];
    *out_max = samples[measure_reps - 1];
    return samples[measure_reps / 2]; /* median */
}


double calib_median_of(const double *v, int n) {
    double t[TN_CALIB_MAX_ROUNDS];
    for (int i = 0; i < n; i++) t[i] = v[i];
    qsort(t, (size_t)n, sizeof(double), cmp_double_asc);
    return t[n / 2];
}
