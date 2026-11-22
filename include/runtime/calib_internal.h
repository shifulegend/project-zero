#ifndef TN_CALIB_INTERNAL_H
#define TN_CALIB_INTERNAL_H

/* Shared by calibration.c (cache, report), calib_bench.c (the timed matmul benchmark) and
 * calib_run.c (the calibration phases). Not part of the public interface. */

#include <stdbool.h>
#include <stdint.h>

#include "core/platform.h"
#include "threading/thread_pool.h"

/* Buffers of the benchmark matmul (one ternary projection, CALIB_DIM x CALIB_DIM). */
typedef struct {
    float *x;
    tn_u8 *pw;
    float *outv;
} BenchBuf;

int64_t calib_now_ns(void);

/* Percent of CPU time other processes used over 500 ms; -1 where it cannot be read. */
int calib_sample_background_load(void);

bool calib_bench_buf_alloc(BenchBuf *b);
void calib_bench_buf_free(BenchBuf *b);

/* Continuous matmuls for warmup_secs (prints a dot a second). */
void calib_bench_warmup(ThreadPool *tp, const BenchBuf *b, double warmup_secs);

/* One timed repetition (a token's worth of matmuls), as tokens per second. */
double calib_bench_one_rep(ThreadPool *tp, const BenchBuf *b);

/* Warmup, then measure_reps timed repetitions: the median tok/s, with min and max. */
double calib_bench_robust(ThreadPool *tp, double warmup_secs, int measure_reps, double *out_min,
                          double *out_max);

/* Median of n values (n <= TN_CALIB_MAX_ROUNDS). */
double calib_median_of(const double *v, int n);

#include "core/cpu_features.h"
#include "runtime/calibration.h"

/* The SIMD backend names tried, [0] being "auto" (see calibration.c for the order). */
extern const char *const TN_CALIB_BACKENDS[TN_CALIB_SIMD_N];

/* Sets avail[i] for the tiers this build compiled in and this CPU can run ("auto" and "scalar"
 * always). Lives in calibration.c, the one calibration file built with the dispatch flag. */
void calib_mark_available(bool avail[TN_CALIB_SIMD_N], const TnCpuFeatures *cpu);

#endif /* TN_CALIB_INTERNAL_H */
