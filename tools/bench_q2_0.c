/*
 * bench_q2_0 — in-binary A/B micro-benchmark for the Q2_0 VNNI row-dot
 * kernel (2026-07-17, docs/architecture/CEILING_CALCULATION.md §7 option A).
 *
 * Benchmarks BOTH kernel variants — parallel_matmul_q2_0_vnni (optimized:
 * per-row float vector accumulator + F16C scale decode) and
 * parallel_matmul_q2_0_vnni_ref (frozen pre-optimization baseline:
 * per-block _mm512_reduce_add_epi32 + scalar fp16 decode) — from ONE binary,
 * with interleaved timing passes, so the comparison is immune to the
 * run-to-run host drift documented in docs/reports/RCA_QWEN_TOKS_DROP_2026-07.md.
 * Also cross-checks the two variants' outputs (max abs diff) each run.
 *
 * Shapes are Ternary-Bonsai-27B's real matmul dims (n = input dim, d = rows):
 *   attn-ish   n=5120  d=5120
 *   ffn-down   n=17408 d=5120
 *   lm-head    n=5120  d=248320
 *
 * Usage: build/tools/bench_q2_0 [threads]   (default: 4, 0 = no thread pool)
 * Synthetic weight packing replicates tests/test_q2_0_matmul.c's generator.
 */

#include "core/platform.h"
#include "math/matmul_q2_0.h"
#include "threading/thread_pool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if !TN_HAS_AVX512VNNI
int main(void) {
    printf("bench_q2_0: build has no AVX-512 VNNI support; nothing to benchmark.\n");
    return 0;
}
#else

#define Q2_0_BLOCK 128
#define Q2_0_BYTES 34

static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign  = (u >> 16) & 0x8000;
    uint32_t exp   = ((u >> 23) & 0xFF);
    uint32_t mant  = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF);
    if (exp <= 102) return (uint16_t)sign;
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

/* Same deterministic block generator as tests/test_q2_0_matmul.c, so bench
 * inputs exercise every 2-bit code value the way the correctness test does. */
static void make_q2_0_block(uint8_t *blk, int seed) {
    float scale = 0.01f + (float)(seed % 13) * 0.007f;
    uint16_t d_h = f32_to_fp16(scale);
    memcpy(blk, &d_h, 2);
    uint8_t *qs = blk + 2;
    for (int i = 0; i < 32; i++) {
        int c0 = (i * 3 + seed) & 0x3;
        int c1 = (i * 5 + seed + 1) & 0x3;
        int c2 = (i * 7 + seed + 2) & 0x3;
        int c3 = (i * 11 + seed + 3) & 0x3;
        qs[i] = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
    }
}

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

typedef int (*Q2Entry)(float *, const float *, const uint8_t *, int, int, ThreadPool *);

static double time_one(Q2Entry fn, float *out, const float *x, const uint8_t *w,
                        int n, int d, ThreadPool *tp) {
    int64_t t0 = now_ns();
    if (!fn(out, x, w, n, d, tp)) {
        fprintf(stderr, "kernel declined shape n=%d d=%d\n", n, d);
        exit(1);
    }
    return (double)(now_ns() - t0) / 1e6; /* ms */
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void bench_shape(const char *name, int n, int d, ThreadPool *tp) {
    int n_blocks = n / Q2_0_BLOCK;
    size_t row_bytes = (size_t)n_blocks * Q2_0_BYTES;
    size_t w_bytes = row_bytes * (size_t)d;

    uint8_t *w = (uint8_t *)malloc(w_bytes);
    float *x = (float *)malloc((size_t)n * sizeof(float));
    float *out_ref = (float *)malloc((size_t)d * sizeof(float));
    float *out_new = (float *)malloc((size_t)d * sizeof(float));
    if (!w || !x || !out_ref || !out_new) {
        fprintf(stderr, "OOM allocating %s buffers\n", name);
        exit(1);
    }
    for (int i = 0; i < d; i++)
        for (int b = 0; b < n_blocks; b++)
            make_q2_0_block(w + (size_t)i * row_bytes + (size_t)b * Q2_0_BYTES,
                            i * 31 + b);
    srand(42);
    for (int j = 0; j < n; j++)
        x[j] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    /* Correctness cross-check first (also serves as cache warmup). */
    time_one(parallel_matmul_q2_0_vnni_ref, out_ref, x, w, n, d, tp);
    time_one(parallel_matmul_q2_0_vnni,     out_new, x, w, n, d, tp);
    double max_abs = 0.0, max_mag = 0.0;
    for (int i = 0; i < d; i++) {
        double diff = fabs((double)out_ref[i] - (double)out_new[i]);
        if (diff > max_abs) max_abs = diff;
        double mag = fabs((double)out_ref[i]);
        if (mag > max_mag) max_mag = mag;
    }
    /* Fail hard on divergence, don't just print it (independent-review
     * finding, 2026-07-17): the two variants share exact int32 block dots
     * and differ only in float accumulation order, so anything beyond a
     * loose relative epsilon is a real kernel bug, and a benchmark that
     * keeps timing a wrong kernel would be worse than useless. */
    if (max_abs > 1e-4 * (max_mag > 1.0 ? max_mag : 1.0)) {
        fprintf(stderr, "%s: variant outputs diverge (maxdiff %.6g vs maxmag %.6g)\n",
                name, max_abs, max_mag);
        exit(1);
    }

    /* Interleaved timing: ref/new alternate so slow host drift hits both. */
    enum { REPS = 7 };
    double t_ref[REPS], t_new[REPS];
    for (int r = 0; r < REPS; r++) {
        t_ref[r] = time_one(parallel_matmul_q2_0_vnni_ref, out_ref, x, w, n, d, tp);
        t_new[r] = time_one(parallel_matmul_q2_0_vnni,     out_new, x, w, n, d, tp);
    }
    qsort(t_ref, REPS, sizeof(double), cmp_double);
    qsort(t_new, REPS, sizeof(double), cmp_double);
    double med_ref = t_ref[REPS / 2], med_new = t_new[REPS / 2];
    double gbps_ref = (double)w_bytes / (med_ref * 1e6);
    double gbps_new = (double)w_bytes / (med_new * 1e6);

    printf("%-22s ref %8.2f ms (%5.2f GB/s)   new %8.2f ms (%5.2f GB/s)   "
           "speedup %.2fx   maxdiff %.3g (maxmag %.3g)\n",
           name, med_ref, gbps_ref, med_new, gbps_new,
           med_ref / med_new, max_abs, max_mag);

    free(w); free(x); free(out_ref); free(out_new);
}

int main(int argc, char **argv) {
    int threads = (argc > 1) ? atoi(argv[1]) : 4;
    ThreadPool *tp = (threads > 0) ? threadpool_create(threads) : NULL;
    if (threads > 0 && !tp) {
        fprintf(stderr, "threadpool_create(%d) failed\n", threads);
        return 1;
    }
    printf("bench_q2_0: Q2_0 VNNI row-dot A/B, threads=%d "
           "(median of 7 interleaved reps; GB/s = packed weight bytes/time)\n",
           threads);

    bench_shape("attn 5120x5120",     5120,  5120,   tp);
    bench_shape("ffn-down 17408x5120", 17408, 5120,  tp);
    bench_shape("lm-head 5120x248320", 5120,  248320, tp);

    if (tp) threadpool_destroy(tp);
    return 0;
}
#endif /* TN_HAS_AVX512VNNI */
