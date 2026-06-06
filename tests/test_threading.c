#include "test_harness.h"
#include "threading/cpu_probe.h"
#include "threading/thread_pool.h"
#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "math/ternary_matmul.h"
#include "memory/aligned_alloc.h"
#include "core/platform.h"
#include <stdint.h>

/* ---- CPU Probe ---- */

static void test_cpu_probe_returns_positive(void) {
    int n = tn_get_optimal_thread_count();
    TEST_ASSERT(n >= 1, "thread count is at least 1");
    TEST_ASSERT(n <= 1024, "thread count is reasonable (<=1024)");
}

/* ---- Thread Pool Basic ---- */

/* Simple task: each thread writes its thread_id into a shared array */
static void write_id_task(void *arg, int thread_id, int start, int end) {
    int *buf = (int *)arg;
    for (int i = start; i < end; i++) {
        buf[i] = thread_id;
    }
}

static void test_threadpool_create_destroy(void) {
    ThreadPool *tp = threadpool_create(2);
    TEST_ASSERT(tp != NULL, "threadpool_create(2) returns non-NULL");
    threadpool_destroy(tp);
}

static void test_threadpool_dispatch_basic(void) {
    int n = 100;
    int buf[100];
    memset(buf, -1, sizeof(buf));

    ThreadPool *tp = threadpool_create(4);
    TEST_ASSERT(tp != NULL, "pool created for dispatch test");

    threadpool_dispatch(tp, write_id_task, buf, n);

    /* Every element should have been assigned a valid thread id */
    int all_valid = 1;
    for (int i = 0; i < n; i++) {
        if (buf[i] < 0 || buf[i] >= 4) {
            all_valid = 0;
            break;
        }
    }
    TEST_ASSERT(all_valid, "all work items processed by valid thread ids");

    threadpool_destroy(tp);
}

static void test_threadpool_single_thread(void) {
    int n = 10;
    int buf[10];
    memset(buf, -1, sizeof(buf));

    ThreadPool *tp = threadpool_create(1);
    TEST_ASSERT(tp != NULL, "single-thread pool created");

    threadpool_dispatch(tp, write_id_task, buf, n);

    /* Single thread => all items processed by thread 0 */
    int all_zero = 1;
    for (int i = 0; i < n; i++) {
        if (buf[i] != 0) { all_zero = 0; break; }
    }
    TEST_ASSERT(all_zero, "single thread processes all items");

    threadpool_destroy(tp);
}

static void test_threadpool_dispatch_multiple(void) {
    /* Dispatch multiple times to ensure pool is reusable */
    int n = 50;
    int buf[50];
    ThreadPool *tp = threadpool_create(3);
    TEST_ASSERT(tp != NULL, "pool created for multi-dispatch");

    for (int round = 0; round < 5; round++) {
        memset(buf, -1, sizeof(buf));
        threadpool_dispatch(tp, write_id_task, buf, n);

        int all_valid = 1;
        for (int i = 0; i < n; i++) {
            if (buf[i] < 0 || buf[i] >= 3) { all_valid = 0; break; }
        }
        TEST_ASSERT(all_valid, "multi-dispatch: all items valid");
    }

    threadpool_destroy(tp);
}

/* ---- Parallel MatMul ---- */

static void test_parallel_matmul_matches_scalar(void) {
    /* Use a moderately sized problem: 64 output rows, 128 input dim */
    int d = 64;
    int n = 128;
    float scale = 0.5f;

    float *x = (float *)tn_aligned_alloc((size_t)n * sizeof(float), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_alloc((size_t)d * sizeof(float), TN_SIMD_ALIGN);
    float *out_parallel = (float *)tn_aligned_alloc((size_t)d * sizeof(float), TN_SIMD_ALIGN);
    tn_i8 *w = (tn_i8 *)tn_aligned_alloc((size_t)d * (size_t)n * sizeof(tn_i8), TN_SIMD_ALIGN);

    TEST_ASSERT(x && out_scalar && out_parallel && w, "parallel matmul alloc OK");

    /* Fill input with a pattern */
    for (int i = 0; i < n; i++) {
        x[i] = (float)(i % 7) - 3.0f;  /* values in [-3, 3] */
    }

    /* Fill weights with ternary values */
    for (int i = 0; i < d * n; i++) {
        w[i] = (tn_i8)((i % 3) - 1);  /* cycles through -1, 0, 1 */
    }

    /* Scalar reference */
    ternary_matmul(out_scalar, x, w, n, d, scale);

    /* Parallel version */
    ThreadPool *tp = threadpool_create(4);
    TEST_ASSERT(tp != NULL, "pool for parallel matmul");

    parallel_ternary_matmul(out_parallel, x, w, n, d, scale, tp);

    /* Compare results */
    int all_match = 1;
    for (int i = 0; i < d; i++) {
        if (fabsf(out_scalar[i] - out_parallel[i]) > 1e-5f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "parallel matmul matches scalar reference");

    threadpool_destroy(tp);
    tn_aligned_free(x);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_parallel);
    tn_aligned_free(w);
}

static void test_parallel_matmul_null_pool_fallback(void) {
    /* NULL pool should fall back to single-threaded */
    int d = 8, n = 4;
    float scale = 1.0f;

    tn_i8 w[] = { 1, -1, 0, 1,
                   0,  1, 1, 0,
                   1,  1, 1, 1,
                  -1, -1, -1, -1,
                   1,  0,  0,  1,
                   0,  0,  0,  0,
                  -1,  1, -1,  1,
                   1, -1,  1, -1 };
    float x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float out_ref[8], out_null[8];

    ternary_matmul(out_ref, x, w, n, d, scale);
    parallel_ternary_matmul(out_null, x, w, n, d, scale, NULL);

    int match = 1;
    for (int i = 0; i < d; i++) {
        if (fabsf(out_ref[i] - out_null[i]) > 1e-5f) { match = 0; break; }
    }
    TEST_ASSERT(match, "NULL pool fallback matches scalar");
}

static void test_parallel_matmul_large(void) {
    /* Larger problem: 512 output rows, 1024 input dim (more realistic) */
    int d = 512;
    int n = 1024;
    float scale = 0.25f;

    float *x = (float *)tn_aligned_alloc((size_t)n * sizeof(float), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_alloc((size_t)d * sizeof(float), TN_SIMD_ALIGN);
    float *out_parallel = (float *)tn_aligned_alloc((size_t)d * sizeof(float), TN_SIMD_ALIGN);
    tn_i8 *w = (tn_i8 *)tn_aligned_alloc((size_t)d * (size_t)n * sizeof(tn_i8), TN_SIMD_ALIGN);

    TEST_ASSERT(x && out_scalar && out_parallel && w, "large matmul alloc OK");

    for (int i = 0; i < n; i++) {
        x[i] = (float)((i * 7 + 3) % 11) - 5.0f;
    }
    for (int i = 0; i < d * n; i++) {
        int v = (i * 13 + 5) % 5;  /* 0..4 mapped to {-1, -1, 0, 1, 1} */
        w[i] = (tn_i8)(v <= 1 ? -1 : (v == 2 ? 0 : 1));
    }

    /* Use the SIMD-dispatched version as reference (same kernel parallel_matmul uses) */
    tn_ternary_matmul(out_scalar, x, w, n, d, scale);

    ThreadPool *tp = threadpool_create(4);
    TEST_ASSERT(tp != NULL, "pool for large matmul");

    parallel_ternary_matmul(out_parallel, x, w, n, d, scale, tp);

    int all_match = 1;
    for (int i = 0; i < d; i++) {
        if (fabsf(out_scalar[i] - out_parallel[i]) > 1e-5f) {
            all_match = 0;
            break;
        }
    }
    TEST_ASSERT(all_match, "large parallel matmul matches single-threaded");

    threadpool_destroy(tp);
    tn_aligned_free(x);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_parallel);
    tn_aligned_free(w);
}

/* ---- FM-006 Stress Tests ---- */

/*
 * FM-006 deadlock regression: rapid sequential dispatches.
 * The original bug (fixed by epoch-based inner wait in 741021a) caused
 * threads trapped in the inner wait to re-enter sleep instead of claiming
 * work when a new dispatch arrived before they fully transitioned to the
 * outer wait. This test hammers the pool with 100 back-to-back dispatches
 * to confirm no deadlock under rapid reuse.
 */
static void test_threadpool_stress_sequential(void) {
    int n = 50;
    int buf[50];
    ThreadPool *tp = threadpool_create(4);
    TEST_ASSERT(tp != NULL, "stress: pool created");

    for (int round = 0; round < 100; round++) {
        memset(buf, -1, sizeof(buf));
        threadpool_dispatch(tp, write_id_task, buf, n);

        int all_valid = 1;
        for (int i = 0; i < n; i++) {
            if (buf[i] < 0 || buf[i] >= 4) { all_valid = 0; break; }
        }
        TEST_ASSERT(all_valid, "stress: all items valid after dispatch");
    }

    threadpool_destroy(tp);
}

/* Accumulator task for verifying correctness under stress */
static void accumulate_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    int *buf = (int *)arg;
    for (int i = start; i < end; i++) {
        buf[i] += 1;
    }
}

/*
 * Stress test: dispatch the same buffer many times, verify cumulative result.
 * Each dispatch increments every element by 1, so after N dispatches
 * every element should equal N.
 */
static void test_threadpool_stress_accumulate(void) {
    int n = 200;
    int buf[200];
    int rounds = 50;
    memset(buf, 0, sizeof(buf));

    ThreadPool *tp = threadpool_create(4);
    TEST_ASSERT(tp != NULL, "accumulate stress: pool created");

    for (int round = 0; round < rounds; round++) {
        threadpool_dispatch(tp, accumulate_task, buf, n);
    }

    int all_correct = 1;
    for (int i = 0; i < n; i++) {
        if (buf[i] != rounds) { all_correct = 0; break; }
    }
    TEST_ASSERT(all_correct, "accumulate stress: all elements equal to round count");

    threadpool_destroy(tp);
}

/*
 * Stress test with varying work sizes per dispatch.
 * Exercises edge cases: total < num_threads, total == 1, large total.
 */
static void test_threadpool_stress_varying_sizes(void) {
    int buf[1024];
    int sizes[] = {1, 2, 3, 4, 5, 7, 10, 16, 50, 100, 512, 1024};
    int num_sizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    ThreadPool *tp = threadpool_create(4);
    TEST_ASSERT(tp != NULL, "varying sizes: pool created");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        memset(buf, -1, (size_t)n * sizeof(int));
        threadpool_dispatch(tp, write_id_task, buf, n);

        int all_valid = 1;
        for (int i = 0; i < n; i++) {
            if (buf[i] < 0 || buf[i] >= 4) { all_valid = 0; break; }
        }
        TEST_ASSERT(all_valid, "varying sizes: all items valid");
    }

    threadpool_destroy(tp);
}

/*
 * Stress test: many sequential parallel matmuls on the same pool.
 * This is the real-world usage pattern: a single thread pool is reused
 * across all layers of a transformer forward pass.
 */
static void test_parallel_matmul_repeated(void) {
    int d = 64;
    int n = 128;
    float scale = 0.5f;

    float *x = (float *)tn_aligned_alloc((size_t)n * sizeof(float), TN_SIMD_ALIGN);
    float *out_scalar = (float *)tn_aligned_alloc((size_t)d * sizeof(float), TN_SIMD_ALIGN);
    float *out_parallel = (float *)tn_aligned_alloc((size_t)d * sizeof(float), TN_SIMD_ALIGN);
    tn_i8 *w = (tn_i8 *)tn_aligned_alloc((size_t)d * (size_t)n * sizeof(tn_i8), TN_SIMD_ALIGN);

    TEST_ASSERT(x && out_scalar && out_parallel && w, "repeated matmul alloc OK");

    for (int i = 0; i < n; i++) {
        x[i] = (float)(i % 7) - 3.0f;
    }
    for (int i = 0; i < d * n; i++) {
        w[i] = (tn_i8)((i % 3) - 1);
    }

    ternary_matmul(out_scalar, x, w, n, d, scale);

    ThreadPool *tp = threadpool_create(4);
    TEST_ASSERT(tp != NULL, "pool for repeated matmul");

    /* Simulate 32 transformer layers reusing the same pool */
    for (int layer = 0; layer < 32; layer++) {
        parallel_ternary_matmul(out_parallel, x, w, n, d, scale, tp);

        int all_match = 1;
        for (int i = 0; i < d; i++) {
            if (fabsf(out_scalar[i] - out_parallel[i]) > 1e-5f) {
                all_match = 0;
                break;
            }
        }
        TEST_ASSERT(all_match, "repeated matmul: layer result matches scalar");
    }

    threadpool_destroy(tp);
    tn_aligned_free(x);
    tn_aligned_free(out_scalar);
    tn_aligned_free(out_parallel);
    tn_aligned_free(w);
}

/* ---- Main ---- */

int main(void) {
    /* Initialize SIMD dispatch (needed for parallel_matmul) */
    tn_simd_init();

    RUN_TEST(test_cpu_probe_returns_positive);
    RUN_TEST(test_threadpool_create_destroy);
    RUN_TEST(test_threadpool_dispatch_basic);
    RUN_TEST(test_threadpool_single_thread);
    RUN_TEST(test_threadpool_dispatch_multiple);
    RUN_TEST(test_parallel_matmul_matches_scalar);
    RUN_TEST(test_parallel_matmul_null_pool_fallback);
    RUN_TEST(test_parallel_matmul_large);
    RUN_TEST(test_threadpool_stress_sequential);
    RUN_TEST(test_threadpool_stress_accumulate);
    RUN_TEST(test_threadpool_stress_varying_sizes);
    RUN_TEST(test_parallel_matmul_repeated);

    TEST_SUMMARY();
}
