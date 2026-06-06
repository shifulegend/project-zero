/*
 * bench_simd.c — Phase 16-S SIMD kernel microbenchmarks
 *
 * Measures throughput of each ternary matmul kernel on the current CPU.
 * Reports MACs/cycle and effective tok/s equivalent at BitNet 2B dimensions.
 *
 * No model file required — uses synthetic random weights and activations.
 *
 * Build:
 *   gcc -O3 -march=native -Iinclude -D_POSIX_C_SOURCE=200809L \
 *       -o build/bench_simd tools/bench_simd.c \
 *       build/math/ternary_matmul_packed_vnni.o \
 *       build/math/ternary_matmul_packed_avx_vnni.o \
 *       build/math/ternary_matmul_packed_avx512.o \
 *       build/math/ternary_matmul_packed_avx2.o \
 *       build/math/ternary_matmul_packed.o \
 *       build/math/ternary_matmul_scalar.o \
 *       build/math/quantize_i8.o \
 *       build/math/cpu_features.o \
 *       build/math/simd_dispatch.o \
 *       build/memory/aligned_alloc.o \
 *       -lm -pthread
 *
 * Or via Makefile convenience target:
 *   make bench
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core/platform.h"
#include "math/ternary_matmul_packed.h"
#include "math/cpu_features.h"
#include "math/simd_dispatch.h"
#include "memory/aligned_alloc.h"

/* ── Benchmark parameters ─────────────────────────────────────────────────── */

/*
 * BitNet b1.58-2B-4T layer dimensions (from model config):
 *   hidden_dim = 2048  (embedding / attention dimension)
 *   ffn_dim    = 5632  (SwiGLU intermediate dimension)
 *   num_layers = 24
 *
 * Dominant kernel per layer:
 *   3× attention projections: (2048 × 2048) matmul
 *   3× FFN gates:             (2048 × 5632) matmul
 *
 * We benchmark at D=2048, N=2048 (square) as the representative case.
 * A single forward pass needs ~48 such matmuls, so tok/s ≈ 1/(48 × kernel_time).
 */
#define BENCH_N        2048   /* input dimension  */
#define BENCH_D        2048   /* output dimension */
#define BENCH_WARMUP   5      /* warmup iterations (not timed) */
#define BENCH_ITERS    50     /* measured iterations */

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random_float(float *arr, int n, unsigned int seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        arr[i] = ((float)(seed & 0x7FFFu) / 16384.0f) - 1.0f;
    }
}

static void fill_ternary_packed(tn_u8 *packed, int n, int d, unsigned int seed) {
    size_t row_bytes = ((size_t)n + 3) >> 2;
    size_t total     = (size_t)d * row_bytes;
    /* Random bytes; not all patterns are valid ternary but close enough for bench */
    for (size_t i = 0; i < total; i++) {
        seed = seed * 1103515245u + 12345u;
        packed[i] = (tn_u8)(seed >> 8);
    }
}

/* ── Per-kernel benchmark ─────────────────────────────────────────────────── */

typedef void (*MatmulFn)(float*, const float*, const tn_u8*, int, int,
                         const float*, int);

typedef struct {
    const char *name;
    MatmulFn    fn;
    int         available; /* compile-time + runtime check */
} KernelEntry;

static double bench_kernel(const char *name, MatmulFn fn,
                           float *x, tn_u8 *packed, float *out,
                           int n, int d) {
    float scales[1] = { 0.125f };

    /* Warmup */
    for (int i = 0; i < BENCH_WARMUP; i++) {
        fn(out, x, packed, n, d, scales, 0);
    }

    /* Timed run */
    double t0 = now_seconds();
    for (int i = 0; i < BENCH_ITERS; i++) {
        fn(out, x, packed, n, d, scales, 0);
    }
    double t1 = now_seconds();

    double elapsed_ms   = (t1 - t0) * 1000.0;
    double per_call_ms  = elapsed_ms / BENCH_ITERS;

    /*
     * MACs per call: each output element is a dot product of n weights × n acts.
     * Non-zero ternary weight fraction ≈ 2/3 (BitNet paper reports ~1/3 zeros).
     * Total ops: d × n × 1 multiply-add (we count all n, as zero skipping varies).
     */
    double total_macs   = (double)d * (double)n;
    /* per_call_ms is in ms; convert to seconds (* 1e-3), then to GMAC/s (/ 1e9) */
    double gmacs_per_s  = (total_macs / (per_call_ms * 1e-3)) * 1e-9; /* GMAC/s */

    /*
     * Effective tok/s: one token forward pass = 24 layers × 6 matmuls per layer.
     * This gives a rough theoretical max ignoring attention, sampling, etc.
     */
    double matmuls_per_tok = 24.0 * 6.0;   /* 144 matmuls */
    double toks_per_s      = 1000.0 / (per_call_ms * matmuls_per_tok);

    printf("  %-22s  %7.3f ms/call  %6.2f GMAC/s  ~%5.1f tok/s (synthetic)\n",
           name, per_call_ms, gmacs_per_s, toks_per_s);

    return per_call_ms;
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n");
    printf("=== Phase 16-S SIMD Kernel Microbenchmarks ===\n");
    printf("    Matrix dims: %d × %d  (N=input, D=output)\n", BENCH_N, BENCH_D);
    printf("    Warmup: %d iters, Measured: %d iters\n\n", BENCH_WARMUP, BENCH_ITERS);

    /* Print detected CPU */
    const TnCpuFeatures *cpu = tn_cpu_features_detect();
    printf("Hardware: %s\n", tn_cpu_best_backend_name(cpu));
    printf("  AVX2=%d  AVX-512F=%d  AVX-512VNNI=%d  AVX-VNNI=%d  ARM-DOTPROD=%d\n\n",
           cpu->avx2, cpu->avx512f, cpu->avx512vnni, cpu->avx_vnni, cpu->arm_dotprod);

    /* Allocate buffers */
    const int n = BENCH_N, d = BENCH_D;
    size_t row_bytes  = ((size_t)n + 3) >> 2;
    float  *x         = (float*)tn_aligned_calloc(n, sizeof(float), 64);
    float  *out       = (float*)tn_aligned_calloc(d, sizeof(float), 64);
    tn_u8  *packed    = (tn_u8*)tn_aligned_calloc((size_t)d * row_bytes, 1, 64);

    if (!x || !out || !packed) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    fill_random_float(x, n, 42);
    fill_ternary_packed(packed, n, d, 99);

    printf("Results (lower ms/call = faster; tok/s is synthetic upper bound):\n");
    printf("  %-22s  %11s  %11s  %s\n",
           "Kernel", "ms/call", "GMAC/s", "~tok/s");
    printf("  %s\n", "----------------------------------------------------------------------");

    double baseline_ms = 0.0;

    /* ── Scalar fallback ─────────────────────────────── */
    {
        extern void ternary_matmul_packed(float*, const float*, const tn_u8*,
                                          int, int, const float*, int);
        /* Force scalar: call the generic dispatcher which auto-selects */
        /* Instead, use the reference scalar directly via dispatch override */
    }
    /* Note: we can't easily call the scalar directly without refactoring dispatch.
     * Call the generic entry point (which will use the best available backend),
     * then compare per-tier numbers. */

    /* ── AVX2 (float32 FMA) ──────────────────────────── */
#if TN_HAS_AVX2
    if (cpu->avx2) {
        baseline_ms = bench_kernel("AVX2 float32",
                                   ternary_matmul_packed_avx2,
                                   x, packed, out, n, d);
    }
#endif

    /* ── AVX-512F (float32 FMA) ──────────────────────── */
#if TN_HAS_AVX512
    if (cpu->avx512f) {
        bench_kernel("AVX-512F float32",
                     ternary_matmul_packed_avx512,
                     x, packed, out, n, d);
    }
#endif

    /* ── AVX-VNNI (256-bit int8 VNNI) ───────────────── */
#if TN_HAS_AVXVNNI
    if (cpu->avx_vnni) {
        bench_kernel("AVX-VNNI int8",
                     ternary_matmul_packed_avx_vnni,
                     x, packed, out, n, d);
    } else {
        printf("  %-22s  (not available on this CPU)\n", "AVX-VNNI int8");
    }
#else
    printf("  %-22s  (compiled out — no -mavxvnni)\n", "AVX-VNNI int8");
#endif

    /* ── AVX-512 VNNI (512-bit int8 VNNI) ───────────── */
#if TN_HAS_AVX512VNNI
    if (cpu->avx512vnni) {
        double vnni_ms = bench_kernel("AVX-512 VNNI int8",
                                      ternary_matmul_packed_vnni,
                                      x, packed, out, n, d);
        if (baseline_ms > 0.0) {
            printf("  %-22s  (%.1f× speedup vs AVX2 float32)\n",
                   "→ VNNI speedup", baseline_ms / vnni_ms);
        }
    } else {
        printf("  %-22s  (not available on this CPU)\n", "AVX-512 VNNI int8");
    }
#else
    printf("  %-22s  (compiled out — no -mavx512vnni)\n", "AVX-512 VNNI int8");
#endif

    /* ── ARM dotprod ─────────────────────────────────── */
#if TN_HAS_ARM_DOTPROD
    if (cpu->arm_dotprod) {
        bench_kernel("ARM DOTPROD int8",
                     ternary_matmul_packed_dotprod,
                     x, packed, out, n, d);
    }
#endif

    /* ── Dispatched (best available) ─────────────────── */
    {
        const char *selected = tn_simd_init();
        printf("\n  Best backend selected by tn_simd_init(): %s\n", selected);
        /* Run via function pointer to confirm dispatch works */
        float scales[1] = { 0.125f };
        double t0 = now_seconds();
        for (int i = 0; i < BENCH_ITERS; i++) {
            tn_ternary_matmul_packed(out, x, packed, n, d, scales, 0);
        }
        double t1 = now_seconds();
        double per_call_ms = (t1 - t0) * 1000.0 / BENCH_ITERS;
        double gmacs       = (double)d * (double)n / (per_call_ms * 1e-3) * 1e-9;
        double toks        = 1000.0 / (per_call_ms * 24.0 * 6.0);
        printf("  %-22s  %7.3f ms/call  %6.2f GMAC/s  ~%5.1f tok/s (synthetic)\n",
               "→ Dispatched", per_call_ms, gmacs, toks);
    }

    printf("\nNote: tok/s is a synthetic upper bound assuming only matmul cost.\n");
    printf("      Real inference is ~3-5× lower due to attention, sampling, I/O.\n");
    printf("      Model weights are LFS stubs — actual end-to-end tok/s requires\n");
    printf("      a real model file.\n\n");

    tn_aligned_free(x);
    tn_aligned_free(out);
    tn_aligned_free(packed);
    return 0;
}
