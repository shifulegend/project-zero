#include "math/simd_dispatch.h"
#include "math/cpu_features.h"
#include "math/ternary_matmul.h"
#include "math/ternary_matmul_packed.h"
#include "core/unpack.h"
#include "math/rmsnorm.h"
#include "math/softmax.h"
#include "math/elementwise.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

/*
 * Portable distribution build (`make dist`): this TU is compiled at a
 * conservative baseline (x86-64-v2) so the always-run startup path emits NO
 * SIMD, while every kernel is force-compiled with its own ISA in its own TU.
 * Force the x86 dispatch branches to be PRESENT here regardless of this TU's
 * own -march; the runtime cpu->* CPUID checks below still gate actual
 * execution, so a tier is only ever called on a CPU that supports it. ARM
 * guards are intentionally left untouched (those kernel TUs are empty on x86).
 * This file contains no intrinsics, so redefining the macros cannot cause the
 * compiler to emit AVX/AVX-512 instructions.
 */
#ifdef TN_FORCE_DISPATCH_ALL
#undef  TN_HAS_AVX2
#define TN_HAS_AVX2 1
#undef  TN_HAS_AVX512
#define TN_HAS_AVX512 1
#undef  TN_HAS_AVX512VNNI
#define TN_HAS_AVX512VNNI 1
#undef  TN_HAS_AVXVNNI
#define TN_HAS_AVXVNNI 1
#endif

/*
 * SIMD dispatch — Phase 16-S.1 (expanded from Phase 3.6)
 *
 * Tier priority for ternary matmul (highest throughput first):
 *   1. AVX-512 VNNI  — 64 int8 MACs/cycle  (Ice Lake, Tiger Lake, Zen4, SP Rapids)
 *   2. AVX-VNNI      — 32 int8 MACs/cycle  (Alder Lake, Raptor Lake, Zen3)
 *   3. AVX-512F      — 16 fp32 MACs/cycle  (Skylake-X, Ice/Tiger Lake without VNNI)
 *   4. AVX2          —  8 fp32 MACs/cycle  (Haswell+, Zen2)
 *   5. ARM dotprod   — 16 int8 MACs/cycle  (Apple M1+, Cortex-A75+, Snapdragon)
 *   6. ARM NEON      —  4 fp32 MACs/cycle  (all ARMv8-A)
 *   7. Scalar        —  1 fp32 MAC/cycle   (fallback for unknown CPUs)
 *
 * Selection uses RUNTIME CPUID probing (tn_cpu_features_detect()) rather than
 * compile-time #ifdefs alone.  This allows a binary compiled with -march=native
 * on a VNNI machine to gracefully fall back when run on a non-VNNI machine.
 * The compile-time guards still prevent generating VNNI instructions on
 * compilers that don't support them — runtime selection only downgrades.
 *
 * Utility ops (rmsnorm, softmax, elementwise) continue using the highest
 * available float SIMD tier: AVX-512F > AVX2 > Scalar.
 */

/* ── AVX-512 VNNI forward declarations ───────────────────────────────────── */
#if TN_HAS_AVX512VNNI
extern void ternary_matmul_packed_vnni(float *out, const float *x,
                                        const tn_u8 *packed_w, int n, int d,
                                        const float *scales, int group_size);
#endif

/* ── AVX-VNNI 256-bit forward declarations ───────────────────────────────── */
#if TN_HAS_AVXVNNI
extern void ternary_matmul_packed_avx_vnni(float *out, const float *x,
                                            const tn_u8 *packed_w, int n, int d,
                                            const float *scales, int group_size);
#endif

/* ── VNNI-256: EVEX-256 via AVX-512VNNI (Tiger Lake — no ZMM throttle) ───── */
#if TN_HAS_AVX512VNNI
extern void ternary_matmul_packed_vnni256(float *out, const float *x,
                                           const tn_u8 *packed_w, int n, int d,
                                           const float *scales, int group_size);
#endif

/* ── AVX-512F forward declarations ───────────────────────────────────────── */
#if TN_HAS_AVX512
extern void ternary_matmul_packed_avx512(float *out, const float *x,
                                          const tn_u8 *packed_w, int n, int d,
                                          const float *scales, int group_size);
extern void rmsnorm_avx512(float *out, const float *x, const float *weight, int size, float eps);
extern void softmax_avx512(float *x, int size);
extern void vec_add_avx512(float *out, const float *a, const float *b, int n);
extern void vec_mul_avx512(float *out, const float *a, const float *b, int n);
extern void vec_scale_avx512(float *x, float s, int n);
extern void silu_avx512(float *x, int n);
extern void relu2_avx512(float *x, int n);
extern float vec_dot_avx512(const float *a, const float *b, int n);
extern void vec_saxpy_avx512(float *out, float scale, const float *v, int n);
#endif

/* ── AVX2 forward declarations ───────────────────────────────────────────── */
#if TN_HAS_AVX2
extern void ternary_matmul_avx2(float *out, const float *x, const tn_i8 *w,
                                 int n, int d, float scale);
extern void ternary_matmul_packed_avx2(float *out, const float *x,
                                        const tn_u8 *packed_w, int n, int d,
                                        const float *scales, int group_size);
extern void unpack_ternary_block_avx2(tn_i8 *out, const tn_u8 *packed, int count);
extern void rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);
extern void softmax_avx2(float *x, int size);
extern void vec_add_avx2(float *out, const float *a, const float *b, int n);
extern void vec_mul_avx2(float *out, const float *a, const float *b, int n);
extern void vec_scale_avx2(float *x, float s, int n);
extern void silu_avx2(float *x, int n);
extern void relu2_avx2(float *x, int n);
extern float vec_dot_avx2(const float *a, const float *b, int n);
extern void vec_saxpy_avx2(float *out, float scale, const float *v, int n);
#endif

/* ── ARM dotprod forward declaration ─────────────────────────────────────── */
#if TN_HAS_ARM_DOTPROD
extern void ternary_matmul_packed_dotprod(float *out, const float *x,
                                           const tn_u8 *packed_w, int n, int d,
                                           const float *scales, int group_size);
#endif

/* ── Global dispatch table ─────────────────────────────────────────────────
 * Initialized to scalar fallback implementations by default.
 */
tn_matmul_fn         tn_ternary_matmul        = ternary_matmul;
tn_matmul_packed_fn  tn_ternary_matmul_packed = ternary_matmul_packed;
tn_unpack_fn         tn_unpack_block          = unpack_ternary_block;
tn_rmsnorm_fn        tn_rmsnorm               = rmsnorm;
tn_softmax_fn        tn_softmax               = softmax;
tn_vec_add_fn        tn_vec_add               = vec_add;
tn_vec_mul_fn        tn_vec_mul               = vec_mul;
tn_vec_scale_fn      tn_vec_scale             = vec_scale;
tn_silu_fn           tn_silu                  = silu;
tn_relu2_fn          tn_relu2                 = relu2_scalar;
tn_vec_dot_fn        tn_vec_dot               = vec_dot;
tn_vec_saxpy_fn      tn_vec_saxpy             = vec_saxpy;

/* Human-readable name of the selected backend (set by tn_simd_init) */
static const char *g_backend_name = "Scalar";
static atomic_int g_simd_init_state = 0; /* 0: uninitialized, 1: initializing, 2: initialized */

static const char *tn_simd_init_impl(void) {
    /*
     * Runtime probe the CPU, then select the best compiled-in kernel.
     * The probe is cached — subsequent calls to tn_simd_init are safe.
     *
     * TN_FORCE_BACKEND override (testing / benchmarking only):
     *   Set this environment variable to cap the dispatch tier:
     *     TN_FORCE_BACKEND=scalar    — scalar reference path
     *     TN_FORCE_BACKEND=avx2      — AVX2 float32 (8-wide FMA)
     *     TN_FORCE_BACKEND=avx512f   — AVX-512F float32 (16-wide FMA)
     *     TN_FORCE_BACKEND=vnni      — AVX-512 VNNI int8 512-bit ZMM (default on VNNI CPUs)
     *     TN_FORCE_BACKEND=vnni256   — VNNI-256 int8 256-bit YMM via AVX-512VNNI (no throttle)
     * Unrecognised values are silently ignored and auto-detect proceeds.
     */
    const char *force = getenv("TN_FORCE_BACKEND");
    int force_scalar  = force && strcmp(force, "scalar")  == 0;
    int force_avx2    = force && strcmp(force, "avx2")    == 0;
    int force_avx512f = force && strcmp(force, "avx512f") == 0;
    int force_vnni256 = force && strcmp(force, "vnni256") == 0;
    /* force_vnni is the default; listed for clarity */
    /* int force_vnni = force && strcmp(force, "vnni") == 0; */

    const TnCpuFeatures *cpu = tn_cpu_features_detect();

    /* ── Utility ops: best float SIMD available ── */
#if TN_HAS_AVX512
    if (!force_scalar && !force_avx2 && cpu->avx512f) {
        tn_rmsnorm   = rmsnorm_avx512;
        tn_softmax   = softmax_avx512;
        tn_vec_add   = vec_add_avx512;
        tn_vec_mul   = vec_mul_avx512;
        tn_vec_scale = vec_scale_avx512;
        tn_silu      = silu_avx512;
        tn_relu2     = relu2_avx512;
        tn_vec_dot   = vec_dot_avx512;
        tn_vec_saxpy = vec_saxpy_avx512;
    } else
#endif
#if TN_HAS_AVX2
    if (!force_scalar && cpu->avx2) {
        tn_rmsnorm   = rmsnorm_avx2;
        tn_softmax   = softmax_avx2;
        tn_vec_add   = vec_add_avx2;
        tn_vec_mul   = vec_mul_avx2;
        tn_vec_scale = vec_scale_avx2;
        tn_silu      = silu_avx2;
        tn_relu2     = relu2_avx2;
        tn_vec_dot   = vec_dot_avx2;
        tn_vec_saxpy = vec_saxpy_avx2;
    } else
#endif
    {
        tn_rmsnorm   = rmsnorm;
        tn_softmax   = softmax;
        tn_vec_add   = vec_add;
        tn_vec_mul   = vec_mul;
        tn_vec_scale = vec_scale;
        tn_silu      = silu;
        tn_relu2     = relu2_scalar;
        tn_vec_dot   = vec_dot;
        tn_vec_saxpy = vec_saxpy;
    }

    /* ── Unpack: AVX2 or scalar ── */
#if TN_HAS_AVX2
    if (!force_scalar && cpu->avx2) {
        tn_unpack_block    = unpack_ternary_block_avx2;
        tn_ternary_matmul  = ternary_matmul_avx2;
    } else
#endif
    {
        tn_unpack_block    = unpack_ternary_block;
        tn_ternary_matmul  = ternary_matmul;
    }

    /* ── Packed matmul: VNNI tier selection ──
     *
     * x86 path (priority: AVX-512 VNNI > AVX-VNNI > AVX-512F > AVX2):
     *   AVX-512 VNNI: 64 int8 MACs/cycle — biggest gain for ternary inference
     *   AVX-VNNI:     32 int8 MACs/cycle — Alder Lake / Zen3 (no AVX-512)
     *   AVX-512F:     16 fp32 MACs/cycle — current baseline
     *   AVX2:          8 fp32 MACs/cycle — Haswell / Zen2
     *
     * ARM path (priority: dotprod > NEON fallback):
     *   dotprod:      16 int8 MACs/cycle — signed×signed, no bias trick needed
     *   NEON:          4 fp32 MACs/cycle — stub, always compiled for ARM
     *
     * TN_FORCE_BACKEND skips upper tiers to reach a specific backend.
     */

#if TN_HAS_AVX512VNNI
    if (!force_scalar && !force_avx2 && !force_avx512f && !force_vnni256 && cpu->avx512vnni) {
        tn_ternary_matmul_packed = ternary_matmul_packed_vnni;
        g_backend_name = "AVX-512 VNNI";
        return g_backend_name;
    }
    /* vnni256: 256-bit EVEX VNNI — same CPU, no ZMM, no frequency throttle */
    if (!force_scalar && !force_avx2 && !force_avx512f && force_vnni256 && cpu->avx512vnni) {
        tn_ternary_matmul_packed = ternary_matmul_packed_vnni256;
        g_backend_name = "VNNI-256 (no throttle)";
        return g_backend_name;
    }
#endif

#if TN_HAS_AVXVNNI
    if (!force_scalar && !force_avx2 && !force_avx512f && !force_vnni256 && cpu->avx_vnni) {
        tn_ternary_matmul_packed = ternary_matmul_packed_avx_vnni;
        g_backend_name = "AVX-VNNI";
        return g_backend_name;
    }
#endif

#if TN_HAS_AVX512
    if (!force_scalar && !force_avx2 && cpu->avx512f) {
        tn_ternary_matmul_packed = ternary_matmul_packed_avx512;
        g_backend_name = "AVX-512F";
        return g_backend_name;
    }
#endif

#if TN_HAS_AVX2
    if (!force_scalar && cpu->avx2) {
        tn_ternary_matmul_packed = ternary_matmul_packed_avx2;
        g_backend_name = "AVX2";
        return g_backend_name;
    }
#endif

#if TN_HAS_ARM_DOTPROD
    if (cpu->arm_dotprod) {
        tn_ternary_matmul_packed = ternary_matmul_packed_dotprod;
        g_backend_name = "NEON+dotprod";
        return g_backend_name;
    }
#endif

    /* Scalar fallback — always compiled, works on any CPU */
    tn_ternary_matmul_packed = ternary_matmul_packed;
    g_backend_name = "Scalar";
    return g_backend_name;
}

const char *tn_simd_init(void) {
    int expected = 0;
    if (atomic_compare_exchange_strong(&g_simd_init_state, &expected, 1)) {
        tn_simd_init_impl();
        atomic_store_explicit(&g_simd_init_state, 2, memory_order_release);
    } else {
        while (atomic_load_explicit(&g_simd_init_state, memory_order_acquire) == 1) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            _mm_pause();
#endif
        }
    }
    return g_backend_name;
}
