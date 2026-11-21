#include "math/simd_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/cpu_features.h"
#include "core/unpack.h"
#include "math/elementwise.h"
#include "math/quantize_i8.h"
#include "math/rmsnorm.h"
#include "math/softmax.h"
#include "math/ternary_matmul.h"
#include "math/ternary_matmul_packed.h"

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
#undef TN_HAS_AVX2
#define TN_HAS_AVX2 1
#undef TN_HAS_AVX512
#define TN_HAS_AVX512 1
#undef TN_HAS_AVX512VNNI
#define TN_HAS_AVX512VNNI 1
#undef TN_HAS_AVX512VBMI
#define TN_HAS_AVX512VBMI 1
#undef TN_HAS_AVXVNNI
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
extern void ternary_matmul_packed_vnni(float *out, const float *x, const tn_u8 *packed_w, int n,
                                       int d, const float *scales, int group_size);
#endif

/* ── AVX-VNNI 256-bit forward declarations ───────────────────────────────── */
#if TN_HAS_AVXVNNI
extern void ternary_matmul_packed_avx_vnni(float *out, const float *x, const tn_u8 *packed_w, int n,
                                           int d, const float *scales, int group_size);
#endif

/* ── VNNI-256: EVEX-256 via AVX-512VNNI (Tiger Lake — no ZMM throttle) ───── */
#if TN_HAS_AVX512VNNI
extern void ternary_matmul_packed_vnni256(float *out, const float *x, const tn_u8 *packed_w, int n,
                                          int d, const float *scales, int group_size);
#endif

/* ── AVX-512F forward declarations ───────────────────────────────────────── */
#if TN_HAS_AVX512
extern void ternary_matmul_packed_avx512(float *out, const float *x, const tn_u8 *packed_w, int n,
                                         int d, const float *scales, int group_size);
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
extern void ternary_matmul_avx2(float *out, const float *x, const tn_i8 *w, int n, int d,
                                float scale);
extern void ternary_matmul_packed_avx2(float *out, const float *x, const tn_u8 *packed_w, int n,
                                       int d, const float *scales, int group_size);
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
extern void ternary_matmul_packed_dotprod(float *out, const float *x, const tn_u8 *packed_w, int n,
                                          int d, const float *scales, int group_size);
#endif

/* ── ARM plain NEON forward declaration ──────────────────────────────────── */
#if TN_HAS_NEON
extern void ternary_matmul_packed_neon(float *out, const float *x, const tn_u8 *packed_w, int n,
                                       int d, const float *scales, int group_size);
#endif

/* ── Global dispatch table ─────────────────────────────────────────────────
 * Every pointer starts on its portable scalar implementation, so a caller that forgets
 * tn_simd_init() (a test, an embedder) gets slow but correct math instead of a NULL call. The
 * exceptions are the three int8-activation pointers at the end: they are bound only for the
 * AVX-512 VNNI tier and gated by tn_simd_ternary_preq(), so NULL means "not available".
 * Call tn_simd_init() once at startup to select the fastest tier.
 */
tn_matmul_fn tn_ternary_matmul               = ternary_matmul_branchless;
tn_matmul_packed_fn tn_ternary_matmul_packed = ternary_matmul_packed;
tn_unpack_fn tn_unpack_block                 = unpack_ternary_block;
tn_rmsnorm_fn tn_rmsnorm                     = rmsnorm;
tn_softmax_fn tn_softmax                     = softmax;
tn_vec_add_fn tn_vec_add                     = vec_add;
tn_vec_mul_fn tn_vec_mul                     = vec_mul;
tn_vec_scale_fn tn_vec_scale                 = vec_scale;
tn_silu_fn tn_silu                           = silu;
tn_relu2_fn tn_relu2                         = relu2_scalar;
tn_vec_dot_fn tn_vec_dot                     = vec_dot;
tn_kvq4_dot_fn tn_kvq4_dot                   = kvq4_dot_scalar;
tn_kvq4_axpy_fn tn_kvq4_axpy                 = kvq4_axpy_scalar;
tn_vec_saxpy_fn tn_vec_saxpy                 = vec_saxpy;
tn_cls_rows_fn tn_cls_i8_rows                = tn_cls_i8_rows_generic;
tn_cls_rows_fn tn_cls_i4_rows                = tn_cls_i4_rows_generic;
tn_quantize_i8_fn tn_quantize_act_i8         = NULL;
tn_sum_i8_fn tn_sum_act_i8                   = NULL;
tn_matmul_preq_fn tn_ternary_matmul_preq     = NULL;

/* Human-readable name of the selected backend (set by tn_simd_init) */
static const char *g_backend_name = "Scalar";

const char *tn_simd_backend_name(void) {
    return g_backend_name;
}
int tn_simd_ternary_preq(void) {
    return tn_ternary_matmul_preq != NULL;
}

static TnActQuant g_act_quant = TN_ACT_AUTO;
static int g_ternary_int8     = 0;  /* the bound ternary kernel quantises its activations to int8 */
static char g_last_force[16]  = ""; /* TN_FORCE_BACKEND of the last tn_simd_init ("" = none) */

int tn_act_quant_parse(const char *name, TnActQuant *out) {
    if (!name || !out) return -1;
    if (strcmp(name, "auto") == 0) {
        *out = TN_ACT_AUTO;
        return 0;
    }
    if (strcmp(name, "int8") == 0) {
        *out = TN_ACT_INT8;
        return 0;
    }
    if (strcmp(name, "float") == 0) {
        *out = TN_ACT_FLOAT;
        return 0;
    }
    return -1;
}
const char *tn_act_quant_name(TnActQuant m) {
    return m == TN_ACT_INT8 ? "int8" : m == TN_ACT_FLOAT ? "float" : "auto";
}
TnActQuant tn_act_quant_get(void) {
    return g_act_quant;
}
const char *tn_act_quant_effective(void) {
    return g_ternary_int8 ? "int8" : "float";
}
void tn_act_quant_set(TnActQuant m) {
    if (m == g_act_quant) return;
    g_act_quant = m;
    /* Re-bind on the same tier: a calibrated or --simd tier is applied through TN_FORCE_BACKEND
     * for the duration of one tn_simd_init() and is not in the environment afterwards. */
    const char *cur = getenv("TN_FORCE_BACKEND");
    int tmp         = (!cur && g_last_force[0]);
    if (tmp) setenv("TN_FORCE_BACKEND", g_last_force, 1);
    (void)tn_simd_init();
    if (tmp) unsetenv("TN_FORCE_BACKEND");
}

/* What the tier choice depends on: the CPU, the TN_FORCE_BACKEND cap and the act-quant mode. */
typedef struct {
    const TnCpuFeatures *cpu;
    int force_scalar, force_avx2, force_avx512f, force_vnni256;
    int float_act; /* act-quant float: skip the tiers whose ternary kernel quantises to int8 */
} SimdSel;

/* Each bind_* sets its group's function pointers; the ternary ones return the backend name, or NULL
 * when none of their tiers is available (built or runnable). */
static void bind_vector_ops(const SimdSel *sel) {
    const TnCpuFeatures *cpu = sel->cpu;
    int force_scalar         = sel->force_scalar;
    int force_avx2           = sel->force_avx2;
    (void)cpu;
    (void)force_scalar;
    (void)force_avx2;
    /* ── Utility ops: best float SIMD available ── */
#if TN_HAS_AVX512
    if (!force_scalar && !force_avx2 && cpu->avx512f) {
        tn_rmsnorm     = rmsnorm_avx512;
        tn_softmax     = softmax_avx512;
        tn_vec_add     = vec_add_avx512;
        tn_vec_mul     = vec_mul_avx512;
        tn_vec_scale   = vec_scale_avx512;
        tn_silu        = silu_avx512;
        tn_relu2       = relu2_avx512;
        tn_vec_dot     = vec_dot_avx512;
        tn_kvq4_dot    = kvq4_dot_avx512;
        tn_kvq4_axpy   = kvq4_axpy_avx512;
        tn_vec_saxpy   = vec_saxpy_avx512;
        tn_cls_i8_rows = tn_cls_i8_rows_avx512;
        tn_cls_i4_rows = tn_cls_i4_rows_avx512;
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
        tn_kvq4_dot  = kvq4_dot_avx2;
        tn_kvq4_axpy = kvq4_axpy_avx2;
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
        tn_kvq4_dot  = kvq4_dot_scalar;
        tn_kvq4_axpy = kvq4_axpy_scalar;
        tn_vec_saxpy = vec_saxpy;
    }
}

static void bind_unpack(const SimdSel *sel) {
    const TnCpuFeatures *cpu = sel->cpu;
    int force_scalar         = sel->force_scalar;
    (void)cpu;
    (void)force_scalar;
    /* ── Unpack: AVX2 or scalar ── */
#if TN_HAS_AVX2
    if (!force_scalar && cpu->avx2) {
        tn_unpack_block   = unpack_ternary_block_avx2;
        tn_ternary_matmul = ternary_matmul_avx2;
    } else
#endif
    {
        tn_unpack_block = unpack_ternary_block;
        /* Phase 3.2: branchless scalar (10-12x faster than the branchy
         * reference on random ternary rows; 2026-10-18 microbench, 2048x2048). */
        tn_ternary_matmul = ternary_matmul_branchless;
    }
}

/* The int8-activation tiers run unless act-quant is float or TN_FORCE_BACKEND caps the tier lower
 * (scalar, avx2, avx512f); vnni256 selects one of them. */
static int int8_tiers_allowed(const SimdSel *sel) {
    return !sel->float_act && !sel->force_scalar && !sel->force_avx2 && !sel->force_avx512f;
}

#if TN_HAS_AVX512VNNI
static const char *bind_vnni_512(const SimdSel *sel) {
    if (!int8_tiers_allowed(sel) || sel->force_vnni256 || !sel->cpu->avx512vnni) return NULL;
    g_ternary_int8           = 1;
    tn_ternary_matmul_packed = ternary_matmul_packed_vnni;
    tn_quantize_act_i8       = quantize_row_to_i8_avx512;
    tn_sum_act_i8            = sum_i8_avx512;
    tn_ternary_matmul_preq   = ternary_matmul_packed_vnni_preq;
#if TN_HAS_AVX512VBMI
    /* The same kernels built with VBMI (3-instruction unpack). cpu->avx512vbmi is
     * execution-verified under a SIGILL trap (cpu_features.c), so this is safe on a
     * hypervisor that advertises VBMI it cannot retire. */
    if (sel->cpu->avx512vbmi) {
        tn_ternary_matmul_packed = ternary_matmul_packed_vnni_vbmi;
        tn_ternary_matmul_preq   = ternary_matmul_packed_vnni_preq_vbmi;
    }
#endif
    return "AVX-512 VNNI";
}

/* vnni256: 256-bit EVEX VNNI — same CPU, no ZMM, no frequency throttle */
static const char *bind_vnni_256(const SimdSel *sel) {
    if (!int8_tiers_allowed(sel) || !sel->force_vnni256 || !sel->cpu->avx512vnni) return NULL;
    g_ternary_int8           = 1;
    tn_ternary_matmul_packed = ternary_matmul_packed_vnni256;
    return "VNNI-256 (no throttle)";
}
#endif

#if TN_HAS_AVXVNNI
static const char *bind_avx_vnni(const SimdSel *sel) {
    if (!int8_tiers_allowed(sel) || sel->force_vnni256 || !sel->cpu->avx_vnni) return NULL;
    g_ternary_int8           = 1;
    tn_ternary_matmul_packed = ternary_matmul_packed_avx_vnni;
    return "AVX-VNNI";
}
#endif

/* x86 tiers that quantise activations to int8: AVX-512 VNNI, VNNI-256, AVX-VNNI. */
static const char *bind_packed_int8_x86(const SimdSel *sel) {
    (void)sel;
    const char *name = NULL;
#if TN_HAS_AVX512VNNI
    if (!name) name = bind_vnni_512(sel);
    if (!name) name = bind_vnni_256(sel);
#endif
#if TN_HAS_AVXVNNI
    if (!name) name = bind_avx_vnni(sel);
#endif
    return name;
}

/* x86 float tiers: AVX-512F, AVX2. */
static const char *bind_packed_float_x86(const SimdSel *sel) {
    (void)sel;
#if TN_HAS_AVX512
    if (!sel->force_scalar && !sel->force_avx2 && sel->cpu->avx512f) {
        tn_ternary_matmul_packed = ternary_matmul_packed_avx512;
        return "AVX-512F";
    }
#endif

#if TN_HAS_AVX2
    if (!sel->force_scalar && sel->cpu->avx2) {
        tn_ternary_matmul_packed = ternary_matmul_packed_avx2;
        return "AVX2";
    }
#endif
    return NULL;
}

/* ARM tiers: dotprod, then NEON. */
static const char *bind_packed_arm(const SimdSel *sel) {
    (void)sel;
#if TN_HAS_ARM_DOTPROD
    if (!sel->float_act && sel->cpu->arm_dotprod) {
        g_ternary_int8           = 1;
        tn_ternary_matmul_packed = ternary_matmul_packed_dotprod;
        return "NEON+dotprod";
    }
#endif

#if TN_HAS_NEON
    if (!sel->force_scalar && sel->cpu->neon) {
        tn_ternary_matmul_packed = ternary_matmul_packed_neon;
        return "NEON";
    }
#endif
    return NULL;
}

const char *tn_simd_init(void) {
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
     *
     * Packed matmul tier priority: AVX-512 VNNI (64 int8 MACs/cycle) > AVX-VNNI (32, Alder Lake /
     * Zen3) > AVX-512F (16 fp32) > AVX2 (8 fp32) on x86; dotprod (16 int8, signed x signed, no bias
     * trick) > NEON (4 fp32, always on ARMv8-A) on ARM; scalar otherwise.
     */
    /* Re-derived on every call: only the AVX-512 VNNI tier binds these. */
    tn_quantize_act_i8     = NULL;
    tn_sum_act_i8          = NULL;
    tn_ternary_matmul_preq = NULL;
    tn_cls_i8_rows         = tn_cls_i8_rows_generic;
    tn_cls_i4_rows         = tn_cls_i4_rows_generic;
    const char *force      = getenv("TN_FORCE_BACKEND");
    snprintf(g_last_force, sizeof(g_last_force), "%s", force ? force : "");
    SimdSel sel;
    sel.cpu           = tn_cpu_features_detect();
    sel.force_scalar  = force && strcmp(force, "scalar") == 0;
    sel.force_avx2    = force && strcmp(force, "avx2") == 0;
    sel.force_avx512f = force && strcmp(force, "avx512f") == 0;
    sel.force_vnni256 = force && strcmp(force, "vnni256") == 0;
    sel.float_act     = (g_act_quant == TN_ACT_FLOAT);
    g_ternary_int8    = 0;

    bind_vector_ops(&sel);
    bind_unpack(&sel);
    const char *name = bind_packed_int8_x86(&sel);
    if (!name) name = bind_packed_float_x86(&sel);
    if (!name) name = bind_packed_arm(&sel);
    if (!name) {
        /* Scalar fallback — always compiled, works on any CPU */
        tn_ternary_matmul_packed = ternary_matmul_packed;
        name                     = "Scalar";
    }
    g_backend_name = name;
    return g_backend_name;
}
