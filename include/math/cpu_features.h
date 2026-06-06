#ifndef TN_CPU_FEATURES_H
#define TN_CPU_FEATURES_H

#include "core/platform.h"
#include <stdbool.h>

/*
 * Runtime CPU feature detection (Phase 16-S.1)
 *
 * Complements the compile-time TN_HAS_* macros in platform.h.
 * Use this at startup to select the correct SIMD kernel for the CPU
 * that is actually RUNNING the binary (which may differ from the CPU
 * that compiled it when cross-compiling or using portable builds).
 *
 * Detection methods:
 *   x86-64 Linux/macOS/Windows : CPUID instruction via GCC __cpuid_count()
 *   ARM Linux                  : getauxval(AT_HWCAP / AT_HWCAP2)
 *   ARM macOS (Apple Silicon)  : sysctlbyname("hw.optional.arm.FEAT_*")
 */

typedef struct {
    /* x86-64 features */
    bool avx2;           /* AVX2 + FMA3 — Haswell 2013+ */
    bool avx512f;        /* AVX-512 Foundation — Ice Lake 2019+, Skylake-X 2017+ */
    bool avx512vnni;     /* AVX-512 VNNI int8 — Ice Lake 2019+, Zen4 2022+ */
    bool avx_vnni;       /* AVX-VNNI 256-bit int8 — Alder Lake 2021+, Zen3 2020+ */
    bool avx512bf16;     /* AVX-512 BF16 — Sapphire Rapids 2023+, Zen4 2022+ */
    bool avx512fp16;     /* AVX-512 FP16 — Sapphire Rapids 2023+, Meteor Lake 2023+ */
    bool avx512vbmi;     /* AVX-512 VBMI (byte permute) — Ice Lake 2019+ */

    /* ARM features */
    bool neon;           /* NEON baseline — all ARMv8-A */
    bool arm_dotprod;    /* SDOT/UDOT int8 — Cortex-A75+ 2017+, Apple M1+ */
    bool arm_sve2;       /* SVE2 scalable vectors — Cortex-X4+, Apple M4+ */
    bool arm_bf16;       /* ARM BF16 — Cortex-A78+, Apple M2+ */

    /* Apple Silicon specific */
    bool apple_silicon;  /* True if running on Apple M-series */

    /* Populated name string for logging */
    char best_backend[32];
} TnCpuFeatures;

/*
 * Probe CPU capabilities at runtime.
 * Safe to call multiple times — detection results are cached.
 * Returns pointer to a static TnCpuFeatures (no allocation).
 */
const TnCpuFeatures *tn_cpu_features_detect(void);

/*
 * Print a formatted table of detected CPU features to stdout.
 * Useful for boot diagnostics and --verbose mode.
 */
void tn_cpu_features_report(const TnCpuFeatures *f);

/*
 * Return a short human-readable string for the best available SIMD tier.
 * E.g. "AVX-512 VNNI", "AVX-VNNI", "AVX-512F", "AVX2", "NEON+dotprod", "NEON", "Scalar"
 */
const char *tn_cpu_best_backend_name(const TnCpuFeatures *f);

#endif /* TN_CPU_FEATURES_H */
