#ifndef TN_HARDWARE_PROFILE_H
#define TN_HARDWARE_PROFILE_H

#include "core/platform.h"
#include "math/cpu_features.h"
#include <stdbool.h>

/*
 * Unified hardware profile — auto-tunes inference parameters at startup.
 *
 * Probes:
 *   1. CPU features (SIMD tier, from cpu_features.h)
 *   2. Physical/logical core count (from cpu_probe.c)
 *   3. L2/L3 cache sizes (from sysfs on Linux, CPUID on others)
 *   4. DRAM bandwidth (50ms sequential read microbenchmark)
 *   5. Available RAM
 *
 * Derives:
 *   - Optimal classifier format (INT4/INT8/BF16) based on SIMD tier
 *   - Optimal thread count
 *   - Prefetch distance based on cache sizes
 *   - Theoretical ceiling tok/s for this hardware
 *
 * Goal: one binary auto-adapts to laptop, Xeon server, or Android terminal
 * with no manual configuration.
 */

typedef enum {
    TN_CLS_BF16 = 0,   /* 2 bytes/weight, full precision, any hardware */
    TN_CLS_INT8 = 1,    /* 1 byte/weight, VNNI or ARM dotprod available */
    TN_CLS_INT4 = 2     /* 0.5 bytes/weight, VNNI+VBMI for fast unpack */
} TnClassifierFormat;

typedef struct {
    /* ── Detected hardware ── */
    const TnCpuFeatures *cpu;
    int    physical_cores;
    int    logical_cores;
    int    optimal_threads;
    size_t l2_cache_bytes;       /* per-core L2 */
    size_t l3_cache_bytes;       /* shared L3 (0 if not detected) */
    size_t free_ram_bytes;
    double measured_bw_gbps;     /* DRAM bandwidth in GB/s */

    /* ── Derived tuning parameters ── */
    TnClassifierFormat classifier_fmt;
    bool   classifier_explicit;  /* true only if the user passed --classifier
                                    (vs. classifier_fmt being an auto-picked
                                    default) — gates whether Q2_0-native models
                                    pay to materialize a separate classifier
                                    copy (see gguf_loader.c) instead of always
                                    doing so regardless of user intent. */
    int    prefetch_rows;        /* rows to prefetch ahead in matmul */
    bool   use_nt_stores;        /* non-temporal stores for large outputs */
    bool   model_fits_l3;        /* true if ternary weights fit in L3 */

    /* ── Performance projections ── */
    double weight_bytes_per_tok; /* total bytes read per token */
    double cls_bytes_per_tok;    /* classifier bytes per token */
    double theoretical_ceiling;  /* tok/s at 100% BW utilization */

    /* ── Backend name ── */
    char   summary[128];         /* human-readable "AVX-512 VNNI | INT4 | 4T | 45.8 GB/s" */
} TnHardwareProfile;

/*
 * Probe hardware and populate the profile.
 * Call once at startup, before weights_map.
 * The returned pointer is to a static — no allocation needed.
 */
const TnHardwareProfile *tn_hardware_profile_init(void);

/*
 * Print a formatted hardware profile report to stdout.
 */
void tn_hardware_profile_report(const TnHardwareProfile *hp);

/*
 * Get the globally initialized hardware profile (NULL if init not called).
 */
const TnHardwareProfile *tn_hardware_profile_get(void);

/*
 * Override the classifier format (user --classifier flag).
 * Must be called after tn_hardware_profile_init().
 * Recalculates weight_bytes_per_tok and theoretical_ceiling.
 */
void tn_hardware_profile_set_classifier(TnClassifierFormat fmt);

/*
 * Correct the per-token weight-traffic estimate with the REAL loaded model
 * (init runs pre-load and seeds it with compile-time BitNet-2B constants —
 * ~6x-overstated ceiling for large GGUF models; see docs/ai/mistakes.md
 * 2026-07-17). Recomputes theoretical_ceiling and the summary string.
 * Must be called after tn_hardware_profile_init(); no-op on
 * weight_bytes_per_tok <= 0.
 */
void tn_hardware_profile_set_model_bytes(double weight_bytes_per_tok,
                                          double cls_bytes_per_tok);

#endif /* TN_HARDWARE_PROFILE_H */
