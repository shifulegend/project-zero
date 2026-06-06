#ifndef TN_CALIBRATION_H
#define TN_CALIBRATION_H

#include "core/platform.h"
#include "core/hardware_profile.h"
#include <stdbool.h>

/*
 * Auto-calibration system — finds optimal engine settings per hardware.
 *
 * On first run (or --calibrate), runs micro-benchmarks to determine:
 *   1. Best SIMD backend for ternary matmul (scalar / AVX2 / AVX-512F / VNNI)
 *   2. Optimal thread count via full T=1..logical_cores sweep
 *   3. Classifier bandwidth estimates (BF16 / INT8 / INT4)
 *
 * Each benchmark uses:
 *   - Time-based thermal warmup (not iteration-based) to avoid turbo-burst
 *     false positives (Tiger Lake PL2 window is ~20-28 s; we run long enough
 *     that subsequent measurements are in steady-state frequency).
 *   - Per-rep individual timing → min / median / max reported so variance
 *     from background load is visible, not silently averaged away.
 *   - Background CPU load sampled via /proc/stat BEFORE each test so noisy
 *     measurements are flagged [!] in the output.
 *   - parallel_ternary_matmul_packed (not the single-threaded dispatch) so
 *     the thread pool is actually exercised for the thread-count sweep.
 *   - Live progress dots printed to stdout so the terminal doesn't look stuck.
 *
 * Results are cached to ~/.project-zero/calibration.bin with a hardware
 * fingerprint (CPU model string + core count + cache sizes). If the
 * fingerprint changes, calibration re-runs automatically.
 *
 * Design principle: the engine defaults to BF16 (full intelligence).
 * Calibration RECOMMENDS faster settings but never applies them silently.
 * Users opt in via --classifier auto-fast or --classifier int8.
 */

#define TN_CALIB_MAGIC    0x505A4341  /* "PZCA" */
#define TN_CALIB_VERSION  2           /* bumped: added thread sweep arrays */

#define TN_CALIB_MAX_THREADS 16       /* max thread counts tracked in sweep */

typedef struct {
    /* Hardware fingerprint (for cache invalidation) */
    char cpu_model[128];       /* /proc/cpuinfo model name */
    int  physical_cores;
    int  logical_cores;
    size_t l2_cache_bytes;
    size_t l3_cache_bytes;

    /* SIMD backend comparison: median tok/s per backend (0 = not tested) */
    double simd_tokps[4];      /* [0]=scalar, [1]=avx2, [2]=avx512f, [3]=vnni */
    int    best_simd_idx;      /* index into simd_tokps[] */
    char   best_simd_name[32]; /* e.g. "avx2", "vnni" */

    /* Classifier bandwidth estimates (tok/s) */
    double cls_tokps[3];       /* [0]=bf16, [1]=int8, [2]=int4 */
    int    best_cls_idx;       /* index: fastest classifier */

    /* Thread count sweep: per-T min/median/max tok/s and background load */
    int    thread_sweep_n;                       /* number of T values tested */
    double thread_tokps_min[TN_CALIB_MAX_THREADS];
    double thread_tokps_med[TN_CALIB_MAX_THREADS];
    double thread_tokps_max[TN_CALIB_MAX_THREADS];
    int    thread_sysload_pct[TN_CALIB_MAX_THREADS]; /* -1 = unavailable */

    /* Best thread count (from sweep median, not heuristic) */
    int    best_threads;
    double best_thread_tokps;

    /* Metadata */
    int    valid;              /* 1 if calibration completed successfully */
} TnCalibrationResult;

/*
 * Run calibration micro-benchmarks.
 * Takes ~40-60 seconds depending on hardware and thread count.
 * Does NOT change any global engine state — only populates `result`.
 * Prints live progress to stdout throughout.
 */
void tn_calibrate(TnCalibrationResult *result, const TnHardwareProfile *hw);

/*
 * Try to load cached calibration from ~/.project-zero/calibration.bin.
 * Returns true if loaded AND hardware fingerprint matches current system.
 * Returns false if no cache, stale cache, or fingerprint mismatch.
 */
bool tn_calibration_load(TnCalibrationResult *result, const TnHardwareProfile *hw);

/*
 * Save calibration results to ~/.project-zero/calibration.bin.
 */
void tn_calibration_save(const TnCalibrationResult *result);

/*
 * Print calibration results as a formatted table.
 */
void tn_calibration_report(const TnCalibrationResult *result);

/*
 * Get the SIMD backend name to apply from calibration.
 * Returns NULL if calibration is invalid.
 * Caller should pass the returned name to TN_FORCE_BACKEND or tn_simd_init.
 */
const char *tn_calibration_best_simd(const TnCalibrationResult *result);

/*
 * Get the recommended classifier for --classifier auto-fast.
 * Returns the fastest classifier format from calibration.
 */
TnClassifierFormat tn_calibration_best_classifier(const TnCalibrationResult *result);

#endif /* TN_CALIBRATION_H */
