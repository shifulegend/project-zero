#ifndef TN_CALIBRATION_H
#define TN_CALIBRATION_H

#include "core/platform.h"
#include "runtime/hardware_profile.h"
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

#define TN_CALIB_MAGIC 0x505A4341 /* "PZCA" */
#define TN_CALIB_VERSION                                                                           \
    4 /* 4: interleaved paired rounds + sign test, ISA mask in the key (2026-11-18); 3: "auto" candidate + margin */

#define TN_CALIB_MAX_THREADS 16 /* max thread counts tracked in sweep */

/* SIMD candidates: "auto" (the dispatcher's CPUID pick, e.g. AVX-VNNI, which no
 * forced value selects) + every forceable tier. A forced tier replaces "auto"
 * only when clearly faster: median > auto median * TN_CALIB_OVERRIDE_MARGIN and
 * its slowest rep beats auto's median — otherwise run-to-run noise decides. */
#define TN_CALIB_SIMD_N 6
#define TN_CALIB_OVERRIDE_MARGIN 1.05

/* 2026-11-18: candidates are measured in interleaved rounds (every candidate once
 * per round, rotating the order), so thermal drift and background load hit all of
 * them alike. A forced tier replaces "auto" only if it ALSO beats auto's paired
 * sample in enough rounds for a one-sided sign test at p <= 0.005 (13 of 15): a wrong override costs speed, a missed small win costs almost nothing, so the default wins ties and near-ties. The
 * pre-2026-11-18 sweep ran each candidate in its own block and picked "avx2" on an
 * AVX-512 VNNI host from noise (docs/ai/mistakes.md). */
#define TN_CALIB_ROUNDS 15
#define TN_CALIB_MAX_ROUNDS 31

typedef struct {
    /* Hardware fingerprint (for cache invalidation) */
    char cpu_model[128]; /* /proc/cpuinfo model name */
    int physical_cores;
    int logical_cores;
    size_t l2_cache_bytes;
    size_t l3_cache_bytes;
    unsigned isa_mask; /* tn_calibration_isa_mask() of the build that measured */

    /* SIMD backend comparison: median tok/s per backend (0 = not tested) */
    double simd_tokps[TN_CALIB_SIMD_N]; /* see tn_calibration_simd_names(); [0] = "auto" */
    int simd_wins[TN_CALIB_SIMD_N];     /* paired rounds each candidate beat "auto" in */
    int simd_rounds;                    /* rounds measured (0 = sweep not run) */
    int simd_sweep_noisy;    /* 1 = high background load: "auto" kept, nothing overridden */
    int best_simd_idx;       /* index into simd_tokps[] */
    char best_simd_name[32]; /* "auto" = keep the dispatcher's own choice */
    char auto_backend[32];   /* what "auto" resolved to, e.g. "AVX-VNNI" */

    /* Classifier bandwidth estimates (tok/s) */
    double cls_tokps[3]; /* [0]=bf16, [1]=int8, [2]=int4 */
    int best_cls_idx;    /* index: fastest classifier */

    /* Thread count sweep: per-T min/median/max tok/s and background load */
    int thread_sweep_n; /* number of T values tested */
    double thread_tokps_min[TN_CALIB_MAX_THREADS];
    double thread_tokps_med[TN_CALIB_MAX_THREADS];
    double thread_tokps_max[TN_CALIB_MAX_THREADS];
    int thread_sysload_pct[TN_CALIB_MAX_THREADS]; /* -1 = unavailable */

    /* Best thread count (from sweep median, not heuristic) */
    int best_threads;
    double best_thread_tokps;

    /* Metadata */
    int valid; /* 1 if calibration completed successfully */
} TnCalibrationResult;

/*
 * Bit mask of the SIMD tiers this build can run (AVX2, AVX-512, AVX-512 VNNI,
 * AVX-VNNI, NEON, dotprod). Part of the cache key: a result measured by a build
 * with other tiers compiled in is never reused.
 */
unsigned tn_calibration_isa_mask(void);

/*
 * Smallest number of paired-round wins (out of n_rounds) that is significant in
 * a one-sided sign test at p <= 0.005; n_rounds + 1 (unreachable) when n_rounds
 * is too small for any result to be significant.
 */
int tn_calib_wins_needed(int n_rounds);

/*
 * Decide which SIMD candidate replaces "auto" (index 0). samples[c][r] is the
 * tok/s of candidate c in round r (same round = measured back to back).
 * Returns 0 (keep auto) unless a candidate has the best median AND a median above
 * auto's median * margin AND beats auto's paired sample in at least
 * tn_calib_wins_needed(n_rounds) rounds. Pure function: no I/O, no globals.
 */
int tn_calib_pick_simd(const double samples[][TN_CALIB_MAX_ROUNDS], const bool avail[], int n_cand,
                       int n_rounds, double margin);

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

/* Activation numerics the calibrated tier runs, for --act-quant auto: "int8" when the tier quantises
 * activations (vnni, vnni256), "float" for the float tiers (scalar, avx2, avx512f), NULL when the
 * result is invalid or keeps the dispatcher's own choice ("auto"). The decision is the measured one:
 * the sweep times the int8 tier against the float tiers and a float tier wins only on a clear margin. */
const char *tn_calibration_act_quant(const TnCalibrationResult *result);

/* The measured evidence behind that choice: the best median tok/s among the int8 tiers (vnni, vnni256 and,
 * when the dispatcher's own choice "auto" quantises activations, auto) and among the float tiers (scalar,
 * avx2, avx512f and, otherwise, auto) of the sweep. Returns 1 and fills both when the
 * result is valid and both groups were measured, else 0. */
int tn_calibration_act_timing(const TnCalibrationResult *result, int auto_is_int8,
                              double *int8_tokps, double *float_tokps);

/*
 * Get the recommended classifier for --classifier auto-fast.
 * Returns the fastest classifier format from calibration.
 */
TnClassifierFormat tn_calibration_best_classifier(const TnCalibrationResult *result);

#endif /* TN_CALIBRATION_H */
