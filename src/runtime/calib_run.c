/* calib_run.c — the calibration run: SIMD backends compared in interleaved rounds, then a thread
 * count sweep, then the analytic classifier estimates. Measurements come from calib_bench.c; the
 * decisions that need no hardware (which count wins, the classifier estimates) are plain
 * functions in this file so tests can check them. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/cpu_features.h"
#include "core/log.h"
#include "core/sys_probe.h"
#include "math/simd_dispatch.h"
#include "runtime/calib_internal.h"
#include "runtime/calibration.h"
#include "threading/thread_pool.h"

/* ── Pure decisions ───────────────────────────────────────────────────── */

int tn_calib_prefer_physical(const double med[], const double min_[], int best_t, double best_tokps,
                             int phys_t, double *out_tokps) {
    *out_tokps = best_tokps;
    if (best_t == phys_t || !(med[phys_t - 1] > 0.0)) return best_t;
    double pm  = med[phys_t - 1];
    bool clear = best_tokps > pm * TN_CALIB_OVERRIDE_MARGIN && min_[best_t - 1] > pm;
    if (clear) return best_t;
    *out_tokps = pm;
    return phys_t;
}

/* ANALYTIC pre-load estimates, not kernel measurements: calibration runs before any model file is
 * opened, so the byte figures below are the BitNet-2B compile-time geometry (522 MB ternary,
 * 128256x2560 classifier) -- the same pre-load limitation as hardware_profile.c's startup box, and
 * wrong in absolute terms for any other model (2026-07-17, see
 * docs/architecture/CEILING_CALCULATION.md §4 site 4). The *ranking* between formats (what
 * --classifier auto-fast consumes via best_cls_idx) only depends on the relative classifier byte
 * sizes, so it survives the wrong absolute base better than the tok/s figures do. TODO(PZ-TODO-2):
 * make this model-aware (needs a calibration redesign -- the cache is keyed on hardware, not
 * model). */
void tn_calib_classifier_estimates(double base_bw, double l3_bytes, double cls_tokps[3],
                                   int *best_idx) {
    double ternary_mb  = 522.0;
    double bf16_cls_mb = (double)(128256ULL * 2560 * 2) / (1024.0 * 1024.0);
    double int8_cls_mb = bf16_cls_mb / 2.0;
    double int4_cls_mb = bf16_cls_mb / 4.0;
    double l3_mb       = l3_bytes / (1024.0 * 1024.0);
    double cached      = (l3_mb >= ternary_mb) ? ternary_mb : l3_mb * 0.5;
    double dram_mb     = ternary_mb - cached;
    /* base_bw is decimal GB/s (bytes/ns from the probe); the *_mb sizes are MiB, so the conversion
     * is 1e9/2^20 ≈ 953.67 MiB/s per GB/s -- NOT 1024 (2026-07-17, independent-review finding: the
     * old *1024 inflated every printed cls_tokps ~7.4%; rankings were unaffected since the error
     * was a common factor). */
    double bw_mbps = base_bw * (1e9 / (1024.0 * 1024.0));

    cls_tokps[0] = bw_mbps / (dram_mb + bf16_cls_mb);
    cls_tokps[1] = bw_mbps / (dram_mb + int8_cls_mb);
    cls_tokps[2] = bw_mbps / (dram_mb + int4_cls_mb);

    *best_idx = 0;
    for (int i = 1; i < 3; i++)
        if (cls_tokps[i] > cls_tokps[*best_idx]) *best_idx = i;
}

/* ── Printing ─────────────────────────────────────────────────────────── */

static void print_header(int phys) {
    TN_OUT_INF("calibration",
               "┌──────────────────────────────────────────────────────────────────┐\n");
    TN_OUT_INF("calibration",
               "│  Auto-Calibration  —  first run, result cached for future use   │\n");
    TN_OUT_INF("calibration",
               "│  Re-triggers only when hardware changes.  ~40-60 seconds.       │\n");
    TN_OUT_INF("calibration",
               "├──────────────────────────────────────────────────────────────────┤\n");
    fflush(stdout);
    TN_OUT_INF("calibration",
               "│  Phase 1/2  SIMD backends at T=%d  (thermal stabilisation)       │\n", phys);
    TN_OUT_INF("calibration",
               "├──────────────────────────────────────────────────────────────────┤\n");
    fflush(stdout);
}

static void print_phase2_header(const char *backend) {
    TN_OUT_INF("calibration",
               "├──────────────────────────────────────────────────────────────────┤\n");
    TN_OUT_INF("calibration", "│  Phase 2/2  Thread sweep  (backend: %-10s)                │\n",
               backend);
    TN_OUT_INF("calibration",
               "├──────────────────────────────────────────────────────────────────┤\n");
    fflush(stdout);
}

/* The "Speed opt" line: the actually-computed best format + gain (2026-07-17; it used to be the
 * hardcoded literal "(INT8, ~+36%)"). "est." because these are analytic pre-load estimates. */
static void speed_opt_line(const TnCalibrationResult *result, char *line, size_t cap) {
    if (!(result->cls_tokps[0] > 0.0)) {
        snprintf(line, cap, "Speed opt : --classifier auto-fast");
        return;
    }
    static const char *cls_names[3] = {"BF16", "INT8", "INT4"};
    int best                        = result->best_cls_idx;
    if (best < 0 || best > 2) best = 0;
    double gain_pct = (result->cls_tokps[best] / result->cls_tokps[0] - 1.0) * 100.0;
    snprintf(line, cap, "Speed opt : --classifier auto-fast  (%s, ~%+.0f%% est.)", cls_names[best],
             gain_pct);
}

static void print_result(const TnCalibrationResult *result, int phys, int max_t) {
    TN_OUT_INF("calibration",
               "├──────────────────────────────────────────────────────────────────┤\n");
    TN_OUT_INF("calibration",
               "│  Result                                                          │\n");
    TN_OUT_INF("calibration", "│    SIMD      : %-10s (%5.1f tok/s at T=%d)                 │\n",
               result->best_simd_name, result->simd_tokps[result->best_simd_idx], phys);
    TN_OUT_INF("calibration", "│    Threads   : T=%-2d  (%5.1f tok/s, swept T=1..%d)           │\n",
               result->best_threads, result->best_thread_tokps, max_t);
    TN_OUT_INF("calibration",
               "│    Classifier: BF16 (default — full intelligence)               │\n");
    char opt_line[80];
    speed_opt_line(result, opt_line, sizeof(opt_line));
    TN_OUT_INF("calibration", "│    %-62s│\n", opt_line);
    TN_OUT_INF("calibration",
               "└──────────────────────────────────────────────────────────────────┘\n");
    fflush(stdout);
}

/* ── Phase 1: SIMD backends ───────────────────────────────────────────── */

/* Forces backend `c` ("auto" = no force) and re-binds the kernels. */
static void bind_backend(int c) {
    if (c == 0)
        unsetenv("TN_FORCE_BACKEND");
    else
        setenv("TN_FORCE_BACKEND", TN_CALIB_BACKENDS[c], 1);
    tn_simd_init();
}

/* ── Which candidates really differ from "auto"? ─────────────────────────
 * A forced value that lands on the same kernel as "auto" adds nothing. */
static void drop_duplicates_of_auto(TnCalibrationResult *result, bool avail[TN_CALIB_SIMD_N]) {
    for (int i = 0; i < TN_CALIB_SIMD_N; i++) {
        if (!avail[i]) continue;
        if (i == 0)
            unsetenv("TN_FORCE_BACKEND");
        else
            setenv("TN_FORCE_BACKEND", TN_CALIB_BACKENDS[i], 1);
        const char *picked = tn_simd_init();
        if (i == 0)
            snprintf(result->auto_backend, sizeof(result->auto_backend), "%s", picked);
        else if (strcmp(picked, result->auto_backend) == 0)
            avail[i] = false;
    }
    unsetenv("TN_FORCE_BACKEND");
    tn_simd_init();
}

/* ── Interleaved rounds ──────────────────────────────────────────────
 * Every candidate once per round, the starting candidate rotating, so
 * thermal drift and background load hit all of them alike (2026-11-18). */
static void run_simd_rounds(ThreadPool *tp, BenchBuf *bb, const int *idx, int nc, int rounds,
                            double samples[][TN_CALIB_MAX_ROUNDS]) {
    printf("  warmup[");
    fflush(stdout);
    calib_bench_warmup(tp, bb, 4.0); /* "auto" is bound: the CPU starts cold */
    printf("]  rounds[");
    fflush(stdout);
    for (int r = 0; r < rounds; r++) {
        for (int k = 0; k < nc; k++) {
            int c = idx[(k + r) % nc];
            bind_backend(c);
            samples[c][r] = calib_bench_one_rep(tp, bb);
        }
        putchar('.');
        fflush(stdout);
    }
    TN_OUT_INF("calibration", "]\n");
    unsetenv("TN_FORCE_BACKEND");
    tn_simd_init();
}

/* Medians, wins over "auto" and one printed line per candidate. */
static void report_simd(TnCalibrationResult *result, double samples[][TN_CALIB_MAX_ROUNDS],
                        const int *idx, int nc, int rounds, int best_simd_idx) {
    for (int k = 0; k < nc; k++) {
        int c     = idx[k];
        double lo = samples[c][0], hi = samples[c][0];
        int wins = 0;
        for (int r = 0; r < rounds; r++) {
            if (samples[c][r] < lo) lo = samples[c][r];
            if (samples[c][r] > hi) hi = samples[c][r];
            if (c > 0 && samples[c][r] > samples[0][r]) wins++;
        }
        result->simd_tokps[c] = calib_median_of(samples[c], rounds);
        result->simd_wins[c]  = wins;
        if (c == 0)
            TN_OUT_INF("calibration", "  %-14s %7.1f  [%5.1f – %5.1f] tok/s%s\n", "auto",
                       result->simd_tokps[c], lo, hi, best_simd_idx == 0 ? "  ← BEST" : "");
        else
            TN_OUT_INF("calibration",
                       "  %-14s %7.1f  [%5.1f – %5.1f] tok/s  beat auto in %2d/%d rounds%s\n",
                       TN_CALIB_BACKENDS[c], result->simd_tokps[c], lo, hi, wins, rounds,
                       c == best_simd_idx ? "  ← BEST" : "");
    }
}

/* Phase 1: returns the index of the backend to use (0 = keep auto). */
static int simd_phase(TnCalibrationResult *result, ThreadPool *tp, bool avail[TN_CALIB_SIMD_N]) {
    drop_duplicates_of_auto(result, avail);

    int bg     = calib_sample_background_load(); /* before we start: our threads are idle */
    bool noisy = (bg >= 0 && bg > 30);
    result->simd_sweep_noisy = noisy;
    printf("  auto = %s;  candidates:", result->auto_backend);
    for (int i = 1; i < TN_CALIB_SIMD_N; i++)
        if (avail[i]) printf(" %s", TN_CALIB_BACKENDS[i]);
    TN_OUT_INF("calibration", "\n");

    double samples[TN_CALIB_SIMD_N][TN_CALIB_MAX_ROUNDS];
    memset(samples, 0, sizeof(samples));
    int idx[TN_CALIB_SIMD_N], nc = 0;
    for (int i = 0; i < TN_CALIB_SIMD_N; i++)
        if (avail[i]) idx[nc++] = i;

    BenchBuf bb;
    const int rounds  = TN_CALIB_ROUNDS;
    int best_simd_idx = 0;
    if (nc >= 2 && calib_bench_buf_alloc(&bb)) {
        run_simd_rounds(tp, &bb, idx, nc, rounds, samples);
        calib_bench_buf_free(&bb);

        result->simd_rounds = rounds;
        if (!noisy)
            best_simd_idx = tn_calib_pick_simd((const double(*)[TN_CALIB_MAX_ROUNDS])samples, avail,
                                               TN_CALIB_SIMD_N, rounds, TN_CALIB_OVERRIDE_MARGIN);
        report_simd(result, samples, idx, nc, rounds, best_simd_idx);
        if (noisy)
            TN_OUT_INF("calibration",
                       "  [!] background load %d%%: keeping auto, no tier is overridden\n", bg);
        fflush(stdout);
    }
    return best_simd_idx;
}

/* ── Phase 2: thread count sweep ──────────────────────────────────────── */

/* One thread count: bench it, record min/median/max and the background load. Returns the median
 * tok/s (0 when the pool could not be created). */
static double sweep_one(TnCalibrationResult *result, int t, double best_so_far) {
    ThreadPool *tp = threadpool_create(t);
    if (!tp) {
        result->thread_tokps_min[t - 1]   = 0.0;
        result->thread_tokps_med[t - 1]   = 0.0;
        result->thread_tokps_max[t - 1]   = 0.0;
        result->thread_sysload_pct[t - 1] = -1;
        tn_log_out(TN_LOG_ERROR, "calibration", "  T=%-2d  [failed to create thread pool]\n", t);
        fflush(stdout);
        return 0.0;
    }

    int bg                            = calib_sample_background_load();
    result->thread_sysload_pct[t - 1] = bg;

    char bg_str[16];
    if (bg >= 0)
        snprintf(bg_str, sizeof(bg_str), "%3d%%", bg);
    else
        snprintf(bg_str, sizeof(bg_str), " n/a");

    printf("  T=%-2d  bg:%4s  ", t, bg_str);
    fflush(stdout);

    /* 1.5 s warmup per T — CPU is already hot from phase 1 and earlier thread-sweep iterations, so
     * a short warmup is sufficient. */
    double lo, hi;
    double tokps = calib_bench_robust(tp, 1.5, 5, &lo, &hi);
    threadpool_destroy(tp);

    result->thread_tokps_min[t - 1] = lo;
    result->thread_tokps_med[t - 1] = tokps;
    result->thread_tokps_max[t - 1] = hi;

    bool noisy = (bg >= 0 && bg > 30);
    TN_OUT_INF("calibration", "  %6.1f  [%5.1f – %5.1f] tok/s%s%s\n", tokps, lo, hi,
               (tokps > best_so_far) ? "  ← PEAK" : "", noisy ? "  [!] high bg load" : "");
    fflush(stdout);
    return tokps;
}

/* Phase 2: sweeps T=1..max_t and records the best count (the physical-core count unless another is
 * clearly faster). */
static void thread_phase(TnCalibrationResult *result, int phys, int max_t) {
    double best_thread_tokps = 0.0;
    int best_thread_t        = 1;
    result->thread_sweep_n   = max_t;

    for (int t = 1; t <= max_t; t++) {
        double tokps = sweep_one(result, t, best_thread_tokps);
        if (tokps > best_thread_tokps) {
            best_thread_tokps = tokps;
            best_thread_t     = t;
        }
    }

    /* 2026-11-18: prefer the physical-core count unless another count is clearly faster (same
     * margin and slowest-rep rule as the SIMD override): a T=3 sample range of 42-69 tok/s next to
     * T=4 at 96 is noise, not a preference. */
    int phys_t = phys < max_t ? phys : max_t;
    best_thread_t =
        tn_calib_prefer_physical(result->thread_tokps_med, result->thread_tokps_min, best_thread_t,
                                 best_thread_tokps, phys_t, &best_thread_tokps);
    result->best_threads      = best_thread_t;
    result->best_thread_tokps = best_thread_tokps;
}

/* ── The run ──────────────────────────────────────────────────────────── */

static void fingerprint(TnCalibrationResult *result, const TnHardwareProfile *hw) {
    tn_sys_cpu_model(NULL, result->cpu_model, sizeof(result->cpu_model));
    result->physical_cores = hw->physical_cores;
    result->logical_cores  = hw->logical_cores;
    result->l2_cache_bytes = hw->l2_cache_bytes;
    result->l3_cache_bytes = hw->l3_cache_bytes;
    result->isa_mask       = tn_calibration_isa_mask();
}

void tn_calibrate(TnCalibrationResult *result, const TnHardwareProfile *hw) {
    memset(result, 0, sizeof(*result));
    fingerprint(result, hw);

    int phys  = hw->physical_cores > 0 ? hw->physical_cores : 1;
    int logi  = hw->logical_cores > 0 ? hw->logical_cores : phys;
    int max_t = logi < TN_CALIB_MAX_THREADS ? logi : TN_CALIB_MAX_THREADS;

    print_header(phys);
    const TnCpuFeatures *cpu = tn_cpu_features_detect();

    ThreadPool *tp_phys = threadpool_create(phys);
    if (!tp_phys) {
        TN_LOG_ERR("calibration", "[Calibration] Failed to create thread pool\n");
        return;
    }

    /* 2026-10-18: "auto" added. Before, the tier the dispatcher actually picks on AVX-VNNI CPUs
     * was never measured, and the noisy winner of the forced tiers always overrode it — on an
     * AVX-VNNI Xeon "scalar" (auto-vectorised by -march=native) won and silently replaced
     * AVX-VNNI. */
    bool avail[TN_CALIB_SIMD_N];
    calib_mark_available(avail, cpu);
    int best_simd_idx = simd_phase(result, tp_phys, avail);
    threadpool_destroy(tp_phys);

    result->best_simd_idx = best_simd_idx;
    snprintf(result->best_simd_name, sizeof(result->best_simd_name), "%s",
             TN_CALIB_BACKENDS[best_simd_idx]);

    /* Restore best backend for phase 2 */
    if (best_simd_idx > 0) setenv("TN_FORCE_BACKEND", TN_CALIB_BACKENDS[best_simd_idx], 1);
    tn_simd_init();
    unsetenv("TN_FORCE_BACKEND");

    print_phase2_header(result->best_simd_name);
    thread_phase(result, phys, max_t);

    if (hw->measured_bw_gbps > 0.0)
        tn_calib_classifier_estimates(hw->measured_bw_gbps, (double)hw->l3_cache_bytes,
                                      result->cls_tokps, &result->best_cls_idx);
    result->valid = 1;
    print_result(result, phys, max_t);
}
