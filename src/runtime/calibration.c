#include "runtime/calibration.h"

#include "core/log.h"
#include "core/sys_probe.h"
#include "runtime/calib_internal.h"

/* Portable dist build: this TU is compiled at the baseline ISA, so the
 * TN_HAS_* macros below would hide every SIMD backend from calibration even
 * though simd_dispatch can run them (runtime CPUID decides, via cpu->*).
 * Same treatment as simd_dispatch.c — this file contains no intrinsics.
 * (2026-10-18: dist auto-calibration only ever benchmarked "scalar".) */
#ifdef TN_FORCE_DISPATCH_ALL
#undef TN_HAS_AVX2
#define TN_HAS_AVX2 1
#undef TN_HAS_AVX512
#define TN_HAS_AVX512 1
#undef TN_HAS_AVX512VNNI
#define TN_HAS_AVX512VNNI 1
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "core/clamp.h"
#include "core/cpu_features.h"
#include "math/parallel_matmul.h"
#include "math/simd_dispatch.h"
#include "threading/cpu_probe.h"
#include "threading/thread_pool.h"

#if TN_POSIX
#include <pwd.h>
#include <unistd.h>
#endif

/* ── Hardware fingerprint ─────────────────────────────────────────────── */

static void read_cpu_model(char *buf, size_t buflen) {
    tn_sys_cpu_model(NULL, buf, buflen);
}

unsigned tn_calibration_isa_mask(void) {
    unsigned m = 0;
#if TN_HAS_AVX2
    m |= 1u << 0;
#endif
#if TN_HAS_AVX512
    m |= 1u << 1;
#endif
#if TN_HAS_AVX512VNNI
    m |= 1u << 2;
#endif
#if TN_HAS_AVXVNNI
    m |= 1u << 3;
#endif
#if TN_HAS_NEON
    m |= 1u << 4;
#endif
#if TN_HAS_ARM_DOTPROD
    m |= 1u << 5;
#endif
    return m;
}

static bool fingerprint_matches(const TnCalibrationResult *cached, const TnHardwareProfile *hw) {
    char current_model[128];
    read_cpu_model(current_model, sizeof(current_model));

    if (strcmp(cached->cpu_model, current_model) != 0) return false;
    if (cached->physical_cores != hw->physical_cores) return false;
    if (cached->logical_cores != hw->logical_cores) return false;
    if (cached->l2_cache_bytes != hw->l2_cache_bytes) return false;
    if (cached->l3_cache_bytes != hw->l3_cache_bytes) return false;
    if (cached->isa_mask != tn_calibration_isa_mask()) return false;
    return true;
}

/* ── Cache file path ──────────────────────────────────────────────────── */

static void resolve_cache_path(char *buf, size_t buflen) {
#if TN_POSIX
    const char *home = getenv("HOME");
    if (!home) {
        struct passwd *pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (home) {
        snprintf(buf, buflen, "%s/.project-zero", home);
        mkdir(buf, 0755);
        snprintf(buf, buflen, "%s/.project-zero/calibration.bin", home);
    } else {
        snprintf(buf, buflen, "/tmp/pz_calibration.bin");
    }
#else
    snprintf(buf, buflen, "calibration.bin");
#endif
}

/* ── Choosing a SIMD tier (pure; tests/test_calibration_pick.c) ─────────── */


int tn_calib_wins_needed(int n_rounds) {
    if (n_rounds < 1 || n_rounds > TN_CALIB_MAX_ROUNDS) return n_rounds + 1;
    /* P(X >= k) for X ~ Binomial(n, 1/2), smallest k with P <= 0.005. */
    double c    = 1.0; /* C(n, j), built downwards from j = n */
    double tail = 0.0, total = 1.0;
    for (int i = 0; i < n_rounds; i++) total *= 2.0;
    int need = n_rounds + 1;
    for (int j = n_rounds; j >= 0; j--) {
        tail += c / total;
        if (tail > 0.005) break;
        need = j;
        c    = c * (double)j / (double)(n_rounds - j + 1); /* C(n, j-1) from C(n, j) */
    }
    return need;
}

int tn_calib_pick_simd(const double samples[][TN_CALIB_MAX_ROUNDS], const bool avail[], int n_cand,
                       int n_rounds, double margin) {
    if (n_cand < 2 || n_rounds < 1 || n_rounds > TN_CALIB_MAX_ROUNDS || !avail[0]) return 0;
    const int need    = tn_calib_wins_needed(n_rounds);
    const double med0 = calib_median_of(samples[0], n_rounds);
    int best          = 0;
    double best_med   = med0;
    for (int c = 1; c < n_cand; c++) {
        if (!avail[c]) continue;
        int wins = 0;
        for (int r = 0; r < n_rounds; r++)
            if (samples[c][r] > samples[0][r]) wins++;
        double med = calib_median_of(samples[c], n_rounds);
        if (wins >= need && med > med0 * margin && med > best_med) {
            best     = c;
            best_med = med;
        }
    }
    return best;
}

/* ── Candidates ───────────────────────────────────────────────────────── */

const char *const TN_CALIB_BACKENDS[TN_CALIB_SIMD_N] = {"auto",    "scalar", "avx2",
                                                        "avx512f", "vnni",   "vnni256"};

void calib_mark_available(bool avail[TN_CALIB_SIMD_N], const TnCpuFeatures *cpu) {
    for (int i = 0; i < TN_CALIB_SIMD_N; i++) avail[i] = (i < 2); /* auto, scalar */
#if TN_HAS_AVX2
    if (cpu->avx2) avail[2] = true;
#endif
#if TN_HAS_AVX512
    if (cpu->avx512f) avail[3] = true;
#endif
#if TN_HAS_AVX512VNNI
    if (cpu->avx512vnni) {
        avail[4] = true;
        avail[5] = true;
    }
#endif
    (void)cpu;
}

/* ── Cache I/O ────────────────────────────────────────────────────────── */

bool tn_calibration_load(TnCalibrationResult *result, const TnHardwareProfile *hw) {
    char path[512];
    resolve_cache_path(path, sizeof(path));

    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* Read and validate magic + version */
    uint32_t magic = 0;
    int version    = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != TN_CALIB_MAGIC) {
        fclose(f);
        return false;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != TN_CALIB_VERSION) {
        fclose(f);
        return false;
    }

    if (fread(result, sizeof(*result), 1, f) != 1) {
        fclose(f);
        return false;
    }
    fclose(f);

    if (!result->valid) return false;

    /* Check hardware fingerprint */
    if (!fingerprint_matches(result, hw)) {
        TN_OUT_INF("calibration", "[Calibration] Hardware changed — re-calibration needed.\n");
        return false;
    }

    return true;
}

void tn_calibration_save(const TnCalibrationResult *result) {
    char path[512];
    resolve_cache_path(path, sizeof(path));

    FILE *f = fopen(path, "wb");
    if (!f) {
        TN_LOG_WRN("calibration", "[Calibration] Warning: could not save to %s\n", path);
        return;
    }

    uint32_t magic = TN_CALIB_MAGIC;
    int version    = TN_CALIB_VERSION;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(result, sizeof(*result), 1, f);
    fclose(f);

    TN_OUT_INF("calibration", "[Calibration] Saved to %s\n", path);
}

void tn_calibration_report(const TnCalibrationResult *result) {
    if (!result->valid) {
        TN_OUT_INF("calibration", "[Calibration] No valid calibration data.\n");
        return;
    }

    TN_OUT_INF("calibration", "\n[Calibration Results]\n");
    TN_OUT_INF("calibration", "  CPU      : %s\n", result->cpu_model);
    TN_OUT_INF("calibration", "  Cores    : %d physical / %d logical\n", result->physical_cores,
               result->logical_cores);
    TN_OUT_INF("calibration", "  SIMD     : %s (%.1f tok/s)\n", result->best_simd_name,
               result->simd_tokps[result->best_simd_idx]);
    TN_OUT_INF("calibration", "  Threads  : T=%d (%.1f tok/s, swept T=1..%d)\n",
               result->best_threads, result->best_thread_tokps, result->thread_sweep_n);

    if (result->thread_sweep_n > 0) {
        TN_OUT_INF("calibration", "  Thread sweep:\n");
        for (int t = 1; t <= result->thread_sweep_n; t++) {
            int bg = result->thread_sysload_pct[t - 1];
            char bg_str[16];
            if (bg >= 0)
                snprintf(bg_str, sizeof(bg_str), "%3d%%", bg);
            else
                snprintf(bg_str, sizeof(bg_str), " n/a");
            tn_log_out(
                TN_LOG_INFO, "calibration", "    T=%-2d  bg:%4s  %6.1f  [%5.1f – %5.1f] tok/s%s\n",
                t, bg_str, result->thread_tokps_med[t - 1], result->thread_tokps_min[t - 1],
                result->thread_tokps_max[t - 1], t == result->best_threads ? "  ← PEAK" : "");
        }
    }

    const char *cls_names[] = {"BF16", "INT8", "INT4"};
    TN_OUT_INF("calibration", "  Classifier estimates:\n");
    for (int i = 0; i < 3; i++) {
        if (result->cls_tokps[i] <= 0.0) continue;
        TN_OUT_INF("calibration", "    %-4s  %.1f tok/s%s%s\n", cls_names[i], result->cls_tokps[i],
                   i == 0 ? " (default, full intelligence)" : "",
                   i == result->best_cls_idx ? " [FASTEST]" : "");
    }
    TN_OUT_INF("calibration", "\n");
}

const char *tn_calibration_best_simd(const TnCalibrationResult *result) {
    if (!result->valid) return NULL;
    /* "auto": the dispatcher's own CPUID choice stands — nothing to force. */
    if (strcmp(result->best_simd_name, "auto") == 0) return NULL;
    return result->best_simd_name;
}

const char *tn_calibration_act_quant(const TnCalibrationResult *result) {
    const char *t = tn_calibration_best_simd(result);
    if (!t) return NULL;
    if (strcmp(t, "vnni") == 0 || strcmp(t, "vnni256") == 0) return "int8";
    return "float";
}

int tn_calibration_act_timing(const TnCalibrationResult *result, int auto_is_int8,
                              double *int8_tokps, double *float_tokps) {
    if (!result || !result->valid) return 0;
    /* indices follow tn_calibration_simd_names(): 0 auto, 1 scalar, 2 avx2, 3 avx512f, 4 vnni, 5 vnni256 */
    double i8 = result->simd_tokps[4] > result->simd_tokps[5] ? result->simd_tokps[4]
                                                              : result->simd_tokps[5];
    double fl = result->simd_tokps[1];
    if (auto_is_int8) {
        if (result->simd_tokps[0] > i8) i8 = result->simd_tokps[0];
    } else if (result->simd_tokps[0] > fl)
        fl = result->simd_tokps[0];
    for (int i = 2; i <= 3; i++)
        if (result->simd_tokps[i] > fl) fl = result->simd_tokps[i];
    if (!(i8 > 0.0) || !(fl > 0.0)) return 0;
    if (int8_tokps) *int8_tokps = i8;
    if (float_tokps) *float_tokps = fl;
    return 1;
}

TnClassifierFormat tn_calibration_best_classifier(const TnCalibrationResult *result) {
    if (!result->valid) return TN_CLS_BF16;
    return (TnClassifierFormat)result->best_cls_idx;
}
