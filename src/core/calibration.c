#include "core/calibration.h"

/* Portable dist build: this TU is compiled at the baseline ISA, so the
 * TN_HAS_* macros below would hide every SIMD backend from calibration even
 * though simd_dispatch can run them (runtime CPUID decides, via cpu->*).
 * Same treatment as simd_dispatch.c — this file contains no intrinsics.
 * (2026-10-18: dist auto-calibration only ever benchmarked "scalar".) */
#ifdef TN_FORCE_DISPATCH_ALL
#undef  TN_HAS_AVX2
#define TN_HAS_AVX2 1
#undef  TN_HAS_AVX512
#define TN_HAS_AVX512 1
#undef  TN_HAS_AVX512VNNI
#define TN_HAS_AVX512VNNI 1
#endif
#include "math/cpu_features.h"
#include "math/simd_dispatch.h"
#include "math/parallel_matmul.h"
#include "threading/thread_pool.h"
#include "threading/cpu_probe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#if TN_POSIX
#include <unistd.h>
#include <pwd.h>
#endif

/* ── Hardware fingerprint ─────────────────────────────────────────────── */

static void get_cpu_model(char *buf, size_t buflen) {
    buf[0] = '\0';
#if TN_POSIX
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *p = strchr(line, ':');
            if (p) {
                p++;
                while (*p == ' ') p++;
                size_t len = strlen(p);
                if (len > 0 && p[len - 1] == '\n') p[len - 1] = '\0';
                snprintf(buf, buflen, "%s", p);
            }
            break;
        }
    }
    fclose(f);
#endif
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

static bool fingerprint_matches(const TnCalibrationResult *cached,
                                 const TnHardwareProfile *hw) {
    char current_model[128];
    get_cpu_model(current_model, sizeof(current_model));

    if (strcmp(cached->cpu_model, current_model) != 0) return false;
    if (cached->physical_cores != hw->physical_cores) return false;
    if (cached->logical_cores != hw->logical_cores) return false;
    if (cached->l2_cache_bytes != hw->l2_cache_bytes) return false;
    if (cached->l3_cache_bytes != hw->l3_cache_bytes) return false;
    if (cached->isa_mask != tn_calibration_isa_mask()) return false;
    return true;
}

/* ── Cache file path ──────────────────────────────────────────────────── */

static void get_cache_path(char *buf, size_t buflen) {
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

/* ── Monotonic clock ─────────────────────────────────────────────────── */

static int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ── Background system load ──────────────────────────────────────────── */

/*
 * Read aggregate CPU counters from /proc/stat (Linux only).
 * Returns 1 on success, 0 on failure (non-Linux, permission error, etc.).
 */
#if TN_POSIX
typedef struct { long long user, nice, sys, idle, iowait, irq, softirq; } CpuStat;

static int read_cpu_stat(CpuStat *s) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return 0;
    char line[256];
    memset(s, 0, sizeof(*s));
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu ", 4) == 0) {
            sscanf(line + 4, "%lld %lld %lld %lld %lld %lld %lld",
                   &s->user, &s->nice, &s->sys, &s->idle,
                   &s->iowait, &s->irq, &s->softirq);
            found = 1;
            break;
        }
    }
    fclose(f);
    return found;
}
#endif

/*
 * Sample background CPU load over 500 ms BEFORE the benchmark starts.
 * Since this runs before our benchmark threads are active, the result
 * reflects load from OTHER processes only — a clean background signal.
 *
 * Returns 0-100 (percent), or -1 if /proc/stat is unavailable.
 */
static int sample_background_load(void) {
#if TN_POSIX
    CpuStat s0, s1;
    if (!read_cpu_stat(&s0)) return -1;   /* /proc/stat absent — no sleep */
    { struct timespec ts = {0, 500000000L}; nanosleep(&ts, NULL); } /* 500 ms */
    if (!read_cpu_stat(&s1)) return -1;

    long long busy  = (s1.user    - s0.user)    + (s1.nice    - s0.nice)
                    + (s1.sys     - s0.sys)     + (s1.irq     - s0.irq)
                    + (s1.softirq - s0.softirq);
    long long idle  = (s1.idle    - s0.idle)    + (s1.iowait  - s0.iowait);
    long long total = busy + idle;
    if (total <= 0) return -1;
    int pct = (int)(busy * 100LL / total);
    return pct < 0 ? 0 : (pct > 100 ? 100 : pct);
#else
    return -1;
#endif
}

/* ── Robust parallel benchmark ───────────────────────────────────────── */

/*
 * Benchmark constants — simulate one full token worth of ternary matmuls.
 * 7 matmuls/layer × 30 layers = 210 per "token equivalent".
 * Dimension 2560×2560 matches BitNet b1.58 2B hidden size.
 */
#define CALIB_DIM             2560
#define CALIB_MATMULS_PER_TOK 210
#define CALIB_MAX_REPS        9

static int cmp_double_asc(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

typedef struct { float *x; tn_u8 *pw; float *outv; } BenchBuf;

static bool bench_buf_alloc(BenchBuf *b) {
    const int d = CALIB_DIM, n = CALIB_DIM, packed_bytes = d * (n / 4);
    b->x    = (float *)malloc((size_t)n * sizeof(float));
    b->pw   = (tn_u8 *)malloc((size_t)packed_bytes * sizeof(tn_u8));
    b->outv = (float *)calloc((size_t)d, sizeof(float));
    if (!b->x || !b->pw || !b->outv) {
        free(b->x); free(b->pw); free(b->outv);
        b->x = NULL; b->pw = NULL; b->outv = NULL;
        return false;
    }
    /* Non-trivial data so SIMD kernels cannot short-circuit */
    for (int i = 0; i < n; i++) b->x[i] = 0.01f * (float)((i % 100) + 1);
    for (int i = 0; i < packed_bytes; i++) b->pw[i] = (tn_u8)((i & 0xFE) | 0x01);
    return true;
}

static void bench_buf_free(BenchBuf *b) { free(b->x); free(b->pw); free(b->outv); b->x = NULL; b->pw = NULL; b->outv = NULL; }

/* Time-based warmup: continuous matmuls for warmup_secs (one dot per second). */
static void bench_warmup(ThreadPool *tp, const BenchBuf *b, double warmup_secs) {
    int64_t warmup_end = now_ns() + (int64_t)(warmup_secs * 1.0e9);
    int64_t next_dot   = now_ns() + 1000000000LL;
    while (now_ns() < warmup_end) {
        parallel_ternary_matmul_packed(b->outv, b->x, b->pw, CALIB_DIM, CALIB_DIM, 1.0f, tp);
        if (now_ns() >= next_dot) { putchar('.'); fflush(stdout); next_dot += 1000000000LL; }
    }
}

/* One timed repetition: CALIB_MATMULS_PER_TOK matmuls, returned as tokens/s. */
static double bench_one_rep(ThreadPool *tp, const BenchBuf *b) {
    int64_t t0 = now_ns();
    for (int i = 0; i < CALIB_MATMULS_PER_TOK; i++)
        parallel_ternary_matmul_packed(b->outv, b->x, b->pw, CALIB_DIM, CALIB_DIM, 1.0f, tp);
    int64_t elapsed = now_ns() - t0;
    return (elapsed > 0) ? (1.0e9 / (double)elapsed) : 0.0;
}

/*
 * Run a robust multi-threaded ternary matmul benchmark.
 *
 * Key design decisions vs the old bench_matmul():
 *   1. Uses parallel_ternary_matmul_packed (NOT tn_ternary_matmul_packed)
 *      so the thread pool is genuinely exercised.  The old code created a
 *      ThreadPool but discarded it — thread count had zero effect on results.
 *   2. Time-based warmup (warmup_secs) instead of a fixed 3-iteration warmup.
 *      Three iterations at T=4/VNNI is ~135 ms — nowhere near long enough to
 *      exit Tiger Lake's 20-28 s PL2 turbo window.  Subsequent measurements
 *      would reflect burst frequency, not sustained throughput.
 *   3. Each of measure_reps reps is timed individually → min/median/max.
 *      The old code timed all reps together so variance was invisible.
 *   4. Progress dots are flushed to stdout once per second during warmup and
 *      once per rep during measurement so the terminal never looks frozen.
 *
 * @param tp            Pre-created thread pool (caller manages lifetime)
 * @param warmup_secs   Seconds of continuous matmuls before timing begins
 * @param measure_reps  Individual timed repetitions (must be <= CALIB_MAX_REPS)
 * @param out_min       Written with lowest observed tok/s across reps
 * @param out_max       Written with highest observed tok/s across reps
 * @return              Median tok/s (robust against single-rep outliers)
 */
static double bench_robust(ThreadPool *tp,
                            double warmup_secs,
                            int measure_reps,
                            double *out_min, double *out_max) {
    BenchBuf b;
    if (!bench_buf_alloc(&b)) { *out_min = *out_max = 0.0; return 0.0; }

    printf("warmup[");
    fflush(stdout);
    bench_warmup(tp, &b, warmup_secs);
    printf("] reps[");
    fflush(stdout);

    if (measure_reps < 1) measure_reps = 1;
    if (measure_reps > CALIB_MAX_REPS) measure_reps = CALIB_MAX_REPS;

    double samples[CALIB_MAX_REPS];
    for (int r = 0; r < measure_reps; r++) {
        samples[r] = bench_one_rep(tp, &b);
        putchar('.');
        fflush(stdout);
    }
    printf("]");
    fflush(stdout);

    bench_buf_free(&b);

    qsort(samples, (size_t)measure_reps, sizeof(double), cmp_double_asc);
    *out_min = samples[0];
    *out_max = samples[measure_reps - 1];
    return samples[measure_reps / 2];   /* median */
}

/* ── Choosing a SIMD tier (pure; tests/test_calibration_pick.c) ─────────── */

static double median_of(const double *v, int n) {
    double t[TN_CALIB_MAX_ROUNDS];
    for (int i = 0; i < n; i++) t[i] = v[i];
    qsort(t, (size_t)n, sizeof(double), cmp_double_asc);
    return t[n / 2];
}

int tn_calib_wins_needed(int n_rounds) {
    if (n_rounds < 1 || n_rounds > TN_CALIB_MAX_ROUNDS) return n_rounds + 1;
    /* P(X >= k) for X ~ Binomial(n, 1/2), smallest k with P <= 0.005. */
    double c = 1.0;                     /* C(n, j), built downwards from j = n */
    double tail = 0.0, total = 1.0;
    for (int i = 0; i < n_rounds; i++) total *= 2.0;
    int need = n_rounds + 1;
    for (int j = n_rounds; j >= 0; j--) {
        tail += c / total;
        if (tail > 0.005) break;
        need = j;
        c = c * (double)j / (double)(n_rounds - j + 1);   /* C(n, j-1) from C(n, j) */
    }
    return need;
}

int tn_calib_pick_simd(const double samples[][TN_CALIB_MAX_ROUNDS], const bool avail[],
                       int n_cand, int n_rounds, double margin) {
    if (n_cand < 2 || n_rounds < 1 || n_rounds > TN_CALIB_MAX_ROUNDS || !avail[0]) return 0;
    const int need = tn_calib_wins_needed(n_rounds);
    const double med0 = median_of(samples[0], n_rounds);
    int best = 0;
    double best_med = med0;
    for (int c = 1; c < n_cand; c++) {
        if (!avail[c]) continue;
        int wins = 0;
        for (int r = 0; r < n_rounds; r++) if (samples[c][r] > samples[0][r]) wins++;
        double med = median_of(samples[c], n_rounds);
        if (wins >= need && med > med0 * margin && med > best_med) { best = c; best_med = med; }
    }
    return best;
}

/* ── Main calibration ─────────────────────────────────────────────────── */

void tn_calibrate(TnCalibrationResult *result, const TnHardwareProfile *hw) {
    memset(result, 0, sizeof(*result));

    /* Fingerprint */
    get_cpu_model(result->cpu_model, sizeof(result->cpu_model));
    result->physical_cores = hw->physical_cores;
    result->logical_cores  = hw->logical_cores;
    result->l2_cache_bytes = hw->l2_cache_bytes;
    result->l3_cache_bytes = hw->l3_cache_bytes;
    result->isa_mask       = tn_calibration_isa_mask();

    int phys  = hw->physical_cores > 0 ? hw->physical_cores : 1;
    int logi  = hw->logical_cores  > 0 ? hw->logical_cores  : phys;
    int max_t = logi < TN_CALIB_MAX_THREADS ? logi : TN_CALIB_MAX_THREADS;

    printf("┌──────────────────────────────────────────────────────────────────┐\n");
    printf("│  Auto-Calibration  —  first run, result cached for future use   │\n");
    printf("│  Re-triggers only when hardware changes.  ~40-60 seconds.       │\n");
    printf("├──────────────────────────────────────────────────────────────────┤\n");
    fflush(stdout);

    const TnCpuFeatures *cpu = tn_cpu_features_detect();

    /* ── Phase 1/2: SIMD backend comparison at T=physical_cores ──────────── */
    printf("│  Phase 1/2  SIMD backends at T=%d  (thermal stabilisation)       │\n", phys);
    printf("├──────────────────────────────────────────────────────────────────┤\n");
    fflush(stdout);

    ThreadPool *tp_phys = threadpool_create(phys);
    if (!tp_phys) {
        fprintf(stderr, "[Calibration] Failed to create thread pool\n");
        return;
    }

    /* 2026-10-18: "auto" added. Before, the tier the dispatcher actually picks
     * on AVX-VNNI CPUs was never measured, and the noisy winner of the forced
     * tiers always overrode it — on an AVX-VNNI Xeon "scalar" (auto-vectorised
     * by -march=native) won and silently replaced AVX-VNNI. */
    const char *backends[TN_CALIB_SIMD_N] = {"auto", "scalar", "avx2", "avx512f", "vnni", "vnni256"};
    bool        avail[TN_CALIB_SIMD_N]    = {true, true, false, false, false, false};

#if TN_HAS_AVX2
    if (cpu->avx2)       avail[2] = true;
#endif
#if TN_HAS_AVX512
    if (cpu->avx512f)    avail[3] = true;
#endif
#if TN_HAS_AVX512VNNI
    if (cpu->avx512vnni) { avail[4] = true; avail[5] = true; }
#endif

    /* ── Which candidates really differ from "auto"? ─────────────────────
     * A forced value that lands on the same kernel as "auto" adds nothing. */
    for (int i = 0; i < TN_CALIB_SIMD_N; i++) {
        if (!avail[i]) continue;
        if (i == 0) unsetenv("TN_FORCE_BACKEND");
        else        setenv("TN_FORCE_BACKEND", backends[i], 1);
        const char *picked = tn_simd_init();
        if (i == 0) snprintf(result->auto_backend, sizeof(result->auto_backend), "%s", picked);
        else if (strcmp(picked, result->auto_backend) == 0) avail[i] = false;
    }
    unsetenv("TN_FORCE_BACKEND");
    tn_simd_init();

    int bg = sample_background_load();   /* before we start: our threads are idle */
    bool noisy = (bg >= 0 && bg > 30);
    result->simd_sweep_noisy = noisy;
    printf("  auto = %s;  candidates:", result->auto_backend);
    for (int i = 1; i < TN_CALIB_SIMD_N; i++) if (avail[i]) printf(" %s", backends[i]);
    printf("\n");

    /* ── Interleaved rounds ──────────────────────────────────────────────
     * Every candidate once per round, the starting candidate rotating, so
     * thermal drift and background load hit all of them alike (2026-11-18). */
    static double samples[TN_CALIB_SIMD_N][TN_CALIB_MAX_ROUNDS];
    memset(samples, 0, sizeof(samples));
    int idx[TN_CALIB_SIMD_N], nc = 0;
    for (int i = 0; i < TN_CALIB_SIMD_N; i++) if (avail[i]) idx[nc++] = i;

    BenchBuf bb;
    const int rounds = TN_CALIB_ROUNDS;
    int best_simd_idx = 0;
    if (nc >= 2 && bench_buf_alloc(&bb)) {
        printf("  warmup[");
        fflush(stdout);
        bench_warmup(tp_phys, &bb, 4.0);       /* "auto" is bound: the CPU starts cold */
        printf("]  rounds[");
        fflush(stdout);
        for (int r = 0; r < rounds; r++) {
            for (int k = 0; k < nc; k++) {
                int c = idx[(k + r) % nc];
                if (c == 0) unsetenv("TN_FORCE_BACKEND");
                else        setenv("TN_FORCE_BACKEND", backends[c], 1);
                tn_simd_init();
                samples[c][r] = bench_one_rep(tp_phys, &bb);
            }
            putchar('.');
            fflush(stdout);
        }
        printf("]\n");
        bench_buf_free(&bb);
        unsetenv("TN_FORCE_BACKEND");
        tn_simd_init();

        result->simd_rounds = rounds;
        if (!noisy)
            best_simd_idx = tn_calib_pick_simd((const double (*)[TN_CALIB_MAX_ROUNDS])samples,
                                               avail, TN_CALIB_SIMD_N, rounds,
                                               TN_CALIB_OVERRIDE_MARGIN);
        for (int k = 0; k < nc; k++) {
            int c = idx[k];
            double lo = samples[c][0], hi = samples[c][0];
            int wins = 0;
            for (int r = 0; r < rounds; r++) {
                if (samples[c][r] < lo) lo = samples[c][r];
                if (samples[c][r] > hi) hi = samples[c][r];
                if (c > 0 && samples[c][r] > samples[0][r]) wins++;
            }
            result->simd_tokps[c] = median_of(samples[c], rounds);
            result->simd_wins[c]  = wins;
            if (c == 0) printf("  %-14s %7.1f  [%5.1f – %5.1f] tok/s%s\n", "auto",
                               result->simd_tokps[c], lo, hi, best_simd_idx == 0 ? "  ← BEST" : "");
            else        printf("  %-14s %7.1f  [%5.1f – %5.1f] tok/s  beat auto in %2d/%d rounds%s\n",
                               backends[c], result->simd_tokps[c], lo, hi, wins, rounds,
                               c == best_simd_idx ? "  ← BEST" : "");
        }
        if (noisy)
            printf("  [!] background load %d%%: keeping auto, no tier is overridden\n", bg);
        fflush(stdout);
    }

    threadpool_destroy(tp_phys);

    result->best_simd_idx = best_simd_idx;
    snprintf(result->best_simd_name, sizeof(result->best_simd_name),
             "%s", backends[best_simd_idx]);

    /* Restore best backend for phase 2 */
    if (best_simd_idx > 0) setenv("TN_FORCE_BACKEND", backends[best_simd_idx], 1);
    tn_simd_init();
    unsetenv("TN_FORCE_BACKEND");

    /* ── Phase 2/2: Thread count sweep with best SIMD backend ────────────── */
    printf("├──────────────────────────────────────────────────────────────────┤\n");
    printf("│  Phase 2/2  Thread sweep  (backend: %-10s)                │\n",
           result->best_simd_name);
    printf("├──────────────────────────────────────────────────────────────────┤\n");
    fflush(stdout);

    double best_thread_tokps = 0.0;
    int    best_thread_t     = 1;
    result->thread_sweep_n   = max_t;

    for (int t = 1; t <= max_t; t++) {
        ThreadPool *tp = threadpool_create(t);
        if (!tp) {
            result->thread_tokps_min[t - 1] = 0.0;
            result->thread_tokps_med[t - 1] = 0.0;
            result->thread_tokps_max[t - 1] = 0.0;
            result->thread_sysload_pct[t - 1] = -1;
            printf("  T=%-2d  [failed to create thread pool]\n", t);
            fflush(stdout);
            continue;
        }

        int bg = sample_background_load();
        result->thread_sysload_pct[t - 1] = bg;

        char bg_str[16];
        if (bg >= 0) snprintf(bg_str, sizeof(bg_str), "%3d%%", bg);
        else         snprintf(bg_str, sizeof(bg_str), " n/a");

        printf("  T=%-2d  bg:%4s  ", t, bg_str);
        fflush(stdout);

        /*
         * 1.5 s warmup per T — CPU is already hot from phase 1 and earlier
         * thread-sweep iterations, so a short warmup is sufficient.
         */
        double lo, hi;
        double tokps = bench_robust(tp, 1.5, 5, &lo, &hi);
        threadpool_destroy(tp);

        result->thread_tokps_min[t - 1] = lo;
        result->thread_tokps_med[t - 1] = tokps;
        result->thread_tokps_max[t - 1] = hi;

        bool is_best = (tokps > best_thread_tokps);
        if (is_best) { best_thread_tokps = tokps; best_thread_t = t; }

        bool noisy = (bg >= 0 && bg > 30);
        printf("  %6.1f  [%5.1f – %5.1f] tok/s%s%s\n",
               tokps, lo, hi,
               is_best ? "  ← PEAK" : "",
               noisy   ? "  [!] high bg load" : "");
        fflush(stdout);
    }

    /* 2026-11-18: prefer the physical-core count unless another count is clearly
     * faster (same margin and slowest-rep rule as the SIMD override): a T=3 sample
     * range of 42-69 tok/s next to T=4 at 96 is noise, not a preference. */
    int phys_t = phys < max_t ? phys : max_t;
    if (best_thread_t != phys_t && result->thread_tokps_med[phys_t - 1] > 0.0) {
        double pm = result->thread_tokps_med[phys_t - 1];
        bool clear = best_thread_tokps > pm * TN_CALIB_OVERRIDE_MARGIN &&
                     result->thread_tokps_min[best_thread_t - 1] > pm;
        if (!clear) { best_thread_t = phys_t; best_thread_tokps = pm; }
    }

    result->best_threads      = best_thread_t;
    result->best_thread_tokps = best_thread_tokps;

    /* ── Classifier bandwidth estimates ───────────────────────────────────────
     * ANALYTIC pre-load estimates, not kernel measurements: calibration runs
     * before any model file is opened, so the byte figures below are the
     * BitNet-2B compile-time geometry (522 MB ternary, 128256x2560 classifier)
     * — the same pre-load limitation as hardware_profile.c's startup box, and
     * wrong in absolute terms for any other model (2026-07-17, see
     * docs/architecture/CEILING_CALCULATION.md §4 site 4). The *ranking*
     * between formats (what --classifier auto-fast consumes via best_cls_idx)
     * only depends on the relative classifier byte sizes, so it survives the
     * wrong absolute base better than the tok/s figures do. TODO: make this
     * model-aware (needs a calibration redesign — the cache is keyed on
     * hardware, not model). */
    double base_bw = hw->measured_bw_gbps;
    if (base_bw > 0.0) {
        double ternary_mb  = 522.0;
        double bf16_cls_mb = (double)(128256ULL * 2560 * 2) / (1024.0 * 1024.0);
        double int8_cls_mb = bf16_cls_mb / 2.0;
        double int4_cls_mb = bf16_cls_mb / 4.0;
        double l3_mb       = (double)hw->l3_cache_bytes / (1024.0 * 1024.0);
        double cached      = (l3_mb >= ternary_mb) ? ternary_mb : l3_mb * 0.5;
        double dram_mb     = ternary_mb - cached;
        /* base_bw is decimal GB/s (bytes/ns from the probe); the *_mb sizes
         * are MiB, so the conversion is 1e9/2^20 ≈ 953.67 MiB/s per GB/s —
         * NOT 1024 (2026-07-17, independent-review finding: the old *1024
         * inflated every printed cls_tokps ~7.4%; rankings were unaffected
         * since the error was a common factor). */
        double bw_mbps     = base_bw * (1e9 / (1024.0 * 1024.0));

        result->cls_tokps[0] = bw_mbps / (dram_mb + bf16_cls_mb);
        result->cls_tokps[1] = bw_mbps / (dram_mb + int8_cls_mb);
        result->cls_tokps[2] = bw_mbps / (dram_mb + int4_cls_mb);

        result->best_cls_idx = 0;
        for (int i = 1; i < 3; i++)
            if (result->cls_tokps[i] > result->cls_tokps[result->best_cls_idx])
                result->best_cls_idx = i;
    }

    result->valid = 1;

    printf("├──────────────────────────────────────────────────────────────────┤\n");
    printf("│  Result                                                          │\n");
    printf("│    SIMD      : %-10s (%5.1f tok/s at T=%d)                 │\n",
           result->best_simd_name, result->simd_tokps[result->best_simd_idx], phys);
    printf("│    Threads   : T=%-2d  (%5.1f tok/s, swept T=1..%d)           │\n",
           result->best_threads, result->best_thread_tokps, max_t);
    printf("│    Classifier: BF16 (default — full intelligence)               │\n");
    /* Print the actually-computed best format + gain (2026-07-17): this line
     * was previously the hardcoded literal "(INT8, ~+36%)" regardless of the
     * cls_tokps[] computed above. "est." because these are analytic pre-load
     * estimates (see the comment on the block above), not measurements. */
    {
        char opt_line[80];
        if (result->cls_tokps[0] > 0.0) {
            static const char *cls_names[3] = {"BF16", "INT8", "INT4"};
            int best = result->best_cls_idx;
            if (best < 0 || best > 2) best = 0;
            double gain_pct = (result->cls_tokps[best] /
                               result->cls_tokps[0] - 1.0) * 100.0;
            snprintf(opt_line, sizeof(opt_line),
                     "Speed opt : --classifier auto-fast  (%s, ~%+.0f%% est.)",
                     cls_names[best], gain_pct);
        } else {
            snprintf(opt_line, sizeof(opt_line),
                     "Speed opt : --classifier auto-fast");
        }
        printf("│    %-62s│\n", opt_line);
    }
    printf("└──────────────────────────────────────────────────────────────────┘\n");
    fflush(stdout);
}

/* ── Cache I/O ────────────────────────────────────────────────────────── */

bool tn_calibration_load(TnCalibrationResult *result, const TnHardwareProfile *hw) {
    char path[512];
    get_cache_path(path, sizeof(path));

    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* Read and validate magic + version */
    uint32_t magic = 0;
    int version = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != TN_CALIB_MAGIC) {
        fclose(f); return false;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != TN_CALIB_VERSION) {
        fclose(f); return false;
    }

    if (fread(result, sizeof(*result), 1, f) != 1) {
        fclose(f); return false;
    }
    fclose(f);

    if (!result->valid) return false;

    /* Check hardware fingerprint */
    if (!fingerprint_matches(result, hw)) {
        printf("[Calibration] Hardware changed — re-calibration needed.\n");
        return false;
    }

    return true;
}

void tn_calibration_save(const TnCalibrationResult *result) {
    char path[512];
    get_cache_path(path, sizeof(path));

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Calibration] Warning: could not save to %s\n", path);
        return;
    }

    uint32_t magic = TN_CALIB_MAGIC;
    int version = TN_CALIB_VERSION;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(result, sizeof(*result), 1, f);
    fclose(f);

    printf("[Calibration] Saved to %s\n", path);
}

void tn_calibration_report(const TnCalibrationResult *result) {
    if (!result->valid) {
        printf("[Calibration] No valid calibration data.\n");
        return;
    }

    printf("\n[Calibration Results]\n");
    printf("  CPU      : %s\n", result->cpu_model);
    printf("  Cores    : %d physical / %d logical\n",
           result->physical_cores, result->logical_cores);
    printf("  SIMD     : %s (%.1f tok/s)\n",
           result->best_simd_name,
           result->simd_tokps[result->best_simd_idx]);
    printf("  Threads  : T=%d (%.1f tok/s, swept T=1..%d)\n",
           result->best_threads, result->best_thread_tokps,
           result->thread_sweep_n);

    if (result->thread_sweep_n > 0) {
        printf("  Thread sweep:\n");
        for (int t = 1; t <= result->thread_sweep_n; t++) {
            int bg = result->thread_sysload_pct[t - 1];
            char bg_str[16];
            if (bg >= 0) snprintf(bg_str, sizeof(bg_str), "%3d%%", bg);
            else         snprintf(bg_str, sizeof(bg_str), " n/a");
            printf("    T=%-2d  bg:%4s  %6.1f  [%5.1f – %5.1f] tok/s%s\n",
                   t, bg_str,
                   result->thread_tokps_med[t - 1],
                   result->thread_tokps_min[t - 1],
                   result->thread_tokps_max[t - 1],
                   t == result->best_threads ? "  ← PEAK" : "");
        }
    }

    const char *cls_names[] = {"BF16", "INT8", "INT4"};
    printf("  Classifier estimates:\n");
    for (int i = 0; i < 3; i++) {
        if (result->cls_tokps[i] <= 0.0) continue;
        printf("    %-4s  %.1f tok/s%s%s\n",
               cls_names[i], result->cls_tokps[i],
               i == 0 ? " (default, full intelligence)" : "",
               i == result->best_cls_idx ? " [FASTEST]" : "");
    }
    printf("\n");
}

const char *tn_calibration_best_simd(const TnCalibrationResult *result) {
    if (!result->valid) return NULL;
    /* "auto": the dispatcher's own CPUID choice stands — nothing to force. */
    if (strcmp(result->best_simd_name, "auto") == 0) return NULL;
    return result->best_simd_name;
}

TnClassifierFormat tn_calibration_best_classifier(const TnCalibrationResult *result) {
    if (!result->valid) return TN_CLS_BF16;
    return (TnClassifierFormat)result->best_cls_idx;
}
