/*
 * tools/bench_full.c — Phase 17 Comprehensive Benchmark Tool
 *
 * Sweeps:
 *   Threads       T = 1 .. MAX_THREADS (default 8)
 *   SIMD backends scalar | avx2 | avx512f | vnni256 | vnni | avxvnni
 *   Classifier    BF16 (0) | INT8 (1) | INT4 (2)
 *
 * For each combination runs BENCH_TOKENS inference steps, discards the
 * first BENCH_WARMUP as warmup, and reports average tok/s for the rest.
 *
 * Build:
 *   make build/tools/bench_full
 *
 * Usage:
 *   build/tools/bench_full --model models/bitnet-b1.58-2B-4T.bin
 *   build/tools/bench_full --model models/... --tokens 30 --warmup 5 \
 *                          --max-threads 8 --csv /tmp/results.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>   /* sleep() */

#include "core/config.h"
#include "core/error.h"
#include "core/hardware_profile.h"
#include "core/moe_config.h"
#include "core/platform.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "math/cpu_features.h"
#include "math/simd_dispatch.h"
#include "memory/mapped_file.h"
#include "threading/thread_pool.h"
#include "transformer/forward.h"

/* ── Defaults ──────────────────────────────────────────────────────────────── */
#define DEFAULT_TOKENS       30
#define DEFAULT_WARMUP        5
#define DEFAULT_MAX_THREADS   8

/* ── Timing ─────────────────────────────────────────────────────────────────── */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── SIMD backend descriptors ───────────────────────────────────────────────── */
typedef struct { const char *name; const char *env_val; } SimdDesc;

/* Values match TN_FORCE_BACKEND strings in src/math/simd_dispatch.c.
 * Order is BEST-FIRST so the highest-performance cells run before the CPU
 * warms up, avoiding thermal-throttle skew on long sweeps.
 * Scalar is last — it is only useful as a reference baseline. */
static const SimdDesc SIMD_LEVELS[] = {
    { "AVX-512VNNI", "vnni"     },   /* 64 int8 MACs/cy — best on Ice/Tiger/Zen4 */
    { "VNNI-256",    "vnni256"  },   /* 32 int8 MACs/cy — no ZMM throttle        */
    { "AVX-VNNI",    "avxvnni"  },   /* 32 int8 MACs/cy — Alder Lake / Zen3      */
    { "AVX-512F",    "avx512f"  },   /* 16 fp32 MACs/cy                          */
    { "AVX2",        "avx2"     },   /*  8 fp32 MACs/cy                          */
    { "Scalar",      "scalar"   },   /*  1 fp32 MAC/cy  — reference only         */
};
#define N_SIMD (int)(sizeof(SIMD_LEVELS)/sizeof(SIMD_LEVELS[0]))

static const char *CLS_NAMES[] = { "BF16", "INT8", "INT4" };
static const TnClassifierFormat CLS_FMTS[] = { TN_CLS_BF16, TN_CLS_INT8, TN_CLS_INT4 };
#define N_CLS 3

/* ── Check if SIMD level is supported at runtime ────────────────────────────── */
static int simd_supported(const char *env_val, const TnCpuFeatures *cpu) {
    if (!strcmp(env_val, "scalar"))  return 1;
    if (!strcmp(env_val, "avx2"))    return cpu->avx2;
    if (!strcmp(env_val, "avx512f")) return cpu->avx512f;
    if (!strcmp(env_val, "vnni256")) return cpu->avx512vnni;
    if (!strcmp(env_val, "vnni"))    return cpu->avx512vnni;
    if (!strcmp(env_val, "avxvnni")) return cpu->avx_vnni;
    return 0;
}

/* ── Activate a SIMD backend ────────────────────────────────────────────────── */
static void activate_simd(const char *env_val) {
    setenv("TN_FORCE_BACKEND", env_val, 1);
    tn_simd_init();
}

/* ── Result record ──────────────────────────────────────────────────────────── */
typedef struct {
    int        threads;
    const char *simd_name;
    const char *cls_name;
    double     tok_per_sec;
    int        valid;
} BenchResult;

/*
 * Run a timed benchmark.
 *
 * Adaptive mode: do 1 warmup token, measure it. If that single token takes
 * longer than MAX_CELL_SEC, the backend is too slow for a full sweep —
 * return the speed from the probe token alone and move on.
 * This caps scalar cells at ~12s each instead of 5+ minutes.
 *
 * NOTE: bench order is BEST-FIRST (VNNI before Scalar) so the fastest
 * cells execute while the CPU is cool, giving accurate peak numbers.
 */
#define MAX_CELL_SEC   5.0   /* cap per-cell probe; scalar T=1 ≈ 12s → capped */

static double run_bench(const Config *cfg, const TransformerWeights *w,
                        int n_tokens, int warmup, ThreadPool *tp) {
    RunState s;
    if (run_state_alloc(&s, cfg, cfg->seq_len) != TN_OK) return -1.0;

    int vocab  = cfg->vocab_size > 0 ? cfg->vocab_size : 1;
    int seqlen = cfg->seq_len    > 0 ? cfg->seq_len    : 1;

    /* --- 1 adaptive probe token --- */
    double t_probe0 = now_sec();
    transformer_forward(0, 0, cfg, w, &s, NULL, tp);
    double probe_sec = now_sec() - t_probe0;

    if (probe_sec > MAX_CELL_SEC) {
        /* Backend too slow — report from probe token, move on */
        run_state_free(&s);
        return 1.0 / probe_sec;
    }

    /* --- Full warmup (minus the 1 probe already done) --- */
    for (int t = 1; t < warmup; t++)
        transformer_forward(t % vocab, t % seqlen, cfg, w, &s, NULL, tp);

    int measured = n_tokens - warmup;
    if (measured <= 0) { run_state_free(&s); return -1.0; }

    double t0 = now_sec();
    for (int t = warmup; t < n_tokens; t++)
        transformer_forward(t % vocab, t % seqlen, cfg, w, &s, NULL, tp);
    double elapsed = now_sec() - t0;

    run_state_free(&s);
    return (elapsed > 0.0) ? (double)measured / elapsed : -1.0;
}

/* ── CSV writer ─────────────────────────────────────────────────────────────── */
static void write_csv(BenchResult *res, int n, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "warn: cannot write CSV to %s\n", path); return; }
    fprintf(f, "threads,simd,classifier,tok_per_sec\n");
    for (int i = 0; i < n; i++) {
        if (res[i].valid)
            fprintf(f, "%d,%s,%s,%.4f\n",
                    res[i].threads, res[i].simd_name, res[i].cls_name,
                    res[i].tok_per_sec);
        else
            fprintf(f, "%d,%s,%s,NA\n",
                    res[i].threads, res[i].simd_name, res[i].cls_name);
    }
    fclose(f);
    printf("CSV written → %s\n", path);
}

/* ── Table printer ──────────────────────────────────────────────────────────── */
static void print_table(BenchResult *res, int n, const char *model_path,
                        const char *best_simd) {
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("  Phase 17 Benchmark  |  %s\n", model_path);
    printf("  Auto-detected best backend: %s\n", best_simd);
    printf("══════════════════════════════════════════════════════════════════════\n");
    printf("  %-7s  %-13s  %-7s  %9s\n", "Threads", "SIMD", "Cls", "tok/s");
    printf("  ─────────────────────────────────────────────────────────────────\n");

    double peak = 0.0;
    for (int i = 0; i < n; i++) {
        if (res[i].valid && res[i].tok_per_sec > peak)
            peak = res[i].tok_per_sec;
    }

    for (int i = 0; i < n; i++) {
        BenchResult *r = &res[i];
        if (!r->valid) {
            printf("  %-7d  %-13s  %-7s  %9s\n",
                   r->threads, r->simd_name, r->cls_name, "N/A");
        } else {
            char marker = (r->tok_per_sec == peak) ? '*' : ' ';
            printf("  %-7d  %-13s  %-7s  %8.2f%c\n",
                   r->threads, r->simd_name, r->cls_name,
                   r->tok_per_sec, marker);
        }
    }
    printf("  ─────────────────────────────────────────────────────────────────\n");
    printf("  Peak: %.2f tok/s   (* marks peak row)\n\n", peak);
}

/* ── main ───────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    const char *model_path   = NULL;
    const char *csv_path     = "/tmp/bench_full_results.csv";
    int n_tokens   = DEFAULT_TOKENS;
    int warmup     = DEFAULT_WARMUP;
    int max_threads = DEFAULT_MAX_THREADS;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")       && i+1 < argc) model_path   = argv[++i];
        else if (!strcmp(argv[i], "--tokens")      && i+1 < argc) n_tokens     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")      && i+1 < argc) warmup       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-threads") && i+1 < argc) max_threads  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")         && i+1 < argc) csv_path     = argv[++i];
    }

    if (!model_path) {
        fprintf(stderr,
            "Usage: bench_full --model <path.bin>\n"
            "         [--tokens N (default %d)]\n"
            "         [--warmup N (default %d)]\n"
            "         [--max-threads N (default %d)]\n"
            "         [--csv path]\n",
            DEFAULT_TOKENS, DEFAULT_WARMUP, DEFAULT_MAX_THREADS);
        return 1;
    }

    /* ── Detect hardware ── */
    const TnCpuFeatures *cpu = tn_cpu_features_detect();
    const char *best_simd    = tn_cpu_best_backend_name(cpu);

    /* ── Load model (once, shared across all runs) ── */
    printf("Loading: %s\n", model_path);
    MappedFile mf;
    if (mapped_file_open(&mf, model_path) != TN_OK) {
        fprintf(stderr, "Cannot open model: %s\n", model_path);
        return 1;
    }

    Config cfg;
    if (config_read(&cfg, mf.data, mf.size) != TN_OK) {
        fprintf(stderr, "Cannot read config from %s\n", model_path);
        mapped_file_close(&mf);
        return 1;
    }
    config_print(&cfg);

    /* Use default SIMD for initial alloc */
    const char *saved_backend = getenv("TN_FORCE_BACKEND");
    if (!saved_backend) saved_backend = "";
    setenv("TN_FORCE_BACKEND", best_simd, 1);
    tn_simd_init();
    const TnHardwareProfile *hw = tn_hardware_profile_init();
    (void)hw;

    TransformerWeights w;
    memset(&w, 0, sizeof(w));
    if (weights_alloc_pointers(&w, &cfg) != TN_OK) {
        fprintf(stderr, "OOM in weights_alloc_pointers\n");
        return 1;
    }

    tn_i8 *wdata = (tn_i8 *)mf.data + 64;  /* skip 64-byte header */
    if (weights_map(&w, &cfg, wdata, mf.size - 64) != TN_OK) {
        fprintf(stderr, "weights_map failed\n");
        return 1;
    }

    printf("Model ready.  Sweep: T=1..%d × %d SIMD levels × %d classifiers\n"
           "              tokens=%d  warmup=%d  measured=%d\n\n",
           max_threads, N_SIMD, N_CLS,
           n_tokens, warmup, n_tokens - warmup);

    /* ── Allocate results array ── */
    int max_results = max_threads * N_SIMD * N_CLS;
    BenchResult *results = (BenchResult *)calloc((size_t)max_results, sizeof(BenchResult));
    int nr = 0;

    /* ── Sweep ── */
    for (int t = 1; t <= max_threads; t++) {
        /* Brief cooldown between thread-count groups so turbo boost can
         * recover after the previous group's scalar cells ran hot. */
        if (t > 1) {
            printf("  [cooldown 2s]\n");
            fflush(stdout);
            sleep(2);
        }

        ThreadPool *tp = (t > 1) ? threadpool_create(t) : NULL;

        for (int si = 0; si < N_SIMD; si++) {
            if (!simd_supported(SIMD_LEVELS[si].env_val, cpu)) {
                for (int ci = 0; ci < N_CLS; ci++) {
                    results[nr++] = (BenchResult){
                        t, SIMD_LEVELS[si].name, CLS_NAMES[ci], 0.0, 0 };
                }
                continue;
            }
            activate_simd(SIMD_LEVELS[si].env_val);

            for (int ci = 0; ci < N_CLS; ci++) {
                /* INT4 requires VNNI for meaningful acceleration */
                if (CLS_FMTS[ci] == TN_CLS_INT4 && !cpu->avx512vnni && !cpu->avx_vnni) {
                    results[nr++] = (BenchResult){
                        t, SIMD_LEVELS[si].name, CLS_NAMES[ci], 0.0, 0 };
                    continue;
                }

                tn_hardware_profile_set_classifier(CLS_FMTS[ci]);

                printf("  T=%-2d  %-13s  %-5s ... ", t, SIMD_LEVELS[si].name, CLS_NAMES[ci]);
                fflush(stdout);

                double tps = run_bench(&cfg, &w, n_tokens, warmup, tp);
                printf("%.2f tok/s\n", tps > 0 ? tps : 0.0);

                results[nr++] = (BenchResult){
                    t, SIMD_LEVELS[si].name, CLS_NAMES[ci], tps, (tps > 0)
                };
            }
        }

        if (tp) threadpool_destroy(tp);
    }

    /* ── Restore best backend ── */
    setenv("TN_FORCE_BACKEND", best_simd, 1);
    tn_simd_init();

    print_table(results, nr, model_path, best_simd);
    write_csv(results, nr, csv_path);

    free(results);
    weights_free_pointers(&w);
    mapped_file_close(&mf);
    return 0;
}
