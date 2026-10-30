#include "core/hardware_profile.h"
#include "core/bw_cache.h"
#include "threading/cache_probe.h"
#include "kv_cache/kv_strategy.h"
#include "math/cpu_features.h"
#include "threading/cpu_probe.h"
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if TN_POSIX
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

/*
 * Model constants for BitNet-b1.58-2B-4T:
 *   30 layers, dim=2560, hidden_dim=6912, vocab=128256
 *   Ternary weights: 2 bits/weight packed as 4 weights/byte
 *   Embedding table: vocab * dim * 2 bytes (BF16)
 *
 * 2026-07-17: these constants are only a PRE-LOAD ESTIMATE (profile init
 * runs before the model file is even opened, so no metadata exists yet).
 * They are correct for the native BitNet-2B path they were written for,
 * but were silently used for every model — a 7.16 GB Ternary-Bonsai-27B
 * GGUF got Data/token "1149 MB" and a ~6x-overstated tok/s ceiling (see
 * docs/ai/mistakes.md). main.c now calls
 * tn_hardware_profile_set_model_bytes() with the real loaded-model sizes
 * to correct weight_bytes_per_tok and theoretical_ceiling post-load.
 */
#define MODEL_TERNARY_BYTES   (522ULL * 1024 * 1024)  /* ~522 MB packed ternary */
#define MODEL_EMBED_BYTES     (128256ULL * 2560 * 2)   /* ~628 MB BF16 embedding */
#define MODEL_NORM_BYTES      (1ULL * 1024 * 1024)     /* ~1 MB norms */
#define MODEL_DIM             2560
#define MODEL_VOCAB           128256

/* ── Cache size detection ──────────────────────────────────────────────── */

/* Delegates to threading/cache_probe (Phase 27.1), which reads each sysfs
 * cache index's own level/type instead of assuming index2 = L2 / index1 =
 * L2 fallback (2026-10-18: on typical x86 index1 is the L1 *instruction*
 * cache, so a missing index2 reported L1i as L2) and adds a CPUID path for
 * non-Linux hosts. */
static size_t detect_l2_cache(void) {
    const CacheInfo *ci = tn_cache_info();
    return ci->source ? (size_t)ci->l2_bytes : 0;
}

static size_t detect_l3_cache(void) {
    const CacheInfo *ci = tn_cache_info();
    return ci->source ? (size_t)ci->l3_bytes : 0;
}

/* ── DRAM bandwidth probe ──────────────────────────────────────────────── */

static int64_t bw_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/*
 * Multi-threaded DRAM bandwidth probe.
 *
 * Uses N threads each reading a buffer > L3 cache to measure true DRAM
 * bandwidth (not L3). This is critical for systems with large L3 caches
 * (e.g., Xeon Emerald Rapids with 260 MB L3) where a small single-threaded
 * probe would measure L3 speed instead of DRAM speed.
 *
 * Each thread reads its own buffer sequentially; total is aggregated.
 * 3 passes, best aggregate result returned.
 */

#if TN_POSIX
#include <pthread.h>
#endif

/* Shared by the POSIX multi-threaded probe and the single-threaded
 * fallback — no pthread dependency in the pass function itself. */
typedef struct {
    const unsigned char *buf;
    size_t size;
    double gbps;
    uint64_t sink;   /* per-thread accumulator result, folded by the caller */
} BwThreadArg;

/* Defeats dead-code elimination of the read loops: each thread deposits its
 * accumulator sum into its own arg (no shared writes from workers — a
 * previous version had all threads += into this global directly, a benign
 * but real data race; independent-review finding, 2026-07-17); the caller
 * folds the per-thread sinks in here after join, single-threaded. */
static volatile uint64_t g_bw_probe_sink;

/*
 * One full-buffer read pass. TWO fixes on 2026-07-17 (both had been
 * under-stating measured bandwidth, compounding to ~3x total — see
 * docs/architecture/CEILING_CALCULATION.md §2 and docs/ai/mistakes.md):
 *
 * 1. ACCOUNTING: this function used to run its own 3 passes internally,
 *    while the caller timed the whole create/join round but divided only
 *    ONE pass worth of bytes by that wall time — a systematic ~3x
 *    under-measurement (the correctly-accounted per-thread bytes/elapsed
 *    figures were computed and then discarded). Now each thread runs
 *    exactly one pass per round; the caller's existing 3-round best-of
 *    loop provides the repetition.
 *
 * 2. ACCESS PATTERN: reads every byte via 8 independent 64-bit accumulator
 *    chains (one cache line per iteration) instead of one volatile char
 *    per line. The volatile-scalar loop serialized a 1-byte load + a
 *    volatile store per line and couldn't keep enough line fills in
 *    flight. Independent non-volatile accumulators let the compiler
 *    vectorize and the core overlap misses; the volatile sink store
 *    happens once per pass, outside the timed loop.
 */
#if TN_POSIX
/* First-touch of a probe buffer, run on the probing thread itself so the
 * page faults are taken in parallel (single-threaded touching of the old
 * 2 GB buffer cost up to 15 s on VMs; 2026-10-19). */
static void *bw_touch_fn(void *arg) {
    BwThreadArg *a = (BwThreadArg *)arg;
    unsigned char *m = (unsigned char *)(uintptr_t)a->buf;
    for (size_t j = 0; j < a->size; j += 4096) m[j] = (unsigned char)j;
    return NULL;
}
#endif

static void *bw_thread_fn(void *arg) {
    BwThreadArg *a = (BwThreadArg *)arg;
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0,
             s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    const uint64_t *p = (const uint64_t *)(const void *)a->buf;
    size_t n_words = a->size / 8;
    int64_t t0 = bw_now_ns();
    for (size_t i = 0; i + 8 <= n_words; i += 8) {
        s0 += p[i];     s1 += p[i + 1]; s2 += p[i + 2]; s3 += p[i + 3];
        s4 += p[i + 4]; s5 += p[i + 5]; s6 += p[i + 6]; s7 += p[i + 7];
    }
    int64_t elapsed = bw_now_ns() - t0;
    a->sink = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    a->gbps = (elapsed > 0) ? (double)a->size / (double)elapsed : 0.0;
    return NULL;
}

/* Forward declaration — defined below after /proc/cpuinfo parsing */
static int count_physical_cores(void);

/* Free RAM seen by tn_hardware_profile_init() before the probe runs (caps
 * the probe buffer); 0 = unknown. */
static size_t g_free_ram_hint;

static double probe_dram_bandwidth(void) {
    size_t l3 = detect_l3_cache();
    int n_threads = count_physical_cores();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 8) n_threads = 8;

    /* The buffers together must dwarf the shared L3 to measure DRAM, not
     * cache: total = 4x L3, at least 512 MB, at most 2 GB and a quarter of
     * free RAM. (Was 512 MB per thread regardless — 2 GB on 4 cores, which
     * also showed up as every run's max RSS; 2026-10-19.) */
    size_t total = (l3 > 0) ? 4 * l3 : 0;
    if (total < 512ULL * 1024 * 1024) total = 512ULL * 1024 * 1024;
    if (total > 2048ULL * 1024 * 1024) total = 2048ULL * 1024 * 1024;
    if (g_free_ram_hint > 0 && total > g_free_ram_hint / 4) total = g_free_ram_hint / 4;
    size_t per_thread = (total / (size_t)n_threads) & ~(size_t)4095;
    if (per_thread < 4096) return 0.0;

#if TN_POSIX
    BwThreadArg args[8];
    pthread_t threads[8];

    /* Allocate, then touch all pages from the probing threads */
    for (int i = 0; i < n_threads; i++) {
        unsigned char *mem = (unsigned char *)malloc(per_thread);
        if (!mem) { n_threads = i; break; }
        args[i].buf = mem;
        args[i].size = per_thread;
        args[i].gbps = 0.0;
    }

    if (n_threads == 0) return 0.0;
    for (int i = 0; i < n_threads; i++)
        pthread_create(&threads[i], NULL, bw_touch_fn, &args[i]);
    for (int i = 0; i < n_threads; i++)
        pthread_join(threads[i], NULL);

    /* Measure WALL TIME for all threads to complete ONE pass each — true
     * aggregate bandwidth including bus contention (plus a small
     * thread-create/join overhead, which biases slightly conservative).
     * 5 rounds, best kept (the buffers are smaller than they used to be). */
    size_t total_bytes = per_thread * (size_t)n_threads;
    double best_gbps = 0.0;

    for (int pass = 0; pass < 5; pass++) {
        int64_t wall_t0 = bw_now_ns();
        for (int i = 0; i < n_threads; i++)
            pthread_create(&threads[i], NULL, bw_thread_fn, &args[i]);
        for (int i = 0; i < n_threads; i++)
            pthread_join(threads[i], NULL);
        int64_t wall_elapsed = bw_now_ns() - wall_t0;
        for (int i = 0; i < n_threads; i++)
            g_bw_probe_sink += args[i].sink;   /* single-threaded fold */

        if (wall_elapsed > 0) {
            double gbps = (double)total_bytes / (double)wall_elapsed;
            if (gbps > best_gbps) best_gbps = gbps;
        }
    }

    for (int i = 0; i < n_threads; i++)
        free((void *)args[i].buf);
    return best_gbps;
#else
    /* Fallback: single-threaded probe (same one-pass-per-round
     * multi-accumulator read as bw_thread_fn — see its comment). */
    unsigned char *buf = (unsigned char *)malloc(per_thread);
    if (!buf) return 0.0;
    for (size_t i = 0; i < per_thread; i += 4096)
        buf[i] = (unsigned char)i;

    double best = 0.0;
    for (int pass = 0; pass < 3; pass++) {
        BwThreadArg a = { .buf = buf, .size = per_thread, .gbps = 0.0, .sink = 0 };
        bw_thread_fn(&a);
        g_bw_probe_sink += a.sink;   /* keep the reads observable here too */
        if (a.gbps > best) best = a.gbps;
    }
    free(buf);
    return best;
#endif
}

/* ── Physical core count (duplicated lite version for profile) ─────── */

#if defined(__linux__)
static int count_physical_cores(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;

    int phys[512], core[512], count = 0;
    int cur_phys = -1, cur_core = -1;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "physical id", 11) == 0) {
            char *p = strchr(line, ':');
            if (p) cur_phys = (int)strtol(p + 1, NULL, 10);
        } else if (strncmp(line, "core id", 7) == 0) {
            char *p = strchr(line, ':');
            if (p) cur_core = (int)strtol(p + 1, NULL, 10);
        } else if (line[0] == '\n') {
            if (cur_phys >= 0 && cur_core >= 0 && count < 512) {
                int found = 0;
                for (int i = 0; i < count; i++) {
                    if (phys[i] == cur_phys && core[i] == cur_core) {
                        found = 1; break;
                    }
                }
                if (!found) { phys[count] = cur_phys; core[count] = cur_core; count++; }
            }
            cur_phys = -1; cur_core = -1;
        }
    }
    fclose(f);
    return count;
}

static int count_logical_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}

#elif defined(__APPLE__)
static int count_physical_cores(void) {
    /* Use popen to avoid sys/sysctl.h conflicts with strict C99 mode */
    FILE *f = popen("sysctl -n hw.physicalcpu 2>/dev/null", "r");
    if (f) {
        int n = 0;
        if (fscanf(f, "%d", &n) == 1 && n > 0) { pclose(f); return n; }
        pclose(f);
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 1) ? (int)(n / 2) : 1;
}

static int count_logical_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}

#elif TN_POSIX
static int count_physical_cores(void) { return (int)(sysconf(_SC_NPROCESSORS_ONLN) / 2); }
static int count_logical_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}
#else
static int count_physical_cores(void) { return 1; }
static int count_logical_cores(void) { return 1; }
#endif

/* ── Classifier format selection ───────────────────────────────────────── */

/*
 * Select the optimal classifier quantization based on available SIMD:
 *
 * INT4: Needs VNNI (for dpbusds) + ideally VBMI (for 3-instruction unpack).
 *       Without VBMI, INT4 still works via SSE interleave (slower unpack)
 *       but still 2x less bandwidth than INT8 → net win on bandwidth-bound.
 *
 * INT8: Needs VNNI (dpbusds) or ARM dotprod (vdotq_s32).
 *       4x compute throughput over float FMA.
 *
 * BF16: Fallback when no int8 acceleration is available.
 *       Works on any hardware with float SIMD.
 */
static TnClassifierFormat select_classifier(const TnCpuFeatures *cpu) {
    /*
     * Default classifier: BF16.
     * BF16 provides full precision with zero intelligence loss and runs on
     * all hardware.  INT8 and INT4 are available via --classifier int8/int4.
     *
     * Per-classifier throughput peaks (from benchmark sweeps):
     *   BF16 ≈  26 tok/s  (T=6, VNNI-256)
     *   INT8 ≈  34 tok/s  (T=3, AVX2)
     *   INT4 ≈  48 tok/s  (T=6, VNNI-256)
     */
    (void)cpu;
    return TN_CLS_BF16;
}

/* ── Prefetch tuning ───────────────────────────────────────────────────── */

/*
 * Prefetch distance in rows, tuned to L2 cache size.
 * Each ternary weight row = dim/4 bytes = 640 bytes for dim=2560.
 * We want to prefetch enough rows to hide DRAM latency (~100ns)
 * without evicting useful data from L2.
 *
 * Target: ~25% of L2 for prefetch buffer.
 */
static int select_prefetch_rows(size_t l2_bytes) {
    if (l2_bytes == 0) return 4;  /* safe default */
    size_t row_bytes = MODEL_DIM / 4;  /* packed ternary: 4 weights/byte */
    size_t prefetch_budget = l2_bytes / 4;
    int rows = (int)(prefetch_budget / row_bytes);
    if (rows < 2) rows = 2;
    if (rows > 32) rows = 32;
    return rows;
}

/* ── Cached bandwidth ─────────────────────────────────────────────────── */

static size_t total_ram_bytes(void) {
#if TN_POSIX && defined(_SC_PHYS_PAGES)
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psz > 0) return (size_t)pages * (size_t)psz;
#endif
    return 0;
}

/* Probe result from the per-host cache when the fingerprint matches and the
 * entry is fresh; otherwise probe and store. PZ_BW_REPROBE=1 (set by
 * --calibrate) forces a new measurement. */
static double cached_dram_bandwidth(int physical_cores, int logical_cores, size_t l3_bytes,
                                    bool *from_cache) {
    *from_cache = false;
    char model[128], key[256], path[512];
    tn_bw_cache_cpu_model(model, sizeof(model));
    int have_key = tn_bw_cache_key(key, sizeof(key), model, physical_cores, logical_cores,
                                   l3_bytes, total_ram_bytes()) == 0;
    int have_path = have_key && tn_bw_cache_path(path, sizeof(path)) == 0;
    const char *re = getenv("PZ_BW_REPROBE");
    int reprobe = re && *re && strcmp(re, "0") != 0;
    long long now = (long long)time(NULL);
    double gbps = 0.0;
    if (have_path && !reprobe &&
        tn_bw_cache_load(path, key, now, TN_BW_CACHE_MAX_AGE_S, &gbps)) {
        *from_cache = true;
        return gbps;
    }
    gbps = probe_dram_bandwidth();
    if (have_path && gbps > 0.0) (void)tn_bw_cache_save(path, key, gbps, now);
    return gbps;
}

/* ── Main initialization ──────────────────────────────────────────────── */

static TnHardwareProfile g_profile;
static bool g_initialized = false;

static void rebuild_summary(void) {
    const char *cls_name = "BF16";
    if (g_profile.classifier_fmt == TN_CLS_INT8) cls_name = "INT8";
    if (g_profile.classifier_fmt == TN_CLS_INT4) cls_name = "INT4";

    snprintf(g_profile.summary, sizeof(g_profile.summary),
             "%s | %s cls | %dT | %.1f GB/s | ceiling %.0f tok/s",
             g_profile.cpu->best_backend,
             cls_name,
             g_profile.optimal_threads,
             g_profile.measured_bw_gbps,
             g_profile.theoretical_ceiling);
}

const TnHardwareProfile *tn_hardware_profile_init(void) {
    if (g_initialized) return &g_profile;

    memset(&g_profile, 0, sizeof(g_profile));

    /* 1. CPU features */
    g_profile.cpu = tn_cpu_features_detect();

    /* 2. Core topology */
    g_profile.physical_cores = count_physical_cores();
    g_profile.logical_cores  = count_logical_cores();
    if (g_profile.physical_cores <= 0)
        g_profile.physical_cores = g_profile.logical_cores;
    g_profile.optimal_threads = tn_get_optimal_thread_count();

    /* 3. Cache sizes */
    g_profile.l2_cache_bytes = detect_l2_cache();
    g_profile.l3_cache_bytes = detect_l3_cache();

    /* 4. RAM — use MemAvailable from /proc/meminfo (same as kv_strategy.c).
     *
     * sysinfo().freeram = MemFree: only unowned pages (~0.3–2 GB on a loaded system).
     * MemAvailable = MemFree + reclaimable buffers/cache: what the kernel will actually
     * hand to a new allocation.  Using MemFree caused KV strategy to see ~2 GB on a
     * 16 GB laptop and fall into KV_SLIDING_I4 (1024 ctx) instead of KV_QUANT_I8
     * (full 4096 ctx), collapsing effective context window unnecessarily.
     */
#if TN_POSIX
    {
        tn_i64 mem_avail = tn_get_free_ram();   /* reads MemAvailable */
        if (mem_avail > 0) {
            g_profile.free_ram_bytes = (size_t)mem_avail;
        } else {
#if defined(__linux__)
            /* Fallback to sysinfo MemFree if /proc/meminfo is unavailable */
            struct sysinfo si;
            if (sysinfo(&si) == 0)
                g_profile.free_ram_bytes = (size_t)si.freeram * si.mem_unit;
#elif defined(__APPLE__)
            /* macOS: parse "vm_stat" for free + inactive pages */
            {
                FILE *vf = popen("vm_stat 2>/dev/null", "r");
                long page_sz = sysconf(_SC_PAGESIZE);
                if (!page_sz || page_sz < 0) page_sz = 4096;
                if (vf && page_sz > 0) {
                    char vline[128];
                    size_t free_p = 0, inactive_p = 0;
                    while (fgets(vline, sizeof(vline), vf)) {
                        size_t v = 0;
                        if (sscanf(vline, "Pages free: %zu.", &v) == 1)       free_p += v;
                        if (sscanf(vline, "Pages inactive: %zu.", &v) == 1)   inactive_p += v;
                    }
                    pclose(vf);
                    g_profile.free_ram_bytes = (free_p + inactive_p) * (size_t)page_sz;
                } else if (vf) { pclose(vf); }
            }
#else
            /* Generic POSIX fallback — _SC_PHYS_PAGES is widely supported */
            {
                long pages = sysconf(_SC_PHYS_PAGES);
                long page_sz = sysconf(_SC_PAGESIZE);
                if (pages > 0 && page_sz > 0)
                    g_profile.free_ram_bytes = (size_t)pages * (size_t)page_sz / 2;
            }
#endif
        }
    }
#endif

    /* 5. Bandwidth: per-host cache, else probe (core/bw_cache.h) */
    g_free_ram_hint = g_profile.free_ram_bytes;
    g_profile.measured_bw_gbps = cached_dram_bandwidth(g_profile.physical_cores, g_profile.logical_cores,
                                                       g_profile.l3_cache_bytes, &g_profile.bw_cached);

    /* 6. Classifier selection */
    g_profile.classifier_fmt = select_classifier(g_profile.cpu);

    /* 7. Prefetch tuning */
    g_profile.prefetch_rows = select_prefetch_rows(g_profile.l2_cache_bytes);

    /* 8. NT stores: useful when output is large and won't be read soon */
    g_profile.use_nt_stores = false;  /* conservative default */

    /* 9. Model-in-cache check */
    g_profile.model_fits_l3 = (g_profile.l3_cache_bytes >= MODEL_TERNARY_BYTES);

    /* 10. Performance projections */
    double cls_bytes;
    switch (g_profile.classifier_fmt) {
        case TN_CLS_INT4: cls_bytes = (double)MODEL_VOCAB * MODEL_DIM / 2.0; break;
        case TN_CLS_INT8: cls_bytes = (double)MODEL_VOCAB * MODEL_DIM;       break;
        default:          cls_bytes = (double)MODEL_VOCAB * MODEL_DIM * 2.0;  break;
    }
    g_profile.cls_bytes_per_tok = cls_bytes;
    g_profile.weight_bytes_per_tok = (double)MODEL_TERNARY_BYTES + cls_bytes +
                                      (double)MODEL_NORM_BYTES;

    if (g_profile.measured_bw_gbps > 0.0) {
        /*
         * L3-aware ceiling estimate (pre-load only; the post-load
         * set_model_bytes() correction uses plain bandwidth division).
         *
         * In steady-state autoregressive inference, L3 retains recently
         * accessed weights between tokens. The effective DRAM read per
         * token depends on how much weight data fits in L3.
         *
         * Heuristic (2026-07-17: comment corrected to match the code — it
         * previously described a "~half stay cached" model that only ever
         * existed in calibration.c's own copy):
         *  1. Classifier counted fully cached if cls_bytes < L3
         *  2. Ternary layers: min(L3 - classifier, ternary_bytes) counted
         *     cached; the remainder billed to DRAM
         *
         * Because the cache credit REDUCES billed DRAM bytes, it RAISES the
         * projected ceiling — this is an optimistic estimate of the
         * bandwidth bound, not a lower bound on performance (the previous
         * comment had that backwards; measured throughput "exceeding the
         * ceiling" was in fact a symptom of the probe under-measuring
         * bandwidth, fixed the same day — see bw_thread_fn above).
         */
        double l3_usable = (double)g_profile.l3_cache_bytes;
        double dram_per_tok = g_profile.weight_bytes_per_tok;

        if (l3_usable > 0.0 && l3_usable < dram_per_tok) {
            /* Classifier: fully cached if it fits in L3 */
            double cls_cached = 0.0;
            if (cls_bytes < l3_usable) {
                cls_cached = cls_bytes;
            }
            /* Ternary layers: cache (L3 - classifier) worth of layers */
            double l3_for_layers = l3_usable - cls_cached;
            double ternary_cached = 0.0;
            if (l3_for_layers > 0.0 && l3_for_layers < (double)MODEL_TERNARY_BYTES) {
                ternary_cached = l3_for_layers;
            } else if (l3_for_layers >= (double)MODEL_TERNARY_BYTES) {
                ternary_cached = (double)MODEL_TERNARY_BYTES;
            }
            dram_per_tok -= (cls_cached + ternary_cached);
        } else if (l3_usable >= dram_per_tok) {
            dram_per_tok = 0.0;
        }

        if (dram_per_tok > 0.0) {
            double seconds_per_tok = dram_per_tok /
                                      (g_profile.measured_bw_gbps * 1e9);
            g_profile.theoretical_ceiling = 1.0 / seconds_per_tok;
        } else {
            g_profile.theoretical_ceiling = 999.0;  /* compute-limited */
        }
    }

    /* 11. Summary string */
    rebuild_summary();

    g_initialized = true;
    return &g_profile;
}

const TnHardwareProfile *tn_hardware_profile_get(void) {
    if (!g_initialized) return NULL;
    return &g_profile;
}

void tn_hardware_profile_set_classifier(TnClassifierFormat fmt) {
    if (!g_initialized) return;

    g_profile.classifier_fmt = fmt;
    g_profile.classifier_explicit = true;

    /* Recalculate data-per-token and ceiling */
    double cls_bytes;
    switch (fmt) {
        case TN_CLS_INT4: cls_bytes = (double)MODEL_VOCAB * MODEL_DIM / 2.0; break;
        case TN_CLS_INT8: cls_bytes = (double)MODEL_VOCAB * MODEL_DIM;       break;
        default:          cls_bytes = (double)MODEL_VOCAB * MODEL_DIM * 2.0;  break;
    }
    g_profile.cls_bytes_per_tok = cls_bytes;
    g_profile.weight_bytes_per_tok = (double)MODEL_TERNARY_BYTES + cls_bytes +
                                      (double)MODEL_NORM_BYTES;

    if (g_profile.measured_bw_gbps > 0.0) {
        double seconds_per_tok = g_profile.weight_bytes_per_tok /
                                  (g_profile.measured_bw_gbps * 1e9);
        g_profile.theoretical_ceiling = 1.0 / seconds_per_tok;
    }

    rebuild_summary();
}

void tn_hardware_profile_set_model_bytes(double weight_bytes_per_tok,
                                          double cls_bytes_per_tok) {
    if (!g_initialized || weight_bytes_per_tok <= 0.0) return;

    g_profile.weight_bytes_per_tok = weight_bytes_per_tok;
    g_profile.cls_bytes_per_tok = cls_bytes_per_tok;
    g_profile.model_fits_l3 =
        ((double)g_profile.l3_cache_bytes >= weight_bytes_per_tok);

    if (g_profile.measured_bw_gbps > 0.0) {
        /* Plain bandwidth division, no L3-retention heuristic: the init-time
         * heuristic was tuned for a ~1.2 GB BitNet working set; for the
         * multi-GB GGUF models this correction exists for, L3 covers well
         * under 1% of the per-token stream and the adjustment is noise. */
        g_profile.theoretical_ceiling =
            (g_profile.measured_bw_gbps * 1e9) / weight_bytes_per_tok;
    }

    rebuild_summary();
}

/* One row of the report box: label + value, padded to the frame's inner width so the right
 * border lines up whatever the value's length (2026-10-30: rows used hand-counted widths and
 * the border was ragged). */
#define TN_HW_BOX_INNER 53
#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
static void hw_box_row(const char *label, const char *fmt, ...) {
    char value[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(value, sizeof value, fmt, ap);
    va_end(ap);
    /* " " + 14-char label + ": " = 17 columns, the value fills the rest. */
    printf("│ %-14s: %-*.*s│\n", label, TN_HW_BOX_INNER - 17, TN_HW_BOX_INNER - 17, value);
}

void tn_hardware_profile_report(const TnHardwareProfile *hp) {
    printf("┌─────────────────────────────────────────────────────┐\n");
    printf("│          Hardware Profile (auto-detected)           │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    hw_box_row("SIMD backend", "%s", hp->cpu->best_backend);
    hw_box_row("Cores", "%d physical / %d logical", hp->physical_cores, hp->logical_cores);
    hw_box_row("Threads", "%d", hp->optimal_threads);
    hw_box_row("L2 cache", "%-6zu KiB per core", hp->l2_cache_bytes / 1024);
    hw_box_row("L3 cache", "%-6zu MiB shared", hp->l3_cache_bytes / (1024 * 1024));
    hw_box_row("Free RAM", "%-6zu MiB", hp->free_ram_bytes / (1024 * 1024));
    hw_box_row("DRAM bandwidth", "%-6.1f GB/s %s", hp->measured_bw_gbps,
               hp->bw_cached ? "(cached)" : "(measured)");
    printf("├─────────────────────────────────────────────────────┤\n");
    const char *cls_name = "BF16 (full precision)";
    if (hp->classifier_fmt == TN_CLS_INT8) cls_name = "INT8 (1 byte/weight)";
    if (hp->classifier_fmt == TN_CLS_INT4) cls_name = "INT4 (0.5 byte/weight)";
    hw_box_row("Classifier", "%s", cls_name);
    hw_box_row("Prefetch rows", "%d", hp->prefetch_rows);
    hw_box_row("Model in L3", "%s", hp->model_fits_l3 ? "YES" : "no");
    /* Data/token + Ceiling are a pre-load estimate at this point in the
     * startup sequence (this report prints before the model file is read);
     * main.c re-derives both from the real loaded model and prints the
     * corrected figures after the weights are loaded. */
    hw_box_row("Data/token", "%-6.0f MB (pre-load est.)", hp->weight_bytes_per_tok / (1024.0 * 1024.0));
    hw_box_row("Ceiling", "%-6.1f tok/s (at 100%% BW)", hp->theoretical_ceiling);
    printf("└─────────────────────────────────────────────────────┘\n");
}
