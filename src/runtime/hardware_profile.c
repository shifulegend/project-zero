#include "runtime/hardware_profile.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/bw_cache.h"
#include "core/cpu_features.h"
#include "core/sys_probe.h"
#include "kv_cache/kv_strategy.h"
#include "runtime/hw_model_constants.h"
#include "threading/cache_probe.h"
#include "threading/cpu_probe.h"

#if TN_POSIX
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/sysinfo.h>
#endif


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

/* Forward declaration — defined below after /proc/cpuinfo parsing */
static int probe_physical_cores(void);

static size_t g_free_ram_hint;

static double probe_dram_bandwidth(void) {
    size_t l3     = detect_l3_cache();
    int n_threads = probe_physical_cores();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 8) n_threads = 8;
    size_t per_thread = tn_hw_bw_buffer_bytes(l3, g_free_ram_hint, n_threads);
    if (per_thread == 0) return 0.0;
    return tn_hw_probe_bandwidth(n_threads, per_thread);
}

/* ── Physical core count (duplicated lite version for profile) ─────── */

#if defined(__linux__)
static int probe_physical_cores(void) {
    return tn_sys_physical_cores(NULL);
}

static int count_logical_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}

#elif defined(__APPLE__)
static int probe_physical_cores(void) {
    /* Use popen to avoid sys/sysctl.h conflicts with strict C99 mode */
    FILE *f = popen("sysctl -n hw.physicalcpu 2>/dev/null", "r");
    if (f) {
        int n = 0;
        if (fscanf(f, "%d", &n) == 1 && n > 0) {
            pclose(f);
            return n;
        }
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
static int probe_physical_cores(void) {
    return (int)(sysconf(_SC_NPROCESSORS_ONLN) / 2);
}
static int count_logical_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
}
#else
static int probe_physical_cores(void) {
    return 1;
}
static int count_logical_cores(void) {
    return 1;
}
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
    if (l2_bytes == 0) return 4;            /* safe default */
    size_t row_bytes       = MODEL_DIM / 4; /* packed ternary: 4 weights/byte */
    size_t prefetch_budget = l2_bytes / 4;
    int rows               = (int)(prefetch_budget / row_bytes);
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
    int have_key = tn_bw_cache_key(key, sizeof(key), model, physical_cores, logical_cores, l3_bytes,
                                   total_ram_bytes()) == 0;
    int have_path = have_key && tn_bw_cache_path(path, sizeof(path)) == 0;
    int reprobe   = tn_sys_env_flag(NULL, "PZ_BW_REPROBE");
    long long now = (long long)time(NULL);
    double gbps   = 0.0;
    if (have_path && !reprobe && tn_bw_cache_load(path, key, now, TN_BW_CACHE_MAX_AGE_S, &gbps)) {
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

/* Free RAM — use MemAvailable from /proc/meminfo (same as kv_strategy.c).
 *
 * sysinfo().freeram = MemFree: only unowned pages (~0.3–2 GB on a loaded system).
 * MemAvailable = MemFree + reclaimable buffers/cache: what the kernel will actually
 * hand to a new allocation.  Using MemFree caused KV strategy to see ~2 GB on a
 * 16 GB laptop and fall into KV_SLIDING_I4 (1024 ctx) instead of KV_QUANT_I8
 * (full 4096 ctx), collapsing effective context window unnecessarily.
 */
#if TN_POSIX && defined(__APPLE__)
/* macOS: parse "vm_stat" for free + inactive pages */
static size_t free_ram_vm_stat(void) {
    FILE *vf     = popen("vm_stat 2>/dev/null", "r");
    long page_sz = sysconf(_SC_PAGESIZE);
    if (!page_sz || page_sz < 0) page_sz = 4096;
    if (!vf) return 0;
    char vline[128];
    size_t free_p = 0, inactive_p = 0;
    while (fgets(vline, sizeof(vline), vf)) {
        size_t v = 0;
        if (sscanf(vline, "Pages free: %zu.", &v) == 1) free_p += v;
        if (sscanf(vline, "Pages inactive: %zu.", &v) == 1) inactive_p += v;
    }
    pclose(vf);
    return (free_p + inactive_p) * (size_t)page_sz;
}
#endif

#if TN_POSIX
/* When MemAvailable is unreadable: the best per-OS estimate. */
static size_t free_ram_fallback(void) {
#if defined(__linux__)
    /* Fallback to sysinfo MemFree if /proc/meminfo is unavailable */
    struct sysinfo si;
    if (sysinfo(&si) == 0) return (size_t)si.freeram * si.mem_unit;
    return 0;
#elif defined(__APPLE__)
    return free_ram_vm_stat();
#else
    /* Generic POSIX fallback — _SC_PHYS_PAGES is widely supported */
    long pages   = sysconf(_SC_PHYS_PAGES);
    long page_sz = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_sz > 0) return (size_t)pages * (size_t)page_sz / 2;
    return 0;
#endif
}
#endif

static size_t detect_free_ram(void) {
#if TN_POSIX
    tn_i64 mem_avail = tn_get_free_ram(); /* reads MemAvailable */
    if (mem_avail > 0) return (size_t)mem_avail;
    return free_ram_fallback();
#else
    return 0;
#endif
}

static void rebuild_summary(void) {
    const char *cls_name = "BF16";
    if (g_profile.classifier_fmt == TN_CLS_INT8) cls_name = "INT8";
    if (g_profile.classifier_fmt == TN_CLS_INT4) cls_name = "INT4";

    snprintf(g_profile.summary, sizeof(g_profile.summary),
             "%s | %s cls | %dT | %.1f GB/s | ceiling %.0f tok/s", g_profile.cpu->best_backend,
             cls_name, g_profile.optimal_threads, g_profile.measured_bw_gbps,
             g_profile.theoretical_ceiling);
}

const TnHardwareProfile *tn_hardware_profile_init(void) {
    if (g_initialized) return &g_profile;

    memset(&g_profile, 0, sizeof(g_profile));

    /* 1. CPU features */
    g_profile.cpu = tn_cpu_features_detect();

    /* 2. Core topology */
    g_profile.physical_cores = probe_physical_cores();
    g_profile.logical_cores  = count_logical_cores();
    if (g_profile.physical_cores <= 0) g_profile.physical_cores = g_profile.logical_cores;
    g_profile.optimal_threads = tn_get_optimal_thread_count();

    /* 3. Cache sizes */
    g_profile.l2_cache_bytes = detect_l2_cache();
    g_profile.l3_cache_bytes = detect_l3_cache();

    /* 4. RAM: MemAvailable, with per-OS fallbacks (see detect_free_ram) */
    g_profile.free_ram_bytes = detect_free_ram();

    /* 5. Bandwidth: per-host cache, else probe (core/bw_cache.h) */
    g_free_ram_hint = g_profile.free_ram_bytes;
    g_profile.measured_bw_gbps =
        cached_dram_bandwidth(g_profile.physical_cores, g_profile.logical_cores,
                              g_profile.l3_cache_bytes, &g_profile.bw_cached);

    /* 6. Classifier selection */
    g_profile.classifier_fmt = select_classifier(g_profile.cpu);

    /* 7. Prefetch tuning */
    g_profile.prefetch_rows = select_prefetch_rows(g_profile.l2_cache_bytes);

    /* 8. NT stores: useful when output is large and won't be read soon */
    g_profile.use_nt_stores = false; /* conservative default */

    /* 9. Model-in-cache check */
    g_profile.model_fits_l3 = (g_profile.l3_cache_bytes >= MODEL_TERNARY_BYTES);

    /* 10. Performance projections */
    double cls_bytes            = tn_hw_classifier_bytes(g_profile.classifier_fmt);
    g_profile.cls_bytes_per_tok = cls_bytes;
    g_profile.weight_bytes_per_tok =
        (double)MODEL_TERNARY_BYTES + cls_bytes + (double)MODEL_NORM_BYTES;

    if (g_profile.measured_bw_gbps > 0.0)
        g_profile.theoretical_ceiling =
            tn_hw_cached_ceiling((double)g_profile.l3_cache_bytes, g_profile.measured_bw_gbps,
                                 g_profile.weight_bytes_per_tok, cls_bytes);

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

    g_profile.classifier_fmt      = fmt;
    g_profile.classifier_explicit = true;

    /* Recalculate data-per-token and ceiling */
    double cls_bytes            = tn_hw_classifier_bytes(fmt);
    g_profile.cls_bytes_per_tok = cls_bytes;
    g_profile.weight_bytes_per_tok =
        (double)MODEL_TERNARY_BYTES + cls_bytes + (double)MODEL_NORM_BYTES;

    if (g_profile.measured_bw_gbps > 0.0) {
        double seconds_per_tok =
            g_profile.weight_bytes_per_tok / (g_profile.measured_bw_gbps * 1e9);
        g_profile.theoretical_ceiling = 1.0 / seconds_per_tok;
    }

    rebuild_summary();
}

TnClassifierFormat tn_hardware_profile_swap_classifier(TnClassifierFormat fmt) {
    if (!g_initialized) return TN_CLS_BF16;
    TnClassifierFormat old   = g_profile.classifier_fmt;
    g_profile.classifier_fmt = fmt;
    return old;
}

void tn_hardware_profile_set_model_bytes(double weight_bytes_per_tok, double cls_bytes_per_tok) {
    if (!g_initialized || weight_bytes_per_tok <= 0.0) return;

    g_profile.weight_bytes_per_tok = weight_bytes_per_tok;
    g_profile.cls_bytes_per_tok    = cls_bytes_per_tok;
    g_profile.model_fits_l3        = ((double)g_profile.l3_cache_bytes >= weight_bytes_per_tok);

    if (g_profile.measured_bw_gbps > 0.0) {
        /* Plain bandwidth division, no L3-retention heuristic: the init-time
         * heuristic was tuned for a ~1.2 GB BitNet working set; for the
         * multi-GB GGUF models this correction exists for, L3 covers well
         * under 1% of the per-token stream and the adjustment is noise. */
        g_profile.theoretical_ceiling = (g_profile.measured_bw_gbps * 1e9) / weight_bytes_per_tok;
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
static void
hw_box_row(const char *label, const char *fmt, ...) {
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
    hw_box_row("Data/token", "%-6.0f MB (pre-load est.)",
               hp->weight_bytes_per_tok / (1024.0 * 1024.0));
    hw_box_row("Ceiling", "%-6.1f tok/s (at 100%% BW)", hp->theoretical_ceiling);
    printf("└─────────────────────────────────────────────────────┘\n");
}
