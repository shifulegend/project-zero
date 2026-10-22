#ifndef TN_CACHE_PROBE_H
#define TN_CACHE_PROBE_H

/*
 * cache_probe.h — Phase 27.1: CPU cache geometry.
 *
 * Sources, in order: Linux sysfs (/sys/devices/system/cpu/cpu0/cache/index*),
 * x86 CPUID leaf 4 (deterministic cache parameters), then documented
 * fallbacks (L1d 32 KiB, L2 1 MiB, L3 0 = unknown, line 64 B). Results are
 * probed once and cached. Used to size cache-blocked kernels (tiled
 * attention) and reported by GET /v1/config.
 */

typedef struct {
    int l1d_bytes;
    int l2_bytes;
    long long l3_bytes;  /* can exceed 2 GiB-ish on big server parts; 0 = unknown */
    int line_bytes;
    int source;          /* 0 = fallback, 1 = sysfs, 2 = cpuid */
} CacheInfo;

#define CACHE_PROBE_FALLBACK_L1D  (32 * 1024)
#define CACHE_PROBE_FALLBACK_L2   (1024 * 1024)
#define CACHE_PROBE_FALLBACK_LINE 64

/* Probed once (thread-safe after the first call returns). */
const CacheInfo *tn_cache_info(void);

/* Parses a sysfs size string ("48K", "2048K", "260M", "64") into bytes; -1 on error. */
long long tn_cache_parse_size(const char *s);

/* Number of rows of `row_bytes` that fit in `fraction` of L1d (>= 1).
 * fraction <= 0 means 0.5 (leave room for the query/accumulators). */
int tn_cache_rows_fitting_l1(int row_bytes, double fraction);

/* "sysfs" | "cpuid" | "fallback" */
const char *tn_cache_source_name(int source);

#endif /* TN_CACHE_PROBE_H */
