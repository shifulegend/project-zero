#ifndef TN_BW_CACHE_H
#define TN_BW_CACHE_H

/*
 * core/bw_cache — per-host cache of the DRAM bandwidth probe.
 *
 * The probe (hardware_profile.c) streams a buffer several times larger than
 * L3; first-touching it costs 1–15 s on VMs and inflates every run's max RSS
 * (2026-10-19, found by the head-to-head sweep). The measured value only
 * changes with the hardware, so it is cached keyed by a host fingerprint
 * (CPU model, cores, L3, total RAM) and re-probed when the fingerprint
 * changes, the entry is older than TN_BW_CACHE_MAX_AGE_S, `--calibrate` runs,
 * or PZ_BW_REPROBE=1 is set.
 *
 * Location follows the update-check cache: $PZ_CACHE_DIR/membw,
 * $XDG_CACHE_HOME/project-zero/membw, ~/.cache/project-zero/membw.
 * File format (one line, text): "pz-membw 1\t<key>\t<gbps>\t<unix time>".
 */

#include <stddef.h>

#define TN_BW_CACHE_MAX_AGE_S (30L * 24 * 3600)   /* re-measure monthly */

/* Host fingerprint string. Returns 0, or -1 if buf is too small. */
int tn_bw_cache_key(char *buf, size_t n, const char *cpu_model, int physical_cores,
                    int logical_cores, size_t l3_bytes, size_t total_ram_bytes);

/* CPU model name (/proc/cpuinfo "model name"; "" where unavailable). */
void tn_bw_cache_cpu_model(char *buf, size_t n);

/* Default cache file path. Returns 0, or -1 when no cache dir is available. */
int tn_bw_cache_path(char *buf, size_t n);

/* 1 and *gbps set when path holds a valid entry for key no older than
 * max_age_s at time now; 0 otherwise (missing, stale, other host, malformed). */
int tn_bw_cache_load(const char *path, const char *key, long long now, long max_age_s,
                     double *gbps);

/* Writes the entry atomically (tmp + rename). Returns 0 or -1. */
int tn_bw_cache_save(const char *path, const char *key, double gbps, long long now);

#endif /* TN_BW_CACHE_H */
