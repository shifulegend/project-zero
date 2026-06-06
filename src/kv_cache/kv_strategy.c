#include "kv_cache/kv_strategy.h"
#include "core/platform.h"
#include "core/config.h"

#include <stdio.h>

#if TN_POSIX
#include <unistd.h>
#endif

#if TN_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

/*
 * RAM threshold constants — calibrated for MemAvailable (not MemFree).
 *
 * MemAvailable on a loaded 16 GB desktop is typically 6–12 GB.
 * MemFree on the same machine is typically 0.3–1 GB.
 *
 * K-3 threshold correction (Phase 16-S.6):
 *   The original GB_12 threshold (12 GB) was too high for 16 GB systems where
 *   MemAvailable is typically 7–9 GB after the OS + model load.  This caused
 *   a 16 GB machine with 8 GB available to fall into KV_SLIDING_I4 instead of
 *   the superior KV_SLIDING_I8 or KV_QUANT_I8, losing context window quality.
 *
 *   New thresholds (calibrated for MemAvailable, all system sizes):
 *   > 32 GB avail  →  KV_FULL_F32   (server / workstation: ~600 MB KV @ 4096 ctx)
 *   >  8 GB avail  →  KV_QUANT_I8   (≥16 GB desktop after load: full ctx, int8)
 *   >  5 GB avail  →  KV_SLIDING_I8 (8–16 GB desktop: sliding 1024, int8)
 *   >  2 GB avail  →  KV_SLIDING_I4 (4–8 GB: sliding 1024, int4)
 *   <= 2 GB avail  →  KV_SLIDING_I4 (minimal: 512 ctx)
 *
 * THROUGHPUT NOTE: KV_SLIDING_I4 is preferred over I8 on bandwidth-limited hardware.
 * On a bandwidth-constrained laptop (i5-11300H, 13.6 GB/s DRAM), I8 @ 1024 ctx
 * costs ~1–2 tok/s vs I4 @ same ctx.  Only use I8 when you have the bandwidth.
 *
 * KV cache allocation = n_layers * n_kv_heads * max_seq_len * head_dim * sizeof(float).
 * For BitNet b1.58-2B-4T: 24 * 8 * max_seq * 128 * 2 bytes (×2 for K+V).
 *   @ 4096 ctx: ~504 MB — exceeds L3, causes page-fault overhead at startup
 *   @ 1024 ctx: ~126 MB — 4× less startup cost and 4× less attention bandwidth
 *
 * Performance cliff: full QUANT_I8 @ 4096 context drops from ~15 tok/s to ~2.4 tok/s
 * on a 16 GB laptop (i5-11300H) due to calloc touching all pages at init.
 */
#define GB_32 ((tn_i64)32 * 1024 * 1024 * 1024)
#define GB_8  ((tn_i64) 8 * 1024 * 1024 * 1024)   /* K-3: was GB_12 (too high for 16 GB) */
#define GB_5  ((tn_i64) 5 * 1024 * 1024 * 1024)   /* K-3: was GB_10 */
#define GB_2  ((tn_i64) 2 * 1024 * 1024 * 1024)

/* Default sliding window sizes */
#define SW_WINDOW_LARGE 1024
#define SW_WINDOW_SMALL 512

tn_i64 tn_get_free_ram(void) {
#if TN_POSIX
  /*
   * Use MemAvailable from /proc/meminfo, not _SC_AVPHYS_PAGES.
   *
   * _SC_AVPHYS_PAGES returns MemFree — only pages with no current owner.
   * On a typical Linux system with 16 GB RAM and active page cache, MemFree
   * may read 400–700 MB while MemAvailable (what the kernel will actually
   * hand to a new allocation, including reclaimable buffers/cache) reads
   * 7–14 GB. Using MemFree causes the KV strategy selector to pick I4
   * Sliding Window on systems that have ample RAM, collapsing throughput
   * from ~15 tok/s to ~1.8 tok/s.
   *
   * MemAvailable was added in Linux 3.14 (2014) and is always present on
   * any kernel we care about. Fall back to _SC_AVPHYS_PAGES only if the
   * parse fails.
   */
  FILE *f = fopen("/proc/meminfo", "r");
  if (f) {
    char line[128];
    while (fgets(line, sizeof(line), f)) {
      long long kb = 0;
      if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) {
        fclose(f);
        return (tn_i64)kb * 1024LL;
      }
    }
    fclose(f);
  }
  /* Fallback: use page count * page size.
   * _SC_AVPHYS_PAGES (free pages) is Linux/glibc; on macOS use _SC_PHYS_PAGES
   * (total) as a conservative over-estimate — the vm_stat path above is the
   * accurate macOS path, so reaching here means vm_stat also failed. */
#if defined(_SC_AVPHYS_PAGES)
  long pages = sysconf(_SC_AVPHYS_PAGES);
#else
  long pages = sysconf(_SC_PHYS_PAGES);  /* macOS fallback — total, not free */
#endif
  long page_size = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page_size > 0) {
    return (tn_i64)pages * (tn_i64)page_size;
  }
  return -1;
#elif TN_WIN32
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms)) {
    return (tn_i64)ms.ullAvailPhys;
  }
  return -1;
#else
  return -1;
#endif
}

KVStrategyResult select_kv_strategy(const Config *cfg, tn_i64 free_ram) {
  KVStrategyResult result;
  int seq_len = cfg->seq_len;

  if (free_ram > GB_32) {
    /* Server / workstation (>32 GB avail) — full float32 KV cache */
    result.strategy = KV_FULL_F32;
    result.max_seq_len = seq_len;
  } else if (free_ram > GB_8) {
    /* ≥16 GB desktop with >8 GB avail — full context, int8 KV quality.
     * K-3 fix: was GB_12 which excluded 16 GB machines (typically 7–9 GB avail).
     * Lowering to GB_8 allows 16 GB systems to use full context correctly. */
    result.strategy = KV_QUANT_I8;
    result.max_seq_len = seq_len;
  } else if (free_ram > GB_5) {
    /* 8–16 GB machines with 5–8 GB avail — sliding int8, 1024 ctx.
     * K-3 fix: was GB_10 — same rationale, range adjusted down. */
    result.strategy = KV_SLIDING_I8;
    result.max_seq_len = seq_len < SW_WINDOW_LARGE ? seq_len : SW_WINDOW_LARGE;
  } else if (free_ram > GB_2) {
    /* 4–8 GB machines — sliding int4, 1024 ctx.
     * I4 uses half the KV bandwidth — faster on bandwidth-limited hardware. */
    result.strategy = KV_SLIDING_I4;
    result.max_seq_len = seq_len < SW_WINDOW_LARGE ? seq_len : SW_WINDOW_LARGE;
  } else {
    /* Very low RAM (<2 GB avail) — sliding int4, 512 ctx */
    result.strategy = KV_SLIDING_I4;
    result.max_seq_len = seq_len < 512 ? seq_len : 512;
  }

  /*
   * Safety cap: compute the actual F32 KV cache cost for the chosen max_seq_len
   * and reduce it if the allocation would exceed 60% of free_ram.
   * This guards against models with large native seq_len (e.g. DeepSeek 163840)
   * on machines that have enough RAM to pass the GB_8 threshold but not enough
   * to hold the resulting cache (27 layers × 16 heads × 163840 × 128 × 4B × 2).
   *
   * Per-token KV cost (F32, K+V combined):
   *   n_layers * n_kv_heads * head_dim * 2 * sizeof(float)
   */
  {
    int head_dim = config_head_dim(cfg);
    tn_i64 per_token = (tn_i64)cfg->n_layers
                     * (tn_i64)(cfg->n_kv_heads > 0 ? cfg->n_kv_heads : cfg->n_heads)
                     * (tn_i64)head_dim * 2 * (tn_i64)sizeof(float);
    if (per_token > 0) {
      tn_i64 budget        = (free_ram / 10) * 6;  /* 60% of free_ram */
      tn_i64 max_safe_ctx  = budget / per_token;
      if (max_safe_ctx < 512) max_safe_ctx = 512;  /* never below 512 */
      if (result.max_seq_len > (int)max_safe_ctx)
        result.max_seq_len = (int)max_safe_ctx;
    }
  }

  return result;
}

const char *kv_strategy_name(KVStrategy s) {
  switch (s) {
  case KV_FULL_F32:
    return "Full F32";
  case KV_QUANT_I8:
    return "Quantized I8";
  case KV_QUANT_I4:
    return "Quantized I4";
  case KV_SLIDING_I8:
    return "Sliding Window I8";
  case KV_SLIDING_I4:
    return "Sliding Window I4";
  default:
    return "Unknown";
  }
}
