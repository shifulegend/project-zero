#include "kv_cache/kv_strategy.h"
#include "memory/file_backed.h"
#include "core/platform.h"
#include "core/config.h"

#include <stdio.h>
#include <string.h>

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

static KVStrategyOptions g_kv_opts; /* zero = automatic */

void kv_strategy_set_options(const KVStrategyOptions *opt) {
  if (opt) g_kv_opts = *opt;
  else memset(&g_kv_opts, 0, sizeof(g_kv_opts));
}

int kv_strategy_kv_type(void) { return g_kv_opts.kv_type; }
const char *kv_strategy_offload_dir(void) { return g_kv_opts.offload_dir; }

KVStrategyResult select_kv_strategy(const Config *cfg, tn_i64 free_ram, const MoEConfig *mc) {
  return select_kv_strategy_ex(cfg, free_ram, mc, &g_kv_opts);
}

void kv_strategy_print(const KVStrategyResult *r, int max_seq_len) {
  /* 2026-10-18: this used to print e.g. "KV Strategy: Quantized I8" although
   * no attention path read a quantized cache. It now reports the storage the
   * generic cache really uses (--kv-type) and the tier for what it is: the
   * context-sizing tier. */
  float frac = g_kv_opts.ram_fraction > 0.0f ? g_kv_opts.ram_fraction : KV_DEFAULT_RAM_FRACTION;
  printf("KV cache: %s%s%s, max context: %d tokens (sizing tier: %s%s, %s budget %.0f%%)\n",
         g_kv_opts.kv_type == 1 ? "I8" : "F32",
         g_kv_opts.offload_dir ? " on disk in " : "",
         g_kv_opts.offload_dir ? g_kv_opts.offload_dir : "",
         max_seq_len, r ? kv_strategy_name(r->strategy) : "n/a",
         g_kv_opts.ctx_size > 0 ? ", --ctx-size" : "",
         g_kv_opts.offload_dir ? "disk" : "RAM", frac * 100.0f);
}

KVStrategyResult select_kv_strategy_ex(const Config *cfg, tn_i64 free_ram, const MoEConfig *mc,
                                       const KVStrategyOptions *opt) {
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

  /* Explicit --ctx-size replaces the tier's default window (still bounded by
   * the model's own context and the RAM safety cap below). */
  if (opt && opt->ctx_size > 0)
    result.max_seq_len = opt->ctx_size < seq_len ? opt->ctx_size : seq_len;

  /*
   * Safety cap: compute the actual F32 KV cache cost for the chosen max_seq_len
   * and reduce it if the allocation would exceed 60% of free_ram.
   * This guards against models with large native seq_len (e.g. DeepSeek 163840)
   * on machines that have enough RAM to pass the GB_8 threshold but not enough
   * to hold the resulting cache (27 layers × 16 heads × 163840 × 128 × 4B × 2).
   *
   * Per-token KV cost (F32, K+V combined):
   *   n_layers * n_kv_heads * head_dim * 2 * sizeof(float)
   *
   * head_dim: use mc->attn_head_dim when set (qwen35 hybrid / qwen3moe —
   * these architectures' real per-head KV dimension is independent of
   * dim/n_heads; config_head_dim() silently underestimates it, e.g. 64 vs
   * the real 128 for Qwen3-30B-A3B, which under-clamps max_seq_len and can
   * still OOM on constrained hosts — GitHub issue #32, confirmed still
   * broken after the qwen3moe loader fix because this cap never saw the
   * true head_dim before). MLA (mc->has_mla) intentionally keeps the naive
   * formula — its real cache is a smaller low-rank latent, not
   * n_kv_heads*head_dim, so the naive formula over-clamps (conservative,
   * not unsafe) rather than under-clamps; see kv_strategy.h for why that's
   * left as a separate follow-up.
   */
  {
    int head_dim = (mc && mc->attn_head_dim > 0) ? mc->attn_head_dim : config_head_dim(cfg);
    /* Bytes per element: F32 = 4; I8 (--kv-type i8) = 1 plus one F32 scale
     * per head_dim-row. */
    int i8 = opt && opt->kv_type == 1;
    tn_i64 row_bytes = i8 ? (tn_i64)head_dim + (tn_i64)sizeof(float)
                          : (tn_i64)head_dim * (tn_i64)sizeof(float);
    tn_i64 per_token = (tn_i64)cfg->n_layers
                     * (tn_i64)(cfg->n_kv_heads > 0 ? cfg->n_kv_heads : cfg->n_heads)
                     * row_bytes * 2;
    if (per_token > 0) {
      float frac = (opt && opt->ram_fraction > 0.0f) ? opt->ram_fraction : KV_DEFAULT_RAM_FRACTION;
      if (frac > 0.95f) frac = 0.95f;
      /* Phase 31: a disk-backed cache is bounded by free disk space instead
       * (RAM then only holds the hot pages). Unknown disk space keeps RAM. */
      tn_i64 base = free_ram;
      if (opt && opt->offload_dir) {
        tn_i64 disk = file_backed_free_space(opt->offload_dir);
        if (disk > 0) base = disk;
      }
      tn_i64 budget        = (tn_i64)((double)base * (double)frac);
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
