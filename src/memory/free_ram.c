/*
 * src/memory/free_ram.c
 *
 * tn_get_free_ram(): bytes of RAM the OS would hand to a new allocation. Split out of
 * kv_cache/kv_strategy.c so memory/ (safe_alloc.c) does not depend on kv_cache/; the code is unchanged.
 */

#include "memory/free_ram.h"
#include "core/platform.h"
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
    long pages = sysconf(_SC_PHYS_PAGES); /* macOS fallback — total, not free */
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
