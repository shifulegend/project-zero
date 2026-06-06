#include "threading/cpu_probe.h"
#include "core/platform.h"
#include <string.h>
#include <stdlib.h>

#if TN_POSIX
#include <unistd.h>
#include <stdio.h>
#endif

#if TN_WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

/*
 * Count unique physical cores on Linux by reading /proc/cpuinfo.
 * Returns 0 if the file is unreadable or parsing fails.
 */
#if TN_POSIX
#include <stdio.h>
static int count_physical_cores_linux(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;

    /* Collect up to 512 unique (physical id, core id) pairs */
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
            /* End of a CPU block — record this (phys, core) pair if unseen */
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
#endif /* TN_POSIX */

int tn_get_optimal_thread_count(void) {
    int logical = 1;

#if TN_POSIX
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) logical = (int)n;
#elif TN_WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    logical = (int)si.dwNumberOfProcessors;
#endif

#if TN_HAS_AVX512 && TN_POSIX
    /*
     * On AVX-512 VNNI capable x86 CPUs, hyperthreading causes severe
     * performance degradation due to contention on the shared AVX-512
     * execution port and L2 cache pressure.
     *
     * Empirical IPC sweep (Addendum M, 2026-03-17, i5-11300H, VNNI):
     *   T=4 (physical only): 19.48 tok/s, IPC 1.28  ← optimal
     *   T=5 (1 HT pair):     16.92 tok/s, IPC n/a   ← dip (scheduler asymmetry)
     *   T=6 (2 HT pairs):    19.05 tok/s, IPC 0.92  ← 2.3% slower, 40% more cycles
     *   T=7 (3 HT pairs):    19.44 tok/s, IPC n/a   ← high variance ±0.80
     *   T=8 (4 HT pairs):     2.61 tok/s, IPC n/a   ← HT cliff (port contention)
     *
     * Prior (Addendum E, AVX-512F scalar dispatch): T=6 was optimal because lower
     * per-core FLOP rate meant HT siblings provided a net gain. With VNNI dispatch,
     * each physical core's memory bandwidth demand saturates the L2/L3 at T=4.
     * HT siblings (T=6+) introduce cache-line contention and drop IPC from 1.28
     * to 0.92 — burning 40% more CPU cycles for <1% wall-clock difference.
     *
     * Optimal = physical cores only (= 4 on a 4-core HT CPU).
     *
     * Safety cap: never exceed logical - 2, leaving at least 2 logical
     * CPUs free for the OS and background tasks.
     */
    int physical = count_physical_cores_linux();
    if (physical <= 0) {
        /* Fallback: half of logical cores */
        int threads = logical / 2;
        if (threads < 1) threads = 1;
        return threads;
    }
    int optimal = physical;   /* physical cores only — VNNI saturates at T=4 */

    /*
     * Safety cap: leave headroom for OS and background tasks.
     *
     * On HT systems (logical > physical): cap at logical - 2 to leave 2
     * logical CPUs free.  Example: 4-core/8-thread → cap=6 → optimal=4.
     *
     * On non-HT systems (logical == physical): use all physical cores.
     * The old cap of logical-2 was wrong here — on a 4-core/4-thread
     * system it returned 2, halving throughput.  There are no HT siblings
     * to contend with, so every core should run a worker.
     */
    if (logical > physical) {
        int cap = logical - 2;
        if (cap < 1) cap = 1;
        if (optimal > cap) optimal = cap;
    }
    if (optimal < 1) optimal = 1;
    return optimal;

#elif TN_HAS_NEON
    /*
     * ARM (Apple Silicon, Cortex-A): no AVX-512 throttle cliff.
     * Use all logical cores — the OS scheduler handles heterogeneous
     * big.LITTLE clusters correctly for compute-bound workloads.
     */
    return logical;

#else
    /*
     * Generic x86 without AVX-512 (AVX2 or scalar), or unknown architecture.
     * Half of logical cores is a safe heuristic for HT systems; on AMD Zen
     * (SMT without throttle cliff) this is conservative but never wrong.
     */
    int threads = logical / 2;
    if (threads < 1) threads = 1;
    return threads;
#endif
}
