#include "threading/cpu_probe.h"
#include "core/platform.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

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
 * Count unique physical cores on Linux by reading /proc/cpuinfo.
 * Handles any line order ("core id" before "physical id"), trailing whitespace,
 * and missing trailing newlines at EOF.
 */
#if TN_POSIX
static bool is_blank_line(const char *s) {
    while (*s) {
        if (!isspace((unsigned char)*s)) return false;
        s++;
    }
    return true;
}

static void commit_cpu_pair(int cur_phys, int cur_core, int *phys, int *core, int *count) {
    if (cur_phys >= 0 && cur_core >= 0 && *count < 512) {
        for (int i = 0; i < *count; i++) {
            if (phys[i] == cur_phys && core[i] == cur_core) {
                return;
            }
        }
        phys[*count] = cur_phys;
        core[*count] = cur_core;
        (*count)++;
    }
}

static int count_physical_cores_linux(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;

    int phys[512], core[512], count = 0;
    int cur_phys = -1, cur_core = -1;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (is_blank_line(line) || strncmp(line, "processor", 9) == 0) {
            commit_cpu_pair(cur_phys, cur_core, phys, core, &count);
            cur_phys = -1;
            cur_core = -1;
            continue;
        }

        char *colon = strchr(line, ':');
        if (!colon) continue;

        char key[128];
        size_t key_len = (size_t)(colon - line);
        if (key_len >= sizeof(key)) key_len = sizeof(key) - 1;
        memcpy(key, line, key_len);
        key[key_len] = '\0';

        long val = strtol(colon + 1, NULL, 10);

        if (strstr(key, "physical id") != NULL) {
            cur_phys = (int)val;
        } else if (strstr(key, "core id") != NULL) {
            cur_core = (int)val;
        }
    }
    commit_cpu_pair(cur_phys, cur_core, phys, core, &count);
    fclose(f);
    return count;
}
#endif /* TN_POSIX */

#if TN_WIN32
static int count_physical_cores_win32(void) {
    DWORD returnLength = 0;
    GetLogicalProcessorInformation(NULL, &returnLength);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || returnLength == 0) {
        return 0;
    }

    DWORD count = returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(returnLength);
    if (!buffer) return 0;

    if (!GetLogicalProcessorInformation(buffer, &returnLength)) {
        free(buffer);
        return 0;
    }

    int physical_cores = 0;
    for (DWORD i = 0; i < count; i++) {
        if (buffer[i].Relationship == RelationProcessorCore) {
            physical_cores++;
        }
    }
    free(buffer);
    return physical_cores;
}
#endif /* TN_WIN32 */

static int count_physical_cores(void) {
#if TN_POSIX
    return count_physical_cores_linux();
#elif TN_WIN32
    return count_physical_cores_win32();
#else
    return 0;
#endif
}

int tn_get_optimal_thread_count(void) {
    (void)count_physical_cores;
    int logical = 1;

#if TN_POSIX
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) logical = (int)n;
#elif TN_WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    logical = (int)si.dwNumberOfProcessors;
#endif

#if TN_HAS_AVX512
    int physical = count_physical_cores();
    if (physical <= 0) {
        int threads = logical / 2;
        if (threads < 1) threads = 1;
        return threads;
    }
    int optimal = physical;

    if (logical > physical) {
        int cap = logical - 2;
        if (cap < 1) cap = 1;
        if (optimal > cap) optimal = cap;
    }
    if (optimal < 1) optimal = 1;
    return optimal;

#elif TN_HAS_NEON
    return logical;

#else
    int threads = logical / 2;
    if (threads < 1) threads = 1;
    return threads;
#endif
}
