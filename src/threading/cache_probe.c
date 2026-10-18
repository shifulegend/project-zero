#include "threading/cache_probe.h"
#include "core/platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#if TN_ARCH_X86
#include <cpuid.h>
#endif

long long tn_cache_parse_size(const char *s) {
    if (!s) return -1;
    while (isspace((unsigned char)*s)) s++;
    if (!isdigit((unsigned char)*s)) return -1;
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (v < 0) return -1;
    while (end && isspace((unsigned char)*end)) end++;
    if (end && (*end == 'K' || *end == 'k')) v *= 1024LL;
    else if (end && (*end == 'M' || *end == 'm')) v *= 1024LL * 1024LL;
    else if (end && (*end == 'G' || *end == 'g')) v *= 1024LL * 1024LL * 1024LL;
    else if (end && *end && !isspace((unsigned char)*end)) return -1;
    return v;
}

static int read_line(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int ok = fgets(buf, (int)cap, f) != NULL;
    fclose(f);
    if (ok) buf[strcspn(buf, "\r\n")] = '\0';
    return ok;
}

static int probe_sysfs(CacheInfo *ci) {
    int found = 0;
    for (int idx = 0; idx < 8; idx++) {
        char path[128], level[16], type[32], size[32], line[16];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        if (!read_line(path, level, sizeof(level))) break;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        if (!read_line(path, type, sizeof(type))) continue;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        if (!read_line(path, size, sizeof(size))) continue;
        long long bytes = tn_cache_parse_size(size);
        if (bytes <= 0) continue;
        int lv = atoi(level);
        if (strcmp(type, "Instruction") == 0) continue;
        if (lv == 1) ci->l1d_bytes = (int)bytes;
        else if (lv == 2) ci->l2_bytes = (int)bytes;
        else if (lv == 3) ci->l3_bytes = bytes;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size", idx);
        if (read_line(path, line, sizeof(line)) && atoi(line) > 0) ci->line_bytes = atoi(line);
        found = 1;
    }
    return found && ci->l1d_bytes > 0;
}

static int probe_cpuid(CacheInfo *ci) {
#if TN_ARCH_X86
    unsigned eax, ebx, ecx, edx;
    int found = 0;
    for (unsigned sub = 0; sub < 8; sub++) {
        if (!__get_cpuid_count(4, sub, &eax, &ebx, &ecx, &edx)) break;
        unsigned type = eax & 0x1f;          /* 0 = no more caches, 2 = instruction */
        if (type == 0) break;
        if (type == 2) continue;
        unsigned level = (eax >> 5) & 0x7;
        long long ways = ((ebx >> 22) & 0x3ff) + 1, parts = ((ebx >> 12) & 0x3ff) + 1;
        long long line = (ebx & 0xfff) + 1, sets = (long long)ecx + 1;
        long long bytes = ways * parts * line * sets;
        if (level == 1) ci->l1d_bytes = (int)bytes;
        else if (level == 2) ci->l2_bytes = (int)bytes;
        else if (level == 3) ci->l3_bytes = bytes;
        ci->line_bytes = (int)line;
        found = 1;
    }
    return found && ci->l1d_bytes > 0;
#else
    (void)ci;
    return 0;
#endif
}

static CacheInfo g_cache;
static int g_cache_ready;

const CacheInfo *tn_cache_info(void) {
    if (g_cache_ready) return &g_cache;
    CacheInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.line_bytes = CACHE_PROBE_FALLBACK_LINE;
    if (probe_sysfs(&ci)) ci.source = 1;
    else {
        memset(&ci, 0, sizeof(ci));
        ci.line_bytes = CACHE_PROBE_FALLBACK_LINE;
        if (probe_cpuid(&ci)) ci.source = 2;
    }
    if (ci.l1d_bytes <= 0) ci.l1d_bytes = CACHE_PROBE_FALLBACK_L1D;
    if (ci.l2_bytes <= 0) ci.l2_bytes = CACHE_PROBE_FALLBACK_L2;
    g_cache = ci;
    g_cache_ready = 1;
    return &g_cache;
}

int tn_cache_rows_fitting_l1(int row_bytes, double fraction) {
    if (row_bytes <= 0) return 1;
    if (fraction <= 0.0) fraction = 0.5;
    long long rows = (long long)((double)tn_cache_info()->l1d_bytes * fraction) / row_bytes;
    return rows < 1 ? 1 : (rows > 1 << 20 ? 1 << 20 : (int)rows);
}

const char *tn_cache_source_name(int source) {
    return source == 1 ? "sysfs" : source == 2 ? "cpuid" : "fallback";
}
