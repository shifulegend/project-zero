/*
 * core/bw_cache.c — per-host DRAM bandwidth cache (see core/bw_cache.h).
 */
#include "core/bw_cache.h"
#include "core/platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if TN_POSIX
#include <sys/stat.h>
#endif

#define BW_CACHE_TAG "pz-membw 1"

int tn_bw_cache_key(char *buf, size_t n, const char *cpu_model, int physical_cores,
                    int logical_cores, size_t l3_bytes, size_t total_ram_bytes) {
    if (!buf || n == 0) return -1;
    /* Tabs and newlines would break the one-line format. */
    char model[128];
    size_t i = 0;
    for (const char *p = cpu_model ? cpu_model : ""; *p && i + 1 < sizeof(model); p++)
        model[i++] = (*p == '\t' || *p == '\n' || *p == '\r') ? ' ' : *p;
    model[i] = '\0';
    int w = snprintf(buf, n, "%s|%dc|%dt|l3=%zu|ram=%zuMB", model[0] ? model : "unknown",
                     physical_cores, logical_cores, l3_bytes, total_ram_bytes >> 20);
    return (w < 0 || (size_t)w >= n) ? -1 : 0;
}

void tn_bw_cache_cpu_model(char *buf, size_t n) {
    if (!buf || n == 0) return;
    buf[0] = '\0';
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) != 0) continue;
        char *p = strchr(line, ':');
        if (p) {
            p++;
            while (*p == ' ') p++;
            p[strcspn(p, "\r\n")] = '\0';
            snprintf(buf, n, "%s", p);
        }
        break;
    }
    fclose(f);
}

int tn_bw_cache_path(char *buf, size_t n) {
    const char *d = getenv("PZ_CACHE_DIR");
    int w;
    if (d && *d) w = snprintf(buf, n, "%s/membw", d);
    else if ((d = getenv("XDG_CACHE_HOME")) && *d) w = snprintf(buf, n, "%s/project-zero/membw", d);
    else if ((d = getenv("HOME")) && *d) w = snprintf(buf, n, "%s/.cache/project-zero/membw", d);
    else return -1;
    return (w < 0 || (size_t)w >= n) ? -1 : 0;
}

int tn_bw_cache_load(const char *path, const char *key, long long now, long max_age_s,
                     double *gbps) {
    if (!path || !key || !gbps) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[512];
    int hit = 0;
    if (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = '\0';
        char *tag = strtok(line, "\t");
        char *k = strtok(NULL, "\t");
        char *v = strtok(NULL, "\t");
        char *t = strtok(NULL, "\t");
        if (tag && k && v && t && strcmp(tag, BW_CACHE_TAG) == 0 && strcmp(k, key) == 0) {
            char *end = NULL;
            double g = strtod(v, &end);
            long long ts = strtoll(t, NULL, 10);
            /* A clock that went backwards counts as stale. */
            if (end != v && isfinite(g) && g > 0.0 && g < 1e5 &&
                ts > 0 && ts <= now && now - ts <= max_age_s) {
                *gbps = g;
                hit = 1;
            }
        }
    }
    fclose(f);
    return hit;
}

#if TN_POSIX
static void mkdir_parents(const char *path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        mkdir(tmp, 0700);
        *p = '/';
    }
}
#endif

int tn_bw_cache_save(const char *path, const char *key, double gbps, long long now) {
    if (!path || !key || !isfinite(gbps) || gbps <= 0.0) return -1;
    if (strchr(key, '\t') || strchr(key, '\n')) return -1;
#if TN_POSIX
    mkdir_parents(path);
#endif
    char tmp[600];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) return -1;
    int ok = fprintf(f, "%s\t%s\t%.3f\t%lld\n", BW_CACHE_TAG, key, gbps, now) > 0;
    ok = (fclose(f) == 0) && ok;
    if (!ok || rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}
