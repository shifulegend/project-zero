/*
 * core/sys_probe — the environment and /proc lookups shared by the hardware profile, the
 * calibration and the bandwidth cache (see include/core/sys_probe.h).
 */
#include "core/sys_probe.h"

#include <stdlib.h>
#include <string.h>

static const char *real_getenv(const char *name) {
    return getenv(name);
}

static const TnSysProbe k_default_probe = {real_getenv, fopen};

const TnSysProbe *tn_sys_probe_default(void) {
    return &k_default_probe;
}

void tn_sys_cpu_model(const TnSysProbe *probe, char *buf, size_t n) {
    if (!buf || n == 0) return;
    buf[0] = '\0';
    if (!probe) probe = &k_default_probe;
    FILE *f = probe->fopen_fn("/proc/cpuinfo", "r");
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

enum { MAX_CORES = 512 };

typedef struct {
    int phys[MAX_CORES], core[MAX_CORES];
    int count;
} CoreSet;

/* Records one (physical id, core id) pair; repeats (hardware threads of a core) count once. */
static void core_set_add(CoreSet *set, int phys, int core) {
    if (phys < 0 || core < 0 || set->count >= MAX_CORES) return;
    for (int i = 0; i < set->count; i++)
        if (set->phys[i] == phys && set->core[i] == core) return;
    set->phys[set->count]   = phys;
    set->core[set->count++] = core;
}

/* The number after the colon of a "key : value" line, or `keep` when the line has no colon. */
static int cpuinfo_value(const char *line, int keep) {
    const char *colon = strchr(line, ':');
    return colon ? (int)strtol(colon + 1, NULL, 10) : keep;
}

/* One /proc/cpuinfo line: a blank line closes a logical CPU's block, the two id lines fill it. */
static void cpuinfo_line(CoreSet *set, int *phys, int *core, const char *line) {
    if (line[0] == '\n') {
        core_set_add(set, *phys, *core);
        *phys = *core = -1;
    } else if (strncmp(line, "physical id", 11) == 0) {
        *phys = cpuinfo_value(line, *phys);
    } else if (strncmp(line, "core id", 7) == 0) {
        *core = cpuinfo_value(line, *core);
    }
}

int tn_sys_physical_cores(const TnSysProbe *probe) {
    if (!probe) probe = &k_default_probe;
    FILE *f = probe->fopen_fn("/proc/cpuinfo", "r");
    if (!f) return 0;
    CoreSet set;
    memset(&set, 0, sizeof(set));
    int phys = -1, core = -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) cpuinfo_line(&set, &phys, &core, line);
    fclose(f);
    return set.count;
}

int tn_sys_env_flag(const TnSysProbe *probe, const char *name) {
    if (!probe) probe = &k_default_probe;
    const char *v = probe->getenv_fn(name);
    return v && *v && strcmp(v, "0") != 0;
}

int tn_sys_cache_dir(const TnSysProbe *probe, char *buf, size_t n) {
    if (!probe) probe = &k_default_probe;
    const char *d = probe->getenv_fn("PZ_CACHE_DIR");
    int w;
    if (d && *d)
        w = snprintf(buf, n, "%s", d);
    else if ((d = probe->getenv_fn("XDG_CACHE_HOME")) && *d)
        w = snprintf(buf, n, "%s/project-zero", d);
    else if ((d = probe->getenv_fn("HOME")) && *d)
        w = snprintf(buf, n, "%s/.cache/project-zero", d);
    else
        return -1;
    return (w < 0 || (size_t)w >= n) ? -1 : 0;
}
