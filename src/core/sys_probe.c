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
