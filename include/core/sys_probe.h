#ifndef TN_SYS_PROBE_H
#define TN_SYS_PROBE_H

#include <stddef.h>
#include <stdio.h>

/*
 * The operating-system lookups of the hardware profile, calibration and bandwidth cache (the
 * environment and text files such as /proc/cpuinfo) behind one small table, so a test can hand
 * in a fake instead of reading the real machine. The functions below take the table as a
 * parameter; NULL means the real system (tn_sys_probe_default()).
 */
typedef struct {
    const char *(*getenv_fn)(const char *name);
    FILE *(*fopen_fn)(const char *path, const char *mode);
} TnSysProbe;

/* The real getenv() and fopen(). */
const TnSysProbe *tn_sys_probe_default(void);

/* "model name" of the first CPU in /proc/cpuinfo (empty string when unavailable). */
void tn_sys_cpu_model(const TnSysProbe *probe, char *buf, size_t n);

/* Directory for per-host caches: $PZ_CACHE_DIR, else $XDG_CACHE_HOME/project-zero, else
 * $HOME/.cache/project-zero. Returns 0 and writes `buf`, or -1 when none is set or it does not
 * fit. */
int tn_sys_cache_dir(const TnSysProbe *probe, char *buf, size_t n);

#endif /* TN_SYS_PROBE_H */
