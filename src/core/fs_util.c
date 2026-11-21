#include "core/fs_util.h"

#include <stdio.h>
#include <string.h>

#include "core/platform.h"

#if TN_POSIX
#include <sys/stat.h>
#include <sys/types.h>
#endif

void tn_mkdir_parents(const char *path) {
#if TN_POSIX
    if (!path) return;
    char tmp[4096];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof(tmp)) return;
    memcpy(tmp, path, n + 1);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        (void)mkdir(tmp, 0700);
        *p = '/';
    }
#else
    (void)path;
#endif
}
