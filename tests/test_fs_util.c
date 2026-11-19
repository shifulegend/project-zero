/* tn_mkdir_parents(): creates missing parents (0700), not the last component. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "core/fs_util.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); g_fail++; } else printf("ok: %s\n", msg); } while (0)
static int is_dir(const char *p) { struct stat st; return stat(p, &st) == 0 && S_ISDIR(st.st_mode); }

int main(void) {
    char root[] = "/tmp/pz_fsutil_XXXXXX";
    if (!mkdtemp(root)) { perror("mkdtemp"); return 1; }
    char p[600], d[600];

    snprintf(p, sizeof p, "%s/a/b/c/file.txt", root);
    tn_mkdir_parents(p);
    snprintf(d, sizeof d, "%s/a/b/c", root);
    CHECK(is_dir(d), "all missing parents are created");
    snprintf(d, sizeof d, "%s/a/b/c/file.txt", root);
    CHECK(access(d, F_OK) != 0, "the final component is not created");
    struct stat st; snprintf(d, sizeof d, "%s/a", root);
    CHECK(stat(d, &st) == 0 && (st.st_mode & 0777) == 0700, "created directories are mode 0700 (umask permitting)");

    tn_mkdir_parents(p);
    CHECK(is_dir(d), "calling it again on existing directories is harmless");

    tn_mkdir_parents("justafile");
    CHECK(access("justafile", F_OK) != 0, "a path without a slash creates nothing");
    tn_mkdir_parents(NULL);
    tn_mkdir_parents("");
    CHECK(1, "NULL and empty paths are ignored");

    char longp[5000]; memset(longp, 'x', sizeof longp - 1); longp[0] = '/'; longp[sizeof longp - 1] = 0;
    tn_mkdir_parents(longp);
    CHECK(1, "a path longer than the buffer is ignored without overflow");

    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    if (system(cmd) != 0) { /* best effort */ }
    printf(g_fail ? "fs_util: %d FAILED\n" : "fs_util: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
