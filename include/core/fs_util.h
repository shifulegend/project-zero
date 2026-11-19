#ifndef TN_CORE_FS_UTIL_H
#define TN_CORE_FS_UTIL_H

/* Create every missing parent directory of `path` (mode 0700), like `mkdir -p
 * $(dirname path)`. The final component is NOT created. Existing directories
 * and creation failures are ignored: the caller's next open()/fopen() reports
 * any real problem. Does nothing on platforms without POSIX mkdir, and for a
 * path too long to copy (the later open fails the same way). */
void tn_mkdir_parents(const char *path);

#endif
