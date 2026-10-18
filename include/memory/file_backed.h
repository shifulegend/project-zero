#ifndef TN_FILE_BACKED_H
#define TN_FILE_BACKED_H

/*
 * file_backed.h — Phase 31: disk-backed (NVMe/SSD) memory for large buffers.
 *
 * file_backed_alloc() creates an anonymous (unlinked) file in `dir`, reserves
 * `bytes` on disk and maps it MAP_SHARED. The kernel page cache becomes the
 * hot tier: recently used pages stay in RAM, cold ones are written back to
 * the file and dropped under memory pressure, so the buffer may exceed RAM.
 * Contents start zeroed. The file disappears when the mapping is released or
 * the process exits. POSIX only; returns TN_ERR_UNSUPPORTED elsewhere.
 */

#include <stddef.h>
#include "core/error.h"
#include "core/platform.h"

typedef struct {
    void  *base;   /* page-aligned mapping, NULL when unused */
    size_t bytes;
} FileBacked;

TernaryError file_backed_alloc(FileBacked *fb, const char *dir, size_t bytes);
void         file_backed_free(FileBacked *fb);

/* Bytes available to an unprivileged writer in `dir`, or -1 if unknown. */
tn_i64 file_backed_free_space(const char *dir);

/* Hint that [p, p+len) will be read soon (async read-ahead from disk).
 * Rounds out to whole pages; a no-op where unsupported. */
void file_backed_prefetch(const void *p, size_t len);

#endif /* TN_FILE_BACKED_H */
