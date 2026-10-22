#include "memory/file_backed.h"
#include <string.h>
#include <stdio.h>

#if TN_POSIX

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>

/* Reserve `bytes` of real disk blocks for fd. Returns 0 or an errno value.
 * 2026-10-21: posix_fallocate() does not exist on macOS (the macOS CI build
 * failed to compile); F_PREALLOCATE + ftruncate is its equivalent there. */
static int reserve_blocks(int fd, off_t bytes) {
#if defined(__APPLE__)
    fstore_t fs = { .fst_flags = F_ALLOCATECONTIG, .fst_posmode = F_PEOFPOSMODE,
                    .fst_offset = 0, .fst_length = bytes, .fst_bytesalloc = 0 };
    if (fcntl(fd, F_PREALLOCATE, &fs) == -1) {
        fs.fst_flags = F_ALLOCATEALL;            /* fragmented is fine */
        if (fcntl(fd, F_PREALLOCATE, &fs) == -1) return errno;
    }
    return ftruncate(fd, bytes) == 0 ? 0 : errno;
#else
    return posix_fallocate(fd, 0, bytes);
#endif
}

TernaryError file_backed_alloc(FileBacked *fb, const char *dir, size_t bytes) {
    if (!fb || !dir || !*dir || bytes == 0) return TN_ERR_INVALID_ARGS;
    memset(fb, 0, sizeof(*fb));
    if ((off_t)bytes < 0) return TN_ERR_OOM;

    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/pz-kv-XXXXXX", dir);
    if (n <= 0 || (size_t)n >= sizeof(path)) return TN_ERR_INVALID_ARGS;
    int fd = mkstemp(path);
    if (fd < 0) return TN_ERR_FILE_OPEN;
    unlink(path); /* lives only as long as the fd / mapping */

    /* Reserve the blocks now: a sparse file would SIGBUS on page write-back
     * once the disk fills up, long after this call reported success. */
    int rc = reserve_blocks(fd, (off_t)bytes);
    if (rc != 0) {
        close(fd);
        return rc == ENOSPC ? TN_ERR_OOM : TN_ERR_FILE_STAT;
    }
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd); /* the mapping keeps the file alive */
    if (p == MAP_FAILED) return TN_ERR_MMAP_FAILED;
    fb->base = p;
    fb->bytes = bytes;
    return TN_OK;
}

void file_backed_free(FileBacked *fb) {
    if (!fb) return;
    if (fb->base) munmap(fb->base, fb->bytes);
    fb->base = NULL;
    fb->bytes = 0;
}

tn_i64 file_backed_free_space(const char *dir) {
    struct statvfs sv;
    if (!dir || statvfs(dir, &sv) != 0) return -1;
    return (tn_i64)sv.f_bavail * (tn_i64)sv.f_frsize;
}

void file_backed_prefetch(const void *p, size_t len) {
    if (!p || len == 0) return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 4096;
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(pg - 1);
    uintptr_t e = (uintptr_t)p + len;
    posix_madvise((void *)a, (size_t)(e - a), POSIX_MADV_WILLNEED);
}

#else /* !TN_POSIX */

TernaryError file_backed_alloc(FileBacked *fb, const char *dir, size_t bytes) {
    (void)dir; (void)bytes;
    if (fb) memset(fb, 0, sizeof(*fb));
    return TN_ERR_UNSUPPORTED;
}
void   file_backed_free(FileBacked *fb) { if (fb) memset(fb, 0, sizeof(*fb)); }
tn_i64 file_backed_free_space(const char *dir) { (void)dir; return -1; }
void   file_backed_prefetch(const void *p, size_t len) { (void)p; (void)len; }

#endif
