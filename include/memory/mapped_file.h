#ifndef TN_MAPPED_FILE_H
#define TN_MAPPED_FILE_H

#include "core/error.h"
#include <stddef.h>

typedef struct {
    void  *data;    /* pointer to mapped region */
    size_t size;    /* file size in bytes */
#ifdef _WIN32
    void  *handle;  /* Windows HANDLE (pointer-sized, avoids truncation on x64) */
#else
    int    fd;      /* POSIX file descriptor */
#endif
} MappedFile;

/**
 * Memory-map a file read-only. Uses mmap (POSIX) or MapViewOfFile (Windows).
 * On success, mf->data and mf->size are set.
 */
TernaryError mapped_file_open(MappedFile *mf, const char *path);

/**
 * Drop the resident pages of a sub-range of a read-only mapping whose bytes
 * have been converted into an engine-owned buffer (I2_S repack, embedding
 * conversion). The mapping stays valid — a later read refaults from the file —
 * so this only lowers RSS (the file is mapped with MAP_POPULATE). Partial pages
 * at either end are kept. Linux only; a no-op elsewhere.
 */
void mapped_file_release(const void *p, size_t n);

/**
 * Unmap and close the file. Safe to call on a zeroed MappedFile.
 */
void mapped_file_close(MappedFile *mf);

#endif /* TN_MAPPED_FILE_H */
