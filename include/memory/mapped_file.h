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
 * Unmap and close the file. Safe to call on a zeroed MappedFile.
 */
void mapped_file_close(MappedFile *mf);

#endif /* TN_MAPPED_FILE_H */
