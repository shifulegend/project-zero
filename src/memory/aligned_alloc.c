/* _DEFAULT_SOURCE for madvise()/MADV_DONTNEED under strict -std=c99 on Linux. */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "memory/aligned_alloc.h"
#include <string.h>

/* Returns non-zero if alignment is a power of two and >= sizeof(void*).
 * posix_memalign returns EINVAL for invalid alignments; we validate here
 * to surface the error early rather than silently returning NULL (FM-004). */
static int alignment_valid(size_t alignment) {
    return alignment >= sizeof(void *) && (alignment & (alignment - 1)) == 0;
}

#if TN_POSIX
#include <stdlib.h>

void *tn_aligned_alloc(size_t size, size_t alignment) {
    if (size == 0 || !alignment_valid(alignment)) return NULL;
    if (size > (size_t)-1 - alignment) return NULL;
    void *ptr = NULL;
    int ret = posix_memalign(&ptr, alignment, size);
    return (ret == 0) ? ptr : NULL;
}

void tn_aligned_free(void *ptr) {
    free(ptr);  /* posix_memalign pointers are free()-compatible */
}

#elif TN_WIN32
#include <malloc.h>

void *tn_aligned_alloc(size_t size, size_t alignment) {
    if (size == 0 || !alignment_valid(alignment)) return NULL;
    if (size > (size_t)-1 - alignment) return NULL;
    return _aligned_malloc(size, alignment);
}

void tn_aligned_free(void *ptr) {
    _aligned_free(ptr);  /* MUST use _aligned_free, not free() */
}

#endif

#if defined(__linux__)
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

/* Large buffers: zero the whole pages inside [p, p+n) with MADV_DONTNEED
 * (private anonymous memory reads back as zero-fill-on-demand) and memset
 * only the partial head/tail pages. An explicit memset of the whole buffer
 * committed every page up front: a 135M model's 8192-token F32 KV cache
 * cost ~5 s of startup and its full size in RSS before a single token was
 * generated (2026-10-19, head-to-head sweep). Returns 0 if it did nothing. */
static int zero_lazily(void *p, size_t n) {
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0 || n < (size_t)pg * 16) return 0;
    uintptr_t a = (uintptr_t)p, e = a + n;
    uintptr_t lo = (a + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi = e & ~((uintptr_t)pg - 1);
    if (hi <= lo || madvise((void *)lo, hi - lo, MADV_DONTNEED) != 0) return 0;
    memset(p, 0, lo - a);
    memset((void *)hi, 0, e - hi);
    return 1;
}
#endif

void *tn_aligned_calloc(size_t count, size_t elem_size, size_t alignment) {
    size_t total = count * elem_size;
    if (total == 0) return NULL;
    /* overflow check */
    if (elem_size != 0 && total / elem_size != count) return NULL;
    void *ptr = tn_aligned_alloc(total, alignment);
    if (ptr) {
#if defined(__linux__)
        if (!zero_lazily(ptr, total))
#endif
            memset(ptr, 0, total);
    }
    return ptr;
}
