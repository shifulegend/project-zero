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

void *tn_aligned_calloc(size_t count, size_t elem_size, size_t alignment) {
    size_t total = count * elem_size;
    if (total == 0) return NULL;
    /* overflow check */
    if (elem_size != 0 && total / elem_size != count) return NULL;
    void *ptr = tn_aligned_alloc(total, alignment);
    if (ptr) {
        memset(ptr, 0, total);
    }
    return ptr;
}
