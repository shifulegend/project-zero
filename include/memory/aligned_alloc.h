#ifndef TN_ALIGNED_ALLOC_H
#define TN_ALIGNED_ALLOC_H

#include <stddef.h>
#include "core/platform.h"

/**
 * Multiply two size_t values with overflow detection.
 * Returns 1 on overflow, 0 on success. Result stored in *result.
 */
static inline int tn_size_mul_overflow(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > (size_t)-1 / a) return 1;
    *result = a * b;
    return 0;
}

/**
 * Multiply 3 or 4 size_t factors with overflow detection.
 * Returns 0 on success, 1 on overflow. Result stored in *result.
 */
static inline int tn_size_mul3(size_t a, size_t b, size_t c, size_t *result) {
    size_t tmp;
    if (tn_size_mul_overflow(a, b, &tmp)) return 1;
    return tn_size_mul_overflow(tmp, c, result);
}

static inline int tn_size_mul4(size_t a, size_t b, size_t c, size_t d, size_t *result) {
    size_t tmp;
    if (tn_size_mul3(a, b, c, &tmp)) return 1;
    return tn_size_mul_overflow(tmp, d, result);
}

/**
 * Allocate memory aligned to `alignment` bytes.
 * Alignment must be a power of 2 and >= sizeof(void*).
 * Returns NULL on failure.
 */
void *tn_aligned_alloc(size_t size, size_t alignment);

/**
 * Allocate and zero-fill memory aligned to `alignment` bytes.
 */
void *tn_aligned_calloc(size_t count, size_t elem_size, size_t alignment);

/**
 * Free memory allocated by tn_aligned_alloc or tn_aligned_calloc.
 * Safe to call with NULL.
 */
void tn_aligned_free(void *ptr);

#endif /* TN_ALIGNED_ALLOC_H */
