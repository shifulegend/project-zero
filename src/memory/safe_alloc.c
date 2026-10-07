#include "memory/safe_alloc.h"
#include "kv_cache/kv_strategy.h"  /* tn_get_free_ram() */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tn_alloc_too_large(size_t count, size_t elem_size) {
    tn_i64 ram = tn_get_free_ram();
    size_t bytes;
    if (ram <= 0) return 0; /* unknown RAM: defer to the allocator's NULL path */
    if (tn_size_mul_overflow(count, elem_size, &bytes)) return 1;
    return bytes > (size_t)ram * 32;
}

/* One-line diagnostic then abort: a required buffer could not be satisfied.
 * Kept in one place so the message format is uniform and greppable. */
static void tn_alloc_fail(const char *what, size_t bytes) {
    fprintf(stderr,
            "[fatal] allocation failed: %s (%zu bytes) — required buffer; "
            "see docs/ai/mistakes.md\n",
            what ? what : "(unnamed)", bytes);
    fflush(stderr);
    abort();
}

void *tn_safe_malloc(size_t size, const char *what) {
    if (size == 0) return NULL;
    if (tn_alloc_too_large(1, size)) tn_alloc_fail(what, size);
    void *p = malloc(size);
    if (!p) tn_alloc_fail(what, size);
    return p;
}

void *tn_safe_calloc(size_t count, size_t elem_size, const char *what) {
    if (count == 0 || elem_size == 0) return NULL;
    size_t bytes;
    if (tn_size_mul_overflow(count, elem_size, &bytes)) tn_alloc_fail(what, 0);
    if (tn_alloc_too_large(count, elem_size)) tn_alloc_fail(what, bytes);
    void *p = calloc(count, elem_size);
    if (!p) tn_alloc_fail(what, bytes);
    return p;
}

void *tn_safe_realloc(void *ptr, size_t size, const char *what) {
    if (size == 0) { free(ptr); return NULL; }
    if (ptr == NULL) return tn_safe_malloc(size, what);
    if (tn_alloc_too_large(1, size)) tn_alloc_fail(what, size);
    void *p = realloc(ptr, size);
    if (!p) tn_alloc_fail(what, size);
    return p;
}

void *tn_safe_aligned_alloc(size_t size, size_t alignment, const char *what) {
    if (size == 0) return NULL;
    if (tn_alloc_too_large(1, size)) tn_alloc_fail(what, size);
    void *p = tn_aligned_alloc(size, alignment);
    if (!p) tn_alloc_fail(what, size);
    return p;
}
