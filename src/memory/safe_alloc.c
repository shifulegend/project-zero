#include "memory/safe_alloc.h"
#include "kv_cache/kv_strategy.h"  /* tn_get_free_ram() */
#include "core/platform.h"
#if TN_POSIX
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int tn_alloc_too_large(size_t count, size_t elem_size) {
    size_t bytes;
    if (tn_size_mul_overflow(count, elem_size, &bytes)) return 1;
#if defined(__linux__)
    /* tn_aligned_calloc zeroes lazily here (MADV_DONTNEED), so untouched pages
     * cost nothing: allow up to 32x free RAM. */
    tn_i64 ram = tn_get_free_ram();
    if (ram <= 0) return 0; /* unknown RAM: defer to the allocator's NULL path */
    return bytes > (size_t)ram * 32;
#else
    /* 2026-10-21: elsewhere tn_aligned_calloc memsets the whole buffer, touching
     * every page, so a buffer larger than physical RAM can never be zeroed --
     * macOS lets the allocation succeed and the memory killer SIGKILLs the
     * process (issue #43; CI: test_redbox's INT_MAX-context case). Deterministic
     * bound: total physical RAM. */
    tn_i64 total = -1;
#if TN_POSIX && defined(_SC_PHYS_PAGES)
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psz > 0) total = (tn_i64)pages * (tn_i64)psz;
#endif
    if (total <= 0) total = tn_get_free_ram();
    if (total <= 0) return 0;
    return bytes > (size_t)total;
#endif
}

/* Required buffers below this size skip the RAM probe: no small buffer can be
 * "absurd", and tn_get_free_ram() is not free (it reads /proc/meminfo, and on
 * macOS spawns vm_stat) -- per-request buffers must not pay that. */
#define TN_SAFE_CHECK_MIN_BYTES ((size_t)64 << 20)

static int too_large_checked(size_t count, size_t elem_size) {
    size_t bytes;
    if (tn_size_mul_overflow(count, elem_size, &bytes)) return 1;
    return bytes >= TN_SAFE_CHECK_MIN_BYTES && tn_alloc_too_large(count, elem_size);
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
    if (too_large_checked(1, size)) tn_alloc_fail(what, size);
    void *p = malloc(size);
    if (!p) tn_alloc_fail(what, size);
    return p;
}

void *tn_safe_calloc(size_t count, size_t elem_size, const char *what) {
    if (count == 0 || elem_size == 0) return NULL;
    size_t bytes;
    if (tn_size_mul_overflow(count, elem_size, &bytes)) tn_alloc_fail(what, 0);
    if (too_large_checked(count, elem_size)) tn_alloc_fail(what, bytes);
    void *p = calloc(count, elem_size);
    if (!p) tn_alloc_fail(what, bytes);
    return p;
}

void *tn_safe_realloc(void *ptr, size_t size, const char *what) {
    if (size == 0) { free(ptr); return NULL; }
    if (ptr == NULL) return tn_safe_malloc(size, what);
    if (too_large_checked(1, size)) tn_alloc_fail(what, size);
    void *p = realloc(ptr, size);
    if (!p) tn_alloc_fail(what, size);
    return p;
}

void *tn_safe_aligned_alloc(size_t size, size_t alignment, const char *what) {
    if (size == 0) return NULL;
    if (too_large_checked(1, size)) tn_alloc_fail(what, size);
    void *p = tn_aligned_alloc(size, alignment);
    if (!p) tn_alloc_fail(what, size);
    return p;
}
