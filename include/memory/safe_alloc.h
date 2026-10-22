#ifndef TN_SAFE_ALLOC_H
#define TN_SAFE_ALLOC_H

#include <stddef.h>
#include "memory/aligned_alloc.h"

/*
 * safe_alloc — checked, trapping allocation for required buffers.
 *
 * The engine treats a failed *required* allocation as fatal, not as a value
 * to branch on. Raw malloc/calloc are still fine for optional buffers (a
 * caller that can fall back), but a required buffer should never be able to
 * leave the process in a half-initialized state: `tn_safe_*` trap
 * deterministically instead of returning NULL.
 *
 * Two hazards this closes (see docs/ai/mistakes.md):
 *   - absurd sizes: on macOS calloc over-commits and the kernel OOM-kills the
 *     process instead of returning NULL, so a size that "looks fine" can kill
 *     the run. `tn_safe_*` rejects anything dwarfing available RAM up front.
 *   - unchecked nullptr: `malloc` result used without a check. `tn_safe_*`
 *     never hands back NULL.
 *
 * Reuses tn_size_mul_overflow / tn_aligned_alloc / tn_get_free_ram — no new
 * allocator, just a checked front door onto the existing one.
 */

/* ---- bit tricks -------------------------------------------------------- */

/* True for a nonzero power of two. */
static inline int tn_is_pow2(size_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

/* Next power of two >= x (x >= 1). Undefined for x > 2^63 like the classic
 * smear; callers pass realistic sizes. */
static inline size_t tn_next_pow2(size_t x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16; x |= x >> 32;
    return x + 1;
}

/* Round x up to a multiple of a (a must be a power of two). */
static inline size_t tn_align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

/* ---- checked size math ------------------------------------------------- */

/*
 * True if count*elem is absurd relative to available RAM (or overflows).
 * A single buffer larger than 32x free RAM can never be part of a runnable
 * configuration, so this only traps pathological requests. Returns 0 when
 * RAM is unknown (fall back to the allocator's own NULL behavior).
 */
int tn_alloc_too_large(size_t count, size_t elem_size);

/* ---- trapping allocators ---------------------------------------------- */

/*
 * Each returns a valid, non-NULL pointer or does not return at all: on
 * absurd size / allocation failure the process prints a one-line diagnostic
 * and aborts. `what` names the buffer for that diagnostic (may be NULL).
 * size/count == 0 returns NULL (an explicitly empty buffer).
 */
void *tn_safe_malloc(size_t size, const char *what);
void *tn_safe_calloc(size_t count, size_t elem_size, const char *what);
void *tn_safe_realloc(void *ptr, size_t size, const char *what);

/* Aligned variant, on tn_aligned_alloc / tn_aligned_free. */
void *tn_safe_aligned_alloc(size_t size, size_t alignment, const char *what);

#endif /* TN_SAFE_ALLOC_H */
