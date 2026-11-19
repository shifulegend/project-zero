#ifndef TN_MEMORY_TLS_SCRATCH_H
#define TN_MEMORY_TLS_SCRATCH_H

/* A growable scratch buffer private to the calling thread. Replaces the "static buffer that is
 * reallocated when it is too small" pattern, which two threads cannot use at once (one frees the
 * buffer the other is reading) and which is never freed. Each thread gets its own buffer, grown on
 * demand and freed when the thread exits. Declare one TnTlsScratch per use site:
 *
 *     static TnTlsScratch g_acts = TN_TLS_SCRATCH_INIT;
 *     TnQ8KActBlock *acts = (TnQ8KActBlock *)tn_tls_scratch(&g_acts, n * sizeof(*acts));
 *
 * The contents are NOT preserved across a growing call and are uninitialised; the pointer stays
 * valid until the same thread calls tn_tls_scratch() on the same object again. */

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>

typedef struct {
    pthread_key_t key;
    atomic_int state; /* 0 = key not created yet, 1 = ready, -1 = pthread_key_create failed */
} TnTlsScratch;

#define TN_TLS_SCRATCH_INIT                                                                        \
    { 0, 0 }

/* Returns at least `bytes` bytes (suitably aligned for any scalar type), or NULL when out of
 * memory or the key could not be created. `bytes` of 0 returns a valid non-NULL pointer. */
void *tn_tls_scratch(TnTlsScratch *s, size_t bytes);

/* Frees the calling thread's buffer for `s` now (it is otherwise freed at thread exit). The next
 * tn_tls_scratch() call on this thread allocates afresh. */
void tn_tls_scratch_release(TnTlsScratch *s);

#endif
