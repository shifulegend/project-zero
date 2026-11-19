#include "memory/tls_scratch.h"

#include <stdlib.h>

/* One record per (thread, TnTlsScratch). The pthread key destructor frees it at thread exit. */
typedef struct {
    size_t cap;
    void *data;
} TlsRecord;

static void tls_record_free(void *p) {
    TlsRecord *r = (TlsRecord *)p;
    if (!r) return;
    free(r->data);
    free(r);
}

static pthread_mutex_t g_key_lock = PTHREAD_MUTEX_INITIALIZER;

/* Creates the key on first use (double-checked: the common path is one atomic load). */
static int tls_key_ready(TnTlsScratch *s) {
    int st = atomic_load_explicit(&s->state, memory_order_acquire);
    if (st != 0) return st > 0;
    pthread_mutex_lock(&g_key_lock);
    st = atomic_load_explicit(&s->state, memory_order_relaxed);
    if (st == 0) {
        st = pthread_key_create(&s->key, tls_record_free) == 0 ? 1 : -1;
        atomic_store_explicit(&s->state, st, memory_order_release);
    }
    pthread_mutex_unlock(&g_key_lock);
    return st > 0;
}

void *tn_tls_scratch(TnTlsScratch *s, size_t bytes) {
    if (!s || !tls_key_ready(s)) return NULL;
    TlsRecord *r = (TlsRecord *)pthread_getspecific(s->key);
    if (!r) {
        r = (TlsRecord *)calloc(1, sizeof(*r));
        if (!r) return NULL;
        if (pthread_setspecific(s->key, r) != 0) {
            free(r);
            return NULL;
        }
    }
    if (r->cap < bytes || !r->data) {
        size_t want = bytes ? bytes : 1;
        void *nd    = malloc(want);
        if (!nd) return NULL;
        free(r->data);
        r->data = nd;
        r->cap  = want;
    }
    return r->data;
}
