#include "transformer/stop_seq.h"
#include <stdlib.h>
#include <string.h>

void stop_matcher_init(StopMatcher *m, const char *const *seqs, int n) {
    if (!m) return;
    memset(m, 0, sizeof(*m));
    for (int i = 0; seqs && i < n && m->n_seqs < STOP_MAX_SEQS; i++) {
        if (!seqs[i] || !seqs[i][0]) continue;
        strncpy(m->seqs[m->n_seqs], seqs[i], STOP_MAX_LEN - 1);
        m->seqs[m->n_seqs][STOP_MAX_LEN - 1] = '\0';
        m->n_seqs++;
    }
}

/* Emits pending[0..n) and shifts the remainder down. Returns emit()'s verdict. */
static int emit_prefix(StopMatcher *m, size_t n, StopEmitFn emit, void *ud) {
    int rc = 0;
    if (n > 0 && emit) {
        char saved = m->pending[n];
        m->pending[n] = '\0';
        rc = emit(m->pending, ud);
        m->pending[n] = saved;
    }
    memmove(m->pending, m->pending + n, m->len - n + 1);
    m->len -= n;
    return rc;
}

/* Longest suffix of pending that is a proper prefix of some stop sequence. */
static size_t held_suffix_len(const StopMatcher *m) {
    size_t best = 0;
    for (int s = 0; s < m->n_seqs; s++) {
        size_t sl = strlen(m->seqs[s]);
        size_t max = sl - 1 < m->len ? sl - 1 : m->len;
        for (size_t k = max; k > best; k--) {
            if (memcmp(m->pending + m->len - k, m->seqs[s], k) == 0) { best = k; break; }
        }
    }
    return best;
}

int stop_matcher_feed(StopMatcher *m, const char *piece, StopEmitFn emit, void *ud) {
    if (!m || !piece) return 0;
    if (m->stopped) return 1;
    if (m->n_seqs == 0) return emit ? (emit(piece, ud) != 0) : 0;

    size_t pl = strlen(piece);
    if (m->len + pl + 1 > m->cap) {
        size_t nc = (m->len + pl + 1) * 2;
        char *nb = (char *)realloc(m->pending, nc);
        if (!nb) return emit ? (emit(piece, ud) != 0) : 0; /* degrade: no matching */
        m->pending = nb;
        m->cap = nc;
    }
    memcpy(m->pending + m->len, piece, pl + 1);
    m->len += pl;

    /* Earliest match across all sequences wins. */
    size_t hit = (size_t)-1;
    for (int s = 0; s < m->n_seqs; s++) {
        const char *f = strstr(m->pending, m->seqs[s]);
        if (f && (size_t)(f - m->pending) < hit) hit = (size_t)(f - m->pending);
    }
    if (hit != (size_t)-1) {
        emit_prefix(m, hit, emit, ud);
        m->len = 0;
        m->pending[0] = '\0';
        m->stopped = 1;
        return 1;
    }
    size_t hold = held_suffix_len(m);
    return emit_prefix(m, m->len - hold, emit, ud) != 0;
}

void stop_matcher_flush(StopMatcher *m, StopEmitFn emit, void *ud) {
    if (!m || m->stopped || m->len == 0) return;
    emit_prefix(m, m->len, emit, ud);
}

void stop_matcher_free(StopMatcher *m) {
    if (!m) return;
    free(m->pending);
    m->pending = NULL;
    m->len = m->cap = 0;
}
