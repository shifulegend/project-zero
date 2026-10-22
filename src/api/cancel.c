/*
 * Phase 22 — src/api/cancel.c
 * See include/api/cancel.h for responsibility.
 */

#include "api/cancel.h"
#include <stdio.h>
#include <string.h>

void cancel_state_init(CancelState *cs) {
    if (!cs) return;
    memset(cs, 0, sizeof(*cs));
    pthread_mutex_init(&cs->lock, NULL);
}

void cancel_state_destroy(CancelState *cs) {
    if (!cs) return;
    pthread_mutex_destroy(&cs->lock);
}

/* Active entry for id, or NULL. Caller holds the lock. */
static CancelEntry *find(CancelState *cs, const char *id) {
    if (!id) return NULL;
    for (int i = 0; i < CANCEL_MAX_ACTIVE; i++)
        if (cs->entries[i].active && strcmp(cs->entries[i].id, id) == 0) return &cs->entries[i];
    return NULL;
}

int cancel_registry_begin(CancelState *cs, const char *id) {
    if (!cs) return -1;
    int rc = -1;
    pthread_mutex_lock(&cs->lock);
    for (int i = 0; i < CANCEL_MAX_ACTIVE; i++) {
        CancelEntry *e = &cs->entries[i];
        if (e->active) continue;
        snprintf(e->id, sizeof(e->id), "%s", id ? id : "");
        e->active = 1;
        e->cancel_requested = 0;
        rc = 0;
        break;
    }
    pthread_mutex_unlock(&cs->lock);
    return rc;
}

void cancel_registry_end(CancelState *cs, const char *id) {
    if (!cs) return;
    pthread_mutex_lock(&cs->lock);
    CancelEntry *e = find(cs, id);
    if (e) memset(e, 0, sizeof(*e));
    pthread_mutex_unlock(&cs->lock);
}

int cancel_registry_check(CancelState *cs, const char *id) {
    if (!cs) return 0;
    pthread_mutex_lock(&cs->lock);
    CancelEntry *e = find(cs, id);
    int result = cs->shutdown || (e && e->cancel_requested);
    pthread_mutex_unlock(&cs->lock);
    return result;
}

int cancel_registry_request(CancelState *cs, const char *id) {
    if (!cs || !id) return 0;
    pthread_mutex_lock(&cs->lock);
    CancelEntry *e = find(cs, id);
    if (e) e->cancel_requested = 1;
    pthread_mutex_unlock(&cs->lock);
    return e != NULL;
}

void cancel_registry_shutdown(CancelState *cs) {
    if (!cs) return;
    pthread_mutex_lock(&cs->lock);
    cs->shutdown = 1;
    pthread_mutex_unlock(&cs->lock);
}
