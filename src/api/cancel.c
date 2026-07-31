/*
 * Phase 22 — src/api/cancel.c
 * See include/api/cancel.h for responsibility.
 */

#include "api/cancel.h"
#include <stdio.h>
#include <string.h>

void cancel_state_init(CancelState *cs) {
    if (!cs) return;
    pthread_mutex_init(&cs->lock, NULL);
    cs->id[0] = '\0';
    cs->active = 0;
    cs->cancel_requested = 0;
}

void cancel_state_destroy(CancelState *cs) {
    if (!cs) return;
    pthread_mutex_destroy(&cs->lock);
}

void cancel_registry_begin(CancelState *cs, const char *id) {
    if (!cs) return;
    pthread_mutex_lock(&cs->lock);
    snprintf(cs->id, sizeof(cs->id), "%s", id ? id : "");
    cs->active = 1;
    cs->cancel_requested = 0;
    pthread_mutex_unlock(&cs->lock);
}

void cancel_registry_end(CancelState *cs) {
    if (!cs) return;
    pthread_mutex_lock(&cs->lock);
    cs->active = 0;
    cs->cancel_requested = 0;
    cs->id[0] = '\0';
    pthread_mutex_unlock(&cs->lock);
}

int cancel_registry_check(CancelState *cs, const char *id) {
    if (!cs) return 0;
    int result = 0;
    pthread_mutex_lock(&cs->lock);
    if (cs->active && cs->cancel_requested &&
        id && strcmp(cs->id, id) == 0) {
        result = 1;
    }
    pthread_mutex_unlock(&cs->lock);
    return result;
}

int cancel_registry_request(CancelState *cs, const char *id) {
    if (!cs || !id) return 0;
    int matched = 0;
    pthread_mutex_lock(&cs->lock);
    if (cs->active && strcmp(cs->id, id) == 0) {
        cs->cancel_requested = 1;
        matched = 1;
    }
    pthread_mutex_unlock(&cs->lock);
    return matched;
}
