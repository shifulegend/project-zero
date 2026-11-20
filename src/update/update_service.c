#include <pthread.h>
#include <string.h>
#include <time.h>

#include "update/update_check.h"

/* Background release check: one UpdateService per owner (the session), no process globals. */

/* ---- background service ---------------------------------------------------- */

/* The first call on a zeroed service creates its mutex; later calls leave it alone. */
static void service_setup(UpdateService *svc) {
    if (svc->ready) return;
    memset(svc, 0, sizeof(*svc));
    pthread_mutex_init(&svc->mu, NULL);
    svc->ready = 1;
}

void update_service_set_fetcher(UpdateService *svc, UpdateFetchFn fn) {
    service_setup(svc);
    svc->fetch = fn;
}

static void *service_thread(void *arg) {
    UpdateService *svc = (UpdateService *)arg;
    UpdateOptions o;
    char cur[64];
    int force;
    pthread_mutex_lock(&svc->mu);
    o = svc->opts;
    update_copy_str(cur, sizeof(cur), svc->current);
    force = svc->force_next;
    pthread_mutex_unlock(&svc->mu);
    UpdateInfo r;
    update_check_with(svc->fetch, &o, cur, force, &r);
    pthread_mutex_lock(&svc->mu);
    svc->info      = r;
    svc->in_flight = 0;
    pthread_mutex_unlock(&svc->mu);
    return NULL;
}

void update_service_init(UpdateService *svc, const UpdateOptions *o, const char *current) {
    service_setup(svc);
    update_service_wait(svc);
    pthread_mutex_lock(&svc->mu);
    svc->opts = *o;
    update_copy_str(svc->current, sizeof(svc->current), current);
    memset(&svc->info, 0, sizeof(svc->info));
    update_copy_str(svc->info.current, sizeof(svc->info.current), current);
    svc->info.channel = o->channel;
    update_copy_str(svc->info.repo, sizeof(svc->info.repo), o->repo);
    svc->info.status = (!o->enabled || o->interval_hours <= 0 || update_env_opt_out())
                           ? UPDATE_STATUS_DISABLED
                           : UPDATE_STATUS_UNKNOWN;
    /* Show the last cached answer immediately (no network). */
    if (svc->info.status != UPDATE_STATUS_DISABLED) {
        char path[512];
        UpdateInfo c;
        memset(&c, 0, sizeof(c));
        if (update_cache_path(o, path, sizeof(path)) == 0 && update_cache_load(path, &c) == 0 &&
            update_cache_matches(&c, o)) {
            svc->info            = c;
            svc->info.from_cache = 1;
            update_compute_status(&svc->info, current);
        }
    }
    svc->configured = 1;
    pthread_mutex_unlock(&svc->mu);
}

/* Whether a background check is due now; the caller holds the lock. */
static int poll_due(const UpdateService *svc, int force, long long now) {
    if (!svc->opts.enabled || svc->opts.interval_hours <= 0 || update_env_opt_out())
        return 0; /* the operator's opt-out also wins over a UI "check now" */
    if (!force) return !update_cache_fresh(&svc->opts, &svc->info, now);
    return svc->info.checked_at <= 0 ||
           now - svc->info.checked_at >= svc->opts.force_min_interval_s;
}

int update_service_poll(UpdateService *svc, int force) {
    if (!svc || !svc->ready) return 0;
    pthread_mutex_lock(&svc->mu);
    if (!svc->configured || svc->in_flight) {
        pthread_mutex_unlock(&svc->mu);
        return 0;
    }
    int start = poll_due(svc, force, (long long)time(NULL));
    if (!start) {
        pthread_mutex_unlock(&svc->mu);
        return 0;
    }
    /* Reap the previous, finished thread before starting another. */
    if (svc->thread_valid) {
        pthread_t old     = svc->thread;
        svc->thread_valid = 0;
        pthread_mutex_unlock(&svc->mu);
        pthread_join(old, NULL);
        pthread_mutex_lock(&svc->mu);
        if (svc->in_flight) {
            pthread_mutex_unlock(&svc->mu);
            return 0;
        }
    }
    svc->in_flight  = 1;
    svc->force_next = force;
    if (pthread_create(&svc->thread, NULL, service_thread, svc) != 0) {
        svc->in_flight = 0;
        pthread_mutex_unlock(&svc->mu);
        return 0;
    }
    svc->thread_valid = 1;
    pthread_mutex_unlock(&svc->mu);
    return 1;
}

void update_service_get(UpdateService *svc, UpdateInfo *out) {
    if (!svc || !svc->ready) {
        memset(out, 0, sizeof(*out));
        return;
    }
    pthread_mutex_lock(&svc->mu);
    *out = svc->info;
    pthread_mutex_unlock(&svc->mu);
}

int update_service_options(UpdateService *svc, UpdateOptions *out) {
    if (!svc || !svc->ready) return 0;
    pthread_mutex_lock(&svc->mu);
    int ok = svc->configured;
    if (ok) *out = svc->opts;
    pthread_mutex_unlock(&svc->mu);
    return ok;
}

void update_service_wait(UpdateService *svc) {
    if (!svc || !svc->ready) return;
    pthread_mutex_lock(&svc->mu);
    int valid         = svc->thread_valid;
    pthread_t t       = svc->thread;
    svc->thread_valid = 0;
    pthread_mutex_unlock(&svc->mu);
    if (valid) pthread_join(t, NULL);
}
