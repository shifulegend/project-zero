#ifndef TN_UPDATE_CHECK_H
#define TN_UPDATE_CHECK_H

/*
 * update_check.h — Phase 38: update notifications for installed builds.
 *
 * Notify-only (never downloads or replaces the binary): the user decides when
 * and how to upgrade, and verifies the release's SHA256SUMS. Design follows
 * common practice for CLI update notifiers (gh, npm update-notifier):
 *   - at most one network check per interval (default 24 h), result cached
 *     on disk; failed checks back off for an hour;
 *   - the check runs off the main path (background thread) and fails silently;
 *   - opt-out by flag (--no-update-check / --update-interval 0) or env
 *     (PZ_NO_UPDATE_CHECK, DO_NOT_TRACK), and off automatically in CI;
 *   - the request carries nothing but "project-zero/<version>" as User-Agent;
 *   - HTTPS only, through the system `curl` (no TLS stack in the engine),
 *     argv passed directly (no shell), response size capped.
 * Source of truth: GitHub Releases of --update-repo (owner/name). Drafts are
 * ignored; pre-releases count on the prerelease channel ("auto" picks it
 * while the running build is itself 0.x or a pre-release).
 */

#include <stddef.h>

typedef enum {
    UPDATE_CHANNEL_AUTO = 0,
    UPDATE_CHANNEL_STABLE,
    UPDATE_CHANNEL_PRERELEASE
} UpdateChannel;

typedef enum {
    UPDATE_STATUS_UNKNOWN = 0,  /* not checked yet, or local version unknown */
    UPDATE_STATUS_DISABLED,
    UPDATE_STATUS_UP_TO_DATE,
    UPDATE_STATUS_AVAILABLE,
    UPDATE_STATUS_ERROR
} UpdateStatus;

typedef struct {
    int  enabled;             /* 0: --no-update-check */
    char repo[128];           /* "owner/name" */
    char api_base[128];       /* "https://api.github.com" (GHE / mirror override) */
    int  interval_hours;      /* between network checks; 0 disables (default 24) */
    int  channel;             /* UpdateChannel */
    int  timeout_s;           /* network timeout (default 5) */
    int  force_min_interval_s;/* spacing for forced checks, e.g. UI "check now" (60) */
    char cache_path[512];     /* "" = $PZ_CACHE_DIR | $XDG_CACHE_HOME | ~/.cache /project-zero/update-check */
} UpdateOptions;

typedef struct {
    int       status;          /* UpdateStatus */
    char      current[64];     /* running build's version string */
    char      latest[64];      /* newest eligible release tag, "" if none */
    char      url[256];        /* its release page */
    char      published[32];   /* ISO-8601 publish time */
    int       latest_prerelease;
    long long checked_at;      /* epoch seconds of the last network attempt */
    int       channel;         /* channel the cached result was fetched for */
    char      repo[128];       /* repo the cached result was fetched for */
    char      error[96];       /* last failure, "" on success */
    int       from_cache;
} UpdateInfo;

#define UPDATE_DEFAULT_INTERVAL_H 24
#define UPDATE_DEFAULT_TIMEOUT_S  5
#define UPDATE_ERROR_RETRY_S      3600
#define UPDATE_MAX_BODY           (2u * 1024u * 1024u)

/* Defaults: enabled, PZ_UPDATE_REPO (build-time) or the upstream repo, 24 h. */
void update_options_default(UpdateOptions *o);

/* "owner/name" with [A-Za-z0-9._-] only. */
int update_repo_valid(const char *repo);
/* https:// URL without whitespace or quotes. */
int update_api_base_valid(const char *base);

/* Name of the environment variable that disables checks, or NULL. */
const char *update_env_opt_out(void);

const char *update_status_name(int status);
const char *update_channel_name(int channel);
int         update_channel_parse(const char *s); /* -1 on error */

/* Pure: pick the newest eligible release from a GitHub /releases JSON array
 * and set latest/url/published/latest_prerelease + status vs `current`.
 * Returns 0, or -1 if the JSON holds no array. */
int update_pick_latest(const char *json, const char *current, int channel, UpdateInfo *out);

/* Recompute out->status from out->latest vs current. */
void update_compute_status(UpdateInfo *out, const char *current);

int update_cache_path(const UpdateOptions *o, char *buf, size_t n);
int update_cache_load(const char *path, UpdateInfo *out); /* 0 = loaded */
int update_cache_save(const char *path, const UpdateInfo *in);

/* Transport hook (tests inject a fake). Returns 0 and a malloc'd body. */
typedef int (*UpdateFetchFn)(const char *url, int timeout_s, const char *user_agent, char **body);
void update_set_fetcher(UpdateFetchFn fn); /* NULL restores curl */

/* Synchronous, cache-aware check. `force` skips the interval (still honours
 * enabled=0 only via the caller — --check-update is an explicit request). */
int update_check(const UpdateOptions *o, const char *current, int force, UpdateInfo *out);

/* ---- background service (server + interactive CLI) -------------------- */
void update_service_init(const UpdateOptions *o, const char *current);
/* Starts a background check when stale (or forced and the last attempt is
 * older than force_min_interval_s). Returns 1 if a check was started. */
int  update_service_poll(int force);
void update_service_get(UpdateInfo *out);
int  update_service_options(UpdateOptions *out); /* 0 if not initialised */
void update_service_wait(void);   /* join an in-flight check (tests, exit) */

/* One-line notice for stderr, or "" when nothing to say. */
void update_format_notice(const UpdateInfo *in, char *buf, size_t n);

#endif /* TN_UPDATE_CHECK_H */
