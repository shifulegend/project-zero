#include "update/update_check.h"
#include "update/semver.h"
#include "api/json_fields.h"
#include "core/platform.h"
#include <ctype.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if TN_POSIX
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

/* Upstream repository; packagers/forks override at build time
 * (make PZ_UPDATE_REPO=owner/name) or at run time (--update-repo). */
#ifndef PZ_UPDATE_REPO
#define PZ_UPDATE_REPO shifulegend/project-zero
#endif
#define UPD_STR2(x) #x
#define UPD_STR(x) UPD_STR2(x)

static void copy_str(char *dst, size_t n, const char *src) {
    if (!dst || n == 0) return;
    snprintf(dst, n, "%s", src ? src : "");
}

void update_options_default(UpdateOptions *o) {
    memset(o, 0, sizeof(*o));
    o->enabled = 1;
    copy_str(o->repo, sizeof(o->repo), UPD_STR(PZ_UPDATE_REPO));
    copy_str(o->api_base, sizeof(o->api_base), "https://api.github.com");
    o->interval_hours = UPDATE_DEFAULT_INTERVAL_H;
    o->channel = UPDATE_CHANNEL_AUTO;
    o->timeout_s = UPDATE_DEFAULT_TIMEOUT_S;
    o->force_min_interval_s = 60;
}

int update_repo_valid(const char *repo) {
    if (!repo || !*repo || strlen(repo) >= 128) return 0;
    int slash = 0;
    for (const char *p = repo; *p; p++) {
        if (*p == '/') { if (p == repo || !p[1] || ++slash > 1) return 0; continue; }
        if (!isalnum((unsigned char)*p) && *p != '.' && *p != '_' && *p != '-') return 0;
    }
    return slash == 1 && strstr(repo, "..") == NULL;
}

int update_api_base_valid(const char *base) {
    if (!base || strncmp(base, "https://", 8) != 0 || strlen(base) >= 128 || !base[8]) return 0;
    for (const char *p = base; *p; p++)
        if (isspace((unsigned char)*p) || *p == '"' || *p == '\'' || *p == '\\' || (unsigned char)*p < 0x20)
            return 0;
    return 1;
}

const char *update_env_opt_out(void) {
    static const char *vars[] = { "PZ_NO_UPDATE_CHECK", "DO_NOT_TRACK", "CI" };
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); i++) {
        const char *v = getenv(vars[i]);
        if (v && *v && strcmp(v, "0") != 0 && strcmp(v, "false") != 0) return vars[i];
    }
    return NULL;
}

const char *update_status_name(int s) {
    switch (s) {
    case UPDATE_STATUS_DISABLED:   return "disabled";
    case UPDATE_STATUS_UP_TO_DATE: return "up_to_date";
    case UPDATE_STATUS_AVAILABLE:  return "available";
    case UPDATE_STATUS_ERROR:      return "error";
    default:                       return "unknown";
    }
}

const char *update_channel_name(int c) {
    return c == UPDATE_CHANNEL_STABLE ? "stable" : c == UPDATE_CHANNEL_PRERELEASE ? "prerelease" : "auto";
}

int update_channel_parse(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "auto") == 0) return UPDATE_CHANNEL_AUTO;
    if (strcmp(s, "stable") == 0) return UPDATE_CHANNEL_STABLE;
    if (strcmp(s, "prerelease") == 0 || strcmp(s, "pre") == 0) return UPDATE_CHANNEL_PRERELEASE;
    return -1;
}

static int include_prereleases(int channel, const char *current) {
    if (channel == UPDATE_CHANNEL_STABLE) return 0;
    if (channel == UPDATE_CHANNEL_PRERELEASE) return 1;
    SemVer c;
    if (semver_parse(current, &c) != 0) return 1;
    return c.major == 0 || c.pre[0] != '\0';
}

void update_compute_status(UpdateInfo *out, const char *current) {
    copy_str(out->current, sizeof(out->current), current);
    SemVer cur, lat;
    if (!out->latest[0] || semver_parse(out->latest, &lat) != 0) {
        out->status = out->error[0] ? UPDATE_STATUS_ERROR : UPDATE_STATUS_UNKNOWN;
        return;
    }
    if (semver_parse(current, &cur) != 0) { out->status = UPDATE_STATUS_UNKNOWN; return; }
    int c = semver_cmp(&lat, &cur);
    /* A build N commits past tag X is newer than X itself. */
    if (c == 0 && cur.commits_ahead > 0) c = -1;
    out->status = c > 0 ? UPDATE_STATUS_AVAILABLE : UPDATE_STATUS_UP_TO_DATE;
}

/* Next top-level element of a JSON array as [*s, *s + *len). Returns 0 at end. */
static int next_element(const char **p, const char **s, size_t *len) {
    const char *q = *p;
    while (*q && (isspace((unsigned char)*q) || *q == ',')) q++;
    if (!*q || *q == ']') { *p = q; return 0; }
    const char *start = q;
    int depth = 0, in_str = 0;
    for (; *q; q++) {
        if (in_str) {
            if (*q == '\\') { if (q[1]) q++; continue; }
            if (*q == '"') in_str = 0;
            continue;
        }
        if (*q == '"') in_str = 1;
        else if (*q == '{' || *q == '[') depth++;
        else if (*q == '}' || *q == ']') {
            if (depth == 0) break;           /* closing bracket of the array */
            if (--depth == 0) { q++; break; }
        } else if (*q == ',' && depth == 0) break;
    }
    *s = start;
    *len = (size_t)(q - start);
    *p = q;
    return *len > 0;
}

int update_pick_latest(const char *json, const char *current, int channel, UpdateInfo *out) {
    if (!json || !out) return -1;
    const char *p = json;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '[') return -1;
    p++;
    int want_pre = include_prereleases(channel, current);
    SemVer best;
    int have = 0;
    out->latest[0] = out->url[0] = out->published[0] = '\0';
    out->latest_prerelease = 0;
    const char *el;
    size_t n;
    while (next_element(&p, &el, &n)) {
        char *obj = (char *)malloc(n + 1);
        if (!obj) break;
        memcpy(obj, el, n);
        obj[n] = '\0';
        int draft = 0, pre = 0;
        json_field_bool(obj, "draft", &draft);
        json_field_bool(obj, "prerelease", &pre);
        char *tag = json_field_string(obj, "tag_name");
        SemVer v;
        if (tag && !draft && semver_parse(tag, &v) == 0 && (want_pre || (!pre && !v.pre[0]))) {
            if (!have || semver_cmp(&v, &best) > 0) {
                best = v;
                have = 1;
                char *url = json_field_string(obj, "html_url");
                char *pub = json_field_string(obj, "published_at");
                copy_str(out->latest, sizeof(out->latest), tag);
                /* Only https release pages are surfaced as links. */
                copy_str(out->url, sizeof(out->url), url && strncmp(url, "https://", 8) == 0 ? url : "");
                copy_str(out->published, sizeof(out->published), pub);
                out->latest_prerelease = pre || v.pre[0];
                free(url);
                free(pub);
            }
        }
        free(tag);
        free(obj);
    }
    out->channel = channel;
    update_compute_status(out, current);
    return 0;
}

/* ---- cache --------------------------------------------------------------- */

int update_cache_path(const UpdateOptions *o, char *buf, size_t n) {
    if (o && o->cache_path[0]) { copy_str(buf, n, o->cache_path); return 0; }
    const char *d = getenv("PZ_CACHE_DIR");
    if (d && *d) { snprintf(buf, n, "%s/update-check", d); return 0; }
    d = getenv("XDG_CACHE_HOME");
    if (d && *d) { snprintf(buf, n, "%s/project-zero/update-check", d); return 0; }
    d = getenv("HOME");
    if (d && *d) { snprintf(buf, n, "%s/.cache/project-zero/update-check", d); return 0; }
    return -1;
}

int update_cache_load(const char *path, UpdateInfo *out) {
    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f) return -1;
    char line[512];
    int ok = 0;
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = '\0';
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';
        const char *k = line, *v = eq + 1;
        if (strcmp(k, "checked_at") == 0)      { out->checked_at = strtoll(v, NULL, 10); ok = 1; }
        else if (strcmp(k, "latest") == 0)     copy_str(out->latest, sizeof(out->latest), v);
        else if (strcmp(k, "url") == 0)        copy_str(out->url, sizeof(out->url), strncmp(v, "https://", 8) == 0 ? v : "");
        else if (strcmp(k, "published") == 0)  copy_str(out->published, sizeof(out->published), v);
        else if (strcmp(k, "prerelease") == 0) out->latest_prerelease = atoi(v) != 0;
        else if (strcmp(k, "channel") == 0)    out->channel = atoi(v);
        else if (strcmp(k, "repo") == 0)       copy_str(out->repo, sizeof(out->repo), v);
        else if (strcmp(k, "error") == 0)      copy_str(out->error, sizeof(out->error), v);
    }
    fclose(f);
    return ok ? 0 : -1;
}

#if TN_POSIX
static void mkdir_parents(const char *path) {
    char tmp[512];
    copy_str(tmp, sizeof(tmp), path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        mkdir(tmp, 0700);
        *p = '/';
    }
}
#endif

static void sanitize_line(char *dst, size_t n, const char *src) {
    size_t i = 0;
    for (; src && *src && i + 1 < n; src++)
        dst[i++] = (*src == '\n' || *src == '\r') ? ' ' : *src;
    dst[i] = '\0';
}

int update_cache_save(const char *path, const UpdateInfo *in) {
    if (!path || !in) return -1;
#if TN_POSIX
    mkdir_parents(path);
#endif
    char tmp[600];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) return -1;
    char latest[64], url[256], pub[32], err[96], repo[128];
    sanitize_line(latest, sizeof(latest), in->latest);
    sanitize_line(url, sizeof(url), in->url);
    sanitize_line(pub, sizeof(pub), in->published);
    sanitize_line(err, sizeof(err), in->error);
    sanitize_line(repo, sizeof(repo), in->repo);
    fprintf(f, "checked_at=%lld\nlatest=%s\nurl=%s\npublished=%s\nprerelease=%d\nchannel=%d\nrepo=%s\nerror=%s\n",
            in->checked_at, latest, url, pub, in->latest_prerelease, in->channel, repo, err);
    int bad = ferror(f);
    if (fclose(f) != 0 || bad) { remove(tmp); return -1; }
    return rename(tmp, path) == 0 ? 0 : -1; /* atomic replace: readers never see a torn file */
}

/* ---- transport ----------------------------------------------------------- */

#if TN_POSIX
static int curl_fetch(const char *url, int timeout_s, const char *ua, char **body) {
    *body = NULL;
    int fds[2];
    if (pipe(fds) != 0) return -1;
    char tmo[16];
    snprintf(tmo, sizeof(tmo), "%d", timeout_s > 0 ? timeout_s : UPDATE_DEFAULT_TIMEOUT_S);
    char maxsz[24];
    snprintf(maxsz, sizeof(maxsz), "%u", UPDATE_MAX_BODY);
    /* posix_spawnp, not fork+exec: the server is multithreaded, and spawn
     * gives the child a clean start without running anything async-unsafe. */
    posix_spawn_file_actions_t fa;
    if (posix_spawn_file_actions_init(&fa) != 0) { close(fds[0]); close(fds[1]); return -1; }
    posix_spawn_file_actions_addopen(&fa, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    /* argv goes straight to the program — no shell, so nothing in url/ua is interpreted. */
    char *const argv[] = {
        "curl", "-fsSL", "--proto", "=https", "--proto-redir", "=https",
        "--max-time", tmo, "--max-filesize", maxsz,
        "-H", "Accept: application/vnd.github+json",
        "-H", "X-GitHub-Api-Version: 2022-11-28",
        "-A", (char *)ua, (char *)url, NULL
    };
    pid_t pid;
    int rc = posix_spawnp(&pid, "curl", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) { close(fds[0]); close(fds[1]); return -1; }
    close(fds[1]);
    size_t cap = 65536, len = 0;
    char *buf = (char *)malloc(cap + 1);
    int over = 0;
    for (;;) {
        if (!buf) break;
        if (len == cap) {
            if (cap >= UPDATE_MAX_BODY) { over = 1; break; }
            size_t nc = cap * 2 > UPDATE_MAX_BODY ? UPDATE_MAX_BODY : cap * 2;
            char *nb = (char *)realloc(buf, nc + 1);
            if (!nb) { free(buf); buf = NULL; break; }
            buf = nb;
            cap = nc;
        }
        ssize_t r = read(fds[0], buf + len, cap - len);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        len += (size_t)r;
    }
    close(fds[0]);
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    if (!buf || over || !WIFEXITED(st) || WEXITSTATUS(st) != 0 || len == 0) {
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    *body = buf;
    return 0;
}
#else
static int curl_fetch(const char *url, int timeout_s, const char *ua, char **body) {
    (void)url; (void)timeout_s; (void)ua;
    *body = NULL;
    return -1;
}
#endif

static UpdateFetchFn g_fetch = curl_fetch;
void update_set_fetcher(UpdateFetchFn fn) { g_fetch = fn ? fn : curl_fetch; }

/* ---- synchronous check ----------------------------------------------------- */

static int cache_fresh(const UpdateOptions *o, const UpdateInfo *c, long long now) {
    if (c->checked_at <= 0 || c->channel != o->channel || strcmp(c->repo, o->repo) != 0) return 0;
    long long age = now - c->checked_at;
    if (age < 0) return 0; /* clock went backwards: re-check */
    long long ttl = c->error[0] ? UPDATE_ERROR_RETRY_S : (long long)o->interval_hours * 3600;
    return age < ttl;
}

int update_check(const UpdateOptions *o, const char *current, int force, UpdateInfo *out) {
    memset(out, 0, sizeof(*out));
    copy_str(out->current, sizeof(out->current), current);
    if (!o) return -1;
    out->channel = o->channel;
    copy_str(out->repo, sizeof(out->repo), o->repo);
    if (!force && (!o->enabled || o->interval_hours <= 0 || update_env_opt_out())) {
        out->status = UPDATE_STATUS_DISABLED;
        return 0;
    }
    if (!update_repo_valid(o->repo) || !update_api_base_valid(o->api_base)) {
        copy_str(out->error, sizeof(out->error), "invalid --update-repo / --update-api");
        out->status = UPDATE_STATUS_ERROR;
        return -1;
    }
    char path[512];
    int have_path = update_cache_path(o, path, sizeof(path)) == 0;
    long long now = (long long)time(NULL);
    UpdateInfo cached;
    memset(&cached, 0, sizeof(cached));
    if (have_path && update_cache_load(path, &cached) == 0 && !force && cache_fresh(o, &cached, now)) {
        *out = cached;
        out->from_cache = 1;
        update_compute_status(out, current);
        return 0;
    }
    char url[320], ua[96];
    snprintf(url, sizeof(url), "%s/repos/%s/releases?per_page=20", o->api_base, o->repo);
    snprintf(ua, sizeof(ua), "project-zero/%s", current && *current ? current : "unknown");
    char *body = NULL;
    out->checked_at = now;
    if (g_fetch(url, o->timeout_s, ua, &body) != 0 || !body ||
        update_pick_latest(body, current, o->channel, out) != 0) {
        /* Keep the last good answer so a flaky network doesn't hide a known update. */
        if (cached.latest[0] && cached.channel == o->channel && strcmp(cached.repo, o->repo) == 0) {
            copy_str(out->latest, sizeof(out->latest), cached.latest);
            copy_str(out->url, sizeof(out->url), cached.url);
            copy_str(out->published, sizeof(out->published), cached.published);
            out->latest_prerelease = cached.latest_prerelease;
        }
        copy_str(out->error, sizeof(out->error), body ? "unexpected response" : "network check failed");
        update_compute_status(out, current);
        if (!out->latest[0]) out->status = UPDATE_STATUS_ERROR;
    }
    free(body);
    copy_str(out->repo, sizeof(out->repo), o->repo);
    out->channel = o->channel;
    if (have_path) update_cache_save(path, out);
    return 0;
}

/* ---- background service ---------------------------------------------------- */

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static UpdateOptions   g_opts;
static UpdateInfo      g_info;
static char            g_current[64];
static int             g_init, g_in_flight, g_thread_valid;
static pthread_t       g_thread;
static int             g_force_next;

static void *service_thread(void *arg) {
    (void)arg;
    UpdateOptions o;
    char cur[64];
    int force;
    pthread_mutex_lock(&g_mu);
    o = g_opts;
    copy_str(cur, sizeof(cur), g_current);
    force = g_force_next;
    pthread_mutex_unlock(&g_mu);
    UpdateInfo r;
    update_check(&o, cur, force, &r);
    pthread_mutex_lock(&g_mu);
    g_info = r;
    g_in_flight = 0;
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

void update_service_init(const UpdateOptions *o, const char *current) {
    update_service_wait();
    pthread_mutex_lock(&g_mu);
    g_opts = *o;
    copy_str(g_current, sizeof(g_current), current);
    memset(&g_info, 0, sizeof(g_info));
    copy_str(g_info.current, sizeof(g_info.current), current);
    g_info.channel = o->channel;
    copy_str(g_info.repo, sizeof(g_info.repo), o->repo);
    g_info.status = (!o->enabled || o->interval_hours <= 0 || update_env_opt_out())
                    ? UPDATE_STATUS_DISABLED : UPDATE_STATUS_UNKNOWN;
    /* Show the last cached answer immediately (no network). */
    if (g_info.status != UPDATE_STATUS_DISABLED) {
        char path[512];
        UpdateInfo c;
        memset(&c, 0, sizeof(c));
        if (update_cache_path(o, path, sizeof(path)) == 0 && update_cache_load(path, &c) == 0 &&
            c.channel == o->channel && strcmp(c.repo, o->repo) == 0) {
            g_info = c;
            g_info.from_cache = 1;
            update_compute_status(&g_info, current);
        }
    }
    g_init = 1;
    pthread_mutex_unlock(&g_mu);
}

int update_service_poll(int force) {
    pthread_mutex_lock(&g_mu);
    if (!g_init || g_in_flight) { pthread_mutex_unlock(&g_mu); return 0; }
    long long now = (long long)time(NULL);
    int disabled = !g_opts.enabled || g_opts.interval_hours <= 0 || update_env_opt_out();
    int start;
    if (disabled) {
        start = 0; /* the operator's opt-out also wins over a UI "check now" */
    } else if (force) {
        start = g_info.checked_at <= 0 || now - g_info.checked_at >= g_opts.force_min_interval_s;
    } else {
        start = !cache_fresh(&g_opts, &g_info, now);
    }
    if (!start) { pthread_mutex_unlock(&g_mu); return 0; }
    /* Reap the previous, finished thread before starting another. */
    if (g_thread_valid) {
        pthread_t old = g_thread;
        g_thread_valid = 0;
        pthread_mutex_unlock(&g_mu);
        pthread_join(old, NULL);
        pthread_mutex_lock(&g_mu);
        if (g_in_flight) { pthread_mutex_unlock(&g_mu); return 0; }
    }
    g_in_flight = 1;
    g_force_next = force;
    if (pthread_create(&g_thread, NULL, service_thread, NULL) != 0) {
        g_in_flight = 0;
        pthread_mutex_unlock(&g_mu);
        return 0;
    }
    g_thread_valid = 1;
    pthread_mutex_unlock(&g_mu);
    return 1;
}

void update_service_get(UpdateInfo *out) {
    pthread_mutex_lock(&g_mu);
    *out = g_info;
    int in_flight = g_in_flight;
    pthread_mutex_unlock(&g_mu);
    (void)in_flight;
}

int update_service_options(UpdateOptions *out) {
    pthread_mutex_lock(&g_mu);
    int ok = g_init;
    if (ok) *out = g_opts;
    pthread_mutex_unlock(&g_mu);
    return ok;
}

void update_service_wait(void) {
    pthread_mutex_lock(&g_mu);
    int valid = g_thread_valid;
    pthread_t t = g_thread;
    g_thread_valid = 0;
    pthread_mutex_unlock(&g_mu);
    if (valid) pthread_join(t, NULL);
}

void update_format_notice(const UpdateInfo *in, char *buf, size_t n) {
    if (!buf || n == 0) return;
    buf[0] = '\0';
    if (!in || in->status != UPDATE_STATUS_AVAILABLE) return;
    snprintf(buf, n,
             ">_ update available: %s -> %s%s%s%s\n"
             "   verify with SHA256SUMS from the release; silence with --no-update-check or PZ_NO_UPDATE_CHECK=1\n",
             in->current, in->latest, in->latest_prerelease ? " (pre-release)" : "",
             in->url[0] ? "  " : "", in->url);
}
