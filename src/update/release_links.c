/*
 * update/release_links.c — see update/release_links.h.
 */
#include "update/release_links.h"
#include "update/semver.h"
#include "update/update_check.h"

#include <stdio.h>
#include <string.h>

int release_web_base(const char *api_base, char *out, size_t n) {
    if (!out || n == 0) return -1;
    out[0] = '\0';
    if (!api_base || strncmp(api_base, "https://", 8) != 0) return -1;
    const char *host = api_base + 8;
    size_t hlen = strcspn(host, "/");
    if (hlen == 0) return -1;
    const char *path = host + hlen;
    int w;
    if (hlen > 4 && strncmp(host, "api.", 4) == 0 && (*path == '\0' || strcmp(path, "/") == 0))
        w = snprintf(out, n, "https://%.*s", (int)(hlen - 4), host + 4);     /* api.github.com */
    else if (strncmp(path, "/api/v3", 7) == 0 && (path[7] == '\0' || path[7] == '/'))
        w = snprintf(out, n, "https://%.*s", (int)hlen, host);               /* GHE */
    else
        return -1;
    if (w < 0 || (size_t)w >= n) { out[0] = '\0'; return -1; }
    return 0;
}

/* Release tag ("v1.2.3[-pre]") when version is exactly a tag, else -1. */
static int exact_tag(const char *version, char *tag, size_t n) {
    SemVer v;
    if (!version || semver_parse(version, &v) != 0 || v.commits_ahead != 0) return -1;
    char core[64];
    semver_format(&v, core, (int)sizeof(core));
    int w = snprintf(tag, n, "v%s", core);
    return (w < 0 || (size_t)w >= n) ? -1 : 0;
}

static int build(const char *api_base, const char *repo, const char *fmt, const char *ref,
                 char *out, size_t n) {
    char web[160];
    if (!out || n == 0) return -1;
    out[0] = '\0';
    if (!repo || !update_repo_valid(repo) || release_web_base(api_base, web, sizeof(web)) != 0)
        return -1;
    int w = snprintf(out, n, fmt, web, repo, ref);
    if (w < 0 || (size_t)w >= n) { out[0] = '\0'; return -1; }
    return 0;
}

int release_notes_url(const char *api_base, const char *repo, const char *version,
                      char *out, size_t n) {
    char tag[80];
    if (out && n) out[0] = '\0';
    if (exact_tag(version, tag, sizeof(tag)) != 0) return -1;
    return build(api_base, repo, "%s/%s/releases/tag/%s", tag, out, n);
}

int release_changelog_url(const char *api_base, const char *repo, const char *version,
                          char *out, size_t n) {
    char tag[80];
    const char *ref = exact_tag(version, tag, sizeof(tag)) == 0 ? tag : "HEAD";
    return build(api_base, repo, "%s/%s/blob/%s/CHANGELOG.md", ref, out, n);
}
