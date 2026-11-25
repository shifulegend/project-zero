#include "update/semver.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_num(const char **p, int *out) {
    if (!isdigit((unsigned char)**p)) return -1;
    long v = 0;
    while (isdigit((unsigned char)**p)) {
        v = v * 10 + (**p - '0');
        if (v > 1000000) return -1;
        (*p)++;
    }
    *out = (int)v;
    return 0;
}

/* "-<n>-g<hex>" (optionally followed by "-dirty") — git describe suffix. */
static int parse_describe(const char *p, int *ahead) {
    if (*p != '-' || !isdigit((unsigned char)p[1])) return -1;
    p++;
    int n = 0;
    if (parse_num(&p, &n) != 0 || p[0] != '-' || p[1] != 'g') return -1;
    p += 2;
    if (!isxdigit((unsigned char)*p)) return -1;
    while (isxdigit((unsigned char)*p)) p++;
    if (*p && strcmp(p, "-dirty") != 0) return -1;
    *ahead = n;
    return 0;
}

/* [vV]<major>.<minor>.<patch>; 0 on success, *pp after the patch number. */
static int parse_core(const char **pp, SemVer *out) {
    const char *p = *pp;
    if (*p == 'v' || *p == 'V') p++;
    if (parse_num(&p, &out->major) || *p++ != '.' || parse_num(&p, &out->minor) || *p++ != '.' ||
        parse_num(&p, &out->patch))
        return -1;
    *pp = p;
    return 0;
}

/* Reads the identifiers of a pre-release tag after its '-': up to '+' or the end, only
 * [A-Za-z0-9.-]. A trailing "-dirty" is dropped. Returns the tag length, -1 on a bad character or
 * an empty tag. *pp is left after the tag. */
static int read_prerelease(const char **pp, char *pre, size_t cap) {
    const char *p = *pp + 1;
    size_t n      = 0;
    while (*p && *p != '+' && n + 1 < cap) {
        if (!isalnum((unsigned char)*p) && *p != '.' && *p != '-') return -1;
        pre[n++] = *p++;
    }
    pre[n] = '\0';
    if (n == 0) return -1;
    if (n > 6 && strcmp(pre + n - 6, "-dirty") == 0) pre[n - 6] = '\0';
    *pp = p;
    return (int)n;
}

/* "v1.0.0-rc.1-3-gabc": a describe suffix glued onto the pre-release moves to commits_ahead. */
static void split_glued_describe(SemVer *out) {
    char *g = strstr(out->pre, "-g");
    if (!g) return;
    char *d = g;
    while (d > out->pre && d[-1] != '-') d--;
    if (d > out->pre && parse_describe(d - 1, &out->commits_ahead) == 0) d[-1] = '\0';
}

/* A pre-release tag after its '-'. 0 on success, *pp after the tag. */
static int parse_prerelease(const char **pp, SemVer *out) {
    if (read_prerelease(pp, out->pre, sizeof(out->pre)) < 0) return -1;
    split_glued_describe(out);
    return 0;
}

/* What follows the patch number ends the version: nothing, a bare "-dirty", or a git-describe
 * suffix ("-<n>-g<hex>", sets commits_ahead). */
static int ends_version(const char *p, SemVer *out) {
    return *p == '\0' || strcmp(p, "-dirty") == 0 || parse_describe(p, &out->commits_ahead) == 0;
}

int semver_parse(const char *s, SemVer *out) {
    if (!s || !out) return -1;
    memset(out, 0, sizeof(*out));
    const char *p = s;
    if (parse_core(&p, out) != 0) return -1;
    if (ends_version(p, out)) return 0;
    if (*p == '-' && parse_prerelease(&p, out) != 0) return -1;
    if (*p == '+') return 0; /* build metadata ignored */
    return *p == '\0' ? 0 : -1;
}

static int is_numeric(const char *s, size_t n) {
    if (n == 0) return 0;
    for (size_t i = 0; i < n; i++)
        if (!isdigit((unsigned char)s[i])) return 0;
    return 1;
}

/* SemVer §11: a release outranks its pre-releases; identifiers compare
 * numerically when both numeric, else lexically; numeric < alphanumeric. */
static int cmp_pre(const char *a, const char *b) {
    if (!*a && !*b) return 0;
    if (!*a) return 1;
    if (!*b) return -1;
    while (*a || *b) {
        if (!*a) return -1;
        if (!*b) return 1;
        size_t la = strcspn(a, "."), lb = strcspn(b, ".");
        int na = is_numeric(a, la), nb = is_numeric(b, lb);
        int c;
        if (na && nb) {
            long x = strtol(a, NULL, 10), y = strtol(b, NULL, 10);
            c = (x > y) - (x < y);
        } else if (na != nb) {
            c = na ? -1 : 1;
        } else {
            size_t m = la < lb ? la : lb;
            c        = strncmp(a, b, m);
            if (c == 0) c = (la > lb) - (la < lb);
        }
        if (c) return c;
        a += la;
        b += lb;
        if (*a == '.') a++;
        if (*b == '.') b++;
    }
    return 0;
}

int semver_cmp(const SemVer *a, const SemVer *b) {
    if (a->major != b->major) return a->major < b->major ? -1 : 1;
    if (a->minor != b->minor) return a->minor < b->minor ? -1 : 1;
    if (a->patch != b->patch) return a->patch < b->patch ? -1 : 1;
    return cmp_pre(a->pre, b->pre);
}

void semver_format(const SemVer *v, char *buf, int buf_size) {
    if (!buf || buf_size <= 0) return;
    if (v->pre[0])
        snprintf(buf, (size_t)buf_size, "%d.%d.%d-%s", v->major, v->minor, v->patch, v->pre);
    else
        snprintf(buf, (size_t)buf_size, "%d.%d.%d", v->major, v->minor, v->patch);
}
