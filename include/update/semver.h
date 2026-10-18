#ifndef TN_SEMVER_H
#define TN_SEMVER_H

/*
 * semver.h — Phase 38: Semantic Versioning 2.0 parse/compare for update checks.
 *
 * Accepts an optional leading 'v' and the `git describe` form the build
 * embeds ("v0.3.1-4-gabc123[-dirty]": parsed as 0.3.1 with commits_ahead=4).
 * A plain commit hash or "dev" does not parse (no local version known).
 */

typedef struct {
    int  major, minor, patch;
    char pre[32];       /* pre-release identifiers ("rc.1"), "" for a release */
    int  commits_ahead; /* git describe distance past the tag, 0 otherwise */
} SemVer;

/* 0 on success, -1 if `s` is not a version. */
int semver_parse(const char *s, SemVer *out);

/* <0, 0, >0 by SemVer precedence (build metadata / commits_ahead ignored). */
int semver_cmp(const SemVer *a, const SemVer *b);

/* "1.2.3" or "1.2.3-rc.1" into buf. */
void semver_format(const SemVer *v, char *buf, int buf_size);

#endif /* TN_SEMVER_H */
