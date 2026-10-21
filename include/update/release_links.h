#ifndef PZ_RELEASE_LINKS_H
#define PZ_RELEASE_LINKS_H

/*
 * update/release_links — web links to the running version's release notes and
 * changelog (2026-10-21), derived from the update settings (--update-repo,
 * --update-api) so forks, mirrors and GitHub Enterprise get their own links and
 * nothing is hardcoded in the web UI.
 *
 * API base -> web base: "https://api.github.com" -> "https://github.com";
 * GitHub Enterprise "https://host/api/v3" -> "https://host".
 */

#include <stddef.h>

/* Web origin for an API base. Returns 0, or -1 when it cannot be derived. */
int release_web_base(const char *api_base, char *out, size_t n);

/* <web>/<repo>/releases/tag/<tag> when `version` is exactly a release tag
 * (`git describe` with no commits past it); "" and -1 otherwise. The tag is
 * written as v<semver>, the repo's tag convention. */
int release_notes_url(const char *api_base, const char *repo, const char *version,
                      char *out, size_t n);

/* <web>/<repo>/blob/<ref>/CHANGELOG.md, ref = the release tag for a release
 * build, else HEAD (default branch). Returns 0, or -1 with out = "". */
int release_changelog_url(const char *api_base, const char *repo, const char *version,
                          char *out, size_t n);

#endif /* PZ_RELEASE_LINKS_H */
