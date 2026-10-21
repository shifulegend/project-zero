/*
 * tests/test_release_links.c — release-notes / changelog links derived from
 * the update settings (GitHub, GitHub Enterprise, invalid inputs, release vs
 * in-between builds).
 */
#include "test_harness.h"
#include "update/release_links.h"
#include <string.h>

static void test_web_base(void) {
    char b[128];
    TEST_ASSERT(release_web_base("https://api.github.com", b, sizeof(b)) == 0 &&
                strcmp(b, "https://github.com") == 0, "api.github.com -> github.com");
    TEST_ASSERT(release_web_base("https://api.github.com/", b, sizeof(b)) == 0 &&
                strcmp(b, "https://github.com") == 0, "trailing slash");
    TEST_ASSERT(release_web_base("https://ghe.corp.example/api/v3", b, sizeof(b)) == 0 &&
                strcmp(b, "https://ghe.corp.example") == 0, "GitHub Enterprise");
    TEST_ASSERT(release_web_base("http://api.github.com", b, sizeof(b)) == -1 && b[0] == '\0',
                "http refused");
    TEST_ASSERT(release_web_base("https://mirror.example/releases", b, sizeof(b)) == -1,
                "unknown layout -> no link");
    TEST_ASSERT(release_web_base("https://api.github.com", b, 8) == -1 && b[0] == '\0',
                "short buffer");
}

static void test_links(void) {
    const char *api = "https://api.github.com", *repo = "owner/project-zero";
    char u[256];
    TEST_ASSERT(release_notes_url(api, repo, "v0.2.0", u, sizeof(u)) == 0 &&
                strcmp(u, "https://github.com/owner/project-zero/releases/tag/v0.2.0") == 0,
                "release build -> its release page");
    TEST_ASSERT(release_notes_url(api, repo, "0.3.0-rc.1", u, sizeof(u)) == 0 &&
                strcmp(u, "https://github.com/owner/project-zero/releases/tag/v0.3.0-rc.1") == 0,
                "pre-release tag, v added");
    TEST_ASSERT(release_notes_url(api, repo, "v0.2.0-5-gabc1234", u, sizeof(u)) == -1 && u[0] == '\0',
                "between releases -> no release notes");
    TEST_ASSERT(release_notes_url(api, repo, "388ca86", u, sizeof(u)) == -1, "bare hash -> none");
    TEST_ASSERT(release_changelog_url(api, repo, "v0.2.0", u, sizeof(u)) == 0 &&
                strcmp(u, "https://github.com/owner/project-zero/blob/v0.2.0/CHANGELOG.md") == 0,
                "release changelog pinned to the tag");
    TEST_ASSERT(release_changelog_url(api, repo, "dev", u, sizeof(u)) == 0 &&
                strcmp(u, "https://github.com/owner/project-zero/blob/HEAD/CHANGELOG.md") == 0,
                "dev build -> default-branch changelog");
    TEST_ASSERT(release_changelog_url(api, "../evil", "dev", u, sizeof(u)) == -1 && u[0] == '\0',
                "invalid repo refused");
    TEST_ASSERT(release_changelog_url(NULL, repo, "dev", u, sizeof(u)) == -1, "NULL api base");
}

int main(void) {
    printf("=== Release links ===\n");
    RUN_TEST(test_web_base);
    RUN_TEST(test_links);
    TEST_SUMMARY();
}
