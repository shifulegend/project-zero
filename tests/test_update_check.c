/*
 * tests/test_update_check.c — Phase 38 update notifications: SemVer parsing
 * and precedence (incl. git-describe builds), release selection from a GitHub
 * /releases payload (drafts, channels), cache round trip + freshness, opt-outs,
 * error back-off keeping the last good answer, the background service, and
 * input validation. Network is replaced by a fake fetcher. Part of TSAN_TESTS.
 */
#include "test_harness.h"
#include "update/semver.h"
#include "update/update_check.h"
#include "api/json_fields.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void test_semver(void) {
    SemVer a, b;
    TEST_ASSERT_EQ(semver_parse("v1.2.3", &a), 0, "v-prefixed release");
    TEST_ASSERT(a.major == 1 && a.minor == 2 && a.patch == 3 && !a.pre[0], "fields");
    TEST_ASSERT_EQ(semver_parse("0.1.0-rc.1", &a), 0, "pre-release");
    TEST_ASSERT(strcmp(a.pre, "rc.1") == 0, "pre identifiers");
    TEST_ASSERT_EQ(semver_parse("v0.1.0-4-gabc123", &a), 0, "git describe");
    TEST_ASSERT(a.commits_ahead == 4 && !a.pre[0], "describe distance, not a pre-release");
    TEST_ASSERT_EQ(semver_parse("v0.1.0-4-gabc123-dirty", &a), 0, "describe dirty");
    TEST_ASSERT_EQ(semver_parse("v1.0.0-rc.1-2-gdeadbee", &a), 0, "describe on a pre-release");
    TEST_ASSERT(strcmp(a.pre, "rc.1") == 0 && a.commits_ahead == 2, "pre kept, distance split off");
    TEST_ASSERT_EQ(semver_parse("1.2.3+build.7", &a), 0, "build metadata ignored");
    TEST_ASSERT(semver_parse("e95dd94-dirty", &a) != 0, "commit hash is not a version");
    TEST_ASSERT(semver_parse("dev", &a) != 0, "dev is not a version");
    TEST_ASSERT(semver_parse("1.2", &a) != 0, "two components rejected");
    TEST_ASSERT(semver_parse("1.2.3-", &a) != 0, "empty pre rejected");

    /* SemVer 2.0 §11 precedence chain. */
    const char *chain[] = { "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
                            "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.0.1", "1.1.0", "2.0.0" };
    int ok = 1;
    for (size_t i = 0; i + 1 < sizeof(chain) / sizeof(chain[0]); i++) {
        semver_parse(chain[i], &a);
        semver_parse(chain[i + 1], &b);
        if (semver_cmp(&a, &b) >= 0 || semver_cmp(&b, &a) <= 0) ok = 0;
    }
    TEST_ASSERT(ok, "SemVer precedence chain");
    char buf[48];
    semver_parse("v3.4.5-rc.2", &a);
    semver_format(&a, buf, sizeof(buf));
    TEST_ASSERT(strcmp(buf, "3.4.5-rc.2") == 0, "format");
}

static const char *RELEASES =
    "[{\"tag_name\":\"v0.3.0\",\"draft\":true,\"prerelease\":false,\"html_url\":\"https://x/d\"},"
    " {\"tag_name\":\"v0.2.0-rc.1\",\"draft\":false,\"prerelease\":true,"
    "  \"html_url\":\"https://github.com/o/r/releases/tag/v0.2.0-rc.1\",\"published_at\":\"2026-10-01T00:00:00Z\","
    "  \"body\":\"notes with \\\"quotes\\\", [brackets] and {braces}\",\"assets\":[{\"name\":\"a\"}]},"
    " {\"tag_name\":\"v0.1.1\",\"draft\":false,\"prerelease\":false,"
    "  \"html_url\":\"https://github.com/o/r/releases/tag/v0.1.1\",\"published_at\":\"2026-09-01T00:00:00Z\"},"
    " {\"tag_name\":\"nightly\",\"draft\":false,\"prerelease\":true,\"html_url\":\"https://x/n\"},"
    " {\"tag_name\":\"v0.1.0\",\"draft\":false,\"prerelease\":false,\"html_url\":\"javascript:alert(1)\"}]";

static void test_pick_latest(void) {
    UpdateInfo u;
    memset(&u, 0, sizeof(u));
    TEST_ASSERT_EQ(update_pick_latest(RELEASES, "v0.1.0", UPDATE_CHANNEL_AUTO, &u), 0, "parse releases");
    TEST_ASSERT(strcmp(u.latest, "v0.2.0-rc.1") == 0, "auto on 0.x includes pre-releases, skips drafts");
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_AVAILABLE, "newer -> available");
    TEST_ASSERT(u.latest_prerelease, "flagged pre-release");
    TEST_ASSERT(strcmp(u.published, "2026-10-01T00:00:00Z") == 0, "published_at");

    memset(&u, 0, sizeof(u));
    update_pick_latest(RELEASES, "v0.1.0", UPDATE_CHANNEL_STABLE, &u);
    TEST_ASSERT(strcmp(u.latest, "v0.1.1") == 0, "stable channel skips pre-releases");

    memset(&u, 0, sizeof(u));
    update_pick_latest(RELEASES, "v0.2.0-rc.1-3-gabc1234", UPDATE_CHANNEL_AUTO, &u);
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_UP_TO_DATE, "build past the latest tag is up to date");

    memset(&u, 0, sizeof(u));
    update_pick_latest(RELEASES, "e95dd94-dirty", UPDATE_CHANNEL_AUTO, &u);
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_UNKNOWN, "dev build: latest known, status unknown");
    TEST_ASSERT(u.latest[0] != '\0', "latest still reported for dev builds");

    memset(&u, 0, sizeof(u));
    update_pick_latest("[{\"tag_name\":\"v0.1.0\",\"html_url\":\"javascript:alert(1)\"}]", "v0.0.1",
                       UPDATE_CHANNEL_AUTO, &u);
    TEST_ASSERT(u.url[0] == '\0', "non-https release URL dropped");
    TEST_ASSERT(update_pick_latest("{\"message\":\"rate limited\"}", "v0.1.0", 0, &u) != 0, "non-array rejected");
    TEST_ASSERT_EQ(update_pick_latest("[]", "v0.1.0", 0, &u), 0, "empty list ok");
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_UNKNOWN, "no releases -> unknown");
}

static void test_validation_and_env(void) {
    TEST_ASSERT(update_repo_valid("shifulegend/project-zero"), "repo ok");
    TEST_ASSERT(!update_repo_valid("a/b/c"), "extra slash");
    TEST_ASSERT(!update_repo_valid("/b"), "empty owner");
    TEST_ASSERT(!update_repo_valid("a/b;rm -rf"), "shell chars");
    TEST_ASSERT(!update_repo_valid("../etc"), "dot-dot");
    TEST_ASSERT(update_api_base_valid("https://api.github.com"), "api ok");
    TEST_ASSERT(!update_api_base_valid("http://api.github.com"), "plain http refused");
    TEST_ASSERT(!update_api_base_valid("https://a b"), "whitespace refused");
    TEST_ASSERT_EQ(update_channel_parse("stable"), UPDATE_CHANNEL_STABLE, "channel parse");
    TEST_ASSERT_EQ(update_channel_parse("nope"), -1, "bad channel");

    unsetenv("PZ_NO_UPDATE_CHECK"); unsetenv("DO_NOT_TRACK"); unsetenv("CI");
    TEST_ASSERT(update_env_opt_out() == NULL, "no opt-out env");
    setenv("CI", "true", 1);
    TEST_ASSERT(update_env_opt_out() && strcmp(update_env_opt_out(), "CI") == 0, "CI disables");
    setenv("CI", "0", 1);
    TEST_ASSERT(update_env_opt_out() == NULL, "CI=0 does not");
    unsetenv("CI");
    setenv("PZ_NO_UPDATE_CHECK", "1", 1);
    TEST_ASSERT(update_env_opt_out() != NULL, "PZ_NO_UPDATE_CHECK disables");
    unsetenv("PZ_NO_UPDATE_CHECK");

    /* Pre-existing JSON parser bug found here: trailing backslash read past the NUL. */
    char *v = json_field_string("{\"a\":\"x\\", "a");
    TEST_ASSERT(v == NULL, "unterminated escape rejected without over-read");
    free(v);
}

/* ---- fake network ---- */
static int g_fetches;
static int g_fail;
static char g_seen_url[400], g_seen_ua[128];

static int fake_fetch(const char *url, int timeout_s, const char *ua, char **body) {
    (void)timeout_s;
    g_fetches++;
    snprintf(g_seen_url, sizeof(g_seen_url), "%s", url);
    snprintf(g_seen_ua, sizeof(g_seen_ua), "%s", ua);
    if (g_fail) { *body = NULL; return -1; }
    *body = strdup(RELEASES);
    return *body ? 0 : -1;
}

static char g_dir[256];

static void setup_opts(UpdateOptions *o) {
    update_options_default(o);
    snprintf(o->repo, sizeof(o->repo), "o/r");
    snprintf(o->cache_path, sizeof(o->cache_path), "%s/sub/update-check", g_dir);
}

static void test_check_and_cache(void) {
    UpdateOptions o;
    UpdateInfo u;
    setup_opts(&o);
    update_set_fetcher(fake_fetch);
    g_fetches = 0; g_fail = 0;

    TEST_ASSERT_EQ(update_check(&o, "v0.1.0", 0, &u), 0, "first check");
    TEST_ASSERT_EQ(g_fetches, 1, "fetched once");
    TEST_ASSERT(strstr(g_seen_url, "https://api.github.com/repos/o/r/releases") == g_seen_url, "releases URL");
    TEST_ASSERT(strcmp(g_seen_ua, "project-zero/v0.1.0") == 0, "UA carries only the version");
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_AVAILABLE, "available");
    TEST_ASSERT(access(o.cache_path, R_OK) == 0, "cache written (parent dirs created)");

    TEST_ASSERT_EQ(update_check(&o, "v0.1.0", 0, &u), 0, "second check");
    TEST_ASSERT_EQ(g_fetches, 1, "served from cache within the interval");
    TEST_ASSERT(u.from_cache && strcmp(u.latest, "v0.2.0-rc.1") == 0, "cached answer");

    TEST_ASSERT_EQ(update_check(&o, "v0.2.0-rc.1", 0, &u), 0, "after upgrading");
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_UP_TO_DATE, "status recomputed vs new current, no fetch");
    TEST_ASSERT_EQ(g_fetches, 1, "still no fetch");

    o.channel = UPDATE_CHANNEL_STABLE;
    update_check(&o, "v0.1.0", 0, &u);
    TEST_ASSERT_EQ(g_fetches, 2, "channel change invalidates the cache");
    TEST_ASSERT(strcmp(u.latest, "v0.1.1") == 0, "stable answer");

    g_fail = 1;
    update_check(&o, "v0.1.0", 1, &u);
    TEST_ASSERT_EQ(g_fetches, 3, "forced check fetches");
    TEST_ASSERT(u.error[0] && strcmp(u.latest, "v0.1.1") == 0, "failure keeps the last good answer");
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_AVAILABLE, "known update stays visible offline");
    update_check(&o, "v0.1.0", 0, &u);
    TEST_ASSERT_EQ(g_fetches, 3, "failed check backs off");

    o.enabled = 0;
    update_check(&o, "v0.1.0", 0, &u);
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_DISABLED, "--no-update-check");
    TEST_ASSERT_EQ(g_fetches, 3, "disabled never fetches");
    o.enabled = 1;
    o.interval_hours = 0;
    update_check(&o, "v0.1.0", 0, &u);
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_DISABLED, "--update-interval 0");

    char notice[512];
    memset(&u, 0, sizeof(u));
    update_pick_latest(RELEASES, "v0.1.0", UPDATE_CHANNEL_AUTO, &u);
    update_format_notice(&u, notice, sizeof(notice));
    TEST_ASSERT(strstr(notice, "v0.1.0 -> v0.2.0-rc.1") && strstr(notice, "--no-update-check"), "notice text");
    u.status = UPDATE_STATUS_UP_TO_DATE;
    update_format_notice(&u, notice, sizeof(notice));
    TEST_ASSERT(notice[0] == '\0', "silent when up to date");
}

static void test_service(void) {
    UpdateOptions o;
    UpdateInfo u;
    setup_opts(&o);
    snprintf(o.cache_path, sizeof(o.cache_path), "%s/svc-cache", g_dir);
    unlink(o.cache_path);
    update_set_fetcher(fake_fetch);
    g_fetches = 0; g_fail = 0;

    update_service_init(&o, "v0.1.0");
    update_service_get(&u);
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_UNKNOWN, "unknown before the first check");
    TEST_ASSERT_EQ(update_service_poll(0), 1, "stale -> background check started");
    TEST_ASSERT_EQ(update_service_poll(0), 0, "no second check while one is in flight");
    update_service_wait();
    update_service_get(&u);
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_AVAILABLE, "result published");
    TEST_ASSERT_EQ(update_service_poll(0), 0, "fresh -> no check");
    TEST_ASSERT_EQ(update_service_poll(1), 0, "forced check rate-limited");
    TEST_ASSERT_EQ(g_fetches, 1, "one fetch total");

    update_service_init(&o, "v0.1.0");
    update_service_get(&u);
    TEST_ASSERT(u.from_cache && u.status == UPDATE_STATUS_AVAILABLE, "restart shows cached answer without network");

    o.enabled = 0;
    update_service_init(&o, "v0.1.0");
    TEST_ASSERT_EQ(update_service_poll(1), 0, "operator opt-out beats a UI check-now");
    update_service_get(&u);
    TEST_ASSERT_EQ(u.status, UPDATE_STATUS_DISABLED, "disabled state");
    update_service_wait();
    update_set_fetcher(NULL);
}

int main(void) {
    printf("=== Update Check (Phase 38) Tests ===\n");
    const char *t = getenv("TMPDIR");
    snprintf(g_dir, sizeof(g_dir), "%s/pz-upd-%d", (t && *t) ? t : "/tmp", (int)getpid());
    RUN_TEST(test_semver);
    RUN_TEST(test_pick_latest);
    RUN_TEST(test_validation_and_env);
    RUN_TEST(test_check_and_cache);
    RUN_TEST(test_service);
    char cmd[300];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", g_dir);
    if (system(cmd) != 0) { /* best effort */ }
    TEST_SUMMARY();
}
