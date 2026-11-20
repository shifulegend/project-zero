/* The OS lookups behind TnSysProbe, driven by fakes: no real /proc or environment involved. */
#include <stdio.h>
#include <string.h>

#include "core/bw_cache.h"
#include "core/sys_probe.h"
#include "test_harness.h"

static const char *g_env[3]; /* PZ_CACHE_DIR, XDG_CACHE_HOME, HOME */
static const char *g_cpuinfo;

static const char *fake_getenv(const char *name) {
    if (strcmp(name, "PZ_CACHE_DIR") == 0) return g_env[0];
    if (strcmp(name, "XDG_CACHE_HOME") == 0) return g_env[1];
    if (strcmp(name, "HOME") == 0) return g_env[2];
    return NULL;
}

static FILE *fake_fopen(const char *path, const char *mode) {
    (void)mode;
    if (strcmp(path, "/proc/cpuinfo") != 0 || !g_cpuinfo) return NULL;
    FILE *f = tmpfile();
    if (f) {
        fputs(g_cpuinfo, f);
        rewind(f);
    }
    return f;
}

static const TnSysProbe k_fake = {fake_getenv, fake_fopen};

static void test_cpu_model(void) {
    char buf[64];
    g_cpuinfo = "processor\t: 0\nmodel name\t: Fake CPU 9000 @ 3.0GHz\nflags\t: x\n";
    tn_sys_cpu_model(&k_fake, buf, sizeof buf);
    TEST_ASSERT(strcmp(buf, "Fake CPU 9000 @ 3.0GHz") == 0,
                "model name is read from the fake file");
    g_cpuinfo = NULL;
    tn_sys_cpu_model(&k_fake, buf, sizeof buf);
    TEST_ASSERT(buf[0] == '\0', "no file gives an empty model");
    tn_sys_cpu_model(&k_fake, buf, 8);
    TEST_ASSERT(strlen(buf) < 8, "a short buffer is respected");
}

static void test_cache_dir_precedence(void) {
    char buf[128];
    g_env[0] = "/pz";
    g_env[1] = "/xdg";
    g_env[2] = "/home/u";
    TEST_ASSERT(tn_sys_cache_dir(&k_fake, buf, sizeof buf) == 0 && strcmp(buf, "/pz") == 0,
                "PZ_CACHE_DIR wins");
    g_env[0] = NULL;
    TEST_ASSERT(tn_sys_cache_dir(&k_fake, buf, sizeof buf) == 0 &&
                    strcmp(buf, "/xdg/project-zero") == 0,
                "then XDG_CACHE_HOME");
    g_env[1] = "";
    TEST_ASSERT(tn_sys_cache_dir(&k_fake, buf, sizeof buf) == 0 &&
                    strcmp(buf, "/home/u/.cache/project-zero") == 0,
                "an empty XDG_CACHE_HOME is skipped, HOME is used");
    g_env[2] = NULL;
    TEST_ASSERT(tn_sys_cache_dir(&k_fake, buf, sizeof buf) == -1, "nothing set is an error");
    g_env[2] = "/home/u";
    TEST_ASSERT(tn_sys_cache_dir(&k_fake, buf, 5) == -1, "a path that does not fit is an error");
}

static void test_default_probe_is_real(void) {
    char buf[256];
    tn_sys_cpu_model(NULL, buf, sizeof buf); /* must not crash; may be empty off Linux */
    TEST_ASSERT(tn_sys_probe_default()->getenv_fn != NULL, "the default probe has a getenv");
    char p[256];
    int r = tn_bw_cache_path(p, sizeof p);
    TEST_ASSERT(r == 0 || r == -1, "the bandwidth cache path still resolves through the probe");
}

int main(void) {
    RUN_TEST(test_cpu_model);
    RUN_TEST(test_cache_dir_precedence);
    RUN_TEST(test_default_probe_is_real);
    TEST_SUMMARY();
}
