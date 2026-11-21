/*
 * test_env_flag.c -- environment switches are read once, cached and safe to query from many threads
 * at once (the hot-path flags used to be plain statics filled on first use).
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "core/env_flag.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

static atomic_int g_shared = TN_ENV_FLAG_UNSET;
static int g_seen[16];

static void *reader(void *arg) {
    int idx = (int)(long)arg;
    int v   = 0;
    for (int i = 0; i < 2000; i++) v = tn_env_on_if_one(&g_shared, "TN_TEST_FLAG_SHARED");
    g_seen[idx] = v;
    return NULL;
}

int main(void) {
    atomic_int a = TN_ENV_FLAG_UNSET, b = TN_ENV_FLAG_UNSET;
    unsetenv("TN_TEST_FLAG_A");
    CHECK(tn_env_on_unless_zero(&a, "TN_TEST_FLAG_A") == 1, "unset: on-unless-zero is on");
    CHECK(tn_env_on_if_one(&b, "TN_TEST_FLAG_A") == 0, "unset: on-if-one is off");

    atomic_int c = TN_ENV_FLAG_UNSET, d = TN_ENV_FLAG_UNSET;
    setenv("TN_TEST_FLAG_B", "0", 1);
    CHECK(tn_env_on_unless_zero(&c, "TN_TEST_FLAG_B") == 0, "0 turns on-unless-zero off");
    CHECK(tn_env_on_if_one(&d, "TN_TEST_FLAG_B") == 0, "0 leaves on-if-one off");
    setenv("TN_TEST_FLAG_B", "1", 1);
    CHECK(tn_env_on_unless_zero(&c, "TN_TEST_FLAG_B") == 0 &&
              tn_env_on_if_one(&d, "TN_TEST_FLAG_B") == 0,
          "the first answer is cached: a later setenv is ignored");

    atomic_int e = TN_ENV_FLAG_UNSET, f = TN_ENV_FLAG_UNSET;
    setenv("TN_TEST_FLAG_C", "1", 1);
    CHECK(tn_env_on_if_one(&e, "TN_TEST_FLAG_C") == 1, "1 turns on-if-one on");
    CHECK(tn_env_on_unless_zero(&f, "TN_TEST_FLAG_C") == 1, "1 leaves on-unless-zero on");
    atomic_int g = TN_ENV_FLAG_UNSET;
    setenv("TN_TEST_FLAG_D", "yes", 1);
    CHECK(tn_env_on_if_one(&g, "TN_TEST_FLAG_D") == 0, "only a leading 1 enables on-if-one");

    setenv("TN_TEST_FLAG_SHARED", "1", 1);
    pthread_t th[16];
    for (long i = 0; i < 16; i++) pthread_create(&th[i], NULL, reader, (void *)i);
    for (int i = 0; i < 16; i++) pthread_join(th[i], NULL);
    int all = 1;
    for (int i = 0; i < 16; i++) all &= g_seen[i] == 1;
    CHECK(all, "16 threads querying a fresh flag at once all see the same value");

    printf("=== %s ===\n", g_fail ? "FAILED" : "env flag tests passed");
    return g_fail ? 1 : 0;
}
