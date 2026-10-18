/*
 * tests/test_tensor_rpc.c — Phase 28: framed tensor transport (round trip over
 * a loopback socket, oversized payload rejected, bad magic rejected) and the
 * layer-plan helpers (dist_plan_parse validation, dist_plan_layers_by_ram).
 * Multi-threaded; part of TSAN_TESTS.
 */
#include "test_harness.h"
#include "distributed/tensor_rpc.h"
#include "distributed/dist_plan.h"
#include <pthread.h>
#include <string.h>
#include <unistd.h>

/* Bind the first free port in a small range so parallel runs don't collide. */
static int listen_any(int *server_fd) {
    for (int port = 39100 + (int)(getpid() % 400); port < 39900; port++)
        if (rpc_listen(port, 1, server_fd) == TN_OK) return port;
    return -1;
}

typedef struct { int server_fd; int ok; } EchoArgs;

/* Echo server: replies to each FORWARD with the payload doubled; stops on BYE. */
static void *echo_thread(void *arg) {
    EchoArgs *a = (EchoArgs *)arg;
    int c = -1;
    if (rpc_accept(a->server_fd, &c) != TN_OK) return NULL;
    float buf[64];
    for (;;) {
        RpcHeader h;
        if (rpc_recv(c, &h, buf, 64) != TN_OK) break;
        if (h.type == RPC_BYE) { a->ok = 1; break; }
        for (uint32_t i = 0; i < h.count; i++) buf[i] *= 2.0f;
        if (rpc_send(c, RPC_FORWARD, h.arg + 1, buf, h.count) != TN_OK) break;
    }
    rpc_close(c);
    return NULL;
}

static void test_round_trip(void) {
    EchoArgs a = { -1, 0 };
    int port = listen_any(&a.server_fd);
    TEST_ASSERT(port > 0, "listen on a loopback port");
    if (port <= 0) return;
    pthread_t th;
    pthread_create(&th, NULL, echo_thread, &a);

    int fd = -1;
    TEST_ASSERT_EQ(rpc_connect("127.0.0.1", port, &fd), TN_OK, "connect");
    float x[16], y[16];
    for (int i = 0; i < 16; i++) x[i] = (float)i - 3.5f;
    TEST_ASSERT_EQ(rpc_send(fd, RPC_FORWARD, 41, x, 16), TN_OK, "send frame");
    RpcHeader h;
    TEST_ASSERT_EQ(rpc_recv(fd, &h, y, 16), TN_OK, "recv frame");
    TEST_ASSERT_EQ((int)h.type, RPC_FORWARD, "reply type");
    TEST_ASSERT_EQ(h.arg, 42, "reply arg");
    TEST_ASSERT_EQ((int)h.count, 16, "reply count");
    int same = 1;
    for (int i = 0; i < 16; i++) if (y[i] != 2.0f * x[i]) same = 0;
    TEST_ASSERT(same, "payload round-trips bit-exactly");

    /* Reply larger than the caller's buffer must be refused, not overflow. */
    TEST_ASSERT_EQ(rpc_send(fd, RPC_FORWARD, 0, x, 16), TN_OK, "send second frame");
    TEST_ASSERT(rpc_recv(fd, &h, y, 8) != TN_OK, "oversized payload rejected");
    rpc_close(fd);
    pthread_join(th, NULL);
    rpc_close(a.server_fd);
}

typedef struct { int server_fd; } RawArgs;

/* Writes a header with the wrong magic. */
static void *bad_magic_thread(void *arg) {
    RawArgs *a = (RawArgs *)arg;
    int c = -1;
    if (rpc_accept(a->server_fd, &c) != TN_OK) return NULL;
    RpcHeader h = { 0xDEADBEEFu, RPC_FORWARD, 0, 0 };
    ssize_t n = write(c, &h, sizeof(h));
    (void)n;
    rpc_close(c);
    return NULL;
}

static void test_bad_magic(void) {
    RawArgs a = { -1 };
    int port = listen_any(&a.server_fd);
    TEST_ASSERT(port > 0, "listen on a loopback port");
    if (port <= 0) return;
    pthread_t th;
    pthread_create(&th, NULL, bad_magic_thread, &a);
    int fd = -1;
    TEST_ASSERT_EQ(rpc_connect("127.0.0.1", port, &fd), TN_OK, "connect");
    RpcHeader h;
    float buf[4];
    TEST_ASSERT(rpc_recv(fd, &h, buf, 4) != TN_OK, "bad magic rejected");
    rpc_close(fd);
    pthread_join(th, NULL);
    rpc_close(a.server_fd);
}

static void test_plan_parse(void) {
    DistPlan dp;
    TEST_ASSERT_EQ(dist_plan_parse(&dp, "127.0.0.1:9101@15-30", 30), 0, "single node");
    TEST_ASSERT_EQ(dp.n, 1, "one node");
    TEST_ASSERT(strcmp(dp.nodes[0].host, "127.0.0.1") == 0, "host parsed");
    TEST_ASSERT_EQ(dp.nodes[0].port, 9101, "port parsed");
    TEST_ASSERT_EQ(dp.nodes[0].start, 15, "start parsed");
    TEST_ASSERT_EQ(dp.nodes[0].end, 30, "end parsed");
    TEST_ASSERT_EQ(dp.nodes[0].fd, -1, "not connected yet");

    TEST_ASSERT_EQ(dist_plan_parse(&dp, "a:1@0-4,b:2@8-12", 12), 0, "two nodes");
    TEST_ASSERT_EQ(dp.n, 2, "two nodes parsed");

    TEST_ASSERT_EQ(dist_plan_parse(&dp, "a:1@0-4,b:2@3-8", 12), -1, "overlap rejected");
    TEST_ASSERT_EQ(dist_plan_parse(&dp, "a:1@0-40", 12), -1, "range past n_layers rejected");
    TEST_ASSERT_EQ(dist_plan_parse(&dp, "a:1@5-5", 12), -1, "empty range rejected");
    TEST_ASSERT_EQ(dist_plan_parse(&dp, "a:0@0-4", 12), -1, "port 0 rejected");
    TEST_ASSERT_EQ(dist_plan_parse(&dp, "a@0-4", 12), -1, "missing port rejected");
    TEST_ASSERT_EQ(dist_plan_parse(&dp, "", 12), -1, "empty spec rejected");

    TEST_ASSERT(dist_plan_parse(&dp, "a:1@0-1,b:1@1-2", 2) == 0 &&
                dist_plan_node_starting_at(&dp, 1) == &dp.nodes[1] &&
                dist_plan_node_starting_at(&dp, 3) == NULL, "node lookup by start layer");
}

static void check_cover(const int *s, const int *e, int n, int n_layers, const char *msg) {
    int ok = s[0] == 0 && e[n - 1] == n_layers;
    for (int i = 0; i < n; i++) {
        if (e[i] <= s[i]) ok = 0;
        if (i > 0 && s[i] != e[i - 1]) ok = 0;
    }
    TEST_ASSERT(ok, msg);
}

static void test_layers_by_ram(void) {
    int s[4], e[4];
    long long eq[2] = { 8LL << 30, 8LL << 30 };
    dist_plan_layers_by_ram(30, eq, 2, s, e);
    check_cover(s, e, 2, 30, "equal RAM covers all layers");
    TEST_ASSERT_EQ(e[0], 15, "equal RAM splits in half");

    long long skew[2] = { 24LL << 30, 8LL << 30 };
    dist_plan_layers_by_ram(32, skew, 2, s, e);
    check_cover(s, e, 2, 32, "skewed RAM covers all layers");
    TEST_ASSERT_EQ(e[0], 24, "3:1 RAM gives 3/4 of the layers");

    long long tiny[4] = { 1LL << 40, 1, 1, 1 };
    dist_plan_layers_by_ram(4, tiny, 4, s, e);
    check_cover(s, e, 4, 4, "every node keeps at least one layer");

    long long unknown[3] = { 0, -1, 0 };
    dist_plan_layers_by_ram(9, unknown, 3, s, e);
    check_cover(s, e, 3, 9, "unknown RAM treated as average");
    TEST_ASSERT(e[0] == 3 && e[1] == 6, "unknown RAM splits evenly");
}

int main(void) {
    printf("=== Tensor RPC / Distributed Plan (Phase 28) Tests ===\n");
    RUN_TEST(test_round_trip);
    RUN_TEST(test_bad_magic);
    RUN_TEST(test_plan_parse);
    RUN_TEST(test_layers_by_ram);
    TEST_SUMMARY();
}
