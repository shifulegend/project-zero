/* HTTP server connection limits: a stalled client is dropped after
 * --client-timeout, connections beyond --max-connections get 503, and a normal
 * request still works. Uses the real listener on a free loopback port; /health
 * needs no model. */
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "api/api_server.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); g_fail++; } else printf("ok: %s\n", msg); } while (0)

static void nap_ms(int ms) { struct timespec t = { ms / 1000, (long)(ms % 1000) * 1000000L }; nanosleep(&t, NULL); }
static void send_str(int fd, const char *s) { if (write(fd, s, strlen(s)) < 0) perror("write"); }

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec / 1e9; }

static int free_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = 0;
    if (bind(fd, (struct sockaddr *)&a, sizeof a) != 0) { close(fd); return -1; }
    socklen_t l = sizeof a; getsockname(fd, (struct sockaddr *)&a, &l);
    int p = ntohs(a.sin_port); close(fd); return p;
}

static int dial(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons((uint16_t)port);
    if (connect(fd, (struct sockaddr *)&a, sizeof a) != 0) { close(fd); return -1; }
    return fd;
}

/* Read until EOF or `ms` elapse; returns bytes read, sets *eof. */
static size_t slurp(int fd, char *buf, size_t cap, int ms, int *eof) {
    size_t n = 0; *eof = 0;
    double end = now_s() + ms / 1000.0;
    while (n + 1 < cap) {
        int left = (int)((end - now_s()) * 1000); if (left <= 0) break;
        struct pollfd p = { fd, POLLIN, 0 };
        if (poll(&p, 1, left) <= 0) break;
        ssize_t r = read(fd, buf + n, cap - 1 - n);
        if (r <= 0) { *eof = 1; break; }
        n += (size_t)r;
    }
    buf[n] = 0; return n;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    int port = free_port();
    if (port < 0) { printf("SKIP: no loopback\n"); return 0; }

    ApiContext ctx;
    api_context_init(&ctx, NULL, NULL, NULL, NULL, NULL, NULL);
    ctx.server_config.client_timeout_s = 1;
    ctx.server_config.max_connections = 2;
    if (api_server_start(port, &ctx) != TN_OK) { printf("FAIL: server did not start\n"); return 1; }
    nap_ms(100);

    char buf[2048]; int eof;

    int ok = dial(port);
    send_str(ok, "GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
    slurp(ok, buf, sizeof buf, 3000, &eof);
    CHECK(strstr(buf, "200") && strstr(buf, "\"ok\""), "normal /health request is served");
    close(ok);

    /* two stalled clients fill the cap */
    int s1 = dial(port), s2 = dial(port);
    send_str(s1, "GET /hea"); send_str(s2, "GET /hea");
    nap_ms(200);
    int over = dial(port);
    size_t n = slurp(over, buf, sizeof buf, 2000, &eof);
    CHECK(n > 0 && strncmp(buf, "HTTP/1.1 503", 12) == 0, "third connection is refused with 503");
    close(over);

    /* the stalled clients are dropped after the timeout */
    double t0 = now_s();
    slurp(s1, buf, sizeof buf, 5000, &eof);
    double dt = now_s() - t0;
    CHECK(eof && dt < 4.0, "stalled client is disconnected after the timeout");
    slurp(s2, buf, sizeof buf, 3000, &eof);
    close(s1); close(s2);

    /* capacity is released again */
    int again = dial(port);
    send_str(again, "GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
    slurp(again, buf, sizeof buf, 3000, &eof);
    CHECK(strstr(buf, "200") != NULL, "capacity is released after the stalled clients leave");
    close(again);

    api_server_stop(&ctx);
    printf(g_fail ? "http_limits: %d FAILED\n" : "http_limits: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
