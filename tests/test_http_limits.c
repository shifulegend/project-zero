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
#include "api/config_endpoint.h"

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

    /* request framing is validated before any body is read */
    struct { const char *req; const char *want; const char *what; } fr[] = {
        { "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", "HTTP/1.1 501", "chunked body refused with 501" },
        { "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n", "HTTP/1.1 400", "non-numeric Content-Length refused with 400" },
        { "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nContent-Length: 6\r\n\r\n", "HTTP/1.1 400", "conflicting Content-Length refused with 400" },
        { "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 99999999999\r\n\r\n", "HTTP/1.1 400", "overflowing Content-Length refused" },
        { "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 9000000\r\n\r\n", "HTTP/1.1 413", "oversized Content-Length refused with 413" },
    };
    for (size_t i = 0; i < sizeof fr / sizeof fr[0]; i++) {
        int c = dial(port);
        send_str(c, fr[i].req);
        slurp(c, buf, sizeof buf, 3000, &eof);
        CHECK(strncmp(buf, fr[i].want, strlen(fr[i].want)) == 0 || (i == 3 && strncmp(buf, "HTTP/1.1 413", 12) == 0), fr[i].what);
        close(c);
    }

    {   /* a well-formed POST with a body still goes through (and the word
         * "content-length:" inside the body does not confuse the framing) */
        const char *body = "{\"id\":\"content-length: 1\"}";
        char req[512];
        snprintf(req, sizeof req, "POST /v1/chat/completions/cancel HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s", strlen(body), body);
        int c = dial(port);
        send_str(c, req);
        slurp(c, buf, sizeof buf, 3000, &eof);
        CHECK(strncmp(buf, "HTTP/1.1 400", 12) != 0 && strncmp(buf, "HTTP/1.1 413", 12) != 0 &&
              strncmp(buf, "HTTP/1.1 501", 12) != 0 && strncmp(buf, "HTTP/1.1", 8) == 0,
              "well-formed POST with a body is still handled");
        close(c);
    }

    {   /* cross-site guard on every POST route */
        char hostline[64], req[640];
        snprintf(hostline, sizeof hostline, "127.0.0.1:%d", port);
        struct { const char *origin_fmt; int want403; const char *what; } cs[] = {
            { "Origin: http://evil.example\r\n", 1, "POST from a foreign Origin is refused with 403" },
            { "Origin: null\r\n", 1, "POST with Origin: null is refused with 403" },
            { "Origin: http://%s\r\n", 0, "POST from the server's own origin is allowed" },
            { "", 0, "POST without Origin (curl, SDKs) is allowed" },
        };
        for (size_t i = 0; i < sizeof cs / sizeof cs[0]; i++) {
            char oh[128];
            snprintf(oh, sizeof oh, cs[i].origin_fmt, hostline);
            snprintf(req, sizeof req, "POST /v1/prompt_cache/clear HTTP/1.1\r\nHost: %s\r\n%sContent-Length: 0\r\n\r\n", hostline, oh);
            int c = dial(port);
            send_str(c, req);
            slurp(c, buf, sizeof buf, 3000, &eof);
            int is403 = strncmp(buf, "HTTP/1.1 403", 12) == 0;
            CHECK(is403 == cs[i].want403 && strncmp(buf, "HTTP/1.1", 8) == 0, cs[i].what);
            close(c);
        }
        CorsConfig none; memset(&none, 0, sizeof none);
        CHECK(api_post_allowed("text/plain", "http://evil.example", "127.0.0.1:1", &none, 5) == 0, "unit: foreign origin refused");
        CHECK(api_post_allowed(NULL, NULL, "127.0.0.1:1", &none, 5) == 1, "unit: no origin allowed");
        CHECK(api_post_allowed("text/plain", "http://127.0.0.1:1", "127.0.0.1:1", &none, 5) == 1, "unit: same origin allowed");
    }

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
