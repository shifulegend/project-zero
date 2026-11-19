/*
 * test_http_response.c -- the response writers: error bodies stay valid JSON whatever the message
 * holds, a header block is never cut off by oversized extra headers, and a heap body is freed even
 * when the peer has already gone (the leak shows up under ASan/LSan in make test).
 */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "api/http_response.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

/* Writes one reply into a socketpair and returns everything the peer reads. */
static size_t capture(char *out, size_t cap, void (*send_fn)(int, const void *), const void *arg) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return 0;
    send_fn(sv[0], arg);
    close(sv[0]);
    size_t n = 0;
    ssize_t r;
    while (n + 1 < cap && (r = read(sv[1], out + n, cap - 1 - n)) > 0) n += (size_t)r;
    out[n] = '\0';
    close(sv[1]);
    return n;
}

static void send_err_quotes(int fd, const void *arg) {
    (void)arg;
    http_send_error_ex(fd, 400, "bad \"x\" \\ path\nline2\ttab", NULL);
}

static void send_err_big_extra(int fd, const void *arg) {
    http_send_error_ex(fd, 429, "slow down", (const char *)arg);
}

static void send_ok_big_extra(int fd, const void *arg) {
    http_send_response_ex(fd, 200, "text/plain", "hello", 0, (const char *)arg);
}

static void send_heap_closed_peer(int fd, const void *arg) {
    (void)arg;
    (void)fd;
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return;
    close(sv[1]); /* the peer is gone before the reply is written */
    http_send_json_heap(sv[0], 400, strdup("{\"error\":1}"), NULL);
    close(sv[0]);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    char buf[4096];

    size_t n         = capture(buf, sizeof buf, send_err_quotes, NULL);
    const char *body = strstr(buf, "\r\n\r\n");
    body             = body ? body + 4 : "";
    CHECK(n > 0 && strstr(body, "bad \\\"x\\\" \\\\ path\\u000aline2\\u0009tab") != NULL,
          "quotes, backslashes and control characters in a message are escaped");
    CHECK(strchr(body, '\n') == NULL && strchr(body, '\t') == NULL,
          "the error body holds no raw control characters");

    char big[1500];
    memset(big, 'a', sizeof(big) - 1);
    big[sizeof(big) - 1] = '\0';
    /* a single header line of 1499 bytes + CRLF cannot fit next to the status line */
    char extra[1600];
    snprintf(extra, sizeof extra, "X-Pad: %s\r\n", big);

    n    = capture(buf, sizeof buf, send_err_big_extra, extra);
    body = strstr(buf, "\r\n\r\n");
    CHECK(body != NULL && strstr(body + 4, "slow down") != NULL,
          "oversized extra headers are dropped: the error reply is complete");
    n    = capture(buf, sizeof buf, send_ok_big_extra, extra);
    body = strstr(buf, "\r\n\r\n");
    CHECK(body != NULL && strcmp(body + 4, "hello") == 0 &&
              strstr(buf, "Content-Length: 5") != NULL,
          "oversized extra headers are dropped: the normal reply is complete");
    (void)n;

    capture(buf, sizeof buf, send_heap_closed_peer, NULL);
    CHECK(1, "a heap body is freed when the peer is already gone (checked by LSan)");

    printf("=== %s ===\n", g_fail ? "FAILED" : "http response tests passed");
    return g_fail ? 1 : 0;
}
