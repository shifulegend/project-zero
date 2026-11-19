/* net_write_all(): full delivery across short writes, and a clean failure when the peer is gone. */
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "api/net_io.h"

#define BIG (4u * 1024u * 1024u) /* far above the socket buffer, forces short writes */

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

typedef struct {
    int fd;
    unsigned char *buf;
    size_t got;
} Reader;

static void *reader_fn(void *arg) {
    Reader *r = (Reader *)arg;
    while (r->got < BIG) {
        ssize_t n = read(r->fd, r->buf + r->got, BIG - r->got);
        if (n <= 0) break;
        r->got += (size_t)n;
        {
            struct timespec ts = {0, 50000};
            nanosleep(&ts, NULL);
        } /* slow reader: the writer must see partial writes */
    }
    return NULL;
}

static void test_peer_masked(void) {
    struct in_addr a;
    char out[32];
    inet_pton(AF_INET, "192.168.1.77", &a);
    net_format_peer_masked(&a, out, sizeof(out));
    CHECK(strcmp(out, "192.168.1.x") == 0, "peer address: last octet masked");
    inet_pton(AF_INET, "127.0.0.1", &a);
    net_format_peer_masked(&a, out, sizeof(out));
    CHECK(strcmp(out, "127.0.0.x") == 0, "peer address: loopback masked");
    char tiny[6];
    net_format_peer_masked(&a, tiny, sizeof(tiny));
    CHECK(strlen(tiny) < sizeof(tiny), "peer address: small buffer stays terminated");
    net_format_peer_masked(NULL, out, sizeof(out));
    CHECK(strcmp(out, "?") == 0, "peer address: NULL gives ?");
}

int main(void) {
    test_peer_masked();
    signal(SIGPIPE, SIG_IGN); /* same as the server */
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        perror("socketpair");
        return 1;
    }

    unsigned char *src = (unsigned char *)malloc(BIG), *dst = (unsigned char *)malloc(BIG);
    for (size_t i = 0; i < BIG; i++) src[i] = (unsigned char)(i * 131u + 7u);
    Reader rd = {sv[1], dst, 0};
    pthread_t th;
    pthread_create(&th, NULL, reader_fn, &rd);
    CHECK(net_write_all(sv[0], src, BIG) == 0, "large write returns 0");
    pthread_join(th, NULL);
    CHECK(rd.got == BIG && memcmp(src, dst, BIG) == 0, "all bytes arrive intact and in order");
    CHECK(net_write_all(sv[0], src, 0) == 0, "zero-length write is a no-op success");

    close(sv[1]); /* peer gone */
    CHECK(net_write_all(sv[0], "x", 1) == -1, "closed peer returns -1 (no crash)");
    close(sv[0]);
    CHECK(net_write_all(-1, "x", 1) == -1, "invalid fd returns -1");
    free(src);
    free(dst);
    printf(g_fail ? "net_io: %d FAILED\n" : "net_io: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
