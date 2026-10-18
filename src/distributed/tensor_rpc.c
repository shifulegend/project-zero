#include "distributed/tensor_rpc.h"
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int send_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return -1;
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

TernaryError rpc_listen(int port, int loopback_only, int *server_fd) {
    if (!server_fd || port <= 0 || port > 65535) return TN_ERR_INVALID_ARGS;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return TN_ERR_SOCKET_CREATE;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(loopback_only ? INADDR_LOOPBACK : INADDR_ANY);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0) { close(fd); return TN_ERR_SOCKET_BIND; }
    if (listen(fd, 4) != 0) { close(fd); return TN_ERR_SOCKET_LISTEN; }
    *server_fd = fd;
    return TN_OK;
}

TernaryError rpc_accept(int server_fd, int *conn_fd) {
    int c = accept(server_fd, NULL, NULL);
    if (c < 0) return TN_ERR_SOCKET_CREATE;
    set_nodelay(c);
    *conn_fd = c;
    return TN_OK;
}

TernaryError rpc_connect(const char *host, int port, int *sockfd) {
    if (!host || !sockfd) return TN_ERR_INVALID_ARGS;
    char ps[16];
    snprintf(ps, sizeof(ps), "%d", port);
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, ps, &hints, &res) != 0 || !res) return TN_ERR_SOCKET_CREATE;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return TN_ERR_SOCKET_CREATE; }
    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        close(fd);
        freeaddrinfo(res);
        return TN_ERR_SOCKET_CREATE;
    }
    freeaddrinfo(res);
    set_nodelay(fd);
    *sockfd = fd;
    return TN_OK;
}

void rpc_close(int fd) {
    if (fd >= 0) close(fd);
}

TernaryError rpc_send(int fd, RpcType type, int32_t arg, const void *data, uint32_t count) {
    RpcHeader h = { RPC_MAGIC, (uint32_t)type, arg, count };
    if (send_all(fd, &h, sizeof(h)) != 0) return TN_ERR_SOCKET_CREATE;
    if (count && (!data || send_all(fd, data, (size_t)count * 4u) != 0)) return TN_ERR_SOCKET_CREATE;
    return TN_OK;
}

TernaryError rpc_recv(int fd, RpcHeader *hdr, void *data, uint32_t max_count) {
    if (!hdr) return TN_ERR_INVALID_ARGS;
    if (recv_all(fd, hdr, sizeof(*hdr)) != 0) return TN_ERR_SOCKET_CREATE;
    if (hdr->magic != RPC_MAGIC) return TN_ERR_INVALID_MAGIC;
    if (hdr->count > max_count || (hdr->count && !data)) return TN_ERR_DIMENSION_MISMATCH;
    if (hdr->count && recv_all(fd, data, (size_t)hdr->count * 4u) != 0) return TN_ERR_SOCKET_CREATE;
    return TN_OK;
}
