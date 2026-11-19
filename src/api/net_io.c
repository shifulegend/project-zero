#include "api/net_io.h"

#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int net_write_all(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1; /* no progress: treat as failure, never spin */
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

void net_format_peer_masked(const struct in_addr *addr, char *out, size_t cap) {
    if (!out || cap == 0) return;
    char full[INET_ADDRSTRLEN];
    if (!addr || !inet_ntop(AF_INET, addr, full, sizeof(full))) {
        snprintf(out, cap, "?");
        return;
    }
    char *dot = strrchr(full, '.');
    if (!dot) {
        snprintf(out, cap, "?");
        return;
    }
    dot[1] = 'x';
    dot[2] = '\0';
    snprintf(out, cap, "%s", full);
}
