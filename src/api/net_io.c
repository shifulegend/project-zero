#include "api/net_io.h"

#include <errno.h>
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
