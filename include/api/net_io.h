#ifndef TN_API_NET_IO_H
#define TN_API_NET_IO_H

#include <stddef.h>

/* Write all `len` bytes to `fd`, retrying on short writes and EINTR.
 * Returns 0 when every byte was written, -1 on error (peer closed, EPIPE,
 * ECONNRESET, ...). The caller must have SIGPIPE ignored (the server does) or
 * pass a socket opened without it, otherwise a closed peer kills the process. */
int net_write_all(int fd, const void *buf, size_t len);

#endif
