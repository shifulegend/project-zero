#ifndef TN_API_NET_IO_H
#define TN_API_NET_IO_H

#include <stddef.h>

/* Write all `len` bytes to `fd`, retrying on short writes and EINTR.
 * Returns 0 when every byte was written, -1 on error (peer closed, EPIPE,
 * ECONNRESET, ...). The caller must have SIGPIPE ignored (the server does) or
 * pass a socket opened without it, otherwise a closed peer kills the process. */
int net_write_all(int fd, const void *buf, size_t len);

/* Writes the IPv4 peer address with its last octet masked ("192.168.1.x") into out, so a log line
 * says which network a client came from without recording the client itself. Always NUL-terminates
 * a non-empty buffer; an address that cannot be formatted becomes "?". */
struct in_addr;
void net_format_peer_masked(const struct in_addr *addr, char *out, size_t cap);

#endif
