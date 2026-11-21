#ifndef TN_API_HTTP_CONNECTION_H
#define TN_API_HTTP_CONNECTION_H

/*
 * http_connection.h — one accepted connection, start to finish: read and parse the request, run
 * the gates (Host check, preflight, API key, rate limit, cross-site POST guard), dispatch to the
 * route, record the metric, close the socket.
 */

#include "api/api_server.h"

/* Serves one request on `fd` and closes it. Takes no ownership of ctx. */
void api_handle_connection(int fd, ApiContext *ctx);

#endif /* TN_API_HTTP_CONNECTION_H */
