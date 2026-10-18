#ifndef TN_TENSOR_RPC_H
#define TN_TENSOR_RPC_H

/*
 * tensor_rpc.h — Phase 28.2: framed tensor transport over TCP.
 *
 * Frame = 16-byte little-endian header + payload:
 *   u32 magic 'PZT1' | u32 type | i32 arg | u32 count  then count floats.
 * Types: INFO (handshake: worker answers its model geometry + layer range),
 * FORWARD (arg = position, payload = hidden state + 1 float holding the
 * coordinator's sliding-window pinned length; reply = hidden state after the
 * worker's layers), ERROR (arg = TernaryError, no payload), BYE.
 * One request is outstanding per connection; plain blocking sockets.
 */

#include <stdint.h>
#include "core/error.h"

#define RPC_MAGIC 0x31545A50u /* "PZT1" */

typedef enum {
    RPC_INFO    = 1,
    RPC_FORWARD = 2,
    RPC_ERROR   = 3,
    RPC_BYE     = 4
} RpcType;

typedef struct {
    uint32_t magic, type;
    int32_t  arg;
    uint32_t count;
} RpcHeader;

/* INFO payload (as floats would lose precision — sent as 8 int32 in place of floats). */
typedef struct {
    int32_t n_layers, dim, vocab_size, layer_start, layer_end, is_ssm, reserved0, reserved1;
} RpcInfo;

TernaryError rpc_listen(int port, int loopback_only, int *server_fd);
TernaryError rpc_accept(int server_fd, int *conn_fd);
TernaryError rpc_connect(const char *host, int port, int *sockfd);
void         rpc_close(int fd);

/* Whole-message send/recv. `data` may be NULL when count == 0. recv checks
 * the magic and that the payload fits `max_count`. */
TernaryError rpc_send(int fd, RpcType type, int32_t arg, const void *data, uint32_t count);
TernaryError rpc_recv(int fd, RpcHeader *hdr, void *data, uint32_t max_count);

#endif /* TN_TENSOR_RPC_H */
