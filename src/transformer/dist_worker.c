/*
 * src/transformer/dist_worker.c
 *
 * The worker side of --remote-layers: serves FORWARD requests for a range of layers. It runs the
 * transformer layers, so it lives in the transformer module (the coordinator side, dist_plan.c, does
 * not need the transformer). Split out of distributed/dist_plan.c; the code is unchanged.
 */

#include "transformer/dist_worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "distributed/dist_plan.h"
#include "distributed/tensor_rpc.h"
#include "kv_cache/sliding_window.h"
#include "transformer/forward.h"

TernaryError dist_worker_serve(int port, int loopback_only, int start, int end, const Config *cfg,
                               const TransformerWeights *w, RunState *s, const MoEConfig *mc,
                               ThreadPool *tp) {
    int srv;
    TernaryError e = rpc_listen(port, loopback_only, &srv);
    if (e != TN_OK) {
        fprintf(stderr, "[dist-worker] cannot listen on port %d: %s\n", port, tn_error_str(e));
        return e;
    }
    printf("[dist-worker] serving layers %d-%d on %s:%d\n", start, end,
           loopback_only ? "127.0.0.1" : "0.0.0.0", port);
    fflush(stdout);
    float *x = (float *)malloc(((size_t)cfg->dim + 1) * sizeof(float));
    if (!x) {
        rpc_close(srv);
        return TN_ERR_OOM;
    }
    int sw_pinned = -1;
    for (;;) {
        int c;
        if (rpc_accept(srv, &c) != TN_OK) continue;
        for (;;) {
            RpcHeader h;
            if (rpc_recv(c, &h, x, (uint32_t)cfg->dim + 1) != TN_OK) break;
            if (h.type == RPC_INFO) {
                RpcInfo info = {cfg->n_layers,
                                cfg->dim,
                                cfg->vocab_size,
                                start,
                                end,
                                mc && mc->is_ssm ? 1 : 0,
                                0,
                                0};
                if (rpc_send(c, RPC_INFO, 0, &info, sizeof(info) / 4) != TN_OK) break;
            } else if (h.type == RPC_FORWARD) {
                if (h.count != (uint32_t)cfg->dim + 1 || h.arg < 0 || h.arg >= cfg->seq_len) {
                    rpc_send(c, RPC_ERROR, TN_ERR_DIMENSION_MISMATCH, NULL, 0);
                    continue;
                }
                int pinned = (int)x[cfg->dim];
                if (h.arg == 0 || pinned != sw_pinned) {
                    sw_init(&s->sw, s->max_seq_len, pinned);
                    sw_pinned = pinned;
                }
                memcpy(s->x, x, (size_t)cfg->dim * sizeof(float));
                for (int l = start; l < end; l++)
                    transformer_layer_forward(s, w, cfg, mc, l, h.arg, tp);
                if (rpc_send(c, RPC_FORWARD, h.arg, s->x, (uint32_t)cfg->dim) != TN_OK) break;
            } else {
                break; /* BYE or unknown */
            }
        }
        rpc_close(c);
    }
}
