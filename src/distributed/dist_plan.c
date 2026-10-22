#include "distributed/dist_plan.h"
#include "distributed/tensor_rpc.h"
#include "transformer/forward.h"
#include "kv_cache/sliding_window.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static DistPlan *g_active;

void dist_plan_set_active(DistPlan *dp) { g_active = dp; }
DistPlan *dist_plan_active(void) { return g_active; }

int dist_plan_parse(DistPlan *dp, const char *spec, int n_layers) {
    if (!dp || !spec) return -1;
    memset(dp, 0, sizeof(*dp));
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", spec);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        if (dp->n >= DIST_MAX_NODES) { fprintf(stderr, "[dist] too many nodes (max %d)\n", DIST_MAX_NODES); return -1; }
        RemoteNode *nd = &dp->nodes[dp->n];
        char *at = strchr(tok, '@'), *colon = strrchr(tok, ':');
        if (!at || !colon || colon > at) { fprintf(stderr, "[dist] bad node '%s' (want host:port@A-B)\n", tok); return -1; }
        *colon = '\0';
        *at = '\0';
        snprintf(nd->host, sizeof(nd->host), "%s", tok);
        nd->port = atoi(colon + 1);
        if (sscanf(at + 1, "%d-%d", &nd->start, &nd->end) != 2 || nd->port <= 0 || nd->port > 65535 ||
            nd->start < 0 || nd->end > n_layers || nd->start >= nd->end) {
            fprintf(stderr, "[dist] bad range/port in '%s:%s@%s' (model has %d layers)\n",
                    nd->host, colon + 1, at + 1, n_layers);
            return -1;
        }
        nd->fd = -1;
        nd->scratch = NULL;
        for (int i = 0; i < dp->n; i++) {
            if (nd->start < dp->nodes[i].end && dp->nodes[i].start < nd->end) {
                fprintf(stderr, "[dist] overlapping layer ranges\n");
                return -1;
            }
        }
        dp->n++;
    }
    return dp->n > 0 ? 0 : -1;
}

void dist_plan_layers_by_ram(int n_layers, const long long *ram, int n_nodes,
                             int *starts, int *ends) {
    if (n_nodes <= 0 || n_layers <= 0) return;
    double total = 0, avg = 0;
    int known = 0;
    for (int i = 0; i < n_nodes; i++) if (ram && ram[i] > 0) { avg += (double)ram[i]; known++; }
    avg = known ? avg / known : 1.0;
    for (int i = 0; i < n_nodes; i++) total += (ram && ram[i] > 0) ? (double)ram[i] : avg;
    int pos = 0;
    double acc = 0;
    for (int i = 0; i < n_nodes; i++) {
        acc += (ram && ram[i] > 0) ? (double)ram[i] : avg;
        int end = (i == n_nodes - 1) ? n_layers : (int)((double)n_layers * acc / total + 0.5);
        int min_end = pos + (n_layers >= n_nodes ? 1 : 0);
        int max_end = n_layers - (n_layers >= n_nodes ? (n_nodes - 1 - i) : 0);
        if (end < min_end) end = min_end;
        if (end > max_end) end = max_end;
        starts[i] = pos;
        ends[i] = end;
        pos = end;
    }
}

TernaryError dist_plan_connect(DistPlan *dp, const Config *cfg, const MoEConfig *mc) {
    dp->dim = cfg->dim;
    for (int i = 0; i < dp->n; i++) {
        RemoteNode *nd = &dp->nodes[i];
        nd->scratch = (float *)malloc(((size_t)cfg->dim + 1) * sizeof(float));
        if (!nd->scratch) return TN_ERR_OOM;
        TernaryError e = rpc_connect(nd->host, nd->port, &nd->fd);
        if (e != TN_OK) {
            fprintf(stderr, "[dist] cannot connect to %s:%d\n", nd->host, nd->port);
            return e;
        }
        RpcInfo info;
        RpcHeader h;
        if (rpc_send(nd->fd, RPC_INFO, 0, NULL, 0) != TN_OK ||
            rpc_recv(nd->fd, &h, &info, sizeof(info) / 4) != TN_OK || h.type != RPC_INFO ||
            h.count != sizeof(info) / 4) {
            fprintf(stderr, "[dist] handshake with %s:%d failed\n", nd->host, nd->port);
            return TN_ERR_INVALID_MAGIC;
        }
        if (info.n_layers != cfg->n_layers || info.dim != cfg->dim ||
            info.vocab_size != cfg->vocab_size || info.is_ssm != (mc && mc->is_ssm ? 1 : 0)) {
            fprintf(stderr, "[dist] %s:%d serves a different model (%d layers, dim %d, vocab %d)\n",
                    nd->host, nd->port, info.n_layers, info.dim, info.vocab_size);
            return TN_ERR_DIMENSION_MISMATCH;
        }
        if (info.layer_start != nd->start || info.layer_end != nd->end) {
            fprintf(stderr, "[dist] %s:%d serves layers %d-%d, expected %d-%d\n",
                    nd->host, nd->port, info.layer_start, info.layer_end, nd->start, nd->end);
            return TN_ERR_INVALID_CONFIG;
        }
        printf("[dist] layers %d-%d -> %s:%d\n", nd->start, nd->end, nd->host, nd->port);
    }
    return TN_OK;
}

void dist_plan_close(DistPlan *dp) {
    if (!dp) return;
    for (int i = 0; i < dp->n; i++) {
        if (dp->nodes[i].fd >= 0) {
            rpc_send(dp->nodes[i].fd, RPC_BYE, 0, NULL, 0);
            rpc_close(dp->nodes[i].fd);
            dp->nodes[i].fd = -1;
        }
        free(dp->nodes[i].scratch);
        dp->nodes[i].scratch = NULL;
    }
    if (g_active == dp) g_active = NULL;
}

RemoteNode *dist_plan_node_starting_at(DistPlan *dp, int layer) {
    if (!dp) return NULL;
    for (int i = 0; i < dp->n; i++) if (dp->nodes[i].start == layer) return &dp->nodes[i];
    return NULL;
}

TernaryError dist_forward_remote(DistPlan *dp, RemoteNode *node, float *x, int pos,
                                 int sw_pinned) {
    RpcHeader h;
    /* Payload: hidden state + the coordinator's sliding-window pinned length,
     * so the worker maps positions to KV slots identically (2026-10-18: the
     * worker never initialised its window and read/wrote the wrong slots). */
    float *msg = node->scratch;
    memcpy(msg, x, (size_t)dp->dim * sizeof(float));
    msg[dp->dim] = (float)sw_pinned;
    TernaryError e = rpc_send(node->fd, RPC_FORWARD, pos, msg, (uint32_t)dp->dim + 1);
    if (e == TN_OK) e = rpc_recv(node->fd, &h, x, (uint32_t)dp->dim);
    if (e == TN_OK && (h.type != RPC_FORWARD || h.count != (uint32_t)dp->dim))
        e = h.type == RPC_ERROR ? (TernaryError)h.arg : TN_ERR_DIMENSION_MISMATCH;
    if (e != TN_OK && !dp->failed) {
        fprintf(stderr, "[dist] layers %d-%d on %s:%d failed: %s\n",
                node->start, node->end, node->host, node->port, tn_error_str(e));
        dp->failed = 1;
    }
    return e;
}

TernaryError dist_worker_serve(int port, int loopback_only, int start, int end,
                               const Config *cfg, const TransformerWeights *w, RunState *s,
                               const MoEConfig *mc, ThreadPool *tp) {
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
    if (!x) { rpc_close(srv); return TN_ERR_OOM; }
    int sw_pinned = -1;
    for (;;) {
        int c;
        if (rpc_accept(srv, &c) != TN_OK) continue;
        for (;;) {
            RpcHeader h;
            if (rpc_recv(c, &h, x, (uint32_t)cfg->dim + 1) != TN_OK) break;
            if (h.type == RPC_INFO) {
                RpcInfo info = { cfg->n_layers, cfg->dim, cfg->vocab_size, start, end,
                                 mc && mc->is_ssm ? 1 : 0, 0, 0 };
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
                for (int l = start; l < end; l++) transformer_layer_forward(s, w, cfg, mc, l, h.arg, tp);
                if (rpc_send(c, RPC_FORWARD, h.arg, s->x, (uint32_t)cfg->dim) != TN_OK) break;
            } else {
                break; /* BYE or unknown */
            }
        }
        rpc_close(c);
    }
}
