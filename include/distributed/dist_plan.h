#ifndef TN_DIST_PLAN_H
#define TN_DIST_PLAN_H

/*
 * dist_plan.h — Phase 28.1 / 28.3: pipeline-parallel layer split across
 * processes or machines.
 *
 * A worker (`--rpc-worker PORT --layers A-B`) loads the same model file and
 * runs layers [A, B) for whoever sends it a hidden state. The coordinator
 * (`--remote-layers host:port@A-B[,...]`) runs embedding, its own layers and
 * the classifier; whenever the layer loop reaches a remote range it ships the
 * hidden state there and continues with the reply. Each worker keeps its own
 * KV/SSM state for its layers, keyed by position exactly like local layers,
 * so caching and resets stay consistent while one coordinator drives it.
 */

#include "core/config.h"
#include "core/moe_config.h"
#include "core/error.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "threading/thread_pool.h"

#define DIST_MAX_NODES 8

typedef struct {
    char host[128];
    int  port;
    int  start, end;   /* layers [start, end) */
    int  fd;           /* -1 until connected */
    float *scratch;    /* dim + 1 floats: outgoing FORWARD payload */
} RemoteNode;

typedef struct {
    int        n;
    RemoteNode nodes[DIST_MAX_NODES];
    int        dim;
    int        failed;  /* set on any transport error; generation stops */
} DistPlan;

/* "host:port@A-B,host:port@C-D". Ranges must be within [0, n_layers), non-empty,
 * non-overlapping. Returns 0 or -1 (message on stderr). */
int dist_plan_parse(DistPlan *dp, const char *spec, int n_layers);

/* 28.1: split n_layers over n_nodes proportionally to ram[i] (bytes; <= 0
 * counts as the average). Writes contiguous [starts[i], ends[i]) ranges that
 * cover every layer, each node >= 1 layer when n_layers >= n_nodes. */
void dist_plan_layers_by_ram(int n_layers, const long long *ram, int n_nodes,
                             int *starts, int *ends);

/* Connects to every node and validates its INFO handshake against cfg/mc. */
TernaryError dist_plan_connect(DistPlan *dp, const Config *cfg, const MoEConfig *mc);
void         dist_plan_close(DistPlan *dp);

/* Process-wide plan used by transformer_forward() (NULL = all layers local). */
void      dist_plan_set_active(DistPlan *dp);
DistPlan *dist_plan_active(void);

/* Node whose range starts at `layer`, else NULL. */
RemoteNode *dist_plan_node_starting_at(DistPlan *dp, int layer);

/* Ships x[dim] through the node's layers at position pos (in place).
 * sw_pinned = the coordinator's sliding-window pinned length (s->sw), which
 * the worker mirrors so both sides map positions to the same KV slots. */
TernaryError dist_forward_remote(DistPlan *dp, RemoteNode *node, float *x, int pos,
                                 int sw_pinned);

/* Worker loop: serves FORWARD requests for layers [start, end) forever
 * (one coordinator connection at a time). Returns only on listen failure. */
TernaryError dist_worker_serve(int port, int loopback_only, int start, int end,
                               const Config *cfg, const TransformerWeights *w, RunState *s,
                               const MoEConfig *mc, ThreadPool *tp);

#endif /* TN_DIST_PLAN_H */
