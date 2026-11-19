#ifndef TN_TRANSFORMER_DIST_WORKER_H
#define TN_TRANSFORMER_DIST_WORKER_H

#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "runtime/run_state.h"
#include "threading/thread_pool.h"

/* Worker loop: serves FORWARD requests for layers [start, end) forever (one coordinator connection
 * at a time). Returns only on listen failure. */
TernaryError dist_worker_serve(int port, int loopback_only, int start, int end, const Config *cfg,
                               const TransformerWeights *w, RunState *s, const MoEConfig *mc,
                               ThreadPool *tp);

#endif /* TN_TRANSFORMER_DIST_WORKER_H */
