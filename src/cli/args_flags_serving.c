/*
 * src/cli/args_flags_serving.c
 *
 * Command-line flags: remote layers, request queue, rate/timeout/connection limits and parallel slots. One group of parse_args() (args.c); each branch is unchanged, and
 * tests/test_args_golden.c compares every flag's parse result with a golden table.
 */

#include <stdio.h>
#include <string.h>

#include "api/chat_compile.h"
#include "api/server_config.h"
#include "cli/arg_values.h"
#include "cli/args.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

TernaryError args_parse_serving(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--rpc-worker") == 0 && i + 1 < argc) {
        ARG_INT(args->rpc_worker_port);
        if (args->rpc_worker_port <= 0 || args->rpc_worker_port > 65535) {
            fprintf(stderr, "Error: --rpc-worker must be a port 1-65535\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
        args->worker_layers = argv[++i];
    } else if (strcmp(argv[i], "--rpc-bind-all") == 0) {
        args->rpc_bind_all = true;
    } else if (strcmp(argv[i], "--remote-layers") == 0 && i + 1 < argc) {
        args->remote_layers = argv[++i];
    } else if (strcmp(argv[i], "--plan-layers") == 0 && i + 1 < argc) {
        args->plan_layers = argv[++i];
    } else if (strcmp(argv[i], "--queue-size") == 0 && i + 1 < argc) {
        ARG_INT(args->queue_size);
        if (args->queue_size < 0) {
            fprintf(stderr, "Error: --queue-size must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--queue-timeout") == 0 && i + 1 < argc) {
        ARG_INT(args->queue_timeout_s);
        if (args->queue_timeout_s < 1) {
            fprintf(stderr, "Error: --queue-timeout must be >= 1 second\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--rate-limit") == 0 && i + 1 < argc) {
        ARG_INT(args->rate_limit);
        if (args->rate_limit < 0 || args->rate_limit > 100000) {
            fprintf(stderr, "Error: --rate-limit must be 0..100000 requests per second\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--client-timeout") == 0 && i + 1 < argc) {
        ARG_INT(args->client_timeout_s);
        if (args->client_timeout_s < 0 || args->client_timeout_s > 86400) {
            fprintf(stderr, "Error: --client-timeout must be 0..86400 seconds\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--max-connections") == 0 && i + 1 < argc) {
        ARG_INT(args->max_connections);
        if (args->max_connections < 0 || args->max_connections > 1048576) {
            fprintf(stderr, "Error: --max-connections must be 0..1048576\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
        ARG_INT(args->parallel);
        if (args->parallel < 1 || args->parallel > PZ_PARALLEL_MAX) {
            fprintf(stderr, "Error: --parallel must be 1..%d\n", PZ_PARALLEL_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--parallel-batch") == 0 && i + 1 < argc) {
        ARG_INT(args->parallel_batch);
        if (args->parallel_batch < 1 || args->parallel_batch > PZ_PARALLEL_BATCH_MAX) {
            fprintf(stderr, "Error: --parallel-batch must be 1..%d\n", PZ_PARALLEL_BATCH_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
