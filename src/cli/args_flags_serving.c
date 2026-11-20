/*
 * src/cli/args_flags_serving.c
 *
 * Command-line flags: remote layers and parallel slots (the request queue and size limits are in args_flags_limits.c). One group of parse_args() (args.c); each branch is unchanged, and
 * tests/test_args_golden.c compares every flag's parse result with a golden table.
 */

#include <stdio.h>
#include <string.h>

#include "api/chat_compile.h"
#include "api/server_config.h"
#include "cli/arg_values.h"
#include "cli/args.h"
#include "cli/args_groups.h"
#include "core/log.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

TernaryError args_parse_serving(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--rpc-worker") == 0 && i + 1 < argc) {
        ARG_INT(args->dist.rpc_worker_port);
        if (args->dist.rpc_worker_port <= 0 || args->dist.rpc_worker_port > 65535) {
            TN_LOG_ERR("args_flags_serving", "Error: --rpc-worker must be a port 1-65535\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
        args->dist.worker_layers = argv[++i];
    } else if (strcmp(argv[i], "--rpc-bind-all") == 0) {
        args->dist.rpc_bind_all = true;
    } else if (strcmp(argv[i], "--remote-layers") == 0 && i + 1 < argc) {
        args->dist.remote_layers = argv[++i];
    } else if (strcmp(argv[i], "--plan-layers") == 0 && i + 1 < argc) {
        args->dist.plan_layers = argv[++i];
    } else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
        ARG_INT(args->server.parallel);
        if (args->server.parallel < 1 || args->server.parallel > PZ_PARALLEL_MAX) {
            TN_LOG_ERR("args_flags_serving", "Error: --parallel must be 1..%d\n", PZ_PARALLEL_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--parallel-batch") == 0 && i + 1 < argc) {
        ARG_INT(args->server.parallel_batch);
        if (args->server.parallel_batch < 1 ||
            args->server.parallel_batch > PZ_PARALLEL_BATCH_MAX) {
            TN_LOG_ERR("args_flags_serving", "Error: --parallel-batch must be 1..%d\n",
                       PZ_PARALLEL_BATCH_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return args_parse_limits(args, argc, argv, ip, handled); /* or not ours at all */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
