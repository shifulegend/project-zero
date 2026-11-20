/* Layer planning and distributed execution: --plan-layers, --rpc-worker and --remote-layers. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/session.h"
#include "core/log.h"
#include "transformer/dist_worker.h"

/* Bytes in a GiB, for the --remote-layers per-node RAM sizes (given and printed in GB). */
#define BYTES_PER_GIB 1073741824.0

/* Returns non-zero when the run ends here: the plan was printed, the worker finished, or the
 * remote layers could not be set up. */
int session_distributed(PzSession *S) {
    /* Phase 28.1: --plan-layers prints a RAM-proportional split and exits. */
    if (S->args.plan_layers) {
        long long ram[DIST_MAX_NODES];
        int starts[DIST_MAX_NODES], ends[DIST_MAX_NODES], n = 0;
        char buf[DIST_SPEC_BUF];
        snprintf(buf, sizeof(buf), "%s", S->args.plan_layers);
        char *save = NULL;
        for (char *tok = strtok_r(buf, ",", &save); tok && n < DIST_MAX_NODES;
             tok       = strtok_r(NULL, ",", &save))
            ram[n++] = (long long)(atof(tok) * BYTES_PER_GIB);
        if (n == 0) {
            S->exit_code = 1;
            TN_LOG_ERR("session_dist", "Error: --plan-layers needs GB,GB,... (max %d nodes)\n",
                       DIST_MAX_NODES);
        } else {
            dist_plan_layers_by_ram(S->p.n_layers, ram, n, starts, ends);
            printf("Layer plan for %d layers over %d node(s):\n", S->p.n_layers, n);
            for (int i = 0; i < n; i++)
                printf("  node %d (%.1f GB): layers %d-%d%s\n", i, (double)ram[i] / BYTES_PER_GIB,
                       starts[i], ends[i], i == 0 ? "  (coordinator)" : "");
            if (n > 1) {
                printf("Coordinator: --remote-layers ");
                for (int i = 1; i < n; i++)
                    printf("%sHOST%d:PORT@%d-%d", i > 1 ? "," : "", i, starts[i], ends[i]);
                printf("\nWorker i:    --rpc-worker PORT --layers <its range>\n");
            }
        }
        S->args.prompt      = NULL;
        S->args.server_mode = 0;
        return 1;
    }

    /* Phase 28: worker mode serves a layer range and never generates. */
    if (S->args.rpc_worker_port > 0) {
        int a = 0, b = S->p.n_layers;
        if (!S->args.worker_layers || sscanf(S->args.worker_layers, "%d-%d", &a, &b) != 2 ||
            a < 0 || b > S->p.n_layers || a >= b) {
            S->exit_code = 1;
            TN_LOG_ERR("session_dist", "Error: --rpc-worker needs --layers A-B within 0-%d\n",
                       S->p.n_layers);
        } else {
            dist_worker_serve(S->args.rpc_worker_port, !S->args.rpc_bind_all, a, b, &S->p, &S->w,
                              S->s, &S->mc, S->tp);
        }
        S->args.prompt      = NULL;
        S->args.server_mode = 0;
        return 1;
    }

    /* Phase 28: coordinator — connect to the workers for remote layer ranges. */
    if (S->args.remote_layers) {
        if (S->draft_active) {
            S->exit_code = 1;
            TN_LOG_ERR("session_dist",
                       "Error: --draft-model cannot be combined with --remote-layers\n");
            return 1;
        }
        if (dist_plan_parse(&S->dist, S->args.remote_layers, S->p.n_layers) != 0 ||
            dist_plan_connect(&S->dist, &S->p, &S->mc) != TN_OK) {
            S->exit_code = 1;
            TN_LOG_ERR("session_dist", "Error: could not set up --remote-layers\n");
            dist_plan_close(&S->dist);
            return 1;
        }
        dist_plan_set_active(&S->dist);
        S->dist_active = 1;
    }
    return 0;
}
