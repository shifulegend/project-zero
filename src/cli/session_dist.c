/* Layer planning and distributed execution: --plan-layers, --rpc-worker and --remote-layers. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/session.h"
#include "core/log.h"
#include "transformer/dist_worker.h"

/* Bytes in a GiB, for the --remote-layers per-node RAM sizes (given and printed in GB). */
#define BYTES_PER_GIB 1073741824.0

/* Phase 28.1: --plan-layers prints a RAM-proportional split and exits. */
static void plan_layers(PzSession *ses) {
    long long ram[DIST_MAX_NODES];
    int starts[DIST_MAX_NODES], ends[DIST_MAX_NODES], n = 0;
    char buf[DIST_SPEC_BUF];
    snprintf(buf, sizeof(buf), "%s", ses->args.dist.plan_layers);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok && n < DIST_MAX_NODES;
         tok       = strtok_r(NULL, ",", &save))
        ram[n++] = (long long)(atof(tok) * BYTES_PER_GIB);
    if (n == 0) {
        ses->exit_code = 1;
        TN_LOG_ERR("session_dist", "Error: --plan-layers needs GB,GB,... (max %d nodes)\n",
                   DIST_MAX_NODES);
    } else {
        dist_plan_layers_by_ram(ses->p.n_layers, ram, n, starts, ends);
        printf("Layer plan for %d layers over %d node(s):\n", ses->p.n_layers, n);
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
    ses->args.input.prompt       = NULL;
    ses->args.server.server_mode = 0;
}

/* Returns non-zero when the run ends here: the plan was printed, the worker finished, or the
 * remote layers could not be set up. */
int session_distributed(PzSession *ses) {
    if (ses->args.dist.plan_layers) {
        plan_layers(ses);
        return 1;
    }

    /* Phase 28: worker mode serves a layer range and never generates. */
    if (ses->args.dist.rpc_worker_port > 0) {
        int a = 0, b = ses->p.n_layers;
        if (!ses->args.dist.worker_layers ||
            sscanf(ses->args.dist.worker_layers, "%d-%d", &a, &b) != 2 || a < 0 ||
            b > ses->p.n_layers || a >= b) {
            ses->exit_code = 1;
            TN_LOG_ERR("session_dist", "Error: --rpc-worker needs --layers A-B within 0-%d\n",
                       ses->p.n_layers);
        } else {
            dist_worker_serve(ses->args.dist.rpc_worker_port, !ses->args.dist.rpc_bind_all, a, b,
                              &ses->p, &ses->w, ses->s, &ses->mc, ses->tp);
        }
        ses->args.input.prompt       = NULL;
        ses->args.server.server_mode = 0;
        return 1;
    }

    /* Phase 28: coordinator — connect to the workers for remote layer ranges. */
    if (ses->args.dist.remote_layers) {
        if (ses->draft_active) {
            ses->exit_code = 1;
            TN_LOG_ERR("session_dist",
                       "Error: --draft-model cannot be combined with --remote-layers\n");
            return 1;
        }
        if (dist_plan_parse(&ses->dist, ses->args.dist.remote_layers, ses->p.n_layers) != 0 ||
            dist_plan_connect(&ses->dist, &ses->p, &ses->mc) != TN_OK) {
            ses->exit_code = 1;
            TN_LOG_ERR("session_dist", "Error: could not set up --remote-layers\n");
            dist_plan_close(&ses->dist);
            return 1;
        }
        ses->s->dist     = &ses->dist;
        ses->dist_active = 1;
    }
    return 0;
}
