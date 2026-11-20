#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/args.h"
#include "cli/config_cmd.h"
#include "cli/program_name.h"
#include "cli/session.h"
#include "core/debug.h"
#include "core/progress.h"
#include "threading/cache_probe.h"

/* A CLI run is a PzSession (cli/session.h) moved through the phases below; each phase lives in
 * src/cli/session_*.c. Every failure path ends in session_release(), which frees whatever stage
 * was reached. */
static int hardware_setup_failed(PzSession *ses) {
    return session_detect_format(ses) != 0 || session_tune_hardware(ses) != 0;
}

static int run(int argc, char **argv, const PzProvenance *provenance) {
    PzSession *ses = (PzSession *)calloc(1, sizeof(*ses));
    if (!ses) {
        TN_LOG_ERR("main", "Error: out of memory\n");
        return 1;
    }
    if (parse_args(&ses->args, argc, argv) != TN_OK) {
        free(ses);
        return 1;
    }
    ses->args.server.config_provenance = *provenance;
    session_apply_log_options(ses);
    (void)tn_cache_info(); /* Phase 27.1: probe once, before any worker threads */

    int early = session_early_command(ses);
    if (early >= 0) {
        free(ses);
        return early;
    }
    if (ses->args.output.verbose) g_tn_verbose = 1;

    session_open_dump(ses);
    if (hardware_setup_failed(ses)) {
        free(ses);
        return 1;
    }
    if (session_load_model(ses) != 0) {
        threadpool_destroy(ses->tp);
        free(ses);
        return 1;
    }
    if (session_setup_tokenizer(ses) != 0) goto fail;

    tn_progress_stage(4, 4, "Ready.", ses->stdout_is_tty);
    tn_progress_done(ses->stdout_is_tty);

    if (session_load_extras(ses) != 0) goto fail;
    session_apply_runtime_options(ses);
    if (!session_distributed(ses)) {
        session_start_update_check(ses);
        session_accuracy_checks(ses);
        session_execute(ses);
        session_finish(ses);
    }
    session_release(ses);
    int exit_code = ses->exit_code;
    free(ses);
    return exit_code;

fail:
    session_release(ses);
    free(ses);
    return 1;
}

int main(int argc, char **argv) {
    pz_warn_if_legacy_invocation(argc > 0 ? argv[0] : NULL);
    /* Phase 40: `pz config ...` manages the settings file; every other run
     * reads it (and PZ_* env) as defaults placed before the user's flags. */
    if (argc > 1 && strcmp(argv[1], "config") == 0) return pz_config_command(argc - 1, argv + 1);
    PzLaunchConfig launch;
    memset(&launch, 0, sizeof(launch));
    argc   = pz_config_prepare_argv(&launch, argc, argv, &argv);
    int rc = run(argc, argv, &launch.provenance);
    pz_launch_config_free(&launch);
    return rc;
}
