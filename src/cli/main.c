#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/args.h"
#include "cli/config_cmd.h"
#include "cli/program_name.h"
#include "cli/session.h"
#include "core/debug.h"
#include "core/progress.h"
#include "kv_cache/kv_strategy.h"
#include "threading/cache_probe.h"

/* A CLI run is a PzSession (cli/session.h) moved through the phases below; each phase lives in
 * src/cli/session_*.c. Every failure path ends in session_release(), which frees whatever stage
 * was reached. */
int main(int argc, char **argv) {
    pz_warn_if_legacy_invocation(argc > 0 ? argv[0] : NULL);
    /* Phase 40: `pz config ...` manages the settings file; every other run
     * reads it (and PZ_* env) as defaults placed before the user's flags. */
    if (argc > 1 && strcmp(argv[1], "config") == 0) return pz_config_command(argc - 1, argv + 1);
    argc = pz_config_prepare_argv(argc, argv, &argv);

    PzSession *S = (PzSession *)calloc(1, sizeof(*S));
    if (!S) {
        TN_LOG_ERR("main", "Error: out of memory\n");
        return 1;
    }
    if (parse_args(&S->args, argc, argv) != TN_OK) {
        free(S);
        return 1;
    }
    session_apply_log_options(S);
    {
        KVStrategyOptions kv_opt = {S->args.ctx_size, S->args.kv_ram_fraction, S->args.kv_type,
                                    S->args.kv_offload_dir};
        kv_strategy_set_options(&kv_opt);
    }
    (void)tn_cache_info(); /* Phase 27.1: probe once, before any worker threads */

    int early = session_early_command(S);
    if (early >= 0) {
        free(S);
        return early;
    }
    if (S->args.verbose) g_tn_verbose = 1;

    session_open_dump(S);
    if (session_detect_format(S) != 0 || session_tune_hardware(S) != 0) {
        free(S);
        return 1;
    }
    if (session_load_model(S) != 0) {
        threadpool_destroy(S->tp);
        free(S);
        return 1;
    }
    if (session_setup_tokenizer(S) != 0) goto fail;

    tn_progress_stage(4, 4, "Ready.", S->stdout_is_tty);
    tn_progress_done(S->stdout_is_tty);

    if (session_load_extras(S) != 0) goto fail;
    session_apply_runtime_options(S);
    if (!session_distributed(S)) {
        session_start_update_check(S);
        session_accuracy_checks(S);
        session_execute(S);
        session_finish(S);
    }
    session_release(S);
    int exit_code = S->exit_code;
    free(S);
    return exit_code;

fail:
    session_release(S);
    free(S);
    return 1;
}
