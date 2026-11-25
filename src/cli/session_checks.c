/* Startup checks that run once the model is loaded: the background update check and the two
 * accuracy probes (LM head quantisation, int8 activations). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cli/session.h"
#include "core/log.h"
#include "core/platform.h"
#include "eval/selfcheck.h"
#include "math/simd_dispatch.h"
#include "runtime/hardware_profile.h"
#include "update/update_check.h"

/* Server always (the web UI shows it), CLI only when a person is watching stderr. Never blocks. */
void session_start_update_check(PzSession *ses) {
    /* Phase 38: background update check — server always (the web UI shows
     * it), CLI only when a person is watching stderr. Never blocks. */
    ses->update.notice = 0;
    if (!ses->args.watermark.watermark_detect && !ses->args.eval.perplexity_file &&
        (ses->args.server.server_mode || isatty(fileno(stderr)))) {
        UpdateOptions uo;
        args_fill_update_options(&ses->args, &uo);
        update_service_init(&ses->update.svc, &uo, PZ_VERSION_STR);
        update_service_poll(&ses->update.svc, 0);
        ses->update.notice = !ses->args.server.server_mode;
    }
}

/* Accuracy probe (2026-11-18): a LM head the user asked to be quantised must still compute
 * the model. Over --quality-tolerance it falls back to BF16, loudly; --quality-tolerance 0
 * keeps the request as given. --self-check below runs the same probe on demand. */
static void check_lossy_head(PzSession *ses) {
    const TnHardwareProfile *hpc = tn_hardware_profile_get();
    int lossy_head = hpc && hpc->classifier_explicit && hpc->classifier_fmt != TN_CLS_BF16;
    if (lossy_head && ses->args.eval.quality_tolerance > 0.0 && !ses->args.eval.self_check &&
        !ses->args.eval.perplexity_file && !ses->args.watermark.watermark_detect) {
        SelfCheckOptions so = {ses->args.eval.quality_tolerance, ses->args.eval.self_check_tokens};
        SelfCheckResult sr;
        if (selfcheck_run(&so, &ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp, &sr) ==
                TN_OK &&
            sr.verdict == SC_FAIL) {
            char line[SELFCHECK_LINE_BUF];
            selfcheck_format(&sr, line, sizeof(line));
            tn_log(TN_LOG_INFO, "session_checks",
                   "[%s]\n[self-check] the %s LM head costs more than --quality-tolerance on this "
                   "model: using a BF16 head. Keep the request with a higher --quality-tolerance "
                   "(0 = no bar).\n",
                   line, sr.classifier);
            tn_hardware_profile_swap_classifier(TN_CLS_BF16);
        }
    }
}

/* A LM head the user asked to be quantised must still compute the model, and the int8
 * activations of a ternary model are accuracy-checked once (verdict cached). */
void session_accuracy_checks(PzSession *ses) {
    check_lossy_head(ses);

    /* --act-quant auto on a ternary model: the int8 numerics are accuracy-checked once per model
     * and setting (verdict cached in ~/.project-zero/selfcheck.txt), and float activations take
     * over if they fail. --no-calibrate, sanitizer builds and an explicit --act-quant skip it. */
    SelfCheckAutoCtx auto_ctx = {ses->args.eval.act_quant,
                                 ses->w.layers_are_ternary,
                                 ses->args.perf.no_calibrate,
                                 TN_SANITIZED,
                                 ses->args.eval.quality_tolerance,
                                 ses->args.eval.self_check || ses->args.eval.perplexity_file ||
                                     ses->args.watermark.watermark_detect,
                                 strcmp(tn_act_quant_effective(), "int8") == 0};
    if (selfcheck_auto_applies(&auto_ctx)) {
        const TnHardwareProfile *hpa = tn_hardware_profile_get();
        const char *cls              = !hpa                                 ? "bf16"
                                       : hpa->classifier_fmt == TN_CLS_INT4 ? "int4"
                                       : hpa->classifier_fmt == TN_CLS_INT8 ? "int8"
                                                                            : "bf16";
        struct stat mst;
        if (stat(ses->args.input.model_path, &mst) == 0) {
            char key[SELFCHECK_KEY_BUF];
            selfcheck_cache_key(key, sizeof(key), ses->args.input.model_path,
                                (long long)mst.st_size, (long long)mst.st_mtime, "int8", cls,
                                ses->args.eval.quality_tolerance, ses->args.eval.self_check_tokens);
            SelfCheckVerdict cv;
            if (!selfcheck_cache_load(key, &cv)) {
                tn_log_out(
                    TN_LOG_INFO, "self_check",
                    "[self-check] first run of this model with int8 activations: checking accuracy "
                    "(cached afterwards)...\n");
                fflush(stdout);
                SelfCheckOptions so = {ses->args.eval.quality_tolerance,
                                       ses->args.eval.self_check_tokens};
                SelfCheckResult sr;
                if (selfcheck_run(&so, &ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp, &sr) ==
                        TN_OK &&
                    sr.verdict != SC_SKIPPED) {
                    char line[SELFCHECK_LINE_BUF];
                    selfcheck_format(&sr, line, sizeof(line));
                    TN_OUT_INF("session_checks", "%s\n", line);
                    selfcheck_cache_store(key, sr.verdict);
                    cv = sr.verdict;
                } else {
                    cv = SC_SKIPPED;
                }
            }
            if (selfcheck_auto_numerics(cv) == TN_ACT_FLOAT) {
                tn_act_quant_set(TN_ACT_FLOAT);
                TN_LOG_INF(
                    "self_check",
                    "[self-check] int8 activations cost more than --quality-tolerance on this "
                    "model: using float activations.\n");
            }
        }
    }
}
