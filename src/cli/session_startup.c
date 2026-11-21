/* Startup phases of a CLI run: commands that need no model, the tensor dump file, the banner and
 * model-format probe, and the hardware tuning (SIMD tier, calibration, classifier, threads). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli/banner.h"
#include "cli/color.h"
#include "cli/session.h"
#include "core/debug.h"
#include "core/log.h"
#include "math/simd_dispatch.h"
#include "runtime/calibration.h"
#include "runtime/hardware_profile.h"
#include "update/update_check.h"

/* Log format and threshold: --log-format / --log-level, and --verbose means debug. */
void session_apply_log_options(PzSession *ses) {
    tn_log_set_format(ses->args.output.log_json ? TN_LOG_JSON : TN_LOG_TEXT);
    if (ses->args.output.log_level >= 0)
        tn_log_set_level((TnLogLevel)ses->args.output.log_level);
    else if (ses->args.output.verbose)
        tn_log_set_level(TN_LOG_DEBUG);
}

/* --version and --check-update need no model. Returns the exit status, or -1 when the run goes on.
 */
int session_early_command(PzSession *ses) {
    /* --version: print version + the SIMD backend selected for this CPU, then
     * exit 0. tn_simd_init() only probes CPUID, so no model is required. */
    if (ses->args.output.show_version) {
        printf("Project Zero %s\n", PZ_VERSION_STR);
        printf("SIMD backend (this CPU): %s\n", tn_simd_init());
        return 0;
    }

    /* Phase 38: --check-update — explicit, synchronous, no model needed. */
    if (ses->args.update.check_update) {
        UpdateOptions uo;
        UpdateInfo ui;
        args_fill_update_options(&ses->args, &uo);
        update_check(&uo, PZ_VERSION_STR, 1, &ui);
        printf("current : %s\n", PZ_VERSION_STR);
        printf("latest  : %s%s\n", ui.latest[0] ? ui.latest : "(none found)",
               ui.latest_prerelease ? " (pre-release)" : "");
        printf("channel : %s   repo: %s\n", update_channel_name(uo.channel), uo.repo);
        printf("status  : %s%s%s\n", update_status_name(ui.status), ui.error[0] ? " - " : "",
               ui.error);
        if (ui.url[0]) printf("release : %s\n", ui.url);
        if (ui.status == UPDATE_STATUS_AVAILABLE)
            printf("verify the download with the release's SHA256SUMS (sha256sum -c SHA256SUMS)\n");
        if (ui.status == UPDATE_STATUS_AVAILABLE) return 3;
        if (ui.status == UPDATE_STATUS_ERROR) return 1;
        return 0;
    }

    return -1;
}

/* --dump-tensors: opens the CSV file the forward pass appends to. */
void session_open_dump(PzSession *ses) {
    /* Open tensor dump file if requested */
    if (ses->args.eval.dump_tensors_path) {
        g_dump_fp = fopen(ses->args.eval.dump_tensors_path, "w");
        if (!g_dump_fp) {
            TN_LOG_WRN("session_startup", "Warning: cannot open dump file '%s' — dumps disabled\n",
                       ses->args.eval.dump_tensors_path);
        } else {
            fprintf(g_dump_fp, "layer,step,n_elem,v0,v1,v2,v3,v4,v5,v6,v7,mean,absmax\n");
            TN_LOG_INF("dump", "[dump] Writing tensors to: %s\n", ses->args.eval.dump_tensors_path);
        }
    }
}

/* Prints the banner and decides GGUF vs native from the file's magic. Returns 1 when the model
 * file cannot be opened. */
int session_detect_format(PzSession *ses) {
    /* Phase 22.5: startup banner — always shown, regardless of invocation
     * path (TTY, piped/redirected, one-shot --prompt, --server, REPL).
     * Branding must not depend on how the CLI is invoked: llama-cli (the
     * reference engine used throughout this project's head-to-head
     * benchmarks) prints its banner unconditionally regardless of -p/
     * single-turn mode — confirmed by reading its actual source
     * (tools/cli/cli.cpp's console::log(LLAMA_ASCII_LOGO) call has no gate
     * at all). tn_banner_print() itself branches on is_tty: animated reveal
     * for a real terminal, a single plain (no-ANSI) block otherwise — so
     * piped/redirected captures (e.g. benchmark screenshots built from
     * tools/make_screenshot.py) are self-identifying too, not just plain
     * generation output. */
    ses->stdout_is_tty = isatty(fileno(stdout));
    int color_enabled_early =
        tn_color_resolve(ses->args.output.color_mode, ses->stdout_is_tty, getenv("NO_COLOR"));
    tn_banner_print(ses->stdout_is_tty, color_enabled_early);

    printf("Project Zero Engine %s — Auto-Tuned Hardware\n", PZ_VERSION_STR);

    /* Detect model format by reading the 4-byte magic number, without a
     * full mmap (load_gguf_model() below does its own mapped_file_open() for
     * the GGUF case; the native case opens its own mf directly below) —
     * a plain 4-byte read is all this decision needs. Done before the
     * hardware probe and first-run calibration (2026-10-22) so a wrong
     * --model path fails at once instead of after a minutes-long sweep. */
    {
        FILE *magic_f = fopen(ses->args.input.model_path, "rb");
        if (!magic_f) {
            TN_LOG_ERR("session_startup", "Failed to open model file: %s\n",
                       ses->args.input.model_path);
            return 1;
        }
        uint32_t file_magic = 0;
        size_t n            = fread(&file_magic, 1, sizeof(file_magic), magic_f);
        fclose(magic_f);
        ses->is_gguf = (n == sizeof(file_magic) && file_magic == GGUF_MAGIC);
    }

    return 0;
}

/* --act-quant auto: say which activation numerics the calibration picked (measured int8 tier
 * against float tier). */
static void announce_act_quant(const TnCalibrationResult *calib, int dispatcher_int8) {
    const char *aq = tn_calibration_act_quant(calib);
    double t_i8 = 0, t_fl = 0;
    if (tn_calibration_act_timing(calib, dispatcher_int8, &t_i8, &t_fl))
        tn_log_out(TN_LOG_INFO, "session_startup",
                   "Activation quantization: %s (auto, %s; measured int8 tier %.1f vs float tier "
                   "%.1f tok/s)\n",
                   aq ? aq : tn_act_quant_effective(), aq ? "calibrated" : "highest tier", t_i8,
                   t_fl);
    else
        TN_OUT_INF("session_startup", "Activation quantization: %s (auto, %s)\n",
                   aq ? aq : tn_act_quant_effective(), aq ? "calibrated" : "highest tier");
}

/* SIMD tier, calibration, classifier, activation numerics and the thread pool. Returns 1 when
 * the pool cannot be created. */
int session_tune_hardware(PzSession *ses) {
    /* ── SIMD backend selection ─────────────────────────────────────────── */
    /* CLI --simd override takes precedence over calibration and auto-detect.
     * Order: --simd flag > TN_FORCE_BACKEND env > calibration cache > auto */
    if (ses->args.perf.simd_override) {
        setenv("TN_FORCE_BACKEND", ses->args.perf.simd_override, 1);
        TN_OUT_INF("session_startup", "SIMD override: %s (user-selected)\n",
                   ses->args.perf.simd_override);
    }

    tn_simd_init();
    const int dispatcher_int8 =
        strcmp(tn_act_quant_effective(), "int8") == 0; /* what "auto" runs, before calibration */

    /* Hardware Profile: auto-detect cores, cache, bandwidth, classifier format.
     * The DRAM bandwidth comes from a per-host cache (core/bw_cache.h);
     * --calibrate re-measures it along with everything else. */
    if (ses->args.perf.calibrate) setenv("PZ_BW_REPROBE", "1", 1);
    const TnHardwareProfile *hw = tn_hardware_profile_init();

    /* ── Calibration ────────────────────────────────────────────────────── */
    TnCalibrationResult calib;
    bool have_calib = false;

    if (ses->args.perf.no_calibrate) {
        /* 2026-11-18: no calibration at all -- nothing is loaded, run or saved, so
         * the dispatcher's own tier and the detected thread count apply and a run
         * does not depend on what an earlier run left in ~/.project-zero. */
        TN_OUT_INF("calibration", "[Calibration] Skipped: --no-calibrate\n");
    } else if (ses->args.perf.calibrate) {
        /* User explicitly requested calibration */
        tn_calibrate(&calib, hw);
        /* A sanitizer build measures 5-20x slower than release: never let
         * it overwrite the cache release builds read (2026-10-22). */
        if (!TN_SANITIZED) tn_calibration_save(&calib);
        have_calib = true;
    } else {
        /* Try to load cached calibration */
        have_calib = tn_calibration_load(&calib, hw);
        if (!have_calib && TN_SANITIZED) {
            /* 2026-10-22: the first-run sweep took >2 min under ASan on a
             * fresh CI runner (the fuzzer counted it as a hang) and would
             * save meaningless timings. Auto-detected defaults instead. */
            TN_OUT_INF("calibration",
                       "[Calibration] Skipped: sanitizer build (timings not representative)\n");
        } else if (!have_calib) {
            /* First run or hardware changed — auto-calibrate */
            TN_OUT_INF("calibration",
                       "[Calibration] First run on this hardware — calibrating...\n");
            tn_calibrate(&calib, hw);
            tn_calibration_save(&calib);
            have_calib = true;
        }
    }

    /* Apply calibrated SIMD backend (if no CLI override) */
    if (have_calib && !ses->args.perf.simd_override && !getenv("TN_FORCE_BACKEND")) {
        const char *best = tn_calibration_best_simd(&calib);
        if (best) {
            setenv("TN_FORCE_BACKEND", best, 1);
            tn_simd_init();
            unsetenv("TN_FORCE_BACKEND");
            TN_OUT_INF("session_startup", "SIMD: %s (calibrated optimal)\n", best);
        }
    }
    /* --act-quant auto: the numerics are the calibrated tier's (measured int8 against float). */
    if (have_calib && ses->args.eval.act_quant == 0) announce_act_quant(&calib, dispatcher_int8);

    /* ── Classifier selection ───────────────────────────────────────────── */
    if (ses->args.perf.classifier_override == 3) {
        /* --classifier auto-fast: use calibration's fastest classifier */
        TnClassifierFormat fast_cls =
            have_calib ? tn_calibration_best_classifier(&calib)
            : (hw->cpu->avx512vnni || hw->cpu->avx_vnni || hw->cpu->arm_dotprod) ? TN_CLS_INT8
                                                                                 : TN_CLS_BF16;
        tn_hardware_profile_set_classifier(fast_cls);
        const char *names[] = {"BF16", "INT8", "INT4"};
        TN_OUT_INF("session_startup", "Classifier: %s (auto-fast, calibrated)\n", names[fast_cls]);
    } else if (ses->args.perf.classifier_override >= 0 && ses->args.perf.classifier_override <= 2) {
        tn_hardware_profile_set_classifier((TnClassifierFormat)ses->args.perf.classifier_override);
        const char *names[] = {"BF16", "INT8", "INT4"};
        TN_OUT_INF("session_startup", "Classifier: %s (user-selected)\n",
                   names[ses->args.perf.classifier_override]);
    }
    /* else: default BF16 (from hardware_profile.c select_classifier) */

    /* ── Activation numerics ────────────────────────────────────────────── */
    if (ses->args.eval.act_quant != 0) {
        tn_act_quant_set((TnActQuant)ses->args.eval.act_quant);
        TN_OUT_INF("session_startup", "Activation quantization: %s (user-selected)\n",
                   tn_act_quant_name((TnActQuant)ses->args.eval.act_quant));
    }

    tn_hardware_profile_report(hw);

    /* Thread count: CLI override > calibration > auto-detected */
    int active_threads;
    if (ses->args.perf.num_threads > 0) {
        active_threads = ses->args.perf.num_threads;
    } else if (have_calib && calib.best_threads > 0) {
        active_threads = calib.best_threads;
    } else {
        active_threads = hw->optimal_threads;
    }
    TN_OUT_INF("session_startup", "Active: %d threads | %s\n", active_threads, hw->summary);

    /* Create Thread Pool */
    ses->tp = threadpool_create(active_threads);
    if (!ses->tp) {
        TN_LOG_ERR("session_startup", "Failed to create thread pool\n");
        return 1;
    }

    return 0;
}
