#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* pause() for --server mode */
#include <signal.h>   /* SIGINT/SIGTERM graceful-shutdown handler, --server mode */
#include "core/platform.h"
#include "core/config.h"
#include "core/moe_config.h"
#include "core/moe_weights.h"
#include "core/weights.h"
#include "core/run_state.h"
#include "core/hardware_profile.h"
#include "core/calibration.h"
#include "core/debug.h"
#include "memory/mapped_file.h"
#include "memory/aligned_alloc.h"
#include "kv_cache/kv_strategy.h"
#include "threading/thread_pool.h"
#include "threading/cpu_probe.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_gguf.h"
#include "cli/args.h"
#include "cli/repl.h"
#include "cli/progress.h"
#include "cli/banner.h"
#include "cli/color.h"
#include "transformer/generate.h"
#include "transformer/forward.h"
#include "transformer/moe_ffn.h"
#include "math/simd_dispatch.h"

/* Phase 15 RAG */
#include "rag/rag_context.h"
#include "rag/vector_db.h"
#include "rag/embedder.h"

/* Phase 34 Multimodal (pipeline extracted to multimodal/vision_pipeline.c
 * in Phase 22.2 — see include/multimodal/vision_pipeline.h) */
#include "multimodal/vision_pipeline.h"

/* Phase 34.2 GGUF loader */
#include "core/gguf_reader.h"
#include "core/gguf_loader.h"

/* Phase 21: OpenAI-compatible API server */
#include "api/api_server.h"

/* Version is injected unquoted by the build (-DPZ_VERSION, see Makefile) and
 * stringified here; falls back to "dev" for builds that don't pass it
 * (e.g. plain `make debug`). Two-level macro so the token is expanded first. */
#ifndef PZ_VERSION
#define PZ_VERSION dev
#endif
#define PZ_STRINGIFY2(x) #x
#define PZ_STRINGIFY(x) PZ_STRINGIFY2(x)
#define PZ_VERSION_STR PZ_STRINGIFY(PZ_VERSION)

/* --server mode graceful shutdown: set only by handle_shutdown_signal (the
 * only async-signal-safe thing to do in a handler is set a sig_atomic_t
 * flag), polled by the main thread's wait loop below. Scoped to --server
 * mode only — installed right before the blocking wait, not at process
 * start, so REPL/one-shot Ctrl+C behavior (immediate kill, default
 * disposition) is unchanged. */
static volatile sig_atomic_t g_shutdown_requested = 0;

static void handle_shutdown_signal(int sig) {
    (void)sig;
    g_shutdown_requested = 1;
}

int main(int argc, char **argv) {
    CliArgs args;
    if (parse_args(&args, argc, argv) != TN_OK) {
        return 1;
    }

    /* --version: print version + the SIMD backend selected for this CPU, then
     * exit 0. tn_simd_init() only probes CPUID, so no model is required. */
    if (args.show_version) {
        printf("Project Zero Engine (adaptive_ai_engine) %s\n", PZ_VERSION_STR);
        printf("SIMD backend (this CPU): %s\n", tn_simd_init());
        return 0;
    }

    if (args.verbose) g_tn_verbose = 1;

    /* Open tensor dump file if requested */
    if (args.dump_tensors_path) {
        g_dump_fp = fopen(args.dump_tensors_path, "w");
        if (!g_dump_fp) {
            fprintf(stderr, "Warning: cannot open dump file '%s' — dumps disabled\n",
                    args.dump_tensors_path);
        } else {
            fprintf(g_dump_fp, "layer,step,n_elem,v0,v1,v2,v3,v4,v5,v6,v7,mean,absmax\n");
            fprintf(stderr, "[dump] Writing tensors to: %s\n", args.dump_tensors_path);
        }
    }

    /* Phase 22.5: animated startup banner — TTY-gated only. Previously also
     * suppressed for one-shot --prompt runs (matching Claude Code's own
     * convention), but llama-cli (the reference engine used throughout this
     * project's head-to-head benchmarks) prints its banner unconditionally
     * regardless of -p/single-turn mode — confirmed by reading its actual
     * source (tools/cli/cli.cpp's console::log(LLAMA_ASCII_LOGO) call has no
     * such gate at all). Matching that here keeps benchmark screenshots
     * visually self-identifying (which engine produced this terminal
     * capture) instead of only showing plain text for one-shot pz runs.
     * Piped/redirected output (not a real TTY) still gets plain text. */
    int stdout_is_tty = isatty(fileno(stdout));
    int color_enabled_early = tn_color_resolve(args.color_mode, stdout_is_tty, getenv("NO_COLOR"));
    if (stdout_is_tty) {
        tn_banner_print(stdout_is_tty, color_enabled_early);
    }

    printf("Project Zero Engine %s — Auto-Tuned Hardware\n", PZ_VERSION_STR);

    /* ── SIMD backend selection ─────────────────────────────────────────── */
    /* CLI --simd override takes precedence over calibration and auto-detect.
     * Order: --simd flag > TN_FORCE_BACKEND env > calibration cache > auto */
    if (args.simd_override) {
        setenv("TN_FORCE_BACKEND", args.simd_override, 1);
        printf("SIMD override: %s (user-selected)\n", args.simd_override);
    }

    if (args.moe_threading_override) {
        moe_set_threading_mode(args.moe_threading_override);
        printf("MoE threading mode: %s\n", args.moe_threading_override);
    }

    tn_simd_init();

    /* Hardware Profile: auto-detect cores, cache, bandwidth, classifier format */
    const TnHardwareProfile *hw = tn_hardware_profile_init();

    /* ── Calibration ────────────────────────────────────────────────────── */
    TnCalibrationResult calib;
    bool have_calib = false;

    if (args.calibrate) {
        /* User explicitly requested calibration */
        tn_calibrate(&calib, hw);
        tn_calibration_save(&calib);
        have_calib = true;
    } else {
        /* Try to load cached calibration */
        have_calib = tn_calibration_load(&calib, hw);
        if (!have_calib) {
            /* First run or hardware changed — auto-calibrate */
            printf("[Calibration] First run on this hardware — calibrating...\n");
            tn_calibrate(&calib, hw);
            tn_calibration_save(&calib);
            have_calib = true;
        }
    }

    /* Apply calibrated SIMD backend (if no CLI override) */
    if (have_calib && !args.simd_override && !getenv("TN_FORCE_BACKEND")) {
        const char *best = tn_calibration_best_simd(&calib);
        if (best) {
            setenv("TN_FORCE_BACKEND", best, 1);
            tn_simd_init();
            unsetenv("TN_FORCE_BACKEND");
            printf("SIMD: %s (calibrated optimal)\n", best);
        }
    }

    /* ── Classifier selection ───────────────────────────────────────────── */
    if (args.classifier_override == 3) {
        /* --classifier auto-fast: use calibration's fastest classifier */
        TnClassifierFormat fast_cls = have_calib
            ? tn_calibration_best_classifier(&calib)
            : (hw->cpu->avx512vnni || hw->cpu->avx_vnni || hw->cpu->arm_dotprod)
                ? TN_CLS_INT8 : TN_CLS_BF16;
        tn_hardware_profile_set_classifier(fast_cls);
        const char *names[] = {"BF16", "INT8", "INT4"};
        printf("Classifier: %s (auto-fast, calibrated)\n", names[fast_cls]);
    } else if (args.classifier_override >= 0 && args.classifier_override <= 2) {
        tn_hardware_profile_set_classifier((TnClassifierFormat)args.classifier_override);
        const char *names[] = {"BF16", "INT8", "INT4"};
        printf("Classifier: %s (user-selected)\n",
               names[args.classifier_override]);
    }
    /* else: default BF16 (from hardware_profile.c select_classifier) */

    tn_hardware_profile_report(hw);

    /* Thread count: CLI override > calibration > auto-detected */
    int active_threads;
    if (args.num_threads > 0) {
        active_threads = args.num_threads;
    } else if (have_calib && calib.best_threads > 0) {
        active_threads = calib.best_threads;
    } else {
        active_threads = hw->optimal_threads;
    }
    printf("Active: %d threads | %s\n", active_threads, hw->summary);

    /* Create Thread Pool */
    ThreadPool *tp = threadpool_create(active_threads);
    if (!tp) {
        fprintf(stderr, "Failed to create thread pool\n");
        return 1;
    }

    /* Phase 22.3: coarse model-load progress (TTY-only in-place updates;
     * degrades to plain one-line-per-stage output otherwise). Stages are
     * named milestones, not fine-grained byte progress — threading a byte
     * callback into the loader internals would be a loader-logic change,
     * out of scope for CLI polish. (stdout_is_tty was already computed
     * above, ahead of the startup banner.) */
    tn_progress_stage(1, 4, "Opening model file...", stdout_is_tty);

    /* Map Model File */
    MappedFile mf;
    if (mapped_file_open(&mf, args.model_path) != TN_OK) {
        fprintf(stderr, "Failed to map model file: %s\n", args.model_path);
        threadpool_destroy(tp);
        return 1;
    }

    /* Detect model format by reading the 4-byte magic number */
    uint32_t file_magic = 0;
    if (mf.size >= 4) memcpy(&file_magic, mf.data, 4);
    bool is_gguf = (file_magic == GGUF_MAGIC);

    tn_progress_stage(2, 4, "Loading weights...", stdout_is_tty);

    Config p;
    TransformerWeights w;
    memset(&w, 0, sizeof(w));   /* required before weights_alloc_pointers */
    MoEConfig mc;
    moe_config_init_dense(&mc); /* initialise to dense; overridden below for MoE models */
    GGUFHeader gguf_hdr;
    GGUFWeightStore *gguf_store = NULL;

    if (is_gguf) {
        printf("Model format: GGUF\n");

        if (gguf_read_header(&gguf_hdr, mf.data, mf.size) != TN_OK) {
            fprintf(stderr, "Failed to parse GGUF header.\n");
            gguf_header_free(&gguf_hdr);
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
        if (config_from_gguf(&p, &gguf_hdr) != TN_OK) {
            fprintf(stderr, "Failed to read config from GGUF metadata.\n");
            gguf_header_free(&gguf_hdr);
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
        config_print(&p);

        /* For DeepSeek-V2 (MLA+MoE) or Qwen3.5/3.6 (hybrid Gated-DeltaNet)
         * GGUF: populate MoEConfig ahead of weight loading. */
        if (strcmp(gguf_hdr.arch, "deepseek2") == 0 || strcmp(gguf_hdr.arch, "qwen35") == 0) {
            if (moe_config_from_gguf(&mc, &gguf_hdr) != TN_OK) {
                fprintf(stderr, "Failed to read MoE config from GGUF.\n");
                mapped_file_close(&mf);
                threadpool_destroy(tp);
                return 1;
            }
            moe_config_print(&mc);
        }

        if (weights_alloc_pointers(&w, &p) != TN_OK) {
            fprintf(stderr, "Failed to allocate weight pointers.\n");
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
        /* For MoE GGUF models (DeepSeek-V2), allocate per-layer/expert arrays. */
        if (mc.is_moe) {
            if (moe_weights_alloc(&w, &p, &mc) != TN_OK) {
                fprintf(stderr, "Failed to allocate MoE weight pointer arrays.\n");
                if (mc.is_moe) moe_weights_free(&w, &mc);
                weights_free_pointers(&w);
                mapped_file_close(&mf);
                threadpool_destroy(tp);
                return 1;
            }
        }
        if (weights_from_gguf(&w, &p, &gguf_hdr, &gguf_store) != TN_OK) {
            fprintf(stderr, "Failed to load GGUF weights.\n");
            if (mc.is_moe) moe_weights_free(&w, &mc);
            weights_free_pointers(&w);
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }

        /* Correct the profiler's per-token traffic + ceiling with the real
         * loaded model (2026-07-17): tn_hardware_profile_init() runs before
         * the model file is opened and seeds Data/token with compile-time
         * BitNet-2B constants (~1149 MB), overstating the ceiling ~6x for
         * multi-GB GGUF models (docs/ai/mistakes.md). Accounting per class:
         * Q2_0-native models subtract the embedding table (read one row per
         * token, not streamed) and swap the raw Q2_0 LM head for the
         * materialized classifier's bytes when one was explicitly requested.
         * Generic GGUF uses the raw file size — an upper bound that still
         * bills the full embedding table, since its tensor format (and thus
         * byte size) isn't known generically here; tight for tied-embedding
         * models, conservative otherwise. MoE models overcount (all experts,
         * not just routed) — TODO: subtract inactive-expert bytes via
         * MoEConfig. */
        {
            double per_tok = (double)mf.size;
            double cls_bytes = 0.0;
            const TnHardwareProfile *hp_m = tn_hardware_profile_get();
            if (w.q35_is_q2_0_model) {
                double q2_row = (double)p.vocab_size * p.dim * (34.0 / 128.0);
                cls_bytes = q2_row; /* zero-copy raw Q2_0 head (default) */
                if (hp_m && hp_m->classifier_explicit) {
                    switch (hp_m->classifier_fmt) {
                    case TN_CLS_INT4:
                        cls_bytes = (double)p.vocab_size * p.dim * 0.5; break;
                    case TN_CLS_INT8:
                        cls_bytes = (double)p.vocab_size * p.dim;       break;
                    default:
                        cls_bytes = (double)p.vocab_size * p.dim * 2.0; break;
                    }
                }
                /* drop embedding (one row/token) + in-file head, add the
                 * head actually used */
                per_tok -= 2.0 * q2_row;
                per_tok += cls_bytes;
            }
            tn_hardware_profile_set_model_bytes(per_tok, cls_bytes);
            if (hp_m) {
                printf("[profile] Data/token (loaded model): %.0f MB -> "
                       "ceiling %.1f tok/s at %.1f GB/s\n",
                       per_tok / (1024.0 * 1024.0),
                       hp_m->theoretical_ceiling, hp_m->measured_bw_gbps);
            }
        }
    } else {
        printf("Model format: native ternary\n");

        if (config_read(&p, mf.data, mf.size) != TN_OK) {
            fprintf(stderr, "Invalid model configuration.\n");
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
        config_print(&p);

        /* MoE model: scale_mode=2 → read 32-byte MoE header at offset 64,
         * weight data starts at byte 128 (64 header + 32 MoE + 32 padding).
         * Dense model: scale_mode=0 → weight data starts at byte 64. */
        tn_i8 *weight_data;
        size_t data_size;
        if (p.scale_mode == 2) {
            const size_t MOE_HDR_OFFSET = 64;
            const size_t WEIGHT_OFFSET  = 128;
            if (mf.size < WEIGHT_OFFSET) {
                fprintf(stderr, "MoE model file too small.\n");
                mapped_file_close(&mf);
                threadpool_destroy(tp);
                return 1;
            }
            size_t moe_size = mf.size - MOE_HDR_OFFSET;
            size_t moe_off  = 0;
            if (moe_config_read(&mc, (char *)mf.data + MOE_HDR_OFFSET, moe_size, &moe_off) != TN_OK) {
                fprintf(stderr, "Failed to read MoE config header.\n");
                mapped_file_close(&mf);
                threadpool_destroy(tp);
                return 1;
            }
            moe_config_print(&mc);
            weight_data = (tn_i8 *)mf.data + WEIGHT_OFFSET;
            data_size   = mf.size - WEIGHT_OFFSET;
        } else {
            weight_data = (tn_i8 *)mf.data + 64;
            data_size   = mf.size - 64;
        }

        if (weights_alloc_pointers(&w, &p) != TN_OK) {
            fprintf(stderr, "Failed to allocate weight pointers.\n");
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }

        if (mc.is_moe) {
            if (moe_weights_alloc(&w, &p, &mc) != TN_OK) {
                fprintf(stderr, "Failed to allocate MoE weight pointers.\n");
                if (mc.is_moe) moe_weights_free(&w, &mc);
                weights_free_pointers(&w);
                mapped_file_close(&mf);
                threadpool_destroy(tp);
                return 1;
            }
        }

        if (weights_map(&w, &p, &mc, weight_data, data_size) != TN_OK) {
            fprintf(stderr, "Failed to map weight structures from file.\n");
            moe_weights_free(&w, &mc);
            weights_free_pointers(&w);
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }

        /* Native-format twin of the GGUF set_model_bytes call above
         * (2026-07-17, independent-review finding: without this, any native
         * model that isn't exactly BitNet-2B kept the hardcoded pre-load
         * estimate forever). Per-token traffic = ternary layers + norms
         * (weight data minus the BF16 embedding table, which is read one
         * row per token) + the classifier at its selected precision (the
         * classifier is derived from that embedding: BF16 = full table,
         * INT8/INT4 = half/quarter). Native MoE (scale_mode==2) overcounts
         * inactive experts here — same TODO as the GGUF branch. */
        {
            double embed_bytes = (double)p.vocab_size * p.dim * 2.0;
            const TnHardwareProfile *hp_n = tn_hardware_profile_get();
            double cls_bytes = embed_bytes; /* BF16 default */
            if (hp_n && hp_n->classifier_fmt == TN_CLS_INT8) cls_bytes = embed_bytes / 2.0;
            if (hp_n && hp_n->classifier_fmt == TN_CLS_INT4) cls_bytes = embed_bytes / 4.0;
            double per_tok = ((double)data_size - embed_bytes) + cls_bytes;
            tn_hardware_profile_set_model_bytes(per_tok, cls_bytes);
            if (hp_n && per_tok > 0.0) {
                printf("[profile] Data/token (loaded model): %.0f MB -> "
                       "ceiling %.1f tok/s at %.1f GB/s\n",
                       per_tok / (1024.0 * 1024.0),
                       hp_n->theoretical_ceiling, hp_n->measured_bw_gbps);
            }
        }
    }

    tn_progress_stage(3, 4, "Preparing runtime...", stdout_is_tty);

    /* KV Strategy — measured AFTER model weights are loaded so that any F32-dequantised
     * weight allocations (MLA projections, shared experts, norms) are already counted in
     * consumed RAM.  For mmap'd GGUF models the raw quantised expert bytes don't use
     * physical RAM until accessed, so only the upfront malloc'd F32 blocks matter here.
     * Re-measuring at this point prevents DeepSeek-style OOM where a 163840-token context
     * would require 7+ GB KV cache on a machine that only has 2–3 GB left after load. */
    {
        tn_i64 post_load_ram = tn_get_free_ram();
        KVStrategyResult kv_res = select_kv_strategy(&p, post_load_ram);
        p.seq_len = kv_res.max_seq_len;
        /* Qwen35 hybrid models keep their own F32 K/V caches
         * (q35_key_cache/q35_value_cache, only the full-attention layers) —
         * the quantized-KV strategy machinery is not wired into that path,
         * so printing e.g. "Quantized I8" for them misreported what actually
         * happens (2026-07-17, found during the ceiling-gap attribution;
         * see docs/ai/mistakes.md). The RAM-aware max-context clamp from
         * select_kv_strategy() still applies either way. */
        if (mc.has_linear_attn) {
            printf("KV Strategy: F32 (Qwen35 hybrid path; quantized-KV "
                   "strategy not wired in), max context: %d tokens\n",
                   p.seq_len);
        } else {
            printf("KV Strategy: %s, max context: %d tokens\n",
                   kv_strategy_name(kv_res.strategy), p.seq_len);
        }
    }

    /* Setup RunState */
    RunState *s = (RunState *)malloc(sizeof(RunState));
    if (!s) {
        fprintf(stderr, "Failed to allocate RunState header.\n");
        if (mc.is_moe) moe_weights_free(&w, &mc);
        weights_free_pointers(&w);
        mapped_file_close(&mf);
        threadpool_destroy(tp);
        return 1;
    }

    /* Qwen3.5/3.6 hybrid models never read the generic key_cache/value_cache
     * (they use their own correctly-sized q35_key_cache/q35_value_cache,
     * allocated below) — skip_kv_cache=true avoids a multi-GB calloc+memset
     * for those models' native (often huge, e.g. 262144) context. See
     * run_state_alloc_ex's header comment and docs/ai/mistakes.md. */
    if (run_state_alloc_ex(s, &p, p.seq_len, mc.has_linear_attn) != TN_OK) {
        fprintf(stderr, "Failed to allocate RunState buffers.\n");
        free(s);
        if (mc.is_moe) moe_weights_free(&w, &mc);
        weights_free_pointers(&w);
        mapped_file_close(&mf);
        threadpool_destroy(tp);
        return 1;
    }

    /* Phase 17.7: MLA k_rope_cache (allocated only when has_mla=1) */
    if (mc.has_mla) {
        if (mla_run_state_alloc(s, &p, &mc, p.seq_len) != TN_OK) {
            fprintf(stderr, "Failed to allocate MLA k_rope_cache.\n");
            run_state_free(s);
            free(s);
            if (mc.is_moe) moe_weights_free(&w, &mc);
            weights_free_pointers(&w);
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
    }

    /* Qwen3.5/3.6 hybrid attention state (allocated only when has_linear_attn=1) */
    if (mc.has_linear_attn) {
        if (q35_run_state_alloc(s, &p, &mc, p.seq_len) != TN_OK) {
            fprintf(stderr, "Failed to allocate Qwen3.5/3.6 hybrid attention state.\n");
            run_state_free(s);
            free(s);
            if (mc.is_moe) moe_weights_free(&w, &mc);
            weights_free_pointers(&w);
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
    }

    /* Tokenizer */
    Tokenizer t;
    memset(&t, 0, sizeof(t));
    if (args.tokenizer_path) {
        if (tokenizer_load(&t, args.tokenizer_path) != TN_OK) {
            fprintf(stderr, "Failed to load tokenizer: %s\n", args.tokenizer_path);
            run_state_free(s);
            free(s);
            if (mc.is_moe) moe_weights_free(&w, &mc);
            weights_free_pointers(&w);
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
        /* When the model is GGUF, pull chat_template and special token IDs
         * from the GGUF metadata if the .bin tokenizer did not supply them.
         * This keeps the tokenizer modular: external .bin files carry vocab/BPE
         * while the model file is the authoritative source for chat format. */
        if (is_gguf) {
            if (!t.chat_template) {
                const GGUFMeta *m = gguf_meta_find(&gguf_hdr, "tokenizer.chat_template");
                if (m && m->val_type == GGUF_VAL_STRING && m->val.string.len > 0) {
                    t.chat_template = (char *)malloc(m->val.string.len + 1);
                    if (t.chat_template) {
                        memcpy(t.chat_template, m->val.string.str, m->val.string.len);
                        t.chat_template[m->val.string.len] = '\0';
                        fprintf(stderr, "[tokenizer] chat_template patched from GGUF metadata\n");
                    }
                }
            }
            /* GGUF metadata is authoritative for BOS/EOS when the .bin vocab
             * scan did not find a match (eos_token_id < 0 means no candidate
             * from the BOS_CANDIDATES / EOS_PRIMARY lists was present in this
             * model's vocabulary, e.g. DeepSeek uses <｜begin▁of▁sentence｜>
             * which is not in the generic candidate lists). */
            if (t.bos_token_id < 0)
                t.bos_token_id = (int)gguf_meta_u32(&gguf_hdr, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
            if (t.eos_token_id < 0) {
                t.eos_token_id = (int)gguf_meta_u32(&gguf_hdr, "tokenizer.ggml.eos_token_id", (uint32_t)-1);
                /* Ensure eos_list is populated so agent_loop.c EOS detection
                 * also fires — agent_loop only iterates eos_list, not eos_token_id */
                if (t.eos_token_id >= 0 && t.n_eos < 8) {
                    int already = 0;
                    for (int _i = 0; _i < t.n_eos; _i++)
                        if (t.eos_list[_i] == t.eos_token_id) { already = 1; break; }
                    if (!already)
                        t.eos_list[t.n_eos++] = t.eos_token_id;
                }
            }
        }
    } else if (is_gguf) {
        /* Phase 34.5: Auto-load tokenizer from GGUF when no external file given */
        TernaryError terr = tokenizer_load_from_gguf(&t, &gguf_hdr);
        if (terr != TN_OK) {
            fprintf(stderr, "[WARN] GGUF tokenizer load failed (err=%d). "
                    "Run with --tokenizer <path> to specify one.\n", (int)terr);
            /* Non-fatal: engine can still run if tokens passed externally */
        }
    }

    tn_progress_stage(4, 4, "Ready.", stdout_is_tty);
    tn_progress_done(stdout_is_tty);

    /* ── Phase 15: RAG initialisation ────────────────────────────────────── */
    RagContext rag;
    memset(&rag, 0, sizeof(rag));
    int rag_ok = 0;

    if (args.memory_db_path) {
        TernaryError err = vector_db_open(&rag.db, args.memory_db_path, p.dim);
        if (err != TN_OK) {
            fprintf(stderr, "[RAG] Warning: could not open vector DB '%s' (err=%d). "
                    "Memory features disabled.\n", args.memory_db_path, (int)err);
        } else {
            err = embedder_init(&rag.emb, &p);
            if (err != TN_OK) {
                fprintf(stderr, "[RAG] Warning: could not initialise embedder (err=%d). "
                        "Memory features disabled.\n", (int)err);
                vector_db_close(&rag.db);
            } else {
                rag.enabled = 1;
                rag_ok = 1;
                printf("[RAG] Memory enabled — %d entries in '%s'\n",
                       rag.db.num_entries, args.memory_db_path);
            }
        }
    }
    /* ─────────────────────────────────────────────────────────────────────── */

    /* ── Phase 34: Vision pipeline (extracted to multimodal/vision_pipeline.c
     * in Phase 22.2 so the HTTP API can reuse it for image_url uploads) ──── */
    static char vis_prompt_buf[8192];
    if (args.image_path) {
        const char *stripped_prompt = args.prompt;
        vision_pipeline_run(args.image_path, args.vision_path, args.proj_path,
                            &p, &w, s, &mc, &t, tp,
                            args.prompt, vis_prompt_buf, sizeof(vis_prompt_buf),
                            &stripped_prompt);
        args.prompt = (char *)stripped_prompt;
    }

    /* Execution Mode */
    if (args.server_mode) {
        /* Phase 21: Start OpenAI-compatible API server */
        ApiContext api_ctx;
        api_context_init(&api_ctx, &p, &w, s, &mc, &t, tp);

        /* Phase 22: API hardening knobs, off by default */
        api_ctx.server_config.cors.enabled = args.cors_enabled;
        for (int ci = 0; ci < args.num_cors_origins; ci++) {
            server_config_add_cors_origin(&api_ctx.server_config, args.cors_origins[ci]);
        }
        if (args.api_key) api_ctx.server_config.auth.api_key = strdup(args.api_key);
        api_ctx.server_config.metrics.enabled = args.metrics_enabled;
        api_ctx.server_config.web_ui = args.web_ui_mode;
        if (args.static_dir) api_ctx.server_config.static_dir = strdup(args.static_dir);
        if (args.vision_path && args.proj_path) {
            api_ctx.server_config.vision_path = strdup(args.vision_path);
            api_ctx.server_config.proj_path   = strdup(args.proj_path);
        }

        TernaryError api_err = api_server_start(args.server_port, &api_ctx);
        if (api_err != TN_OK) {
            fprintf(stderr, "Error: Failed to start API server on port %d: %s\n",
                    args.server_port, tn_error_str(api_err));
        } else {
            printf("Press Ctrl+C to stop.\n");

            /* Graceful shutdown: previously a bare pause() with no signal
             * handler installed meant SIGINT/SIGTERM used the default
             * disposition (terminate immediately) — pause() never actually
             * returns under default disposition, so api_server_stop() and
             * all of main()'s later cleanup (tokenizer_free, gguf_header_free,
             * mapped_file_close, etc.) were unreachable dead code in server
             * mode. Installing a handler makes the signal "caught" instead
             * of fatal, which is what makes pause() return. */
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = handle_shutdown_signal;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = 0;
            sigaction(SIGINT, &sa, NULL);
            sigaction(SIGTERM, &sa, NULL);

            /* Block main thread — listener runs in background thread.
             * Loop (not a single pause()) so a signal that arrived between
             * installing the handler and the first pause() call — setting
             * the flag before we ever wait — is still honored immediately
             * rather than requiring a second signal. */
            while (!g_shutdown_requested) {
                pause();
            }
            printf("\nShutting down...\n");
            api_server_stop(&api_ctx);
        }
    } else if (args.prompt) {
        printf("\n");
        generate(&p, &w, s, &mc, &t, tp, args.prompt, args.max_tokens, args.temperature, args.top_p);
        printf("\n");
    } else {
        run_repl(&p, &w, &mc, NULL, NULL, NULL, s, &t, tp, &args, rag_ok ? &rag : NULL);
    }

    /* Cleanup */
    if (rag_ok) {
        embedder_free(&rag.emb);
        vector_db_close(&rag.db);
    }
    /* tokenizer_free is safe unconditionally: t was memset to zero above,
     * and both load paths (--tokenizer file, or GGUF auto-load) populate
     * the same fields it frees. Previously gated on args.tokenizer_path,
     * which skipped cleanup entirely for the GGUF-auto-load path — the
     * common case when no external --tokenizer is passed — leaking the
     * whole vocab (~49k strings) every run. */
    tokenizer_free(&t);
    /* mla_run_state_free/q35_run_state_free must run before run_state_free()
     * (they free k_rope_cache/q35_* pointer arrays that run_state_free()
     * doesn't know about). mla_run_state_free was previously never called
     * anywhere in the shutdown path — a real leak of k_rope_cache (n_layers
     * malloc'd buffers) + mla_rope_freq on every DeepSeek/MLA run, only its
     * own OOM-cleanup path called it. Fixed here alongside the new
     * q35_run_state_free wiring (see docs/ai/mistakes.md). */
    if (mc.has_mla) mla_run_state_free(s, p.n_layers);
    if (mc.has_linear_attn) q35_run_state_free(s, &p, &mc);
    run_state_free(s);
    free(s);
    if (mc.is_moe) moe_weights_free(&w, &mc);
    weights_free_pointers(&w);
    if (gguf_store) weights_free_gguf(gguf_store);
    if (is_gguf) gguf_header_free(&gguf_hdr);
    mapped_file_close(&mf);
    threadpool_destroy(tp);

    return 0;
}
