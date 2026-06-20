#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* pause() for --server mode */
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
#include "transformer/generate.h"
#include "transformer/forward.h"
#include "math/simd_dispatch.h"

/* Phase 15 RAG */
#include "rag/rag_context.h"
#include "rag/vector_db.h"
#include "rag/embedder.h"

/* Phase 34 Multimodal */
#include "multimodal/image_load.h"
#include "multimodal/patch_extract.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "multimodal/vision_bridge.h"
#include "multimodal/vision_weights_load.h"
#include "memory/aligned_alloc.h"

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

    printf("Project Zero Engine %s — Auto-Tuned Hardware\n", PZ_VERSION_STR);

    /* ── SIMD backend selection ─────────────────────────────────────────── */
    /* CLI --simd override takes precedence over calibration and auto-detect.
     * Order: --simd flag > TN_FORCE_BACKEND env > calibration cache > auto */
    if (args.simd_override) {
        setenv("TN_FORCE_BACKEND", args.simd_override, 1);
        printf("SIMD override: %s (user-selected)\n", args.simd_override);
    }

    const char *simd_backend = tn_simd_init();

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
    tn_i64 free_ram = (tn_i64)hw->free_ram_bytes;
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
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
        if (config_from_gguf(&p, &gguf_hdr) != TN_OK) {
            fprintf(stderr, "Failed to read config from GGUF metadata.\n");
            mapped_file_close(&mf);
            threadpool_destroy(tp);
            return 1;
        }
        config_print(&p);

        /* For DeepSeek-V2 GGUF: populate MoEConfig and allocate MoE weight arrays. */
        if (strcmp(gguf_hdr.arch, "deepseek2") == 0) {
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
            if (moe_config_read(&mc, mf.data + MOE_HDR_OFFSET, moe_size, &moe_off) != TN_OK) {
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
    }

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
        printf("KV Strategy: %s, max context: %d tokens\n",
               kv_strategy_name(kv_res.strategy), p.seq_len);
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

    if (run_state_alloc(s, &p, p.seq_len) != TN_OK) {
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

    /* ── Phase 34: Vision pipeline ──────────────────────────────────────── */
    if (args.image_path) {
        if (!args.vision_path || !args.proj_path) {
            fprintf(stderr,
                "[vision] --image requires --vision <vision.bin> and --proj <projector.bin>\n"
                "[vision] Extract weights first:\n"
                "[vision]   python tools/extract_multimodal.py --repo moondream-hf/moondream2 --out models/\n");
        } else {
            printf("\nVision pipeline: loading %s\n", args.image_path);

            /* Load vision model weights */
            VisionModel vm;
            memset(&vm, 0, sizeof(vm));
            int vm_ok = 0;

            if (vision_model_load_encoder(&vm, args.vision_path) == TN_OK &&
                vision_model_load_projector(&vm, args.proj_path) == TN_OK) {
                vision_model_print_info(&vm);
                vm_ok = 1;
            } else {
                fprintf(stderr, "[vision] failed to load vision weights\n");
            }

            if (vm_ok) {
                /* Load and normalize image */
                float *pixels = NULL;
                int img_w = 0, img_h = 0;
                int target_res = (int)(0.5f + __builtin_sqrtf((float)vm.cfg.num_patches))
                                 * (int)(0.5f + __builtin_sqrtf((float)(vm.cfg.patch_dim / 3)));
                if (target_res <= 0) target_res = 384;

                if (load_image(args.image_path, &pixels, &img_w, &img_h, target_res) != TN_OK) {
                    fprintf(stderr, "[vision] failed to load image: %s\n", args.image_path);
                    vm_ok = 0;
                } else {
                    printf("  Image loaded: %dx%d (SigLIP-normalized)\n", img_w, img_h);
                }

                if (vm_ok) {
                    /* Extract patches */
                    int patch_size = (int)(0.5f + __builtin_sqrtf((float)(vm.cfg.patch_dim / 3)));
                    int max_patches = vm.cfg.num_patches;
                    float *patches = (float *)tn_aligned_alloc(
                        (size_t)max_patches * vm.cfg.patch_dim * sizeof(float), 64);
                    int num_patches = 0;

                    if (!patches) {
                        fprintf(stderr, "[vision] OOM allocating patches\n");
                        vm_ok = 0;
                    } else {
                        extract_patches(pixels, patches, target_res, patch_size, &num_patches);
                        printf("  Extracted %d patches (patch_size=%d)\n", num_patches, patch_size);
                        vm.cfg.num_patches = num_patches;
                    }

                    if (vm_ok) {
                        /* Vision encoder forward */
                        float *vis_emb = (float *)tn_aligned_alloc(
                            (size_t)num_patches * vm.cfg.embed_dim * sizeof(float), 64);
                        if (!vis_emb) {
                            fprintf(stderr, "[vision] OOM: vision embeddings\n");
                            vm_ok = 0;
                        } else {
                            printf("  Running vision encoder (%d layers)...\n", vm.cfg.n_layers);
                            vision_encoder_forward(vis_emb, patches, &vm.cfg, &vm.weights, tp);
                            printf("  Encoder done.\n");

                            /* MLP projector (or pixel-shuffle + single linear) */
                            int out_tokens = vision_projector_output_tokens(&vm.proj, num_patches);
                            float *projected = (float *)tn_aligned_alloc(
                                (size_t)out_tokens * p.dim * sizeof(float), 64);

                            /* Projector llm_dim must match LLM dim; warn if mismatched */
                            if (vm.proj.llm_dim != p.dim) {
                                fprintf(stderr,
                                    "[vision] WARNING: projector llm_dim=%d != LLM dim=%d\n"
                                    "[vision] Embeddings will be truncated/zero-padded.\n",
                                    vm.proj.llm_dim, p.dim);
                            }

                            if (!projected) {
                                fprintf(stderr, "[vision] OOM: projected embeddings\n");
                            } else {
                                /* Use projector's own llm_dim for the projection */
                                float *proj_buf = (float *)tn_aligned_alloc(
                                    (size_t)out_tokens * vm.proj.llm_dim * sizeof(float), 64);
                                if (proj_buf) {
                                    vision_projector_forward_batch(proj_buf, vis_emb,
                                                                   num_patches, &vm.proj, tp);
                                    /* Copy into projected, matching LLM dim */
                                    int copy_dim = vm.proj.llm_dim < p.dim ? vm.proj.llm_dim : p.dim;
                                    memset(projected, 0, (size_t)out_tokens * p.dim * sizeof(float));
                                    for (int pp = 0; pp < out_tokens; pp++) {
                                        memcpy(&projected[pp * p.dim],
                                               &proj_buf[pp * vm.proj.llm_dim],
                                               (size_t)copy_dim * sizeof(float));
                                    }
                                    tn_aligned_free(proj_buf);
                                    printf("  Projector done: %d tokens × %d-dim  (scale_factor=%d)\n",
                                           out_tokens, p.dim, vm.proj.scale_factor);

                                    /* Inject into KV cache.
                                     * For SmolVLM-style models, pre-process a chat prefix
                                     * so visual tokens land at the correct sequence position
                                     * (after "User: "), matching the training layout. */
                                    /* Inject into KV cache.
                                     * For ChatML-style models (chat_template contains "im_start"),
                                     * pre-process a chat prefix so visual tokens land at the correct
                                     * sequence position (after "User: "), matching training layout.
                                     * Detected from the model's own chat_template — no vocab_size
                                     * heuristics. */
                                    if (args.prompt && t.chat_template &&
                                        strstr(t.chat_template, "im_start")) {
                                        /* Build prefix from the actual token strings in this model's vocab */
                                        int im_start_id = tokenizer_find_id(&t, "<|im_start|>");
                                        const char *vis_prefix = (im_start_id >= 0)
                                            ? "<|im_start|>User: "
                                            : "User: ";
                                        int prefix_toks[32];
                                        int n_pre = tokenizer_encode(&t, vis_prefix, strlen(vis_prefix),
                                                                     prefix_toks, 32);
                                        if (n_pre > 0) {
                                            for (int pi = 0; pi < n_pre; pi++) {
                                                transformer_forward(prefix_toks[pi], s->current_pos,
                                                                    &p, &w, s, &mc, tp);
                                                s->current_pos++;
                                            }
                                            /* Strip the prefix from args.prompt for generate() */
                                            const char *stripped = args.prompt;
                                            size_t pfx_len = strlen(vis_prefix);
                                            if (strncmp(stripped, vis_prefix, pfx_len) == 0)
                                                stripped += pfx_len;
                                            /* Store stripped pointer — use a static for simplicity */
                                            static char vis_prompt_buf[8192];
                                            snprintf(vis_prompt_buf, sizeof(vis_prompt_buf), "%s", stripped);
                                            args.prompt = vis_prompt_buf;
                                        }
                                    }

                                    VisionContext vctx;
                                    vctx.patch_embeddings = projected;
                                    vctx.num_patches      = out_tokens;
                                    vctx.embed_dim        = p.dim;
                                    inject_vision_into_kv_cache(s, &p, &w, &vctx, tp);
                                    printf("  Vision context injected (%d KV tokens)\n\n",
                                           out_tokens);
                                }
                                tn_aligned_free(projected);
                            }
                            tn_aligned_free(vis_emb);
                        }
                        tn_aligned_free(patches);
                    }
                    tn_aligned_free(pixels);
                }
            }
            vision_model_free(&vm);
        }
    }

    /* Execution Mode */
    if (args.server_mode) {
        /* Phase 21: Start OpenAI-compatible API server */
        ApiContext api_ctx;
        api_context_init(&api_ctx, &p, &w, s, &mc, &t, tp);
        TernaryError api_err = api_server_start(args.server_port, &api_ctx);
        if (api_err != TN_OK) {
            fprintf(stderr, "Error: Failed to start API server on port %d: %s\n",
                    args.server_port, tn_error_str(api_err));
        } else {
            printf("Press Ctrl+C to stop.\n");
            /* Block main thread — listener runs in background thread */
            pause();
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
    if (args.tokenizer_path) {
        tokenizer_free(&t);
    }
    run_state_free(s);
    free(s);
    if (mc.is_moe) moe_weights_free(&w, &mc);
    weights_free_pointers(&w);
    if (gguf_store) weights_free_gguf(gguf_store);
    mapped_file_close(&mf);
    threadpool_destroy(tp);

    return 0;
}
