# GitNexus Codebase Intelligence Index — project-zero

> **Master index for AI agents.** Read this to understand the entire codebase before making any change.
> All information is derived from actual source code, not inferred.

---

## Project Overview

**project-zero** is a high-performance, portable LLM inference engine written in C (+ one C++ file for
Jinja2 chat templates). It supports BitNet ternary models, GGUF float models (F16, Q4_K, Q5_K, Q6_K,
Q8_0, F32), and DeepSeek-V2-style MoE/MLA models. The primary binary is `adaptive_ai_engine`.

**Core design principles:**
- Zero external dependencies for core inference (no BLAS, no ONNX, no Python)
- Runtime SIMD dispatch: one binary runs on scalar/AVX2/AVX-512/VNNI/ARM-NEON/dotprod
- Data-driven weight dispatch: no model-name hardcoding — dispatch from type flags
- Caller-participates thread pool: N-1 OS workers + calling thread = N HW slots
- mmap-based weight loading: zero-copy for F16/Q4_K tensors from GGUF

---

## Source Module Index

### 📁 `src/cli/` — Command-Line Interface
| File | Description |
|------|-------------|
| `src/cli/main.c` | Entry point: parses CLI, loads model, dispatches to generate/REPL/server/agent |
| `src/cli/args.c` | Argument parser for `--model`, `--prompt`, `--threads`, `--simd`, `--classifier`, etc. |
| `src/cli/repl.c` | Interactive multi-turn REPL loop |
| `src/cli/timer.c` | Wall-clock microsecond timer (`timer_now_us`, `timer_tokens_per_sec`) |

### 📁 `src/core/` — Model Loading & Config
| File | Description |
|------|-------------|
| `src/core/gguf_loader.c` | GGUF→`TransformerWeights` mapping; `config_from_gguf`, `weights_from_gguf` |
| `src/core/gguf_reader.c` | Raw GGUF binary parser: reads header, metadata KV pairs, tensor index |
| `src/core/gguf_quant.c` | GGUF dequantization: Q4_K, Q4_0, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0 → F32 |
| `src/core/config.c` | `config_read` (native .bin), `config_print` |
| `src/core/weights.c` | `weights_alloc_pointers`, `weights_map` (native ternary), `weights_build_classifier_quant` |
| `src/core/moe_config.c` | `moe_config_read`, `moe_config_init_dense`, `moe_config_print` |
| `src/core/moe_weights.c` | `moe_weights_alloc`, `moe_weights_map`, `moe_weights_free` for expert pointer arrays |
| `src/core/run_state.c` | `run_state_alloc`/`run_state_free` — allocates all scratch/KV cache buffers |
| `src/core/mla_run_state.c` | `mla_run_state_alloc`/`mla_run_state_free` — per-layer k_rope_cache for MLA |
| `src/core/calibration.c` | Hardware benchmarking and optimal thread/SIMD selection; result cached at `~/.project-zero/calibration.bin` |
| `src/core/hardware_profile.c` | CPU feature detection, RAM probing, classifier format selection |
| `src/core/unpack.c` | Scalar ternary unpack (2-bit packed → int8) |
| `src/core/unpack_avx2.c` | AVX2 ternary unpack |
| `src/core/debug.c` | `DBG_DUMP` macro support, `g_tn_verbose`, `g_dump_fp` |
| `src/core/error.c` | TernaryError codes and string table |
| `src/core/step_timing.c` | Optional per-step latency profiler (TN_STEP_TIMING=1) |

### 📁 `src/math/` — SIMD Math Kernels
| File | Description |
|------|-------------|
| `src/math/simd_dispatch.c` | 🔥 SIMD init + global dispatch table (`tn_simd_init`); routes to best kernel |
| `src/math/cpu_features.c` | CPUID probe; fills `TnCpuFeatures` (avx2, avx512f, avx512vnni, avx_vnni, arm_dotprod…) |
| `src/math/matmul_f16.c` | 🔥 `parallel_matmul_f16` — F16 zero-copy matmul (AVX-512/AVX2/scalar) |
| `src/math/parallel_matmul.c` | 🔥 `parallel_ternary_matmul_packed` + pre-quantisation dispatch + BF16/INT8/INT4 variants |
| `src/math/ternary_matmul_packed.c` | Scalar ternary packed matmul (reference) |
| `src/math/ternary_matmul_packed_avx2.c` | AVX2 ternary packed matmul (8-wide FP32 FMA) |
| `src/math/ternary_matmul_packed_avx512.c` | AVX-512F ternary matmul (16-wide FP32 FMA) |
| `src/math/ternary_matmul_packed_vnni.c` | AVX-512 VNNI ternary matmul (64 int8 MACs/cycle) |
| `src/math/ternary_matmul_packed_vnni256.c` | VNNI-256 variant (256-bit EVEX, no ZMM throttle) |
| `src/math/ternary_matmul_packed_avx_vnni.c` | AVX-VNNI 256-bit (Alder Lake/Zen3, no AVX-512) |
| `src/math/ternary_matmul_packed_dotprod.c` | ARM dotprod (SDOT, 16 int8 MACs/cycle) |
| `src/math/ternary_matmul_avx2.c` | AVX2 unpacked ternary matmul |
| `src/math/ternary_matmul_scalar.c` | Scalar ternary matmul reference |
| `src/math/rmsnorm.c` | Scalar RMSNorm |
| `src/math/rmsnorm_avx2.c` | AVX2 RMSNorm |
| `src/math/rmsnorm_avx512.c` | AVX-512F RMSNorm |
| `src/math/softmax.c` | Scalar softmax (online safe) |
| `src/math/softmax_avx2.c` | AVX2 softmax |
| `src/math/softmax_avx512.c` | AVX-512F softmax |
| `src/math/elementwise.c` | Scalar: vec_add, vec_mul, vec_scale, silu, relu2, vec_dot, vec_saxpy |
| `src/math/elementwise_avx2.c` | AVX2 versions of all elementwise ops |
| `src/math/elementwise_avx512.c` | AVX-512F versions of all elementwise ops |
| `src/math/rope.c` | `apply_rope` with full YaRN support for standard attention |
| `src/math/matmul_q4k.c` | Zero-copy Q4_K matmul kernel (used in MLA attention) |
| `src/math/matmul_q5_0.c` | Q5_0 matmul kernel |
| `src/math/matmul_q5_1.c` | Q5_1 matmul kernel |
| `src/math/matmul_q5k.c` | Q5_K matmul kernel |
| `src/math/quantize_i8.c` | `quantize_row_to_i8_avx512`, `sum_i8_avx512` for pre-quantisation |

### 📁 `src/transformer/` — Transformer Core
| File | Description |
|------|-------------|
| `src/transformer/forward.c` | 🔥 `transformer_forward` — embed → N layers (attn+ffn) → RMSNorm → lm_head → logits |
| `src/transformer/generate.c` | 🔥 `generate` / `generate_with_callback` — token loop with sampling |
| `src/transformer/attention.c` | 🔥 `attention_forward` — GQA+RoPE+KV cache (routes to MLA when `mc->has_mla`) |
| `src/transformer/mla_attention.c` | 🔥 `mla_attention_forward` — DeepSeek MLA: KV compress → expand → Q → RoPE → attend |
| `src/transformer/ffn.c` | `ffn_forward` — SwiGLU/ReGLU dense FFN (routes to MoE when `moe_layer_is_moe`) |
| `src/transformer/moe_ffn.c` | `moe_ffn_forward` — router → top-k experts → accumulate; expert hit tracking |
| `src/transformer/moe_router.c` | `moe_router_forward` — gate matmul → softmax → top-k selection |
| `src/transformer/embedding.c` | `embed_token` — BF16/F32 token embedding table lookup |

### 📁 `src/kv_cache/` — KV Cache Management
| File | Description |
|------|-------------|
| `src/kv_cache/kv_strategy.c` | `kv_strategy_select` — picks FULL/SLIDING_WINDOW/COMPRESS based on RAM |
| `src/kv_cache/kv_compress.c` | KV cache compression (token eviction) |
| `src/kv_cache/sliding_window.c` | `sw_init`, `sw_map_position`, `sw_advance`, `sw_valid_count` |

### 📁 `src/threading/` — Thread Pool
| File | Description |
|------|-------------|
| `src/threading/thread_pool.c` | Caller-participates thread pool: N-1 OS threads + caller = N HW slots |
| `src/threading/cpu_probe.c` | Physical core count detection from `/proc/cpuinfo` |

### 📁 `src/tokenizer/` — Tokenizer
| File | Description |
|------|-------------|
| `src/tokenizer/tokenizer_load.c` | Load vocab from native .bin file |
| `src/tokenizer/tokenizer_encode.c` | BPE encoding: `tokenizer_encode` |
| `src/tokenizer/tokenizer_decode.c` | `tokenizer_decode` — token ID → UTF-8 piece |
| `src/tokenizer/tokenizer_gguf.c` | `tokenizer_from_gguf` — extract BPE vocab from GGUF |
| `src/tokenizer/chat_template.cpp` | Jinja2 chat template renderer (`chat_template_apply`) |

### 📁 `src/api/` — OpenAI-Compatible API Server
| File | Description |
|------|-------------|
| `src/api/http_server.c` | Embedded HTTP server: listens on port, dispatches to handlers |
| `src/api/chat_compile.c` | Compile `messages[]` array → single prompt string |
| `src/api/json_parse.c` | Minimal JSON parser for OpenAI-format requests |
| `src/api/sse_stream.c` | Server-Sent Events (SSE) streaming for token-by-token output |

### 📁 `src/agent/` — Agentic Loop
| File | Description |
|------|-------------|
| `src/agent/agent_loop.c` | `run_agent_loop` — generation loop with tool call interception |
| `src/agent/tool_interceptor.c` | Detect `<tool_call>` tags in generated text |
| `src/agent/cmd_exec.c` | Execute shell commands from tool calls |
| `src/agent/output_inject.c` | Inject tool output back into context |
| `src/agent/user_approval.c` | Interactive approval prompt before executing commands |

### 📁 `src/rag/` — RAG Memory
| File | Description |
|------|-------------|
| `src/rag/embedder.c` | `embedder_generate` — mean-pool forward pass → L2-normalised embedding |
| `src/rag/vector_db.c` | Append-only VRDB file: header + (embedding, text) records |
| `src/rag/similarity.c` | Cosine similarity search over VectorDB embeddings |
| `src/rag/memory_search.c` | Top-k memory retrieval |
| `src/rag/rag_context.c` | `RagContext` — ties embedder + DB + search together |
| `src/rag/auto_save.c` | Auto-detect and save memorable agent outputs |
| `src/rag/auto_retrieve.c` | Auto-retrieve relevant memories at prompt build time |

### 📁 `src/multimodal/` — Vision
| File | Description |
|------|-------------|
| `src/multimodal/image_load.c` | Load image via stb_image |
| `src/multimodal/patch_extract.c` | Extract fixed-size image patches |
| `src/multimodal/vision_encoder.c` | ViT-style vision encoder forward pass |
| `src/multimodal/vision_projector.c` | Project vision features into LLM embedding space |
| `src/multimodal/vision_bridge.c` | Bridge: inject vision tokens into RunState before generation |
| `src/multimodal/vision_weights_load.c` | Load vision model weights from GGUF |

### 📁 `src/sampling/` — Sampling Strategies
| File | Description |
|------|-------------|
| `src/sampling/argmax.c` | `sample_argmax` — greedy decoding |
| `src/sampling/temperature.c` | `apply_temperature` — logit scaling |
| `src/sampling/top_k.c` | Top-K filtering |
| `src/sampling/top_p.c` | `sample_top_p` — nucleus sampling |
| `src/sampling/rng.c` | `rng_seed`, `rng_float` — xorshift64 RNG |

### 📁 `src/memory/` — Low-level Memory
| File | Description |
|------|-------------|
| `src/memory/aligned_alloc.c` | 64-byte aligned malloc/free (AVX-512 safe) |
| `src/memory/mapped_file.c` | `mapped_file_open`/`mapped_file_close` — mmap wrapper |

### 📁 `src/reasoning/` — Reasoning Mode
| File | Description |
|------|-------------|
| `src/reasoning/reasoning_generate.c` | Chain-of-thought aware generation loop |
| `src/reasoning/prompt_inject.c` | Inject reasoning prompts |
| `src/reasoning/thought_filter.c` | Filter/extract `<think>` tags from output |

---

## Key Data Structures

### `Config` (`include/core/config.h`)
```c
typedef struct {
    int   dim;               // transformer embedding dim
    int   hidden_dim;        // FFN intermediate dim
    int   n_layers;          // number of transformer layers
    int   n_heads;           // number of attention heads
    int   n_kv_heads;        // KV heads (GQA; n_kv_heads ≤ n_heads)
    int   vocab_size;        // vocabulary size
    int   seq_len;           // max sequence length
    int   act_type;          // 0=SiLU, 1=ReLU²
    float rope_theta;        // RoPE base frequency
    int   scale_mode;        // 0=ternary, 1=float32, 2=MoE ternary
    int   bos_token_id;      // BOS token (-1=none)
    int   eos_token_id;      // EOS token (-1=none)
    float rope_freq_scale;   // YaRN: 1/factor (e.g. 0.025 for factor=40)
    float rope_yarn_ext_factor;
    float rope_yarn_attn_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    int   rope_orig_ctx_len; // original training context length
    float rope_yarn_log_mul; // YaRN log multiplier
    float rms_norm_eps;      // 1e-5 for Llama; 1e-6 for DeepSeek
} Config;
```

### `TransformerWeights` (`include/core/weights.h`)
Key fields:
- `token_embedding_table` — BF16 embedding table (vocab_size × dim)
- `embd_f32` — F32 pre-dequanted embeddings (non-NULL for GGUF quantised models)
- `wq/wk/wv/wo[layer]` — Q/K/V/O attention projections (ternary int8 or mmap F16/F32)
- `sq/sk/sv/so[layer]` — per-layer ternary scales
- `w1/w2/w3[layer]` — FFN gate/down/up projections
- `rms_att_weight[layer]`, `rms_ffn_weight[layer]`, `rms_final_weight` — norm weights
- `wcls` — BF16 lm_head; `wcls_i8`/`wcls_i4` — quantised variants
- `moe_gate_w/moe_w1/moe_w2/moe_w3[layer][expert]` — MoE routed expert weights
- `moe_shared_w1/w2/w3[layer]` — DeepSeek shared (always-active) expert weights
- `mla_wq/mla_wkv_a/mla_wkv_b[layer]` — MLA projections
- `layers_are_ternary` — selects packed ternary kernel path
- `layer_weight_type` — `WEIGHT_TYPE_F32/F16/Q4K` for GGUF models
- `has_expert_quant`, `has_mla_quant` — GGUF quantised expert/MLA flags

### `RunState` (`include/core/run_state.h`)
- `x/xb/xb2` — activation scratch buffers (dim floats)
- `hb/hb2` — FFN hidden state buffers (hidden_dim floats)
- `q` — query vector (dim floats)
- `att` — attention scores (n_heads × max_seq_len floats)
- `logits` — output logits (vocab_size floats)
- `key_cache/value_cache` — KV cache `[layer][head][pos][head_dim]`
- `k_rope_cache` — MLA RoPE key cache (NULL for non-MLA)
- `rope_freq/mla_rope_freq` — precomputed RoPE frequencies
- `sw` — SlidingWindow state for circular KV buffer mapping

### `MoEConfig` (`include/core/moe_config.h`)
- `num_experts`, `num_experts_per_tok`, `expert_hidden_dim`
- `is_moe`, `first_k_dense_replace`, `n_shared_experts`
- `has_mla`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`

### `ThreadPool` (`include/threading/thread_pool.h`)
- N-1 OS worker threads + caller participates as Nth worker
- Atomic slice claim (no lock on fast path), spin then cond_wait fallback
- `threadpool_create(n)`, `threadpool_dispatch(tp, task_fn, args, n_rows)`, `threadpool_destroy`

---

## Public API Functions

### Model Loading
```c
// GGUF format
TernaryError gguf_read_header(GGUFHeader *hdr, const void *data, size_t size);
TernaryError config_from_gguf(Config *cfg, const GGUFHeader *hdr);
TernaryError weights_from_gguf(TransformerWeights *w, const Config *cfg,
                                const GGUFHeader *hdr, GGUFWeightStore **store);
// Native ternary format
TernaryError config_read(Config *cfg, const void *mapped_ptr, size_t file_size);
TernaryError weights_map(TransformerWeights *w, const Config *cfg,
                          const MoEConfig *mc, tn_i8 *data, size_t data_size);
// Shared
TernaryError weights_alloc_pointers(TransformerWeights *w, const Config *cfg);
void weights_build_classifier_quant(TransformerWeights *w, const Config *cfg);
```

### Inference
```c
float *transformer_forward(int token, int pos, const Config *cfg,
                            const TransformerWeights *w, RunState *s,
                            const MoEConfig *mc, ThreadPool *tp);

void generate(const Config *cfg, const TransformerWeights *w, RunState *s,
              const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
              const char *prompt, int max_tokens, float temperature, float top_p);

typedef void (*TokenCallback)(const char *piece, void *userdata);
void generate_with_callback(const Config *cfg, const TransformerWeights *w,
                             RunState *s, const MoEConfig *mc,
                             Tokenizer *tok, ThreadPool *tp, const char *prompt,
                             int max_tokens, float temperature, float top_p,
                             TokenCallback callback, void *userdata);
```

### SIMD
```c
const char *tn_simd_init(void);          // init dispatch table, returns backend name
// Global function pointers (set by tn_simd_init):
// tn_rmsnorm, tn_softmax, tn_vec_add, tn_vec_mul, tn_vec_scale,
// tn_silu, tn_relu2, tn_vec_dot, tn_vec_saxpy,
// tn_ternary_matmul, tn_ternary_matmul_packed, tn_unpack_block
```

### Thread Pool
```c
ThreadPool *threadpool_create(int n_threads);
void threadpool_dispatch(ThreadPool *tp, task_fn fn, void *args, int n_rows);
void threadpool_destroy(ThreadPool *tp);
```

### API Server
```c
void api_context_init(ApiContext *ctx, ...);
TernaryError api_server_start(int port, ApiContext *ctx);
void api_server_stop(ApiContext *ctx);
```

---

## Model Format Support Matrix

| Format | Load Path | Weight Kernel | Notes |
|--------|-----------|---------------|-------|
| Native ternary (.bin) | `weights_map` | packed ternary VNNI/AVX2/scalar | Original format |
| GGUF F16 | `weights_from_gguf` | `parallel_matmul_f16` (AVX-512/AVX2) | 🔥 Zero-copy mmap |
| GGUF F32 | `weights_from_gguf` | `parallel_matmul_float32` | Dequant to heap |
| GGUF Q4_K | `weights_from_gguf` | `parallel_matmul_q4k` or F32 dequant | Zero-copy for MLA |
| GGUF Q5_0 | `weights_from_gguf` | dequant → `parallel_matmul_float32` | |
| GGUF Q5_1 | `weights_from_gguf` | dequant → `parallel_matmul_float32` | |
| GGUF Q5_K | `weights_from_gguf` | dequant → `parallel_matmul_float32` | |
| GGUF Q6_K | `weights_from_gguf` | dequant → `parallel_matmul_float32` | |
| GGUF Q8_0 | `weights_from_gguf` | dequant → `parallel_matmul_float32` | |
| GGUF BF16 | `weights_from_gguf` | `parallel_matmul_bf16` | |
| GGUF MoE (DeepSeek-V2) | `weights_from_gguf` + `moe_weights_alloc` | per-expert dequant at inference | |

---

## SIMD Capability Matrix

| Tier | Instruction Set | Kernel | MACs/cycle | CPUs |
|------|----------------|--------|------------|------|
| 1 | AVX-512 VNNI | `ternary_matmul_packed_vnni` | 64 int8 | Ice Lake, Tiger Lake, Zen4, Sapphire Rapids |
| 2 | AVX-VNNI (256) | `ternary_matmul_packed_avx_vnni` | 32 int8 | Alder Lake, Raptor Lake, Zen3 |
| 3 | AVX-512F | `ternary_matmul_packed_avx512` | 16 fp32 | Skylake-X, Ice/Tiger Lake |
| 4 | AVX2 | `ternary_matmul_packed_avx2` | 8 fp32 | Haswell+, Zen2 |
| 5 | ARM dotprod | `ternary_matmul_packed_dotprod` | 16 int8 | Apple M1+, Cortex-A75+ |
| 6 | ARM NEON | (scalar fallback) | 4 fp32 | all ARMv8-A |
| 7 | Scalar | `ternary_matmul_packed` | 1 | any CPU |
| — | F16C (x86) | `parallel_matmul_f16` | — | all AVX2 CPUs |

Override: `TN_FORCE_BACKEND=scalar|avx2|avx512f|vnni|vnni256`

---

## Dependencies & Build System

**Build:** `make -j4` (Makefile) or CMake (`CMakeLists.txt`)
**Compiler:** Apple Clang / GCC, `-O3 -march=native`
**Runtime dependencies:** POSIX only (`pthread`, `mmap`, `clock_gettime`)
**No external libraries** for core inference
**Optional:** `stb_image.h`, `stb_image_resize2.h` (header-only, for multimodal)
**Chat templates:** single C++ file (`src/tokenizer/chat_template.cpp`) using Jinja2-lite

---

*Index generated from source by GitNexus. Refresh: `make -j4 && git add .claude/ && git commit -m 'gitnexus: refresh index'`*
