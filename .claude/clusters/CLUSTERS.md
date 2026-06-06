# GitNexus Functional Cluster Map — project-zero

> Defines functional areas of the codebase. Each cluster is a coherent subsystem.
> Use this to find the right files for any task.

---

## Cluster 1: `cli-entry` — CLI Parsing, Main, REPL, Timer

**Purpose:** Entry point, argument parsing, interactive loop, wall-clock timing.

**Source files:**
- 📁 `src/cli/main.c` — `main()`: SIMD init → calibration → model load → KV alloc → tokenizer → generate/REPL/server/agent dispatch
- 📁 `src/cli/args.c` — `parse_args()`: parses all `--model`, `--prompt`, `--threads`, `--simd`, `--classifier`, `--server`, `--agent`, `--dump-tensors`, etc.
- 📁 `src/cli/repl.c` — `run_repl()`: interactive multi-turn conversation loop
- 📁 `src/cli/timer.c` — `timer_now_us()`, `timer_tokens_per_sec()`

**Public headers:**
- `include/cli/args.h` — `CliArgs` struct
- `include/cli/repl.h`
- `include/cli/timer.h`

**Key functions:**
```c
int main(int argc, char **argv);
TernaryError parse_args(CliArgs *args, int argc, char **argv);
void run_repl(const Config *cfg, ...);
int64_t timer_now_us(void);
double timer_tokens_per_sec(int64_t start_us, int64_t end_us, int tokens);
```

**Internal dependencies:** All other clusters (main.c orchestrates everything)

**External-facing API:** Binary interface — `./adaptive_ai_engine --model ... --prompt ...`

---

## Cluster 2: `core-loader` — GGUF Reader, Config, Weights, Hardware Profile

**Purpose:** Read model files (GGUF or native .bin), populate Config + TransformerWeights, probe hardware.

**Source files:**
- 📁 `src/core/gguf_reader.c` — `gguf_read_header()`: parse GGUF magic, metadata KV, tensor descriptors
- 📁 `src/core/gguf_loader.c` — `config_from_gguf()`, `weights_from_gguf()`: map GGUF tensors to weight pointers
- 📁 `src/core/gguf_quant.c` — dequantization: `gguf_dequant_q4_k`, `q5_0`, `q5_1`, `q5_k`, `q6_k`, `q8_0`, `q4_0`, `q5_k`
- 📁 `src/core/config.c` — `config_read()` (native), `config_print()`
- 📁 `src/core/weights.c` — `weights_alloc_pointers()`, `weights_map()` (native), `weights_build_classifier_quant()`
- 📁 `src/core/moe_config.c` — `moe_config_from_gguf()`, `moe_config_read()` (native), `moe_config_init_dense()`
- 📁 `src/core/moe_weights.c` — `moe_weights_alloc()`, `moe_weights_map()`, `moe_weights_free()`
- 📁 `src/core/hardware_profile.c` — `tn_hardware_profile_init()`, `tn_hardware_profile_set_classifier()`
- 📁 `src/core/calibration.c` — `tn_calibrate()`, `tn_calibration_load()`, `tn_calibration_save()`
- 📁 `src/core/run_state.c` — `run_state_alloc()`, `run_state_free()`
- 📁 `src/core/mla_run_state.c` — `mla_run_state_alloc()`, `mla_run_state_free()`
- 📁 `src/core/unpack.c`, `src/core/unpack_avx2.c` — packed ternary → int8 decompression
- 📁 `src/memory/mapped_file.c` — `mapped_file_open()`, `mapped_file_close()`
- 📁 `src/memory/aligned_alloc.c` — 64-byte aligned malloc/free

**Public headers:**
- `include/core/config.h` — `Config`
- `include/core/weights.h` — `TransformerWeights`
- `include/core/run_state.h` — `RunState`
- `include/core/moe_config.h` — `MoEConfig`
- `include/core/gguf_loader.h` — `weights_from_gguf`, `config_from_gguf`
- `include/core/gguf_reader.h` — `GGUFHeader`, `GGUFTensor`
- `include/core/hardware_profile.h` — `TnHardwareProfile`, `TnClassifierFormat`
- `include/core/calibration.h` — `TnCalibrationResult`

**Key functions:**
```c
TernaryError gguf_read_header(GGUFHeader *hdr, const void *data, size_t size);
TernaryError config_from_gguf(Config *cfg, const GGUFHeader *hdr);
TernaryError weights_from_gguf(TransformerWeights *w, const Config *cfg,
                                 const GGUFHeader *hdr, GGUFWeightStore **store);
TernaryError weights_map(TransformerWeights *w, const Config *cfg,
                          const MoEConfig *mc, tn_i8 *data, size_t data_size);
void weights_build_classifier_quant(TransformerWeights *w, const Config *cfg);
```

**⚠️ Critical rule:** All tensor name lookups must use runtime format strings. Never hardcode e.g. `"blk.0.attn_q.weight"`. Use `snprintf(buf, size, "blk.%d.attn_q.weight", l)`.

---

## Cluster 3: `math-kernels` — All Matmul Variants, RMSNorm, Softmax, RoPE, Elementwise

**Purpose:** Low-level SIMD math. All kernels are parallel (threadpool-dispatched) except utility ops.

**Source files:**
- 📁 `src/math/matmul_f16.c` — 🔥 `parallel_matmul_f16()` — F16 zero-copy (AVX-512/AVX2/scalar)
- 📁 `src/math/parallel_matmul.c` — 🔥 `parallel_ternary_matmul_packed()`, `parallel_matmul_bf16()`, `parallel_matmul_i8()`, `parallel_matmul_i4()`, `parallel_matmul_float32()`, pre-quantisation (`tn_preq_prepare`, `parallel_ternary_matmul_packed_preq`)
- 📁 `src/math/ternary_matmul_packed.c` — scalar reference
- 📁 `src/math/ternary_matmul_packed_avx2.c` — AVX2 8-wide FMA
- 📁 `src/math/ternary_matmul_packed_avx512.c` — AVX-512F 16-wide FMA
- 📁 `src/math/ternary_matmul_packed_vnni.c` — AVX-512 VNNI int8 (64 MACs/cycle)
- 📁 `src/math/ternary_matmul_packed_vnni256.c` — VNNI-256 (no ZMM throttle)
- 📁 `src/math/ternary_matmul_packed_avx_vnni.c` — AVX-VNNI (Alder Lake)
- 📁 `src/math/ternary_matmul_packed_dotprod.c` — ARM SDOT
- 📁 `src/math/ternary_matmul_avx2.c` — AVX2 unpacked ternary
- 📁 `src/math/ternary_matmul_scalar.c` — scalar reference
- 📁 `src/math/rmsnorm.c` + `rmsnorm_avx2.c` + `rmsnorm_avx512.c`
- 📁 `src/math/softmax.c` + `softmax_avx2.c` + `softmax_avx512.c`
- 📁 `src/math/elementwise.c` + `elementwise_avx2.c` + `elementwise_avx512.c`
- 📁 `src/math/rope.c` — `apply_rope()` with YaRN
- 📁 `src/math/matmul_q4k.c` — `parallel_matmul_q4k()` zero-copy Q4_K
- 📁 `src/math/matmul_q5_0.c`, `matmul_q5_1.c`, `matmul_q5k.c`
- 📁 `src/math/quantize_i8.c` — `quantize_row_to_i8_avx512()`, `sum_i8_avx512()`

**Key function signatures:**
```c
void parallel_matmul_f16(float *out, const float *x, const tn_u16 *w,
                          int n, int d, ThreadPool *tp);
void parallel_ternary_matmul_packed(float *out, const float *x, const tn_u8 *w,
                                     int n, int d, float scale, ThreadPool *tp);
void parallel_matmul_float32(float *out, const float *x, const float *w,
                              int n, int d, ThreadPool *tp);
int tn_preq_prepare(TnPreqActivation *preq, int8_t *buf, const float *x, int n);
void parallel_ternary_matmul_packed_preq(float *out, const float *x,
    const tn_u8 *w, int n, int d, float scale, const TnPreqActivation *preq, ThreadPool *tp);
```

**⚠️ Caution:** `parallel_matmul_f16` AVX2 path uses `_mm256_cvtph_ps` (F16C). Present on all AVX2 x86 CPUs.

---

## Cluster 4: `simd-dispatch` — CPU Feature Detection, Runtime SIMD Selection

**Purpose:** Probe CPU at startup, set global function pointers to best available kernels.

**Source files:**
- 📁 `src/math/simd_dispatch.c` — `tn_simd_init()`, global dispatch table, `TN_FORCE_BACKEND` env override
- 📁 `src/math/cpu_features.c` — `tn_cpu_features_detect()` → `TnCpuFeatures` struct (CPUID-based)

**Public headers:**
- `include/math/simd_dispatch.h` — function pointer types + extern declarations
- `include/math/cpu_features.h` — `TnCpuFeatures` struct + `tn_cpu_features_detect()`
- `include/core/platform.h` — compile-time `TN_HAS_AVX2`, `TN_HAS_AVX512`, etc.

**Global dispatch table (after `tn_simd_init()`):**
```c
tn_matmul_fn         tn_ternary_matmul;        // unpacked ternary
tn_matmul_packed_fn  tn_ternary_matmul_packed; // packed ternary (main hot path)
tn_unpack_fn         tn_unpack_block;
tn_rmsnorm_fn        tn_rmsnorm;
tn_softmax_fn        tn_softmax;
tn_vec_add_fn        tn_vec_add;
tn_vec_mul_fn        tn_vec_mul;
tn_vec_scale_fn      tn_vec_scale;
tn_silu_fn           tn_silu;
tn_relu2_fn          tn_relu2;
tn_vec_dot_fn        tn_vec_dot;
tn_vec_saxpy_fn      tn_vec_saxpy;
```

**Selection priority:** AVX-512 VNNI > VNNI-256 > AVX-VNNI > AVX-512F > AVX2 > ARM dotprod > Scalar
**Override:** `TN_FORCE_BACKEND=scalar|avx2|avx512f|vnni|vnni256`

---

## Cluster 5: `transformer-forward` — Forward Pass, Embedding, Generate, Token Loop

**Purpose:** Orchestrates the per-token forward pass and generation loop.

**Source files:**
- 📁 `src/transformer/forward.c` — 🔥 `transformer_forward()`: embed → N layers → final norm → lm_head
- 📁 `src/transformer/generate.c` — 🔥 `generate()`, `generate_with_callback()`: tokenise → forward loop → sample → callback
- 📁 `src/transformer/embedding.c` — `embed_token()`: BF16 or F32 table lookup

**Key functions:**
```c
float *transformer_forward(int token, int pos, const Config *cfg,
    const TransformerWeights *w, RunState *s, const MoEConfig *mc, ThreadPool *tp);
void generate(const Config *cfg, const TransformerWeights *w, RunState *s,
    const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp, const char *prompt,
    int max_tokens, float temperature, float top_p);
void generate_with_callback(..., TokenCallback callback, void *userdata);
void embed_token(float *x, int token, const float *embd_f32,
                  const tn_u16 *embd_bf16, int dim);
```

**lm_head dispatch (in `forward.c`):**
- `wcls_is_ternary=true` → `parallel_ternary_matmul_packed`
- `TN_CLS_INT4` + `wcls_i4` → `parallel_matmul_i4`
- `TN_CLS_INT8` + `wcls_i8` → `parallel_matmul_i8`
- else → `parallel_matmul_bf16`

---

## Cluster 6: `attention` — Standard Attention, MLA Attention (DeepSeek), RoPE

**Purpose:** Per-layer attention with GQA, RoPE, and KV cache writes.

**Source files:**
- 📁 `src/transformer/attention.c` — `attention_forward()`: RMSNorm → Q/K/V proj → RoPE → KV cache write → GQA scores → softmax → V-weighted sum → O proj
- 📁 `src/transformer/mla_attention.c` — `mla_attention_forward()`: KV compress → latent norm → KV expand → Q proj → YaRN RoPE → split-score attention → wo proj

**Public headers:**
- `include/transformer/attention.h`
- `include/transformer/mla_attention.h`

**Key dispatch logic in `attention_forward`:**
```c
// Route to MLA when has_mla=1:
if (mc && mc->has_mla) { mla_attention_forward(...); return; }
// Kernel selection for Q/K/V/O projections:
if (w->layers_are_ternary)    → parallel_ternary_matmul_packed_preq()
else if (WEIGHT_TYPE_F16)     → parallel_matmul_f16()
else                          → parallel_matmul_float32()
```

**⚠️ MLA critical detail:** MLA uses `s->mla_rope_freq` (size = qk_rope_head_dim/2 = 32), NOT `s->rope_freq` (size = head_dim/2 = 64). Using wrong buffer gives incorrect RoPE frequencies.

**⚠️ KV cache mapping:** All KV read/write uses `sw_map_position(&s->sw, hist_logical)`. Direct use of `t` as slot index is a bug (PRE10-BUG-004).

---

## Cluster 7: `ffn` — Dense FFN, MoE FFN, MoE Router, Expert Dispatch

**Purpose:** Per-layer feed-forward network — either dense SwiGLU or MoE expert routing.

**Source files:**
- 📁 `src/transformer/ffn.c` — `ffn_forward()`: SwiGLU dense (RMSNorm → gate+up proj → SiLU → ×up → down proj → residual); routes to MoE via `moe_layer_is_moe(mc, layer)`
- 📁 `src/transformer/moe_ffn.c` — `moe_ffn_forward()`: router → top-k → per-expert SwiGLU → accumulate + shared experts
- 📁 `src/transformer/moe_router.c` — `moe_router_forward()`: gate matmul → softmax → top-k selection

**Key expert dispatch logic in `moe_ffn.c`:**
- `has_expert_quant=true` → `dequant_expert_weight()` on raw mmap bytes per expert per token
- `has_expert_quant=false` → pre-dequanted F32 pointers
- Supports Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0 via `quant_type` field

**⚠️ Expert tracking:** `moe_expert_tracking_reset/print/free` for MoE diagnostic output. All 27 DeepSeek MoE layers should show expert activity — dead layers indicate NaN propagation.

---

## Cluster 8: `kv-cache` — KV Strategy, Compression, Sliding Window

**Purpose:** Manage KV cache memory, select strategy based on RAM, implement circular buffer.

**Source files:**
- 📁 `src/kv_cache/kv_strategy.c` — `kv_strategy_select()`: FULL vs SLIDING_WINDOW vs COMPRESS based on free RAM
- 📁 `src/kv_cache/kv_compress.c` — token eviction for COMPRESS strategy
- 📁 `src/kv_cache/sliding_window.c` — `sw_init`, `sw_advance`, `sw_map_position`, `sw_valid_count`

**KV cache layout (in `run_state.h`):**
```
key_cache[layer][kv_head][pos][head_dim]   // TRANSPOSED for cache-line efficiency
KV_CACHE_IDX(layer, head, pos, d, n_kv_heads, max_seq, head_dim)
```

**Non-temporal stores (`kv_nt_store` in `attention.c`):** Prevents KV writes from evicting weight data from L3 cache.

---

## Cluster 9: `threading` — Thread Pool, CPU Probe, Parallelism

**Purpose:** Efficient CPU parallelism with no oversubscription.

**Source files:**
- 📁 `src/threading/thread_pool.c` — Caller-participates pool: `(N-1)` OS workers + 1 caller = N HW slots. Atomic slice claim, spin-then-sleep fallback.
- 📁 `src/threading/cpu_probe.c` — Physical core count from `/proc/cpuinfo`

**Public headers:**
- `include/threading/thread_pool.h` — `ThreadPool`, `threadpool_create/dispatch/destroy`
- `include/threading/cpu_probe.h`

**Key design:** At T=4 on a 4-core CPU: 3 OS threads + caller = 4 threads, no context switches.
`use_blocking_wait=true` when `n_threads >= physical_cores * 2` to avoid spin waste on SMT.

**Thread pool dispatch:**
```c
typedef void (*task_fn)(void *arg, int thread_id, int start, int end);
void threadpool_dispatch(ThreadPool *tp, task_fn fn, void *args, int n_rows);
```
Each worker receives a `[start, end)` row slice. Caller executes the last slice.

---

## Cluster 10: `tokenizer` — BPE Tokenizer, GGUF Tokenizer, Chat Template

**Purpose:** Encode prompts to token IDs, decode token IDs to text, apply chat templates.

**Source files:**
- 📁 `src/tokenizer/tokenizer_load.c` — load vocab from native .bin format
- 📁 `src/tokenizer/tokenizer_encode.c` — `tokenizer_encode()`: BPE merge encoding
- 📁 `src/tokenizer/tokenizer_decode.c` — `tokenizer_decode()`: token ID → UTF-8 piece
- 📁 `src/tokenizer/tokenizer_gguf.c` — `tokenizer_from_gguf()`: extract BPE from GGUF metadata
- 📁 `src/tokenizer/chat_template.cpp` — `chat_template_apply()`: Jinja2 chat template renderer

**Key function:**
```c
int tokenizer_encode(Tokenizer *tok, const char *text, size_t len,
                      int *tokens, int max_tokens);
const char *tokenizer_decode(Tokenizer *tok, int prev_token, int token);
char *chat_template_apply(const char *tmpl, const char **roles,
    const char **contents, int n_turns, const char *bos, const char *eos,
    int add_gen_prompt);
```

**⚠️ GGUF chat template path:** When `tok->chat_template` is non-NULL, the Jinja2 template renders BOS internally. Do NOT manually prepend BOS — double-BOS corrupts output.

---

## Cluster 11: `api-server` — HTTP Server, SSE Stream, JSON, Chat Compile

**Purpose:** OpenAI-compatible `/v1/chat/completions` endpoint with SSE token streaming.

**Source files:**
- 📁 `src/api/http_server.c` — `api_server_start()`: embedded HTTP listener thread
- 📁 `src/api/chat_compile.c` — `chat_compile()`: `messages[]` → single prompt string
- 📁 `src/api/json_parse.c` — minimal JSON parser for OpenAI request format
- 📁 `src/api/sse_stream.c` — `sse_send_token()`: write `data: {"token":"..."}` chunks

**Public headers:**
- `include/api/api_server.h` — `ApiContext`, `api_server_start/stop`
- `include/api/chat_compile.h`, `chat_request.h`, `sse_stream.h`

**Usage:**
```c
ApiContext ctx;
api_context_init(&ctx, cfg, weights, run_state, moe_cfg, tok, tp);
api_server_start(8080, &ctx);  // background thread
// main thread blocks; Ctrl+C calls api_server_stop()
```

---

## Cluster 12: `agent` — Agent Loop, Tool Interceptor, Cmd Exec, Output Inject

**Purpose:** Autonomous agentic mode: model generates tool calls → engine executes → injects results.

**Source files:**
- 📁 `src/agent/agent_loop.c` — `run_agent_loop()`: generation loop detecting `<tool_call>` tags
- 📁 `src/agent/tool_interceptor.c` — parse tool call XML/JSON from generated text
- 📁 `src/agent/cmd_exec.c` — execute shell command, capture stdout/stderr
- 📁 `src/agent/output_inject.c` — inject tool output into `<tool_result>` context
- 📁 `src/agent/user_approval.c` — `user_approval_prompt()`: interactive Y/N before execution

**Public headers:**
- `include/agent/agent_loop.h` — `run_agent_loop()`
- `include/agent/cmd_exec.h`, `output_inject.h`, `tool_interceptor.h`

**RAG integration:** `run_agent_loop` accepts optional `RagContext *rag` parameter; auto-save/retrieve enabled when non-NULL.

---

## Cluster 13: `rag-memory` — Vector DB, Embedder, Similarity, Memory Search, RAG Context

**Purpose:** Persistent episodic memory: embed text → store in file-backed vector DB → retrieve at prompt build.

**Source files:**
- 📁 `src/rag/vector_db.c` — append-only VRDB file (magic `VRDB`, 16-byte header + records)
- 📁 `src/rag/embedder.c` — `embedder_generate()`: mean-pool forward pass → L2-normalised vector
- 📁 `src/rag/similarity.c` — cosine dot-product search (embeddings are pre-normalised, so cosine = dot)
- 📁 `src/rag/memory_search.c` — top-k retrieval with score threshold
- 📁 `src/rag/rag_context.c` — `RagContext`: ties embedder + DB + search
- 📁 `src/rag/auto_save.c` — auto-detect memorable content in agent output
- 📁 `src/rag/auto_retrieve.c` — auto-prepend relevant memories to prompt

**VRDB file format:**
```
[0..3]  magic: 0x42445256 "VRDB"
[4..7]  version: 1
[8..11] num_entries
[12..15] embed_dim
[Record]: float32[embed_dim] + uint32 text_len + text
```

**Key struct:**
```c
typedef struct {
    int     num_entries, embed_dim, capacity;
    float  *embeddings;   // flat row-major [n × embed_dim]
    char  **texts;
    FILE   *fp;
} VectorDB;
```

---

## Cluster 14: `multimodal` — Vision Encoder, Image Load, Patch Extract, Vision Projector

**Purpose:** Vision-language pipeline: load image → extract patches → ViT encode → project to LLM space → inject.

**Source files:**
- 📁 `src/multimodal/image_load.c` — `image_load()`: via stb_image, returns float pixel buffer
- 📁 `src/multimodal/patch_extract.c` — `extract_patches()`: tile image into fixed-size patches
- 📁 `src/multimodal/vision_encoder.c` — ViT-style encoder forward pass
- 📁 `src/multimodal/vision_projector.c` — project vision features → LLM embedding dim
- 📁 `src/multimodal/vision_bridge.c` — `vision_inject()`: write projected patches into `RunState` before generation (sets `s->current_pos`)
- 📁 `src/multimodal/vision_weights_load.c` — load vision model weights from GGUF

**Public headers:**
- `include/multimodal/image_load.h`, `patch_extract.h`, `vision_encoder.h`
- `include/multimodal/vision_projector.h`, `vision_bridge.h`, `vision_weights_load.h`

---

## Cluster 15: `sampling` — Sampling Strategies, RNG

**Purpose:** Convert logit vector to next-token ID.

**Source files:**
- 📁 `src/sampling/argmax.c` — `sample_argmax()`: greedy decoding (temperature=0)
- 📁 `src/sampling/temperature.c` — `apply_temperature()`: divide logits by T
- 📁 `src/sampling/top_k.c` — top-K filtering
- 📁 `src/sampling/top_p.c` — `sample_top_p()`: nucleus (top-p) sampling
- 📁 `src/sampling/rng.c` — `rng_seed()`, `rng_float()`: xorshift64 RNG

**Public headers:**
- `include/sampling/sampling.h` — all sampling function signatures
- `include/sampling/rng.h`

**Dispatch in `generate_with_callback`:**
```c
if (temperature <= 0.0f) → sample_argmax()
else if (top_p < 1.0f)  → apply_temperature() + sample_top_p()
else                    → apply_temperature() + tn_softmax() + categorical sample
```

---

*Clusters file generated by GitNexus. Refresh: `make -j4 && git add .claude/ && git commit -m 'gitnexus: refresh index'`*
