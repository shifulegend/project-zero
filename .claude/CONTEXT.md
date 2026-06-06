# GitNexus Codebase Context — project-zero

> **For AI agents.** Provides the architectural understanding needed to navigate, modify, and debug this codebase without reading every file.
>
> ⚠️ **Stale index?** Run `make -j4 && git add .claude/ && git commit -m 'gitnexus: refresh index'` to update after major changes.

---

## Architecture Overview (Layer Diagram)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI / API / Agent                           │
│  src/cli/main.c   src/api/http_server.c   src/agent/agent_loop.c   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     Inference Engine                                │
│  generate.c → transformer_forward.c → attention.c / ffn.c         │
│               mla_attention.c / moe_ffn.c / embedding.c            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Math Kernels                                   │
│  simd_dispatch.c → matmul_f16.c, parallel_matmul.c, rmsnorm.c,    │
│                    softmax.c, elementwise.c, rope.c                │
│  (AVX-512 VNNI > AVX-VNNI > AVX-512F > AVX2 > ARM dotprod >       │
│   ARM NEON > Scalar)                                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   Core Services                                     │
│  gguf_loader.c   weights.c   run_state.c   kv_cache/   threading/  │
│  calibration.c   hardware_profile.c   tokenizer/                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                Memory / Platform                                    │
│  mapped_file.c (mmap)   aligned_alloc.c (64-byte)   platform.h    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete File Tree with Roles

```
project-zero/
├── adaptive_ai_engine          ← compiled binary
├── Makefile / CMakeLists.txt
├── include/
│   ├── core/
│   │   ├── config.h            Config struct (all model hyperparams)
│   │   ├── weights.h           TransformerWeights (all weight pointers)
│   │   ├── run_state.h         RunState (KV cache + scratch buffers)
│   │   ├── moe_config.h        MoEConfig (MoE + MLA topology)
│   │   ├── moe_weights.h       MoE expert weight array management
│   │   ├── platform.h          Type aliases, SIMD macros, arch detection
│   │   ├── gguf_loader.h       GGUF weight loading API
│   │   ├── gguf_reader.h       GGUF binary parser API
│   │   ├── gguf_quant.h        GGUF dequantization API
│   │   ├── hardware_profile.h  CPU feature + RAM + classifier format
│   │   ├── calibration.h       Auto-calibration API + result struct
│   │   ├── error.h             TernaryError enum
│   │   ├── debug.h             DBG_DUMP/DBG_MSG macros, g_dump_fp
│   │   └── step_timing.h       Per-step latency profiler
│   ├── math/
│   │   ├── simd_dispatch.h     Global dispatch function pointers
│   │   ├── cpu_features.h      TnCpuFeatures struct + detect()
│   │   ├── matmul_f16.h        parallel_matmul_f16 signature
│   │   ├── parallel_matmul.h   All parallel matmul signatures
│   │   ├── ternary_matmul*.h   Packed/scalar ternary signatures
│   │   ├── rmsnorm.h / rope.h / softmax.h / elementwise.h
│   │   └── quantize_i8.h       INT8 quantisation helpers
│   ├── transformer/
│   │   ├── forward.h           transformer_forward signature
│   │   ├── generate.h          generate / generate_with_callback
│   │   ├── attention.h         attention_forward
│   │   ├── mla_attention.h     mla_attention_forward
│   │   ├── ffn.h / moe_ffn.h / moe_router.h / embedding.h
│   ├── kv_cache/
│   │   ├── kv_strategy.h / kv_compress.h / sliding_window.h
│   ├── threading/
│   │   ├── thread_pool.h       ThreadPool API
│   │   └── cpu_probe.h         Physical core detection
│   ├── tokenizer/
│   │   ├── tokenizer.h / tokenizer_gguf.h / chat_template.h
│   ├── api/ agent/ rag/ multimodal/ sampling/ memory/ reasoning/
│   └── stb_image.h / stb_image_resize2.h   (header-only libs)
└── src/
    ├── cli/main.c args.c repl.c timer.c
    ├── core/ (15 files)
    ├── math/ (25 files: scalar + avx2 + avx512 + vnni variants)
    ├── transformer/ (8 files)
    ├── kv_cache/ (3 files)
    ├── threading/ (2 files)
    ├── tokenizer/ (4 .c + 1 .cpp)
    ├── api/ (4 files)
    ├── agent/ (5 files)
    ├── rag/ (7 files)
    ├── multimodal/ (6 files)
    ├── sampling/ (5 files)
    ├── memory/ (2 files)
    └── reasoning/ (3 files)
```

---

## Key Execution Flows (one-line each)

| Flow | Path |
|------|------|
| CLI prompt → output | `main.c` → `generate()` → `generate_with_callback()` → token loop |
| Per-token forward | `transformer_forward()` → `embed_token()` → N×`attention_forward()` + `ffn_forward()` → RMSNorm → lm_head |
| Ternary attention | `attention_forward()` → `parallel_ternary_matmul_packed_preq()` for Q/K/V/O |
| F16 attention | `attention_forward()` → `parallel_matmul_f16()` for Q/K/V/O |
| MLA attention | `mla_attention_forward()` → compress→expand→Q→YaRN RoPE→scores→wo |
| MoE FFN | `ffn_forward()` → `moe_ffn_forward()` → `moe_router_forward()` → top-k expert SwiGLU → accumulate |
| GGUF load | `gguf_read_header()` → `config_from_gguf()` → `weights_from_gguf()` → lm_head quantise |
| SIMD init | `tn_simd_init()` → `tn_cpu_features_detect()` → set global fn pointers |
| Calibration | `tn_calibrate()` → thread/SIMD sweep → save to `~/.project-zero/calibration.bin` |
| API request | `http_server.c` → `json_parse.c` → `generate_with_callback(SSE callback)` → SSE stream |
| Agent tool | `agent_loop.c` → `tool_interceptor.c` → `cmd_exec.c` → `output_inject.c` |
| RAG memory | `embedder_generate()` → `vector_db_store()` / `memory_search()` → `rag_context.c` |

---

## Data Flow: CLI args → model load → inference → output

```
argv[]
  └─ parse_args() [args.c]
       ├─ args.model_path, args.prompt, args.num_threads
       ├─ args.simd_override → setenv("TN_FORCE_BACKEND")
       └─ args.classifier_override

tn_simd_init()                         → global dispatch table set
tn_hardware_profile_init()             → TnHardwareProfile (cores, RAM, SIMD caps)
tn_calibrate() / tn_calibration_load() → best_threads, best_simd, best_classifier

mapped_file_open(model_path)           → MappedFile {data, size}
  ├─ magic == GGUF_MAGIC →
  │    gguf_read_header()              → GGUFHeader {arch, n_meta, tensors[]}
  │    config_from_gguf()             → Config
  │    moe_config_from_gguf()         → MoEConfig (if deepseek2)
  │    weights_alloc_pointers()       → pointer arrays allocated
  │    moe_weights_alloc()            → expert pointer arrays (if MoE)
  │    weights_from_gguf()            → weight pointers set, embd_f32 built
  │    weights_build_classifier_quant() → wcls_i8 / wcls_i4 built
  └─ magic == TN_MAGIC →
       config_read()                  → Config
       weights_map()                  → weight pointers set

kv_strategy_select()                  → max_seq_len (RAM-capped)
threadpool_create(active_threads)     → ThreadPool
tokenizer_from_gguf() / tokenizer_load() → Tokenizer {vocab, bpe_merges}
run_state_alloc()                     → RunState {x, xb, hb, q, att, logits,
                                                    key_cache, value_cache}
mla_run_state_alloc()                 → RunState.k_rope_cache (if MLA)

generate_with_callback(prompt, ...)
  └─ chat_template_apply() / manual BOS
  └─ tokenizer_encode()              → prompt_tokens[]
  └─ for step in [0..max_steps):
       transformer_forward(token, pos)
         embed_token()               → s->x
         for layer in [0..n_layers):
           attention_forward()       → s->x (via s->xb + residual)
           ffn_forward()             → s->x (via s->xb + residual)
         tn_rmsnorm(s->x)            → final norm
         parallel_matmul_*()         → s->logits (lm_head)
       sample_argmax / sample_top_p → next_token
       tokenizer_decode()            → piece
       callback(piece, userdata)     → stdout / SSE / agent buffer
```

---

## Critical Coupling Points

| Dependency | Files | What It Controls |
|------------|-------|------------------|
| `weights.h` → `config.h` | All transformer files | `TransformerWeights` needs `Config` for dims |
| `simd_dispatch.c` → `cpu_features.c` | Entire math stack | All fn pointers go through dispatch |
| `attention.c` → `mla_attention.h` | `attention_forward` | Single `if (mc->has_mla)` branch |
| `ffn.c` → `moe_ffn.h` | `ffn_forward` | Single `if (moe_layer_is_moe)` branch |
| `forward.c` → `hardware_profile.h` | lm_head dispatch | `TnClassifierFormat` selects BF16/INT8/INT4 |
| `gguf_loader.c` → `weights.h` | All weight flags | Sets `layer_weight_type`, `has_mla_quant`, etc. |
| `parallel_matmul.c` → `simd_dispatch.h` | Every matmul call | Uses `tn_ternary_matmul_packed` fn pointer |
| `run_state.h` → `kv_cache/sliding_window.h` | KV cache indexing | `sw_map_position` maps logical→physical slot |
| `generate.c` → `tokenizer/chat_template.h` | GGUF models | `tok->chat_template` triggers Jinja2 path |
| `moe_ffn.c` → `gguf_quant.h` | Expert dequant | `has_expert_quant` → per-expert dequant at inference |

---

## Known Issues / Active Areas

### From DEBUGGING_JOURNAL.md (most recent first)

**2026-03-24 — F16 prefetch investigation concluded:**
- Software prefetch in `matmul_f16.c` causes regression (−8% T=1, −13% T=2)
- Hardware prefetcher handles sequential access correctly for OS-page-cached model
- ⚠️ Do NOT add `_mm_prefetch` calls to matmul loops

**2026-03-23 — INT8/INT4 classifier wrong weight source (FIXED):**
- `weights_build_classifier_quant()` must read from `w->wcls`, NOT `token_embedding_table`
- DeepSeek has separate `output.weight` → different pointer
- See GOLDEN_RULES Historical Regressions table

**2026-03-23 — MoE expert cache causes regression (FIXED/REVERTED):**
- Routed-expert heap copies: `cache=0 → 1.055 tok/s`, `cache=1 → 0.810 tok/s` — worse
- Do NOT add expert caching without a structural redesign

**Earlier — MLA RoPE wrong frequency buffer (FIXED):**
- `mla_attention_forward` must use `s->mla_rope_freq`, NOT `s->rope_freq`
- `rope_freq` is computed for `head_dim=128`; MLA needs `qk_rope_head_dim=64`

**Earlier — KV value vector wrong slot after sliding window wrap (FIXED):**
- PRE10-BUG-004: used raw `t` instead of `sw_map_position(hist_logical)` for value lookup
- Bug in `attention.c` Step 5c, fixed by using `v_mapped`

**Earlier — Q4_K dequant nibble ordering (FIXED):**
- Wrong nibble grouping → `groupe groupe groupe` repeat output
- Fixed in commit 9801018

---

## Performance Baselines (from BENCHMARK_REPORT.md Addendum AS)

**System:** Intel i5-5250U, 2C/4T, 8GB DDR3, macOS, AVX2 (no AVX-512)
**Model:** SmolLM2-135M-Instruct-f16.gguf (271 MB, F16 dense)

| T | project-zero | bitnet.cpp | llama.cpp |
|---|-------------|------------|-----------|
| 1 | 27.74 tok/s | 28.91 tok/s | 30.71 tok/s |
| 2 | **41.75 tok/s** | 42.05 tok/s | 39.37 tok/s |
| 3 | **27.06 tok/s** | 22.11 tok/s | 21.40 tok/s |
| 4 | **33.73 tok/s** | 22.19 tok/s | 22.48 tok/s |

**T=1 gap root cause:** llama.cpp/bitnet.cpp compiled with Apple Accelerate BLAS. PZ uses pure AVX2 FMA.
At T=1, PZ is operating at DDR3 bandwidth ceiling (~5.66 GB/s vs 5.3 GB/s ceiling). Not fixable without BLAS linkage.

**T=3..4 advantage root cause:** PZ's `(N-1) OS worker + 1 caller` thread pool avoids ggml work-stealing
overhead and OS scheduling oversubscription on 2-physical-core Broadwell.

---

## Quick-Reference Function Signatures

```c
// 🔥 Core inference hot path
float *transformer_forward(int token, int pos, const Config *cfg,
    const TransformerWeights *w, RunState *s, const MoEConfig *mc, ThreadPool *tp);

void generate_with_callback(const Config *cfg, const TransformerWeights *w,
    RunState *s, const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
    const char *prompt, int max_tokens, float temperature, float top_p,
    TokenCallback callback, void *userdata);

// 🔥 Layer-level ops
void attention_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
    const MoEConfig *mc, int layer, int pos, ThreadPool *tp);
void mla_attention_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
    const MoEConfig *mc, int layer, int pos, ThreadPool *tp);
void ffn_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
    const MoEConfig *mc, int layer, ThreadPool *tp);
void moe_ffn_forward(RunState *s, const TransformerWeights *w, const Config *cfg,
    const MoEConfig *mc, int layer, ThreadPool *tp);

// 🔥 Math kernels
void parallel_matmul_f16(float *out, const float *x, const tn_u16 *w,
    int n, int d, ThreadPool *tp);
void parallel_ternary_matmul_packed(float *out, const float *x, const tn_u8 *w,
    int n, int d, float scale, ThreadPool *tp);
void parallel_matmul_float32(float *out, const float *x, const float *w,
    int n, int d, ThreadPool *tp);

// Thread pool
ThreadPool *threadpool_create(int n_threads);
void threadpool_dispatch(ThreadPool *tp, task_fn fn, void *args, int n_rows);

// SIMD
const char *tn_simd_init(void);   // call once at startup

// Calibration
void tn_calibrate(TnCalibrationResult *result, const TnHardwareProfile *hw);
bool tn_calibration_load(TnCalibrationResult *result, const TnHardwareProfile *hw);
void tn_calibration_save(const TnCalibrationResult *result);
```

---

## "How to Add X" Guides

### How to add a new SIMD kernel tier
1. Create `src/math/ternary_matmul_packed_NEWSIMD.c` matching existing signature
2. Add `extern void ternary_matmul_packed_NEWSIMD(...)` in `simd_dispatch.c`
3. Add compile-time guard `#if TN_HAS_NEWSIMD` in `simd_dispatch.c`
4. Add runtime CPUID check `cpu->new_simd` in `tn_simd_init()` selection block
5. Add `TN_HAS_NEWSIMD` macro to `include/core/platform.h`
6. Add `new_simd` field to `TnCpuFeatures` in `cpu_features.h`; detect in `cpu_features.c`

### How to add a new weight format (GGUF quant type)
1. Add `gguf_dequant_NEWTYPE()` in `src/core/gguf_quant.c` + declare in `gguf_quant.h`
2. Add `case GGUF_TYPE_NEWTYPE:` in `tensor_to_f32()` in `gguf_loader.c`
3. If zero-copy is possible: add a flag to `TransformerWeights` (e.g. `has_new_quant`), set it in `gguf_loader.c`, dispatch in `attention.c`/`ffn.c`/`mla_attention.c`
4. Add kernel `src/math/matmul_NEWTYPE.c` + header

### How to add a new model architecture
1. Create `src/core/moe_config.c` analogue for architecture-specific config
2. Add arch-specific metadata key reading in `config_from_gguf()` (check GGUF `arch` field)
3. All tensor name lookups must use `snprintf(name_buf, sizeof(name_buf), "%s.blk.%d.%s", arch, l, suffix)` — never hardcode
4. If the architecture needs new attention variant: add `has_newattn` to `MoEConfig`, dispatch in `attention_forward()`
5. Test with France→Paris, Germany→Berlin, and expert utilization stats (all layers active)

### How to add a new CLI flag
1. Add field to `CliArgs` struct in `include/cli/args.h`
2. Parse in `src/cli/args.c` (add `--new-flag` case)
3. Handle in `src/cli/main.c` after `parse_args()`

### How to extend the API server
1. Add route handler in `src/api/http_server.c` (follow `/v1/chat/completions` pattern)
2. Add JSON parsing logic in `src/api/json_parse.c`
3. If streaming: use SSE via `src/api/sse_stream.c`

---

*Context file generated by GitNexus. Refresh: `make -j4 && git add .claude/ && git commit -m 'gitnexus: refresh index'`*
