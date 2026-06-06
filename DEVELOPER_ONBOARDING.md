# Developer Onboarding: Testing & Progressive Phases

> ⚠️ **READ [`GOLDEN_RULES.md`](../GOLDEN_RULES.md) FIRST.**
> No hardcoding. Test after every change. Document at every step. No exceptions.

## Overview
This document ensures that all future development phases (Phase 11+) adhere to the exhaustive Quality Assurance and Security Strategy established for Project Zero.

## Testing Mandate
Every new feature or performance optimization MUST include:
1.  **Independent Unit Test**: New kernels (especially SIMD) must have a scalar reference and a self-contained test suite in `tests/`.
2.  **Regression Validation**: Run `make test` and `make test-packed` before pushing.
3.  **Boundary Stressing**: Purposefully feed non-power-of-two dimensions and out-of-range weights to verify robustness.

## Working with Packed Weights (Phase 10+)
- **Encoding**: -1 -> 0, 0 -> 1, 1 -> 2.
- **Alignment**: Weight rows are packed at 4 weights per byte.
- **Scaling**: Per-group scaling is the preferred mode for high precision. Ensure `scales` indexing logic is audited.

## Memory Management: Chunked Tensor Loading
To support models of arbitrary size (e.g., Qwen-7B, Llama-70B) on memory-constrained hardware (e.g., 8GB RAM), we have implemented a **Streaming/Chunked Tensor Processing** approach in the conversion pipeline.

### The Solution
Instead of loading a full weight matrix into RAM (which for a 152k vocab embedding in float32 is ~2.2GB), we utilize `safetensors.get_slice` to load and process data in smaller chunks (default: 10,000 rows). This keeps the peak memory footprint extremely low regardless of model width or depth.

### SWOT Analysis: RAM Management
| **Strengths (S)** | **Weaknesses (W)** |
| :--- | :--- |
| - Allows 7B+ models on 8GB RAM devices. <br> - Linear RAM growth relative to chunk size, not model size. <br> - Efficient disk I/O overlap possible. | - Increased complexity in conversion logic. <br> - Multi-pass requirements for global scale factors (AbsMax). |
| **Opportunities (O)** | **Threats (T)** |
| - Deployable on edge devices / consumer laptops. <br> - Support for future "Ultra-Large" model phases. <br> - Foundation for a "Virtual Memory" weight loader in the C engine. | - Potential performance bottle-neck if disk speed is extremely slow. <br> - Risk of rounding discrepancies if not careful with multi-pass abs-max calculations. |

### CRITICAL: Independent Developer Assessment Required
Every developer transitioning to this phase is required to **independently assess** the feasibility of this "Chunked Loading" architecture for the entire project life-cycle. Please answer the following:

1.  **Future Compatibility**: Does this approach hinder any planned functionalities for Phases 11-20 (e.g., dynamic weight updates, KV cache optimizations, or LoRA merging)?
2.  **Performance vs. Functionality**: Does the memory saving come at the cost of essential inference latency or data integrity?
3.  **Generalization**: Is this logic robust enough for non-transformer architectures if the project requirements pivot?

> [!IMPORTANT]
> **Zero-Cost Policy**: We do not want a solution at the cost of functionality. If the answer to any of the above is **NO**, do not proceed with integration.

## Branching Strategy
- **Feature Branches**: Create branches with `phase-X-feature-name`.
- **Audit Branches**: Periodic forensic audits will be merged into `qa-strategy-report-vN` branches.
- **Main**: Keep `master` stable and verified by the latest strategy.

## Vulnerability Reporting
If a memory leak or OOB read is identified:
1. Create a regression test case in `tests/test_bugfixes.c`.
2. Implement the fix.
3. Document the root cause in `WALKTHROUGH_PHASE_X.md`.

---

## Phase 14: Agentic Tool Execution

Phase 14 added an agentic loop so the model can emit `<exec>` (and other) tags that
the engine intercepts, executes safely, and injects results back into the KV cache.

### Running the agentic REPL

```bash
make release
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --verbose
```

Use the `/agent` command at the REPL prompt:

```
/agent You are a focused assistant. For your next single output, print EXACTLY this and nothing else: <exec>echo AGENT-TEST</exec>
```

Type `y` at `Allow? [y/N]`.

### Verified test prompts

```
/agent Run <exec>date</exec> and tell me what day it is.
/agent Run <exec>uname -a</exec> and describe the OS.
/agent Run <exec>pwd</exec> and tell me the working directory.
/agent Run <exec>id</exec> and tell me the current user.
/agent Run <exec>ls models</exec> and list available model files.
```

### Testing without interactive TTY (automation/CI)

```bash
# Requires a PTY; plain pipe will not work (getchar() reads from TTY)
PROJECT_ZERO_AGENT_AUTO_APPROVE=1 python3 tests/agent_pty_runner.py
```

See [WALKTHROUGH_PHASE14.md](WALKTHROUGH_PHASE14.md) for the complete PTY runner script
and a verified run capture.

### Agent unit tests

```bash
make test   # includes test_tool_interceptor and test_cmd_exec
```

### Allow-list (permitted commands)

`echo`, `ls`, `cat`, `pwd`, `uname`, `date`, `id`

Any other command is silently blocked (no fork, no shell, no approval prompt).

### Security model

- `fork` + `execvp` only — no `/bin/sh` invocation, no shell expansion
- 5-second timeout with `SIGKILL`
- Human approval required by default (or `PROJECT_ZERO_AGENT_AUTO_APPROVE=1` for testing)
- `<save_memory>` / `<search_memory>` are no-ops until Phase 15 (RAG)

---

## Pipeline Debugging: Comparing Against llama.cpp

When the engine produces wrong output, the fastest way to find the bug is a step-by-step
numerical comparison with llama.cpp on the same prompt. This section documents the exact
methodology used to find and fix the Q4_K nibble ordering bug (the root cause of
"groupe groupe..." garbage output from DeepSeek-V2-Lite-Chat).

### Prerequisites
- llama.cpp built at `~/llama.cpp` (or any path)
- Python 3 with numpy and struct (no gguf library needed for tensor access)
- Engine built with debug dumps enabled (default; controlled by `--dump-tensors` flag)

### Step 1: Establish a Working Reference

Run llama.cpp on your test prompt and record the output:
```bash
cd ~/llama.cpp
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -c 512 --threads 4 \
  -p "User: What is the capital of France?\n\nAssistant:" \
  -n 20
# Expected: "The capital of France is Paris."
```

### Step 2: Run Our Engine with Tensor Dumps

```bash
cd /home/<USER>/Documents/project-zero
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 1 \
  --dump-tensors /tmp/engine_dump.csv
```

The `--dump-tensors` flag writes a CSV with this format:
```
layer,step,n_elem,v0,v1,v2,v3,v4,v5,v6,v7,mean,absmax
-1,embed,2048,0.01385,-0.01355,0.00989,...
0,attn_norm,2048,0.07419,-0.08989,0.07148,...
0,kv_cmpr,576,-0.00453,0.00122,-0.01009,...
...
```

- `layer=-1`: pre-layer steps (embedding)
- `layer=0..N`: transformer layer index
- `step`: step name matching the `DBG_DUMP(layer, "step_name", ptr, n)` calls in source
- Rows appear in execution order; each unique `(layer, step)` appears once per token position

### Step 3: Load and Parse the Dump in Python

```python
import csv
from collections import defaultdict
import numpy as np

rows = defaultdict(list)
with open('/tmp/engine_dump.csv') as f:
    for r in csv.DictReader(f):
        rows[(int(r['layer']), r['step'])].append(
            [float(r[f'v{i}']) for i in range(8)])

# Access: rows[(layer, step_name)][position_index]
# Example: embedding at position 0
embed_pos0 = rows[(-1, 'embed')][0]   # [v0..v7]
attn_norm_pos0 = rows[(0, 'attn_norm')][0]
k_rope_pos5 = rows[(0, 'k_pe')][5]
```

### Step 4: Write a Python Reference for Each Step

For each step, implement the same computation in Python using weights read directly from
the GGUF file. Compare element by element:

```python
def compare(step_name, py_vals, eng_vals, n_positions=14, threshold_rel=0.005):
    issues = 0
    for pos in range(n_positions):
        for i, (py, eng) in enumerate(zip(py_vals[pos], eng_vals[pos])):
            abs_diff = abs(py - eng)
            rel_diff = abs_diff / max(abs(py), 1e-6)
            if abs_diff > 1e-3 and rel_diff > threshold_rel:
                print(f"  FAIL {step_name} pos={pos} idx={i}: py={py:.6f} eng={eng:.6f} rel={rel_diff:.1%}")
                issues += 1
    if issues == 0:
        print(f"  PASS {step_name}: all {n_positions} positions within {threshold_rel:.1%}")
    return issues == 0
```

### Step 5: Binary-Search the Divergence Point

Start from Step 1 (tokenization) and work forward. The first step that fails is the root cause
(or exposes the root cause in the weights used by that step).

**For DeepSeek-V2-Lite, the step order is:**
1. Tokenization / BOS injection
2. Token embedding lookup (embd weight Q4_K)
3. Pre-attention RMSNorm
4. Q projection (W_q Q4_K)
5. KV-A compression (W_kva Q4_K)
6. KV-A latent RMSNorm
7. KV-B expansion (W_kvb Q4_K)
8. YaRN RoPE
9. KV cache write (memcpy — should always match)

### Critical Lessons Learned

1. **Never compare only the first 8 elements of a weight tensor.** The Q4_K nibble bug
   corrupted elements 16–31, 48–63, etc. Elements 0–15 matched by coincidence.
   Always compare a sample from each 32-element group: indices 0, 16, 32, 48+ within a block.

2. **When Python and engine both fail the same way, check the Python first.**
   We spent time chasing a "match" when in fact both our Python reference and our C code
   had the same Q4_K decode bug. The bug was only found when we ran a THIRD implementation
   (llama.cpp's Python gguf reader) and compared all three.

3. **float32 accumulation noise is expected at ≤0.5% relative error.**
   2048-element dot products in float32 (no compensated summation) accumulate ~1 ULP per
   addition. For matmuls this shows as up to 0.2% relative error on large values.
   Near-zero values can show higher relative error (>10%) with tiny absolute error (< 1e-4).
   Use `max(abs(py), 1e-6)` in the denominator to avoid false alarms.

4. **Add `DBG_DUMP` calls surgically; remove before production commits.**
   The dump macro has zero overhead when the dump file is not open. Adding dumps at
   every attention step adds ~5 KB/token to the dump file but no runtime cost in normal runs.

### Where Dump Points Are Defined

| Source file | Dump label | Step |
|-------------|------------|------|
| `src/transformer/forward.c` | `embed` (layer=-1) | Step 3: token embedding |
| `src/transformer/mla_attention.c` | `attn_norm` | Step 4: pre-attn RMSNorm |
| `src/transformer/mla_attention.c` | `q` | Step 5: Q projection output |
| `src/transformer/mla_attention.c` | `kv_cmpr` | Steps 6+7: KV-A compress + norm |
| `src/transformer/mla_attention.c` | `kv` | Step 8: KV-B expansion (kv_full[0:8]) |
| `src/transformer/mla_attention.c` | `k_pe` | Step 9: k_rope after YaRN |
| `src/transformer/mla_attention.c` | `q_pe` | Step 9: q_rope[h0] after YaRN |

To add a new dump point:
```c
// In any .c file with access to DBG_DUMP (include/core/debug.h):
DBG_DUMP(layer, "my_step", float_ptr, n_elements);
// layer=-1 for pre-layer, 0..N for transformer layer index
```

---

## Modularity Rules (Hard Constraints — Session 2026-03-22)

The engine must adapt to each model dynamically. **No hardcoded token IDs anywhere.**

### Token ID resolution order
1. **GGUF models**: `tokenizer.chat_template`, `tokenizer.ggml.bos_token_id`,
   `tokenizer.ggml.eos_token_id` read directly from GGUF metadata by `gguf_loader.c`.
2. **Native `.bin` models**: `tokenizer_load()` scans vocab for candidates in
   `BOS_CANDIDATES[]` / `EOS_PRIMARY[]` / `EOS_CANDIDATES[]` lists in `tokenizer_load.c`.
3. **GGUF + external `.bin` tokenizer**: GGUF metadata overrides vocab scan results
   in the patch block in `main.c` (~line 353).

### Sentinel convention
`bos_token_id` and `eos_token_id` in both `Tokenizer` and `Config` structs use **-1**
to mean "not set". After any `memset(t, 0, sizeof(*t))`, explicitly set:
```c
t->bos_token_id = -1;
t->eos_token_id = -1;
```
Without this, the zero-check guards `if (t->bos_token_id < 0)` will never fire.

### RoPE defaults for native `.bin` models
The `.bin` header is only 40 bytes (`dim` through `scale_mode`). Fields added later
(`rope_freq_scale`, `rope_yarn_attn_factor`, etc.) are NOT stored in the file.
`config_read()` in `src/core/config.c` sets these safe defaults:
```c
cfg->rope_freq_scale       = 1.0f;  /* no scaling */
cfg->rope_yarn_attn_factor = 1.0f;  /* amplitude = 1 (CRITICAL: 0 zeros all Q/K!) */
cfg->rope_yarn_ext_factor  = 0.0f;  /* no YaRN */
```
If `attn_factor` is left at 0, `apply_rope()` multiplies every Q/K value by zero,
destroying positional encoding and producing completely garbled output.

### Correct tokenizer files per model
| Model | Tokenizer file |
|-------|----------------|
| Bitnet-b1.58-2B-4T | `models/bitnet-b1.58-2B-4T_tokenizer_proper.bin` |
| SmolLM2-135M-Instruct | embedded in `.gguf` (no `--tokenizer` needed) |
| DeepSeek-V2-Lite-Chat | `models/deepseek-v2-lite-tokenizer.bin` |

> **Do not use `qwen_tokenizer.bin`** (root-level legacy file) — it has a different
> vocabulary and will produce wrong token IDs for Bitnet.

### Re-validation: DeepSeek GGUF without `--tokenizer` (2026-03-22)

This path was re-validated from the current `master` branch without any code changes before the
llama.cpp catch-up work began.

Command:
```bash
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 12 \
  --temperature 0.0 \
  --threads 4
```

Observed stderr on the no-tokenizer path:
```text
Warning: --tokenizer <path> not provided. Proceeding anyway, but text generation may fail if vocab is absent.
[gguf-tok] Loaded 102400 tokens (gpt2), 2400 special — BOS=100000 EOS=100001 chat_template=yes
```

Observed output:
```text
The capital of France is Paris.
0.91 tok/s (7 tokens)
```

Control run with explicit `--tokenizer models/deepseek-tokenizer-gguf.bin` produced the same
answer at `0.90 tok/s (7 tokens)`.

Implication:
- GGUF tokenizer auto-loading is currently working as intended.
- The remaining DeepSeek work should focus on benchmarking and llama.cpp gap reduction, not on
  recovering basic DeepSeek correctness.

### DeepSeek benchmark script overrides (2026-03-22)

The DeepSeek benchmarking scripts were widened so they can sweep the full `T=1..8` thread range
without hardcoding a second copy of the script.

`tools/deepseek_bench.sh` now accepts these environment overrides:

- `DEEPSEEK_THREADS_MATRIX`
- `DEEPSEEK_THREADS_SWEEP`
- `DEEPSEEK_SIMD_MODES`
- `DEEPSEEK_CLASSIFIERS`
- `DEEPSEEK_LLAMA_THREADS_CSV`
- `DEEPSEEK_BEST_SIMD`
- `DEEPSEEK_BEST_CLS`

Example smoke run:

```bash
DEEPSEEK_THREADS_MATRIX='1' \
DEEPSEEK_THREADS_SWEEP='1' \
DEEPSEEK_SIMD_MODES='avx512f' \
DEEPSEEK_CLASSIFIERS='bf16 int8' \
DEEPSEEK_LLAMA_THREADS_CSV='1' \
bash tools/deepseek_bench.sh
```

`tools/deepseek_bench_perf.sh` now accepts:

- `DEEPSEEK_PERF_PZ_THREADS`
- `DEEPSEEK_PERF_PZ_SIMDS`
- `DEEPSEEK_PERF_PZ_CLASSIFIERS`
- `DEEPSEEK_PERF_LLAMA_THREADS`

Example perf smoke run:

```bash
DEEPSEEK_PERF_PZ_THREADS='1' \
DEEPSEEK_PERF_PZ_SIMDS='avx512f' \
DEEPSEEK_PERF_PZ_CLASSIFIERS='bf16' \
DEEPSEEK_PERF_LLAMA_THREADS='1' \
bash tools/deepseek_bench_perf.sh
```

Important:
- `perf_event_paranoid` must allow `perf stat` access (`-1` was used in this session).
- The llama perf path now uses `llama-bench` CSV output for tok/s extraction, while perf counters
  still come from the same `perf stat` run.
