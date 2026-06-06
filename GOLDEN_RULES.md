# GOLDEN RULES — Project Zero Development

> **READ THIS FIRST. READ THIS EVERY TIME. THESE RULES ARE NON-NEGOTIABLE.**
>
> Every developer, every AI agent, every contributor MUST follow these rules
> before touching a single line of code. Violating any rule has historically
> caused production regressions, broken correctness, wasted days of debugging.

---

## Rule 1 — NO HARDCODING

**1A. No hardcoded values anywhere in engine code.**

- All tensor names, dimension constants, quantization types, thread counts,
  SIMD modes, and model parameters MUST be derived at runtime from the model
  file (GGUF metadata, config fields, or CLI arguments).
- If a value is in the GGUF header, read it from the header. Period.
- Bad: `if (n_layers == 27)` — Good: `if (cfg->n_layers == mc->first_dense + 1)`
- Bad: `"blk.0.attn_q.weight"` hardcoded — Good: `snprintf(name_buf, ..., "blk.%d.attn_q.weight", l)`

**1B. Maximum modularity — every fix must stand on its own.**

- A fix in `mla_attention.c` must NOT require a corresponding change in
  `moe_ffn.c` to keep working, and vice versa.
- Dispatch logic (float vs. quantized, Q4_K vs. Q5_1, etc.) belongs in one
  place per subsystem. Use flags and type fields, not scattered `if` chains.
- Follow the existing pattern: `has_expert_quant`, `layer_weight_type`,
  `layers_are_ternary` — add a single flag, dispatch on it.
- When adding a new weight format: (a) add a flag to `TransformerWeights`,
  (b) set it in `gguf_loader.c`, (c) dispatch in the compute path. Never
  spread the logic across 5 files.

---

## Rule 2 — TEST AFTER EVERY CHANGE

**Run the engine after EACH code change, no matter how small.**

```bash
cd /home/<USER>/Documents/project-zero
make -j4 && ./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "What is the capital of France?" \
  --max-tokens 12 --temperature 0.0 --threads 4
```

Expected output must contain: `The capital of France is Paris.`

Also test Germany:
```bash
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "What is the capital of Germany?" \
  --max-tokens 12 --temperature 0.0 --threads 4
```
Expected output must contain: `Berlin`

**If the build breaks or output is garbled → REVERT IMMEDIATELY before continuing.**

Do not stack more changes on a broken state. Revert with:
```bash
git diff src/  # inspect what changed
git checkout src/path/to/changed/file.c  # revert specific file
```

---

## Rule 3 — DYNAMIC BEHAVIOR, BEST PERFORMANCE

**The system must always select the best available path at runtime.**

- Weight format (F32, Q4_K, Q5_1, ternary) is detected at load time from the
  GGUF tensor type field. Never assume a format.
- SIMD level (scalar, AVX2, AVX512, VNNI) is detected at startup via CPUID.
  Never `#ifdef __AVX512F__` as the only dispatch — always have a runtime
  fallback path.
- Thread count is set by `--threads` CLI arg. Kernels must work correctly
  for T=1 through T=8 (and beyond) without race conditions.
- KV cache strategy is selected based on available free RAM and model context
  length. Never hardcode `max_seq_len = 163840` when the system has 16 GB RAM.
- Classifier path (BF16/INT8/INT4) is selected based on `--classifier` CLI
  arg or auto-detected for best bandwidth utilization.

---

## Rule 4 — VERIFY OUTPUT ACCURACY, NOT JUST RUNNING

**"It runs" is not the same as "it is correct".**

After every change, verify:
1. Output is semantically correct (e.g., Paris for France, Berlin for Germany)
2. Output is NOT garbled (no `!!!!!!!!`, `groupe groupe groupe`, random bytes)
3. EOS token causes generation to stop at the right place
4. Expert utilization is healthy: ALL 27 layers should show expert invocations,
   not just L01. Dead layers L02-L26 means NaN is propagating from earlier.

Signs of silent corruption to watch for:
- Only the first 1-2 layers show activity in expert stats
- Output repeats the same token endlessly
- `inf` or `nan` in debug tensor dumps
- tok/s drops to near 0 (stuck in a loop or crash-loop)

---

## Rule 5 — NO PERFORMANCE REGRESSION

**After every change, verify tok/s and IPC have not regressed.**

Baseline (established on this machine, 4 threads, DeepSeek-V2-Lite Q4K_S):
- Minimum acceptable: ~0.9 tok/s at T=4 (current F32 dequant path)
- After MLA Q4K zero-copy: target ≥ 1.5 tok/s at T=4
- llama.cpp reference (T=4): check `tools/deepseek_bench_perf.sh` for current numbers

Quick performance check:
```bash
./adaptive_ai_engine \
  --model models/deepseek-v2-lite-chat-Q4_K_S.gguf \
  --prompt "Explain the theory of relativity in detail." \
  --max-tokens 64 --temperature 0.0 --threads 4 2>&1 | grep "tok/s"
```

For full thread sweep (T=1..8) with IPC:
```bash
bash tools/deepseek_bench_perf.sh 2>&1 | tee benchmark_results/latest_perf.csv
```

Regression threshold: >5% drop in tok/s from previous commit baseline = REVERT.

---

## Rule 6 — DOCUMENT AT EACH STEP, NOT AT THE END

**Update docs when you change code. Never batch documentation for "later".**

Required updates at each milestone:
- `DEBUGGING_JOURNAL.md` — add a dated entry for every fix attempt, whether
  it succeeded or failed. Future sessions rely on this to avoid re-discovering
  the same bugs.
- `BENCHMARK_REPORT.md` — append results after every benchmark run, with
  thread count, SIMD mode, classifier mode, and measurement methodology.
- `docs/PERFORMANCE_CEILING_REPORT.md` — update the hot-path analysis table
  when a new kernel path is added or changed.
- `CPU_LLM_TERNARY_ENGINE.md` — update the feature/status table when a
  capability changes from ⏳ planned to ✅ working.
- Commit after each milestone. Commit message format:
  `<subsystem>: <what changed and why>` — one line summary, then details.
  Always include: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`

**Memory compaction warning**: AI agents lose context between sessions. The
docs ARE the memory. If it is not written down and committed, it is lost.

---

## Rule 7 — SEARCH REFERENCE IMPLEMENTATIONS FIRST

**Before implementing a fix, search llama.cpp and bitnet.cpp for how they do it.**

Local reference implementations:
- `~/llama.cpp/src/models/deepseek2.cpp` — DeepSeek-V2 attention, FFN, MoE
- `~/llama.cpp/ggml/src/ggml-common.h` — Q4_K, Q5_K, Q6_K block structs
- `~/llama.cpp/ggml/src/ggml-quants.c` — dequantization reference kernels
- `~/llama.cpp/tools/llama-bench/` — benchmark methodology

Online references to search before implementing:
- llama.cpp: https://github.com/ggml-org/llama.cpp
- bitnet.cpp: https://github.com/microsoft/BitNet
- ggml quantization: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c

Key principle from llama.cpp: **never dequantize weights upfront**.
Pass raw quantized tensor pointer to `ggml_mul_mat`. It dispatches to the
right kernel based on `tensor->type` at compute time. Our engine must do the
same: detect the type at load time, store the raw pointer, dispatch at compute
time.

---

## Historical Regressions (Learn From These)

| Date | Change | Symptom | Root Cause | Fix |
|------|--------|---------|------------|-----|
| 2026-03-23 | INT8/INT4 classifier | Garbled output with `--classifier int8/int4` | `weights_build_classifier_quant()` read from `token_embedding_table` but DeepSeek has separate `output.weight` → wrong logits | Use `w->wcls` in both INT8 and INT4 build loops |
| 2025 | Added MLA Q4K dispatch | `!!!!!!!!!!!` output, layers L02-L26 dead | F32 path still called on raw Q4K bytes → NaN propagation | Revert; implement dispatch flag correctly |
| 2025 | KV strategy | OOM on 16 GB machine | `max_seq_len=163840` × Q4K size = 36 GB alloc | Add RAM cap in kv_strategy.c |
| Earlier | Q4_K dequant | `groupe groupe groupe` repeat | Wrong nibble grouping in `gguf_dequant_q4_k()` | Commit 9801018 |
| Earlier | MLA RoPE | Semantically wrong output | Wrong freq buffer (`s->rope_freq` vs `s->mla_rope_freq`) | Commit 4e6fdfe |
| Earlier | kq_scale/YaRN | Subtle output degradation | Divergence from llama.cpp YaRN mscale formula | Commit 4e6fdfe |

---

## Quick Sanity Checklist Before Any Commit

```
[ ] make -j4 succeeds with no errors
[ ] France → Paris output correct
[ ] Germany → Berlin output correct
[ ] All 27 layers show expert activity (not just L01)
[ ] tok/s >= baseline for T=4
[ ] DEBUGGING_JOURNAL.md updated with what changed
[ ] No hardcoded tensor names, dimensions, or model-specific constants
[ ] New flag/field added to weights.h, NOT scattered if-chains
[ ] Tested with T=1, T=4, T=8 (at minimum)
```

---

## File Map (Read These In Order)

1. **`GOLDEN_RULES.md`** ← YOU ARE HERE
2. **`README.md`** — quick start, model support, CLI reference
3. **`DEVELOPER_ONBOARDING.md`** — testing mandate, phase structure, packed weights
4. **`DEBUGGING_JOURNAL.md`** — full chronological fix history (most recent first)
5. **`docs/PERFORMANCE_CEILING_REPORT.md`** — hot-path analysis, where time is spent
6. **`BENCHMARK_REPORT.md`** — all benchmark results with methodology
7. **`CPU_LLM_TERNARY_ENGINE.md`** — feature status table, architecture overview
8. **`IMPLEMENTATION_PLAN.md`** — roadmap and phase tracking

---

*Last updated: 2026-03-23. This file must be updated when any rule changes.*
*Any AI agent reading this: these rules override any optimization instinct.*
*Correctness first. Performance second. Modularity always.*

---

## Rule 8 — DOCUMENT ALL TEST PARAMETERS EXPLICITLY

**"It works" is meaningless without recording every flag, including defaults.**

Every test result MUST record the exact command with ALL flags (including `--simd`, `--classifier`,
`--temperature`, `--threads`), the exact output text, and the tok/s.

**Why this matters — the regression that caused weeks of wasted bisection (2026-03-23):**

A working result was documented without recording all flags. The actual working command used
no `--simd`/`--classifier` flags (defaulting to bf16 classifier). Later tests used
`--simd avx512f --classifier int8` — a completely different, broken code path.
Multiple sessions bisected commits looking for the regression. The bug was the INT8 classifier
reading from the wrong weight matrix (`token_embedding_table` instead of `wcls`), present
since the classifier was added, invisible because nobody ever tested it with the right command.

**Mandatory test result template:**
```
Date: YYYY-MM-DD HH:MM UTC | Commit: <SHA>
Command: ./adaptive_ai_engine --model <path> --prompt "<text>" \
         --max-tokens N --temperature T --threads N --simd <mode> --classifier <mode>
Output:  <exact text>
tok/s:   X.XX tok/s (N tokens)
```

---

---

## Addendum: Pre-Benchmark Environment Setup (2026-03-23)

Before running any benchmark, these steps are MANDATORY or results will be 0.38–0.57 tok/s
(7.3× below correct baseline) due to disk I/O stalls and kernel memory migration interference.

```bash
# 1. Set CPU governor to performance
echo "<YOUR_SUDO_PASSWORD>" | sudo -S bash -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done'

# 2. Disable THP (prevents try_to_migrate stealing 14% CPU)
echo "<YOUR_SUDO_PASSWORD>" | sudo -S bash -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'

# 3. Pin model in buffer cache (eliminates wa=11–18% disk wait during inference)
echo "<YOUR_SUDO_PASSWORD>" | sudo -S vmtouch -t models/deepseek-v2-lite-chat-Q4_K_S.gguf

# 4. Free swap if > 500 MB is in use
free -m  # check swap usage; if high, restart apps or: sudo swapoff -a && sudo swapon -a
```

**Without step 3, results are invalid.** mmap page faults dominate inference time when
the 9.5 GB model is not resident in buffer cache. This was the root cause of all
prior "regression" reports showing 0.38–0.57 tok/s.

