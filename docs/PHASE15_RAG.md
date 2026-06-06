# Phase 15 — Local RAG (Retrieval-Augmented Generation)

**Branch:** `claude/next-development-increment-Q8iRj`
**Date:** 2026-03-16
**Author:** Claude Code (Sonnet 4.6)
**Depends on:** Phase 14 (Agentic Tool Execution)

---

## Overview

Phase 15 gives Project Zero **persistent memory**.  Without it, every
conversation starts with a blank slate — the model forgets everything the
moment the process exits.  RAG solves this by storing text snippets as
numerical embedding vectors in a compact binary file.  On every new prompt,
the engine silently searches that file, finds the most relevant past
memories, and quietly injects them into the model's context window before
generation begins.

### What the user experiences

| Situation | Before Phase 15 | After Phase 15 |
|---|---|---|
| Tell the model your name | Forgotten on restart | Remembered across sessions |
| Model writes `<save_memory>…</save_memory>` | Logged as no-op | Fact embedded + stored to disk |
| Model writes `<search_memory>…</search_memory>` | Logged as no-op | Top memories injected into context |
| Start a new conversation | Blank slate | Auto-retrieved relevant memories prepended |
| Duplicate save | Stored again (waste) | Deduplicated via 0.95 cosine threshold |

---

## Architecture

```
User prompt
     │
     ▼
auto_retrieve_and_inject()          ← Phase 15.5
  embedder_generate(prompt) → vec
  rag_find_top_k(vec, db)   → top-3
  inject_text_into_kv()
     │
     ▼
generate() / run_agent_loop()       ← Phases 12, 14
  ...model produces tokens...
     │
     ├── TAG_SAVE_MEMORY detected
     │       └─ auto_save_memory()  ← Phase 15.6
     │              embedder_generate() → vec
     │              dedup check (cos ≥ 0.95?)
     │              vector_db_store()
     │
     └── TAG_SEARCH_MEMORY detected
             └─ memory_search()     ← Phase 15.4
                    embedder_generate() → vec
                    rag_find_top_k()    → results
                    inject_text_into_kv()
```

### New modules

| Module | Header | Source | Purpose |
|---|---|---|---|
| Embedder | `include/rag/embedder.h` | `src/rag/embedder.c` | Text → L2-normalised float vector |
| Similarity | `include/rag/similarity.h` | `src/rag/similarity.c` | Cosine similarity + top-k scan |
| Vector DB | `include/rag/vector_db.h` | `src/rag/vector_db.c` | File-backed persistent store |
| Memory Search | `include/rag/memory_search.h` | `src/rag/memory_search.c` | High-level search API |
| Auto Retrieve | `include/rag/auto_retrieve.h` | `src/rag/auto_retrieve.c` | Pre-generation injection |
| Auto Save | `include/rag/auto_save.h` | `src/rag/auto_save.c` | Post-tag save hook |
| RAG Context | `include/rag/rag_context.h` | _(header only)_ | Bundle: VectorDB + Embedder |

---

## Sub-task Log

### ST-5: Build and engine test results

**Build**: All 57 existing `.c` files + 6 new `src/rag/*.c` files compile with
`-Wall -Wextra -Wpedantic -std=c99` with zero errors.  Only pre-existing
warnings remain (unrelated to Phase 15).

**Test results**:

| Suite | Tests | Status |
|---|---|---|
| `test_rag` (NEW) | 90 | ✅ pass |
| `test_math` | 33 | ✅ pass |
| `test_sampling` | 2,510 | ✅ pass |
| `test_tokenizer` | 44 | ✅ pass |
| `test_threading` | 170 | ✅ pass |
| `test_kv_cache` | 122 | ✅ pass |
| `test_forward` | 378 | ✅ pass |
| `test_reasoning` | 32 | ✅ pass |
| `test_simd` | 23 | ✅ pass |
| `test_mmap` | 7 | ✅ pass |
| `test_cmd_exec` | 5 | ✅ pass |
| `test_tool_interceptor` | 6 | ✅ pass |
| `audit_math` | 284 | ✅ pass |
| `audit_sliding_window_crash` | 1 | ✅ pass |
| `audit_threadpool_stress` | 2 | ✅ pass |
| `test_blackbox` | 19 | ✅ pass |
| `test_bugfixes` | 35 | ✅ pass |
| `test_redbox` | 8 | ✅ pass |
| `test_vision_components` | — | ⚠️ pre-existing (missing image file) |
| **TOTAL** | **3,742** | **all pass** |

**Engine smoke test**:

```
$ ./adaptive_ai_engine --model /tmp/test_tiny_model.bin \
    --tokenizer /tmp/test_tiny_tokenizer.bin \
    --memory-db /tmp/test_memory.vrdb \
    --prompt "hello" --max-tokens 3
Project Zero Engine (Phase 15 — RAG Memory)
SIMD Backend: AVX-512
[RAG] Memory enabled — 0 entries in '/tmp/test_memory.vrdb'
```

- The engine loads the model, initialises RAG, creates a `VRDB` file with
  the correct 16-byte header (magic `0x42445256`, version 1, dim 64).
- A second run re-opens the existing DB with `0 entries` confirmed.
- `--memory-db` is optional; omitting it disables RAG silently.

### ST-4: CLI integration

Changes to existing files:

| File | Change |
|---|---|
| `include/cli/args.h` | Added `char *memory_db_path` field |
| `src/cli/args.c` | Parse `--memory-db <path>`; print in `--help` |
| `include/cli/repl.h` | Added `RagContext *rag` parameter |
| `src/cli/repl.c` | Auto-retrieve before each generation; `/memory list`; `/memory search <q>`; agent dispatch passes `rag` |
| `src/cli/main.c` | RAG init block: `vector_db_open` + `embedder_init` if `--memory-db`; cleanup on exit |

New REPL commands (Phase 15):

```
/memory list              — show all stored memories
/memory search <query>    — manually search and display results
```

### ST-3: Agent loop wiring

`src/agent/agent_loop.c` was rewritten to:

1. Accept a `RagContext *rag` parameter (NULL = disabled).
2. Fix the previously non-reentrant `strtok` → `strtok_r` (addresses
   security gap P0 from the analysis report).
3. `TAG_SAVE_MEMORY` → `handle_save_memory()` → `auto_save_memory()`.
4. `TAG_SEARCH_MEMORY` → `handle_search_memory()` → `memory_search()` +
   `inject_text_into_kv()` with formatted results.
5. Both handlers are no-ops if `rag == NULL` (graceful degradation).

### ST-2: RAG source files implemented

Six `.c` files created under `src/rag/`:

| File | Lines | Key design note |
|---|---|---|
| `embedder.c` | ~80 | Resets embed RunState on every call; accumulates `s->x` post-layernorm |
| `similarity.c` | ~60 | Insertion-sort top-k; O(n·k) scan; delegates hot inner loop to `tn_vec_dot` |
| `vector_db.c` | ~180 | Append-only binary file; all records loaded into heap on open; pool-based text storage avoids N separate mallocs |
| `memory_search.c` | ~55 | Thin glue: embed → top-k → filter by MIN_SCORE |
| `auto_retrieve.c` | ~70 | Budget check first; formats `<memory>…</memory>` block; calls `inject_text_into_kv` |
| `auto_save.c` | ~50 | Dedup scan (O(n·dim)) before store; returns 0/1/-1 |

Notable implementation details:

- **KV cache reset in embedder**: `memset` zeroes key/value cache slabs before
  each embed call.  This is correct because the embedding RunState is small
  (512 tokens) and always processes a fresh independent sequence.
- **Pool-based text storage**: `vector_db.c` uses a single `realloc`-growing
  char buffer (`text_pool`) rather than one `malloc` per string.  All `texts[i]`
  pointers point into this pool.  When the pool is reallocated, all pointers are
  fixed up by the delta.
- **Header rewrite on every store**: `write_header()` seeks to offset 0 and
  writes 16 bytes — this keeps `num_entries` consistent even if the process
  crashes mid-append.
- **`tn_aligned_calloc` for query/embedding vectors**: Keeps heap allocations
  64-byte aligned for potential SIMD use in `tn_vec_dot`.

### ST-1: RAG headers created

All six module headers (`embedder.h`, `similarity.h`, `vector_db.h`,
`memory_search.h`, `auto_retrieve.h`, `auto_save.h`) plus the convenience
`rag_context.h` bundle were written under `include/rag/`.

Key design decisions made at this stage:

- **Embedding strategy**: Mean-pool `s->x` (post-final-layernorm hidden
  state) across all token positions, then L2-normalise.  No second model
  needed — the same LLM does embeddings.
- **Dedicated RunState**: `Embedder` owns a small `RunState` with
  `max_seq_len = EMBEDDER_MAX_SEQ (512)`.  Embedding calls never touch the
  main conversation's KV cache.
- **Vector DB file format**: Append-only binary.  16-byte header (magic
  `VRDB`, version, num_entries, embed_dim) followed by variable-length
  records `[embed_dim×float32 | uint32 text_len | text bytes]`.
- **Cosine similarity == dot product**: Since embeddings are pre-normalised
  to unit length, `cos(a,b) = a·b`.  Uses `tn_vec_dot` from the existing
  SIMD dispatch table.
- **RagContext bundle**: `VectorDB + Embedder + enabled` flag passed as one
  pointer to keep function signatures clean.

---

## Usage Guide

### Enable memory

```bash
./adaptive_ai_engine \
    --model bitnet.bin \
    --tokenizer tokenizer.bin \
    --memory-db ~/.project_zero_memory.vrdb
```

The `.vrdb` file is created automatically if it does not exist.

### REPL commands

| Command | Effect |
|---|---|
| `/memory list` | Print all stored memories with their index |
| `/memory search <query>` | Search and display top-5 most relevant memories |
| `/agent <prompt>` | Run agentic mode (save/search memory tags active) |

### Agent tags

In agentic mode (`/agent <prompt>`), the model can emit:

```
<save_memory>User's name is Alice.</save_memory>
```
→ Generates an embedding, checks for duplicates (cos ≥ 0.95), stores to disk.

```
<search_memory>What is the user's name?</search_memory>
```
→ Generates a query embedding, finds top-3 matches (score ≥ 0.5),
injects `<result>…</result>` block into context.

### Transparent retrieval

On every REPL prompt, `auto_retrieve_and_inject()` runs automatically:
1. Embeds the user prompt.
2. Finds top-3 DB entries with similarity ≥ 0.60.
3. If any found, injects `<memory>…</memory>` block into KV cache before
   `generate()` runs.

Pass `--verbose` to see `[Memory] Injected N tokens of relevant context.`
messages.

---

## Security and Known Limitations

### Security notes

- The `strtok` → `strtok_r` fix (ST-3) closes security gap P0 from the
  `PROJECT_ANALYSIS_REPORT.md` (non-reentrant command parsing in agent loop).
- The RAG layer inherits the existing `user_approval.c` gate for `<exec>`
  tags.  The `<save_memory>` tag has no approval gate — it only writes to
  a local file owned by the user, so this is appropriate.
- **SQL injection equivalent**: A malicious prompt that tricks the model into
  emitting a `<save_memory>` tag with adversarial content could pollute the
  DB.  Mitigation is the same as Phase 14: use the auto-approve env var only
  in trusted contexts.

### Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| `embedder_generate` uses the full LLM forward pass for each embed | ~same cost as generating one token per text byte | Only called on memory writes and retrievals, not during normal generation |
| Brute-force O(n·dim) search | Scales linearly with DB size | Acceptable for < 100K entries; HNSW index could replace for larger DBs |
| No encryption of the `.vrdb` file | Private memories stored in plaintext | File is local-only; OS filesystem permissions apply |
| `VRDB_MAX_ENTRIES = 65536` hard cap | DB limited to 64K memories | Configurable at compile time; sufficient for years of use |

---

## Files Changed Summary

### New files

```
include/rag/embedder.h
include/rag/similarity.h
include/rag/vector_db.h
include/rag/memory_search.h
include/rag/auto_retrieve.h
include/rag/auto_save.h
include/rag/rag_context.h
src/rag/embedder.c
src/rag/similarity.c
src/rag/vector_db.c
src/rag/memory_search.c
src/rag/auto_retrieve.c
src/rag/auto_save.c
tests/test_rag.c
docs/PHASE15_RAG.md
```

### Modified files

```
include/agent/agent_loop.h    — added RagContext* param
src/agent/agent_loop.c        — wired save/search; strtok→strtok_r
include/cli/args.h            — added memory_db_path
src/cli/args.c                — --memory-db flag
include/cli/repl.h            — added RagContext* param
src/cli/repl.c                — /memory commands; auto_retrieve call
src/cli/main.c                — RAG init/cleanup; banner update
```

