#ifndef TN_CLI_MODEL_LOAD_H
#define TN_CLI_MODEL_LOAD_H

#include <stdbool.h>
#include "core/config.h"
#include "core/error.h"
#include "core/gguf_loader.h"
#include "core/gguf_reader.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "memory/mapped_file.h"
#include "threading/thread_pool.h"

/**
 * Phase 18 (speculative decoding), Stage 3: the GGUF model-load sequence
 * factored out of main() so it can be reused for a second (draft) model —
 * mmap -> GGUF header -> Config/MoEConfig -> weight allocation/load ->
 * KV-strategy sizing -> RunState (+ MLA/Qwen3.5/Qwen3-MoE state) allocation.
 *
 * Every field a caller needs to keep the loaded model alive and usable is
 * bundled here; `state` is heap-allocated (mirrors main()'s existing
 * `RunState *s = malloc(...)` pattern) so callers hold a stable pointer.
 */
typedef struct {
    Config config;
    TransformerWeights weights;
    MoEConfig moe_config;
    RunState *state;
    MappedFile mf;
    GGUFHeader gguf_hdr;
    GGUFWeightStore *gguf_store; /* NULL if weights_from_gguf() didn't need one */
} LoadedModel;

/**
 * Loads a GGUF model end-to-end. Native (raw ternary `.bin`) models are NOT
 * handled here — that format has no speculative-decoding draft-model use
 * case (Phase 18 targets GGUF-only draft models), so main()'s existing
 * native-`.bin` branch stays as its own separate inline path.
 *
 * @param out              Populated on TN_OK; left in an unspecified but
 *                          safe-to-ignore state on error (nothing needs
 *                          freeing on failure — this function cleans up
 *                          after itself on every error path).
 * @param model_path       Path to the `.gguf` file to load.
 * @param is_primary        true for the main/verifier model: prints the
 *                          usual load-progress/config/profiler output and
 *                          sizes the KV cache from currently-free RAM via
 *                          select_kv_strategy() (tn_hardware_profile_set_model_bytes()
 *                          is also only ever called for the primary model —
 *                          it measures *this* load's bytes/token for the
 *                          profiler's ceiling display, which must reflect
 *                          the verifier, not a draft model loaded alongside
 *                          it). false for a secondary (draft) model: loads
 *                          silently and sizes its context from
 *                          `max_seq_len_hint` instead of probing free RAM —
 *                          a draft model never needs a large context, and
 *                          two independent free-RAM probes during the same
 *                          startup would double-count the same headroom.
 * @param max_seq_len_hint  Only consulted when is_primary is false: the
 *                          draft model's `Config.seq_len` is clamped to
 *                          this value (clamped to at least 1) instead of
 *                          being sized by select_kv_strategy(). Ignored
 *                          when is_primary is true.
 * @param tp               Thread pool used while loading weights (passed
 *                          through to weights_from_gguf()); not owned by
 *                          this function — the caller creates and destroys
 *                          it independently of any load_gguf_model() call.
 * @param stdout_is_tty    Only consulted when is_primary is true: forwarded
 *                          to tn_progress_stage()/tn_progress_done() for
 *                          the in-place vs. line-per-stage progress output.
 */
TernaryError load_gguf_model(LoadedModel *out, const char *model_path,
                              bool is_primary, int max_seq_len_hint,
                              ThreadPool *tp, bool stdout_is_tty);

/**
 * Frees everything load_gguf_model() allocated into `*m` on a TN_OK return,
 * in the correct order (MLA/Qwen3.5/Qwen3-MoE run-state extras before the
 * generic RunState, weights before the GGUF store/header they may still
 * reference, mmap last) — mirrors main()'s existing end-of-program cleanup
 * sequence. Safe to call on a zero-initialized LoadedModel (e.g. if the
 * caller never successfully loaded one) since every underlying free() is
 * itself NULL/zero-safe.
 */
void loaded_model_free(LoadedModel *m);

#endif /* TN_CLI_MODEL_LOAD_H */
