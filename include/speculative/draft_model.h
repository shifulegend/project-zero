#ifndef TN_SPECULATIVE_DRAFT_MODEL_H
#define TN_SPECULATIVE_DRAFT_MODEL_H

#include "cli/model_load.h"
#include "core/config.h"
#include "core/error.h"
#include "core/moe_config.h"
#include "core/run_state.h"
#include "core/weights.h"
#include "memory/mapped_file.h"
#include "threading/thread_pool.h"
#include "tokenizer/tokenizer.h"

/**
 * Phase 18 (speculative decoding), Stage 5: a second, smaller GGUF model
 * loaded alongside the verifier. Implemented entirely via
 * cli/model_load.h's load_gguf_model() (is_primary=false) -- zero
 * duplicated GGUF-parsing code.
 */
typedef struct {
    Config config;
    TransformerWeights weights;
    MoEConfig moe_config;
    RunState *state;
    MappedFile mf;
    GGUFHeader gguf_hdr;
    GGUFWeightStore *gguf_store;
    Tokenizer tokenizer; /* best-effort; token-ID-level accept/reject only needs
                            vocab_size to match -- the verifier's own tokenizer
                            remains authoritative for decode/EOS/chat-template */
} DraftModel;

/**
 * Loads a GGUF draft model. Hard-errors (TN_ERR_INVALID_ARGS) if the draft
 * model's vocab_size doesn't match verifier_vocab_size -- draft and
 * verifier token IDs must mean the same thing for token-level accept/reject
 * to be meaningful. Refuses (TN_ERR_UNSUPPORTED) a draft model with
 * has_linear_attn, matching transformer_forward_batch()'s own refusal.
 *
 * @param max_seq_len_hint  Forwarded to load_gguf_model() -- the draft
 *                          model's context is sized from this hint, not a
 *                          free-RAM probe (see cli/model_load.h).
 */
TernaryError draft_model_load(DraftModel *dm, const char *draft_model_path,
                               int verifier_vocab_size, int max_seq_len_hint,
                               ThreadPool *tp);

/** Frees everything draft_model_load() allocated. Safe on a zeroed DraftModel. */
void draft_model_free(DraftModel *dm);

#endif /* TN_SPECULATIVE_DRAFT_MODEL_H */
