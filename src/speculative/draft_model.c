#include "speculative/draft_model.h"

#include <stdio.h>
#include <string.h>

#include "tokenizer/tokenizer_gguf.h"

TernaryError draft_model_load(DraftModel *dm, const char *draft_model_path,
                               int verifier_vocab_size, int max_seq_len_hint,
                               ThreadPool *tp) {
    memset(dm, 0, sizeof(*dm));

    LoadedModel lm;
    TernaryError err = load_gguf_model(&lm, draft_model_path, /*is_primary=*/false,
                                        max_seq_len_hint, tp, /*stdout_is_tty=*/false);
    if (err != TN_OK) {
        fprintf(stderr, "Failed to load draft model '%s': %s\n",
                draft_model_path, tn_error_str(err));
        return err;
    }

    if (lm.moe_config.has_linear_attn) {
        fprintf(stderr, "Draft model '%s' uses linear attention (Qwen3.5/3.6 hybrid), "
                "which speculative decoding's batched verify pass does not support.\n",
                draft_model_path);
        loaded_model_free(&lm);
        return TN_ERR_UNSUPPORTED;
    }

    if (lm.config.vocab_size != verifier_vocab_size) {
        fprintf(stderr, "Draft model '%s' vocab_size (%d) does not match verifier "
                "vocab_size (%d) -- draft and verifier must share a tokenizer for "
                "token-level accept/reject to be meaningful.\n",
                draft_model_path, lm.config.vocab_size, verifier_vocab_size);
        loaded_model_free(&lm);
        return TN_ERR_INVALID_ARGS;
    }

    dm->config      = lm.config;
    dm->weights     = lm.weights;
    dm->moe_config  = lm.moe_config;
    dm->state       = lm.state;
    dm->mf          = lm.mf;
    dm->gguf_hdr    = lm.gguf_hdr;
    dm->gguf_store  = lm.gguf_store;

    TernaryError tok_err = tokenizer_load_from_gguf(&dm->tokenizer, &dm->gguf_hdr);
    if (tok_err != TN_OK) {
        /* Non-fatal: speculative_generate()'s accept/reject loop compares
         * token IDs directly and never decodes through the draft model's own
         * tokenizer -- the verifier's tokenizer (already loaded by the
         * caller) remains authoritative for output text/EOS/chat-template. */
        fprintf(stderr, "[draft-model] Warning: could not load draft model's own "
                "tokenizer metadata (err=%d) -- continuing without it.\n", (int)tok_err);
    }

    printf("[draft-model] Loaded '%s' (%d layers, vocab_size=%d) for speculative decoding\n",
           draft_model_path, dm->config.n_layers, dm->config.vocab_size);

    return TN_OK;
}

void draft_model_free(DraftModel *dm) {
    tokenizer_free(&dm->tokenizer);
    if (!dm->state) return;

    LoadedModel lm;
    lm.config     = dm->config;
    lm.weights    = dm->weights;
    lm.moe_config = dm->moe_config;
    lm.state      = dm->state;
    lm.mf         = dm->mf;
    lm.gguf_hdr   = dm->gguf_hdr;
    lm.gguf_store = dm->gguf_store;
    loaded_model_free(&lm);
    dm->state = NULL;
}
