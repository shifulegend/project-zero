#ifndef TN_TOKENIZER_GGUF_H
#define TN_TOKENIZER_GGUF_H

/*
 * tokenizer_gguf.h — Universal GGUF Tokenizer Loader (Phase 34.5)
 *
 * Loads any tokenizer embedded in a GGUF model file, eliminating the need
 * for an external .bin tokenizer file.
 *
 * Supported types (auto-detected from tokenizer.ggml.model):
 *   "gpt2"  → BPE (byte-pair encoding), uses tokenizer.ggml.merges for scores
 *   "llama" → SPM (SentencePiece), uses tokenizer.ggml.scores
 *   "bert"  → WPM (WordPiece), uses tokenizer.ggml.scores
 *   "t5"    → UGM (Unigram), uses tokenizer.ggml.scores
 *   "rwkv"  → RWKV trie, scores=0
 *   others  → treat as SPM with zero scores
 */

#include "core/error.h"
#include "core/gguf_reader.h"
#include "tokenizer/tokenizer.h"

/**
 * Load a tokenizer from the GGUF header metadata.
 *
 * Reads tokenizer.ggml.tokens (vocab strings), tokenizer.ggml.scores (SPM),
 * tokenizer.ggml.merges (BPE), and special token IDs.
 *
 * Returns TN_OK on success. On failure, t may be partially initialized —
 * caller should call tokenizer_free() if this function returns non-TN_OK
 * and t->vocab is non-NULL.
 *
 * @param t    Tokenizer to initialize (must be zero-initialized before call)
 * @param hdr  Parsed GGUF header (must remain valid for lifetime of t)
 */
TernaryError tokenizer_load_from_gguf(Tokenizer *t, const GGUFHeader *hdr);

#endif /* TN_TOKENIZER_GGUF_H */
