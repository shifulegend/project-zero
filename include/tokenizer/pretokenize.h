#ifndef TN_PRETOKENIZE_H
#define TN_PRETOKENIZE_H

/*
 * pretokenize.h — BPE pre-tokenization (the regex split that runs before byte
 * pair merges), selected by the GGUF key tokenizer.ggml.pre exactly as
 * llama.cpp's llama-vocab.cpp does. Without it, BPE merges across word,
 * digit and punctuation boundaries the model never saw in training (e.g.
 * "?\n\n" as one token where llama.cpp — and the model's training data —
 * have "?", "\n", "\n"), shifting outputs away from the reference engine.
 *
 * The GPT-2 / Llama-3 / Qwen2 patterns are ports of llama.cpp's hand-written
 * splitters (unicode.cpp unicode_regex_split_custom_*); the extra patterns of
 * the "default" and "smollm" pipelines are simple character-class splits.
 * Character classes (letter / number / whitespace / punctuation) cover ASCII
 * exactly and the common Unicode blocks by range; other code points count as
 * letters, which matches their category for the scripts LLM vocabularies use.
 */

#include <stddef.h>

typedef enum {
    PRETOK_NONE = 0,  /* no split (SentencePiece models, legacy .bin tokenizers) */
    PRETOK_DEFAULT,   /* GGUF without tokenizer.ggml.pre (llama.cpp "default") */
    PRETOK_GPT2,      /* "gpt2", "olmo", "mpt", ... */
    PRETOK_LLAMA3,    /* "llama3", "llama-bpe", "smaug-bpe", "falcon3", ... */
    PRETOK_QWEN2,     /* "qwen2", "stablelm2", "hunyuan" */
    PRETOK_SMOLLM     /* "smollm", "codeshell", "exaone", "minerva" */
} PretokType;

/* tokenizer.ggml.pre value (NULL = key absent) -> type. Unknown names map to
 * DEFAULT, as llama.cpp would refuse/warn; the caller may log it. */
PretokType pretok_type_from_name(const char *pre);
const char *pretok_type_name(PretokType t);

/* Split text[0..len) into fragments. Writes up to max_frags (start, length)
 * byte pairs into starts/lens and returns the fragment count, or -1 when
 * max_frags is too small. Fragments cover the input exactly, in order, and
 * never split a UTF-8 sequence. */
int pretokenize(PretokType type, const char *text, size_t len,
                size_t *starts, size_t *lens, int max_frags);

#endif /* TN_PRETOKENIZE_H */
