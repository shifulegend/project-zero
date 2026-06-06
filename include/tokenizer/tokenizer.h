#ifndef TN_TOKENIZER_H
#define TN_TOKENIZER_H

#include "core/error.h"
#include <stddef.h>

typedef struct {
    /* Core vocabulary (BPE) */
    char **vocab;              /* Array of token strings */
    float *vocab_scores;       /* Merge priority scores (BPE) */
    int vocab_size;
    int max_token_len;         /* Longest token string length */
    int *sorted_vocab_indices; /* Indices sorted by token string for binary search */
    int sorted;                /* Whether sorted index has been built */

    /* Chat template (read from GGUF tokenizer.chat_template, or NULL) */
    char *chat_template;

    /* Special tokens (read from GGUF tokenizer.ggml.{bos,eos}_token_id) */
    int bos_token_id;          /* -1 if not set */
    int eos_token_id;          /* -1 if not set */
    int add_bos_token;         /* 1 = prepend BOS at encode time */

    /* Dynamic special-tokens list built from tokenizer.ggml.token_type.
     * Any token with type != 1 (NORMAL) is matched verbatim before BPE. */
    char **special_tokens;     /* malloc'd array of token strings (owned) */
    int   *special_token_ids;  /* corresponding token IDs */
    int    n_special;          /* count */

    /* All EOS-like token IDs, populated by tokenizer_load via vocab scan.
     * Includes <|eot_id|>, <|end_of_text|>, </s>, <|im_end|>, etc. */
    int eos_list[8];
    int n_eos;
} Tokenizer;

/**
 * Load tokenizer from a binary vocab file.
 * Format: [vocab_size(int)] [max_token_len(int)]
 *         [score(float) len(int) bytes(char*)]...
 */
TernaryError tokenizer_load(Tokenizer *t, const char *path);

/**
 * Binary-search the sorted vocab for a token string.
 * Returns the token ID (index) if found, or -1 if not in vocab.
 * Requires that the sorted index has been built (after tokenizer_load).
 */
int tokenizer_find_id(const Tokenizer *t, const char *str);

/**
 * Free all tokenizer memory.
 */
void tokenizer_free(Tokenizer *t);

/**
 * Encode text into token IDs using BPE.
 * Returns number of tokens written, or negative on error.
 * Tokens are written to `tokens[]`, up to `max_tokens`.
 */
int tokenizer_encode(Tokenizer *t, const char *text, size_t prompt_len, int *tokens, int max_tokens);

/**
 * Decode a single token ID to its string representation.
 * Handles raw byte tokens (e.g., <0x0A> -> newline).
 * If prev_token == -1, leading space is preserved; otherwise
 * leading space is stripped when prev_token is the BOS token.
 */
const char *tokenizer_decode(Tokenizer *t, int prev_token, int token);

#endif /* TN_TOKENIZER_H */
