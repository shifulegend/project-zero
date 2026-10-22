#ifndef TN_TOKENIZER_H
#define TN_TOKENIZER_H

#include "core/error.h"
#include <stddef.h>

#define TN_TOKENIZER_MAX_EOG 16

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
    unsigned char *special_is_control; /* 1 = GGUF CONTROL type (e.g. <|im_start|>), 0 = USER_DEFINED */
    int    n_special;          /* count */

    /* End-of-generation token IDs (tokenizer_collect_eog): the declared
     * EOS/EOT/EOM ids plus known end-of-turn tokens (<|eot_id|>, <|im_end|>,
     * <end_of_turn>, ...), as llama.cpp's EOG set. */
    int eos_list[TN_TOKENIZER_MAX_EOG];
    int n_eos;

    /* Chat-mode turn ends (tokenizer_set_chat_stops, 2026-10-22): 1 for every
     * CONTROL token the active chat template never writes. NULL = raw mode. */
    unsigned char *chat_stop;

    /* BPE pre-tokenization (tokenizer/pretokenize.h PretokType), from GGUF
     * tokenizer.ggml.pre; 0 = none (SentencePiece, legacy .bin tokenizers). */
    int pretok_type;
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
 * Rebuild t->eos_list: first the given ids (declared EOS/EOT/EOM; negative or
 * out-of-range entries skipped), then every vocab token whose text is a known
 * end-of-turn marker. With require_special, a text match only counts when the
 * token is one of t->special_token_ids (GGUF CONTROL/USER_DEFINED types), so
 * an ordinary vocab entry spelled like a marker never stops generation.
 * Needs the sorted index. Duplicates are dropped; at most TN_TOKENIZER_MAX_EOG.
 */
void tokenizer_collect_eog(Tokenizer *t, const int *ids, int n_ids, int require_special);

/**
 * Chat-mode end of turn (2026-10-22). A CONTROL token (GGUF token_type 3) is
 * structure, never answer text; when the model generates one that the active
 * chat template itself never writes, the assistant's turn is over -- e.g.
 * Falcon3-1.58bit ends turns with ">>UNUSED_1<<" though its metadata declares
 * "<|endoftext|>". Tokens a template uses inside answers (Llama-3.1's
 * <|python_tag|>) and USER_DEFINED added tokens (<think>, <tool_call>) stay
 * generatable. Call at the start of every generation with the template in
 * use, or NULL for raw prompts (then only the declared EOG set stops).
 */
void tokenizer_set_chat_stops(Tokenizer *t, const char *chat_template);

/** 1 if `id` ends generation: declared EOS, the EOG set, or a chat stop. */
int tokenizer_is_eog(const Tokenizer *t, int id);

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
 * Prompt-injection defence (2026-10-22). Returns a malloc'd copy of untrusted
 * text (chat message content, tool output) in which every CONTROL special
 * token string -- "<|im_start|>", "<|eot_id|>", ... -- is replaced by an
 * opaque placeholder. tokenizer_encode() turns a placeholder back into the
 * same characters as plain text, never into the control token, so the text
 * cannot open or close turns of the chat template it is placed into. Template
 * string operations (split, trim) still work on the copy. USER_DEFINED added
 * tokens (<think>, <tool_call>) stay matchable, as with llama.cpp's
 * parse_special = false. Placeholders are code points U+F0000 + index
 * (supplementary private use); NULL on OOM or NULL input.
 */
char *tokenizer_neutralize_control(const Tokenizer *t, const char *text);

/**
 * Decode a single token ID to its string representation.
 * Handles raw byte tokens (e.g., <0x0A> -> newline).
 * If prev_token == -1, leading space is preserved; otherwise
 * leading space is stripped when prev_token is the BOS token.
 */
const char *tokenizer_decode(Tokenizer *t, int prev_token, int token);

#endif /* TN_TOKENIZER_H */
