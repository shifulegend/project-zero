/*
 * tokenizer_gguf.c — Universal GGUF Tokenizer Loader (Phase 34.5)
 *
 * Reads tokenizer vocab, scores, and merge rules directly from the GGUF model
 * file's metadata arrays. Eliminates dependency on an external .bin file.
 *
 * Supported tokenizer types (auto-detected from tokenizer.ggml.model):
 *   "gpt2"  → BPE: merges array gives rank ordering (lower index = higher priority)
 *   "llama" → SPM: scores array gives log-prob weights
 *   "bert"  → WPM: scores array; [UNK], [CLS], [SEP] handled as normal tokens
 *   "t5"    → UGM: scores array; same path as SPM
 *   "rwkv"  → RWKV trie: no scores; all 0.0f
 *   others  → treated as SPM with zero scores
 *
 * Merge scores for BPE: vocab_scores[merged_token_id] = -(float)merge_rank
 * so that lower merge ranks (higher priority merges) get higher (less negative) scores.
 * All tokens start at score 0.0f; only explicitly merged tokens get a non-zero score.
 *
 * GGUF string array format:
 *   Each element: [len: uint64_t][bytes: char[len]]   (no NUL in file)
 *
 * GGUF float32 array format:
 *   Packed float32 values — direct pointer cast after verifying elem_type.
 */

#include "tokenizer/tokenizer_gguf.h"
#include "core/gguf_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Internal sort (duplicated from tokenizer_load.c for independence) ──── */

static void swap_int_g(int *a, int *b) { int t = *a; *a = *b; *b = t; }

static void vocab_isort(int *arr, int n, char **vocab) {
    for (int i = 1; i < n; i++) {
        int key = arr[i], j = i - 1;
        while (j >= 0 && strcmp(vocab[arr[j]], vocab[key]) > 0) {
            arr[j + 1] = arr[j]; j--;
        }
        arr[j + 1] = key;
    }
}

static void vocab_qsort(int *arr, int n, char **vocab) {
    while (n > 16) {
        int mid = n / 2;
        if (strcmp(vocab[arr[0]], vocab[arr[mid]])   > 0) swap_int_g(&arr[0], &arr[mid]);
        if (strcmp(vocab[arr[0]], vocab[arr[n-1]])   > 0) swap_int_g(&arr[0], &arr[n-1]);
        if (strcmp(vocab[arr[mid]], vocab[arr[n-1]]) > 0) swap_int_g(&arr[mid], &arr[n-1]);
        swap_int_g(&arr[mid], &arr[n-2]);
        int pivot = arr[n-2], i = 0, j = n-2;
        for (;;) {
            while (strcmp(vocab[arr[++i]], vocab[pivot]) < 0) {}
            while (strcmp(vocab[arr[--j]], vocab[pivot]) > 0) {}
            if (i >= j) break;
            swap_int_g(&arr[i], &arr[j]);
        }
        swap_int_g(&arr[i], &arr[n-2]);
        if (i < n - i - 1) { vocab_qsort(arr, i, vocab); arr += i+1; n -= i+1; }
        else                { vocab_qsort(arr+i+1, n-i-1, vocab); n = i; }
    }
    vocab_isort(arr, n, vocab);
}

static void build_sorted(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) t->sorted_vocab_indices[i] = i;
    vocab_qsort(t->sorted_vocab_indices, t->vocab_size, t->vocab);
    t->sorted = 1;
}

/* ── String array walker ────────────────────────────────────────────────── */

/* Walk a GGUF string array at `data`, advancing *cursor past element `idx`.
 * Returns the idx-th string as a malloc'd NUL-terminated copy.
 * Elements must be read sequentially (cursor-style) for O(n) total cost. */
typedef struct {
    const uint8_t *p;     /* current cursor position within mmap */
    uint64_t       count; /* total elements */
    uint64_t       cur;   /* current index */
} StrArrayCursor;

static void str_cursor_init(StrArrayCursor *c, const void *data, uint64_t count) {
    c->p = (const uint8_t *)data;
    c->count = count;
    c->cur = 0;
}

/* Advance cursor by one string element, returning it as a heap-allocated copy. */
static char *str_cursor_next(StrArrayCursor *c) {
    if (c->cur >= c->count) return NULL;
    uint64_t slen;
    memcpy(&slen, c->p, 8);
    c->p += 8;
    char *s = (char *)malloc(slen + 1);
    if (s) {
        memcpy(s, c->p, slen);
        s[slen] = '\0';
    }
    c->p += slen;
    c->cur++;
    return s;
}

/* ── Find a vocab token by exact string match (linear scan) ─────────────── */
/* Used for BPE merge resolution. Linear O(vocab_size) per merge — acceptable
 * since merge resolution is a one-time initialization cost. */
static int find_vocab_token(char **vocab, int vocab_size, const char *s) {
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i] && strcmp(vocab[i], s) == 0) return i;
    }
    return -1;
}

/* ── Main loader ─────────────────────────────────────────────────────────── */

TernaryError tokenizer_load_from_gguf(Tokenizer *t, const GGUFHeader *hdr) {
    /* 1. Detect tokenizer type */
    const char *tok_model = gguf_meta_str(hdr, "tokenizer.ggml.model");
    if (!tok_model) tok_model = "llama"; /* safe default */

    int is_bpe = (strcmp(tok_model, "gpt2") == 0);
    int is_spm = (!is_bpe);  /* SPM, WPM, UGM, RWKV — all use scores array */

    /* 2. Load vocab tokens array */
    GGUFValType elem_type = GGUF_VAL_STRING;
    const void *tokens_data = gguf_meta_array_data(hdr, "tokenizer.ggml.tokens", &elem_type);
    uint64_t    vocab_count = gguf_meta_array_count(hdr, "tokenizer.ggml.tokens");

    if (!tokens_data || vocab_count == 0) {
        fprintf(stderr, "[gguf-tok] ERROR: no tokenizer.ggml.tokens array found in GGUF\n");
        return TN_ERR_INVALID_ARGS;
    }
    if (elem_type != GGUF_VAL_STRING) {
        fprintf(stderr, "[gguf-tok] ERROR: tokenizer.ggml.tokens is not a string array\n");
        return TN_ERR_INVALID_ARGS;
    }

    t->vocab_size = (int)vocab_count;
    t->vocab      = (char **)calloc((size_t)vocab_count, sizeof(char *));
    t->vocab_scores       = (float *)calloc((size_t)vocab_count, sizeof(float));
    t->sorted_vocab_indices = (int *)calloc((size_t)vocab_count, sizeof(int));

    if (!t->vocab || !t->vocab_scores || !t->sorted_vocab_indices) {
        tokenizer_free(t);
        return TN_ERR_OOM;
    }

    /* Walk the string array sequentially with a cursor for O(n) total */
    StrArrayCursor sc;
    str_cursor_init(&sc, tokens_data, vocab_count);
    t->max_token_len = 0;
    for (uint64_t i = 0; i < vocab_count; i++) {
        char *s = str_cursor_next(&sc);
        if (!s) {
            /* OOM or truncated array — fail gracefully */
            tokenizer_free(t);
            return TN_ERR_OOM;
        }
        t->vocab[i] = s;
        int slen = (int)strlen(s);
        if (slen > t->max_token_len) t->max_token_len = slen;
    }

    /* 3a. SPM path: read scores array */
    if (is_spm) {
        GGUFValType stype = GGUF_VAL_FLOAT32;
        const void *scores_data = gguf_meta_array_data(hdr, "tokenizer.ggml.scores", &stype);
        uint64_t    scores_count = gguf_meta_array_count(hdr, "tokenizer.ggml.scores");

        if (scores_data && stype == GGUF_VAL_FLOAT32 && scores_count == vocab_count) {
            /* Packed float32 in mmap, but NOT necessarily 4-byte aligned —
             * GGUF packs metadata byte-tight with no padding, so this
             * array's start offset depends on whatever preceded it in the
             * file. A raw `((const float *)scores_data)[i]` is an unaligned
             * load, undefined behavior in C (UBSan: "misaligned address...
             * requires 4 byte alignment") and a real crash risk on
             * strict-alignment targets. memcpy into a local is the same
             * fix gguf_reader.c's own read_u32/read_u64 already use for
             * header fields — copies out at whatever alignment, compiles to
             * a single unaligned-load instruction on x86/ARM. */
            const char *bytes = (const char *)scores_data;
            for (uint64_t i = 0; i < vocab_count; i++) {
                float sc;
                memcpy(&sc, bytes + i * sizeof(float), sizeof(float));
                t->vocab_scores[i] = sc;
            }
        } else if (scores_count > 0) {
            fprintf(stderr, "[gguf-tok] WARN: scores array type mismatch or wrong count "
                    "(%llu vs %llu vocab), using zeros\n",
                    (unsigned long long)scores_count, (unsigned long long)vocab_count);
            /* scores remain zero-initialized */
        }
        /* If no scores array: all zeros (valid for uniform-weight SPM or RWKV) */
    }

    /* 3b. BPE path: read merges array, compute scores from merge ranks */
    if (is_bpe) {
        GGUFValType mtype = GGUF_VAL_STRING;
        const void *merges_data = gguf_meta_array_data(hdr, "tokenizer.ggml.merges", &mtype);
        uint64_t    merges_count = gguf_meta_array_count(hdr, "tokenizer.ggml.merges");

        if (merges_data && mtype == GGUF_VAL_STRING && merges_count > 0) {
            /* scores start at 0.0f; merged tokens get -(float)rank so lower-rank
             * merges (higher priority) sort before higher-rank merges in BPE. */
            StrArrayCursor mc2;
            str_cursor_init(&mc2, merges_data, merges_count);

            for (uint64_t rank = 0; rank < merges_count; rank++) {
                char *merge_str = str_cursor_next(&mc2);
                if (!merge_str) break;

                /* Merge format: "token_a token_b" separated by a single space */
                char *space = strchr(merge_str, ' ');
                if (!space) { free(merge_str); continue; }

                /* Build merged token string by concatenating token_a + token_b */
                size_t la = (size_t)(space - merge_str);
                size_t lb = strlen(space + 1);
                size_t lm = la + lb;
                char *merged = (char *)malloc(lm + 1);
                if (merged) {
                    memcpy(merged, merge_str, la);
                    memcpy(merged + la, space + 1, lb);
                    merged[lm] = '\0';

                    int id = find_vocab_token(t->vocab, t->vocab_size, merged);
                    if (id >= 0) {
                        /* Only update if not already set (keep highest-priority merge) */
                        if (t->vocab_scores[id] == 0.0f)
                            t->vocab_scores[id] = -(float)rank;
                    }
                    free(merged);
                }
                free(merge_str);
            }
        } else {
            fprintf(stderr, "[gguf-tok] WARN: BPE tokenizer but no tokenizer.ggml.merges "
                    "array — BPE merge scores will be zero\n");
        }
    }

    /* 4. Build sorted index for binary-search encode */
    build_sorted(t);

    /* 5. Read chat template and special token metadata
     * Use gguf_meta_find + exact length copy to get the full string from the
     * model exactly as llama.cpp does — no reliance on NUL termination. */

    /* chat_template — copy exact bytes from the GGUF metadata string */
    {
        const GGUFMeta *m = gguf_meta_find(hdr, "tokenizer.chat_template");
        if (m && m->val_type == GGUF_VAL_STRING && m->val.string.len > 0) {
            /* string.str is already NUL-terminated (gguf_reader allocates it so),
             * but we use string.len to be explicit about the true length. */
            t->chat_template = (char *)malloc(m->val.string.len + 1);
            if (t->chat_template) {
                memcpy(t->chat_template, m->val.string.str, m->val.string.len);
                t->chat_template[m->val.string.len] = '\0';
            }
        }
    }

    /* BOS / EOS token IDs — prefer tokenizer.ggml.*_token_id; fall back to
     * searching the vocab for the BOS/EOS token strings */
    t->bos_token_id = (int)gguf_meta_u32(hdr, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
    t->eos_token_id = (int)gguf_meta_u32(hdr, "tokenizer.ggml.eos_token_id", (uint32_t)-1);

    /* add_bos_token boolean */
    {
        const GGUFMeta *abm = gguf_meta_find(hdr, "tokenizer.ggml.add_bos_token");
        if (abm && abm->val_type == GGUF_VAL_BOOL)
            t->add_bos_token = abm->val.boolean ? 1 : 0;
        else
            t->add_bos_token = 1; /* default: add BOS */
    }

    /* token_type array → dynamic special tokens list.
     * Tokens with type != 1 (NORMAL) are matched verbatim before BPE.
     * Types: 0=UNDEFINED, 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE */
    {
        GGUFValType tt_elem = GGUF_VAL_UINT32;
        const void *tt_data = gguf_meta_array_data(hdr, "tokenizer.ggml.token_type", &tt_elem);
        uint64_t tt_count   = gguf_meta_array_count(hdr, "tokenizer.ggml.token_type");

        if (tt_data && (tt_elem == GGUF_VAL_INT32 || tt_elem == GGUF_VAL_UINT32)) {
            /* Same unaligned-load hazard as the scores array above — GGUF's
             * byte-tight packing gives no alignment guarantee for this
             * array's start offset, so index via memcpy, not a raw
             * (const int32_t *) cast + subscript. */
            const char *tt_bytes = (const char *)tt_data;
            int cap = 128;
            t->special_tokens    = (char **)malloc((size_t)cap * sizeof(char *));
            t->special_token_ids = (int *)   malloc((size_t)cap * sizeof(int));
            if (t->special_tokens && t->special_token_ids) {
                for (int i = 0; i < t->vocab_size && (uint64_t)i < tt_count; i++) {
                    int32_t ttype;
                    memcpy(&ttype, tt_bytes + (size_t)i * sizeof(int32_t), sizeof(int32_t));
                    /* Include: CONTROL(3), USER_DEFINED(4) — the "added tokens".
                     * Exclude: NORMAL(1), BYTE(6, handled by BPE), UNUSED(5). */
                    if ((ttype == 3 || ttype == 4) && t->vocab[i] && t->vocab[i][0]) {
                        if (t->n_special >= cap) {
                            cap *= 2;
                            t->special_tokens    = (char **)realloc(t->special_tokens,    (size_t)cap * sizeof(char *));
                            t->special_token_ids = (int *)   realloc(t->special_token_ids, (size_t)cap * sizeof(int));
                            if (!t->special_tokens || !t->special_token_ids) break;
                        }
                        t->special_tokens[t->n_special]    = strdup(t->vocab[i]);
                        t->special_token_ids[t->n_special] = i;
                        t->n_special++;
                    }
                }
            }
        }
    }

    /* 6. Log summary */
    fprintf(stderr, "[gguf-tok] Loaded %d tokens (%s), %d special — BOS=%d EOS=%d chat_template=%s\n",
            t->vocab_size, tok_model, t->n_special,
            t->bos_token_id, t->eos_token_id,
            t->chat_template ? "yes" : "no");

    return TN_OK;
}
