#include "tokenizer/tokenizer.h"
#include "core/platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Portable context-aware sort for vocabulary indices.
 * NEW-01 fix: eliminates the non-reentrant g_vocab_ptr global.
 * Uses quicksort with median-of-three pivot + insertion sort for small
 * partitions. Fully reentrant — vocab pointer is passed as a parameter.
 */
static void swap_int(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

static void vocab_insertion_sort(int *arr, int n, char **vocab) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && strcmp(vocab[arr[j]], vocab[key]) > 0) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static void vocab_quicksort(int *arr, int n, char **vocab) {
    while (n > 16) {
        /* Median-of-three pivot selection */
        int mid = n / 2;
        if (strcmp(vocab[arr[0]], vocab[arr[mid]]) > 0) swap_int(&arr[0], &arr[mid]);
        if (strcmp(vocab[arr[0]], vocab[arr[n-1]]) > 0) swap_int(&arr[0], &arr[n-1]);
        if (strcmp(vocab[arr[mid]], vocab[arr[n-1]]) > 0) swap_int(&arr[mid], &arr[n-1]);
        swap_int(&arr[mid], &arr[n-2]);
        int pivot = arr[n-2];

        int i = 0, j = n - 2;
        for (;;) {
            while (strcmp(vocab[arr[++i]], vocab[pivot]) < 0) {}
            while (strcmp(vocab[arr[--j]], vocab[pivot]) > 0) {}
            if (i >= j) break;
            swap_int(&arr[i], &arr[j]);
        }
        swap_int(&arr[i], &arr[n-2]);

        /* Recurse on smaller partition, iterate on larger (tail-call optimization) */
        if (i < n - i - 1) {
            vocab_quicksort(arr, i, vocab);
            arr += i + 1;
            n -= i + 1;
        } else {
            vocab_quicksort(arr + i + 1, n - i - 1, vocab);
            n = i;
        }
    }
    vocab_insertion_sort(arr, n, vocab);
}

static void build_sorted_index(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab_indices[i] = i;
    }
    vocab_quicksort(t->sorted_vocab_indices, t->vocab_size, t->vocab);
    t->sorted = 1;
}

int tokenizer_find_id(const Tokenizer *t, const char *str) {
    if (!t->sorted || !t->sorted_vocab_indices || !str) return -1;
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = t->sorted_vocab_indices[mid];
        int cmp = strcmp(t->vocab[idx], str);
        if (cmp == 0) return idx;
        if (cmp < 0) lo = mid + 1;
        else         hi = mid - 1;
    }
    return -1;
}

TernaryError tokenizer_load(Tokenizer *t, const char *path) {
    memset(t, 0, sizeof(*t));
    /* sentinel -1 means "not found yet"; memset alone sets these to 0 which is a valid token ID */
    t->bos_token_id = -1;
    t->eos_token_id = -1;

    FILE *fp = fopen(path, "rb");
    TN_CHECK(fp != NULL, TN_ERR_TOKENIZER_LOAD);

    /* Read vocab_size and max_token_len */
    if (fread(&t->vocab_size, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return TN_ERR_TOKENIZER_LOAD;
    }
    if (fread(&t->max_token_len, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return TN_ERR_TOKENIZER_LOAD;
    }

    /* Validate ranges — must close fp on failure since TN_CHECK does a
     * bare return that would leak the FILE handle. */
    if (t->vocab_size <= 0 || t->vocab_size > 1000000) {
        fclose(fp);
        return TN_ERR_TOKENIZER_LOAD;
    }
    if (t->max_token_len <= 0 || t->max_token_len > 4096) {
        fclose(fp);
        return TN_ERR_TOKENIZER_LOAD;
    }

    /* Allocate arrays */
    t->vocab = (char **)calloc((size_t)t->vocab_size, sizeof(char *));
    if (!t->vocab) { fclose(fp); return TN_ERR_OOM; }

    t->vocab_scores = (float *)calloc((size_t)t->vocab_size, sizeof(float));
    if (!t->vocab_scores) { fclose(fp); tokenizer_free(t); return TN_ERR_OOM; }

    t->sorted_vocab_indices = (int *)calloc((size_t)t->vocab_size, sizeof(int));
    if (!t->sorted_vocab_indices) { fclose(fp); tokenizer_free(t); return TN_ERR_OOM; }

    /* Read each token: score(float) len(int) bytes(char[len]) */
    for (int i = 0; i < t->vocab_size; i++) {
        float score;
        int len;

        if (fread(&score, sizeof(float), 1, fp) != 1) {
            fclose(fp); tokenizer_free(t); return TN_ERR_TOKENIZER_LOAD;
        }
        if (fread(&len, sizeof(int), 1, fp) != 1) {
            fclose(fp); tokenizer_free(t); return TN_ERR_TOKENIZER_LOAD;
        }
        if (len < 0 || len > t->max_token_len) {
            fclose(fp); tokenizer_free(t); return TN_ERR_TOKENIZER_LOAD;
        }

        t->vocab_scores[i] = score;
        t->vocab[i] = (char *)malloc((size_t)len + 1);
        if (!t->vocab[i]) {
            fclose(fp); tokenizer_free(t); return TN_ERR_OOM;
        }

        if (len > 0 && fread(t->vocab[i], 1, (size_t)len, fp) != (size_t)len) {
            fclose(fp); tokenizer_free(t); return TN_ERR_TOKENIZER_LOAD;
        }
        t->vocab[i][len] = '\0';
    }

    fclose(fp);

    /* Build sorted index for binary search during encoding */
    build_sorted_index(t);

    /* ── Derive BOS/EOS token IDs from vocabulary contents ──────────────────
     * Scan the vocabulary for known BOS/EOS token strings and record their
     * actual indices. This replaces any hardcoded numeric fallbacks.
     *
     * MODULARITY RULE: The tokenizer NEVER sets chat_template. That is model
     * metadata, not tokenizer metadata. Chat templates are set by:
     *   1. tokenizer_gguf.c  — reads "tokenizer.chat_template" from the GGUF
     *   2. main.c patch      — copies GGUF chat_template into an external .bin
     *                          tokenizer when both are provided together
     * This ensures base models (e.g. BitNet) are never wrongly given an
     * instruct chat template just because their BPE vocab contains control
     * tokens from the same family as an instruct model.
     * ─────────────────────────────────────────────────────────────────────── */

    /* BOS candidates — first match wins */
    if (t->bos_token_id < 0) {
        static const char *const BOS_CANDIDATES[] = {
            "<|begin_of_text|>", "<s>", "<bos>", NULL
        };
        for (int k = 0; BOS_CANDIDATES[k]; k++) {
            int id = tokenizer_find_id(t, BOS_CANDIDATES[k]);
            if (id >= 0) { t->bos_token_id = id; break; }
        }
    }

    /* Primary EOS candidate — first match wins */
    if (t->eos_token_id < 0) {
        static const char *const EOS_PRIMARY[] = {
            "<|eot_id|>", "<|end_of_text|>", "<|im_end|>",
            "</s>", "<eos>", "[EOS]", "<|endoftext|>", NULL
        };
        for (int k = 0; EOS_PRIMARY[k]; k++) {
            int id = tokenizer_find_id(t, EOS_PRIMARY[k]);
            if (id >= 0) { t->eos_token_id = id; break; }
        }
    }

    /* ── Populate eos_list: all EOS-like token IDs found in this vocab ─────
     * Used by generate.c / agent_loop.c to stop generation without any
     * hardcoded numeric token IDs.
     * ─────────────────────────────────────────────────────────────────────── */
    static const char *const EOS_CANDIDATES[] = {
        "<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>",
        "<eos>", "[EOS]", "<|endoftext|>", "<end_of_utterance>", NULL
    };
    t->n_eos = 0;
    for (int k = 0; EOS_CANDIDATES[k] && t->n_eos < 8; k++) {
        int id = tokenizer_find_id(t, EOS_CANDIDATES[k]);
        if (id >= 0) {
            t->eos_list[t->n_eos++] = id;
        }
    }

    return TN_OK;
}

void tokenizer_free(Tokenizer *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            free(t->vocab[i]);
        }
        free(t->vocab);
    }
    free(t->vocab_scores);
    free(t->sorted_vocab_indices);
    free(t->chat_template);
    if (t->special_tokens) {
        for (int i = 0; i < t->n_special; i++) free(t->special_tokens[i]);
        free(t->special_tokens);
    }
    free(t->special_token_ids);
    memset(t, 0, sizeof(*t));
}
