/* gguf_meta_str_copy(): caller-owned buffers, so two lookups never overwrite each other
 * (the old gguf_meta_str() returned one shared static buffer). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/gguf_reader.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

static void set_str(GGUFMeta *m, const char *key, char *val) {
    memset(m, 0, sizeof *m);
    snprintf(m->key, sizeof m->key, "%s", key);
    m->val_type       = GGUF_VAL_STRING;
    m->val.string.str = val;
    m->val.string.len = strlen(val);
}

int main(void) {
    GGUFHeader *h = (GGUFHeader *)calloc(1, sizeof *h);
    char v1[] = "gpt2", v2[] = "llama-bpe", v3[] = "0123456789ABCDEF";
    set_str(&h->meta[0], "tokenizer.ggml.model", v1);
    set_str(&h->meta[1], "tokenizer.ggml.pre", v2);
    set_str(&h->meta[2], "long.value", v3);
    memset(&h->meta[3], 0, sizeof h->meta[3]);
    snprintf(h->meta[3].key, sizeof h->meta[3].key, "a.number");
    h->meta[3].val_type = GGUF_VAL_UINT32;
    h->meta[3].val.u32  = 7;
    h->n_meta           = 4;

    char a[32], b[32];
    CHECK(gguf_meta_str_copy(h, "tokenizer.ggml.model", a, sizeof a) == 4,
          "first lookup returns the length");
    CHECK(gguf_meta_str_copy(h, "tokenizer.ggml.pre", b, sizeof b) == 9,
          "second lookup returns its length");
    CHECK(strcmp(a, "gpt2") == 0 && strcmp(b, "llama-bpe") == 0,
          "both results stay valid after the second lookup (no shared buffer)");

    char t[8];
    int n = gguf_meta_str_copy(h, "long.value", t, sizeof t);
    CHECK(n == 16 && strcmp(t, "0123456") == 0,
          "a long value is truncated to cap-1 bytes, NUL-terminated, full length returned");
    char one[1];
    CHECK(gguf_meta_str_copy(h, "tokenizer.ggml.model", one, 1) == 4 && one[0] == '\0',
          "cap 1 gives an empty string");

    char x[8] = "keep";
    CHECK(gguf_meta_str_copy(h, "missing", x, sizeof x) == -1 && strcmp(x, "keep") == 0,
          "an absent key returns -1 and leaves the buffer alone");
    CHECK(gguf_meta_str_copy(h, "a.number", x, sizeof x) == -1, "a non-string value returns -1");
    CHECK(gguf_meta_str_copy(h, "tokenizer.ggml.model", x, 0) == -1, "cap 0 returns -1");
    CHECK(gguf_meta_str_copy(NULL, "k", x, sizeof x) == -1 &&
              gguf_meta_str_copy(h, NULL, x, sizeof x) == -1 &&
              gguf_meta_str_copy(h, "k", NULL, 4) == -1,
          "NULL arguments return -1");
    free(h);
    printf(g_fail ? "gguf_meta_str: %d FAILED\n" : "gguf_meta_str: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
