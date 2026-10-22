#include "kv_cache/prefix_cache.h"
#include <stdlib.h>
#include <string.h>

int prefix_cache_init(PrefixCache *pc, int capacity) {
    if (!pc) return -1;
    memset(pc, 0, sizeof(*pc));
    if (capacity <= 0) return -1;
    pc->tokens = (int *)malloc((size_t)capacity * sizeof(int));
    if (!pc->tokens) return -1;
    pc->capacity = capacity;
    return 0;
}

void prefix_cache_free(PrefixCache *pc) {
    if (!pc) return;
    free(pc->tokens);
    memset(pc, 0, sizeof(*pc));
}

void prefix_cache_reset(PrefixCache *pc) {
    if (pc) pc->n = 0;
}

int prefix_cache_match(const PrefixCache *pc, const int *tokens, int n) {
    if (!pc || !pc->tokens || !tokens || n <= 0) return 0;
    int lim = n < pc->n ? n : pc->n;
    int i = 0;
    while (i < lim && pc->tokens[i] == tokens[i]) i++;
    return i;
}

void prefix_cache_truncate(PrefixCache *pc, int len) {
    if (!pc) return;
    if (len < 0) len = 0;
    if (len < pc->n) pc->n = len;
}

void prefix_cache_set(PrefixCache *pc, int pos, int token) {
    if (!pc || !pc->tokens) return;
    if (pos < 0 || pos >= pc->capacity || pos > pc->n) { pc->n = 0; return; }
    pc->tokens[pos] = token;
    pc->n = pos + 1;
}
