#include "rag/auto_retrieve.h"
#include "rag/memory_search.h"
#include "agent/output_inject.h"
#include "memory/aligned_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum size of the formatted injection string */
#define INJECT_BUF_SIZE 8192

int auto_retrieve_and_inject(const char *user_prompt,
                              VectorDB *db,
                              const Config *cfg,
                              const TransformerWeights *w,
                              RunState *s,
                              Embedder *emb,
                              Tokenizer *tok,
                              ThreadPool *tp) {
    if (!user_prompt || !db || !cfg || !w || !s || !emb || !tok)
        return -1;

    /* Budget check: don't inject if too little space remains */
    int free_tokens = cfg->seq_len - s->current_pos;
    if (free_tokens < AUTO_RETRIEVE_MIN_FREE_TOKENS) {
        return 0; /* skip silently */
    }

    if (db->num_entries == 0) return 0;

    /* Search for relevant memories */
    MemoryResult results[AUTO_RETRIEVE_TOP_N];
    int n = memory_search(db, user_prompt, cfg, w, emb, tok, tp,
                          results, AUTO_RETRIEVE_TOP_N);
    if (n <= 0) return 0;

    /* Filter strictly by AUTO_RETRIEVE_MIN_SCORE (memory_search uses a
     * lower MEMORY_SEARCH_MIN_SCORE — we re-filter here) */
    int kept = 0;
    for (int i = 0; i < n; i++) {
        if (results[i].score >= AUTO_RETRIEVE_MIN_SCORE) {
            results[kept++] = results[i];
        }
    }
    if (kept == 0) return 0;

    /* Format injection text */
    char *buf = (char *)malloc(INJECT_BUF_SIZE);
    if (!buf) return -1;

    int off = 0;
    off += snprintf(buf + off, INJECT_BUF_SIZE - off,
                    "\n<memory>\nRelevant context from previous conversations:\n");
    for (int i = 0; i < kept && off < INJECT_BUF_SIZE - 128; i++) {
        off += snprintf(buf + off, INJECT_BUF_SIZE - off,
                        "- %s\n", results[i].text ? results[i].text : "");
    }
    off += snprintf(buf + off, INJECT_BUF_SIZE - off, "</memory>\n");

    /* Inject into main KV cache */
    int n_injected = inject_text_into_kv(buf, cfg, w, s, tok, tp);
    free(buf);

    if (n_injected < 0) return n_injected;
    return n_injected;
}
