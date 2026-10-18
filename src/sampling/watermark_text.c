#include "sampling/watermark_text.h"
#include <stdlib.h>
#include <string.h>

TernaryError watermark_detect_text(const WatermarkConfig *wm, Tokenizer *tok,
                                   const char *text, WatermarkResult *out) {
    if (!wm || !tok || !text || !out) return TN_ERR_INVALID_ARGS;
    size_t len = strlen(text);
    /* BPE never yields more tokens than input bytes (byte fallback is 1:1). */
    int cap = (int)(len < 1 ? 1 : (len > 1000000 ? 1000000 : len)) + 1;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) return TN_ERR_OOM;
    int n = tokenizer_encode(tok, text, len, ids, cap);
    if (n < 0) { free(ids); return TN_ERR_TOKENIZER_ENCODE; }
    watermark_detect(wm, ids, n, out);
    free(ids);
    return TN_OK;
}
