#include "speculative/spec_scratch.h"
#include "core/platform.h"
#include "memory/aligned_alloc.h"
#include <string.h>

TernaryError spec_batch_scratch_alloc(SpecBatchScratch *sb, const Config *cfg, int n_tokens) {
    memset(sb, 0, sizeof(*sb));
    sb->n_tokens   = n_tokens;
    sb->dim        = cfg->dim;
    sb->hidden_dim = cfg->hidden_dim;
    sb->vocab_size = cfg->vocab_size;

    size_t nt = (size_t)n_tokens;
    size_t dim_count, hidden_count;
    if (tn_size_mul_overflow(nt, (size_t)cfg->dim, &dim_count) ||
        tn_size_mul_overflow(nt, (size_t)cfg->hidden_dim, &hidden_count)) {
        return TN_ERR_OOM;
    }

    sb->x      = (float *)tn_aligned_calloc(dim_count, sizeof(float), TN_SIMD_ALIGN);
    sb->xb     = (float *)tn_aligned_calloc(dim_count, sizeof(float), TN_SIMD_ALIGN);
    sb->xb2    = (float *)tn_aligned_calloc(dim_count, sizeof(float), TN_SIMD_ALIGN);
    sb->hb     = (float *)tn_aligned_calloc(hidden_count, sizeof(float), TN_SIMD_ALIGN);
    sb->hb2    = (float *)tn_aligned_calloc(hidden_count, sizeof(float), TN_SIMD_ALIGN);
    sb->q      = (float *)tn_aligned_calloc(dim_count, sizeof(float), TN_SIMD_ALIGN);

    if (!sb->x || !sb->xb || !sb->xb2 || !sb->hb || !sb->hb2 || !sb->q) {
        spec_batch_scratch_free(sb);
        return TN_ERR_OOM;
    }
    return TN_OK;
}

void spec_batch_scratch_free(SpecBatchScratch *sb) {
    if (!sb) return;
    tn_aligned_free(sb->x);
    tn_aligned_free(sb->xb);
    tn_aligned_free(sb->xb2);
    tn_aligned_free(sb->hb);
    tn_aligned_free(sb->hb2);
    tn_aligned_free(sb->q);
    memset(sb, 0, sizeof(*sb));
}
