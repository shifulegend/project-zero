#ifndef TN_CORE_BATCH_SCRATCH_FWD_H
#define TN_CORE_BATCH_SCRATCH_FWD_H

/* Forward declaration of the N-token batch scratch (full definition and the
 * allocators: runtime/spec_scratch.h). The transformer headers only pass a
 * pointer to it, so they include this leaf header instead of reaching into the
 * speculative module; a .c file that reads its fields includes spec_scratch.h. */
typedef struct SpecBatchScratch SpecBatchScratch;

#endif
