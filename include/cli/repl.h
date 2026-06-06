#ifndef PROJECT_ZERO_REPL_H
#define PROJECT_ZERO_REPL_H

#include "core/config.h"
#include "core/moe_config.h"
#include "core/weights.h"
#include "core/run_state.h"
#include "tokenizer/tokenizer.h"
#include "threading/thread_pool.h"
#include "cli/args.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "rag/rag_context.h"  /* Phase 15 */

/**
 * Run the interactive Read-Eval-Print-Loop (REPL).
 *
 * @param mc   MoE config — NULL or zero-init for dense models.
 * @param rag  Optional RAG context (Phase 15).  Pass NULL to disable memory.
 */
void run_repl(Config *p, TransformerWeights *w,
              const MoEConfig *mc,
              VisionConfig *vc, VisionWeights *vw, VisionProjector *vp,
              RunState *s, Tokenizer *t, ThreadPool *tp,
              CliArgs *args, RagContext *rag);

#endif // PROJECT_ZERO_REPL_H
