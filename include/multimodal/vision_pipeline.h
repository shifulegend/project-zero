#ifndef TN_VISION_PIPELINE_H
#define TN_VISION_PIPELINE_H

/*
 * Phase 22.2 — extracted from the inline vision block that used to live in
 * main.c (Phase 34), so both the CLI (--image) and the HTTP API (image_url
 * content parts) can load an image, run the vision encoder+projector, and
 * inject the result into the KV cache without duplicating ~150 lines.
 *
 * Best-effort like the original inline version: on any failure it logs to
 * stderr and returns a non-OK code, but never leaves partial state that
 * would crash a later generate() call — the caller can still proceed with
 * text-only generation.
 */

#include "core/error.h"
#include "core/config.h"
#include "core/weights.h"
#include "core/run_state.h"
#include "core/moe_config.h"
#include "tokenizer/tokenizer.h"
#include "threading/thread_pool.h"
#include <stddef.h>

/*
 * @param image_path              Path to the image file.
 * @param vision_path             Path to vision.bin encoder weights.
 * @param proj_path               Path to projector.bin weights.
 * @param cfg, w, s, tok, tp       Model/runtime state (s->current_pos is
 *                                 advanced by any chat-prefix tokens fed in;
 *                                 the vision embeddings themselves are
 *                                 injected via inject_vision_into_kv_cache).
 * @param prompt_text              The user's prompt (may be NULL).
 * @param stripped_prompt_buf      Scratch buffer the function may write a
 *                                 stripped copy of prompt_text into (when the
 *                                 tokenizer's chat_template uses ChatML-style
 *                                 "im_start" markers, matching the original
 *                                 CLI behavior of pre-feeding a chat prefix
 *                                 and stripping it from the visible prompt).
 * @param stripped_prompt_buf_cap  Capacity of stripped_prompt_buf.
 * @param out_prompt               Receives either prompt_text unchanged or a
 *                                 pointer into stripped_prompt_buf. Never
 *                                 NULL on return (falls back to prompt_text).
 * @return TN_OK if the image was loaded and injected (even if some
 *         cosmetic step like dimension matching warned); a TN_ERR_* code if
 *         the vision model or image failed to load, in which case
 *         *out_prompt is still set to prompt_text so the caller can
 *         degrade gracefully to text-only generation.
 */
TernaryError vision_pipeline_run(const char *image_path,
                                  const char *vision_path,
                                  const char *proj_path,
                                  Config *cfg,
                                  TransformerWeights *w,
                                  RunState *s,
                                  const MoEConfig *mc,
                                  Tokenizer *tok,
                                  ThreadPool *tp,
                                  const char *prompt_text,
                                  char *stripped_prompt_buf,
                                  size_t stripped_prompt_buf_cap,
                                  const char **out_prompt);

#endif /* TN_VISION_PIPELINE_H */
