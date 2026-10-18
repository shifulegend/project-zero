#ifndef TN_WATERMARK_TEXT_H
#define TN_WATERMARK_TEXT_H

/*
 * watermark_text.h — Phase 33: score raw text (not token IDs) for the
 * watermark by tokenizing it with the model's own tokenizer first. Shared by
 * the CLI (--watermark-detect) and the HTTP API (POST /v1/watermark/detect).
 */

#include "core/error.h"
#include "sampling/watermark.h"
#include "tokenizer/tokenizer.h"

/* z-score at/above which text is reported as watermarked by default
 * (~3e-5 one-sided false-positive rate). API callers may pass their own. */
#define WATERMARK_DETECT_Z_THRESHOLD 4.0

/* Tokenizes `text` (no BOS) and runs watermark_detect(). Returns
 * TN_ERR_INVALID_ARGS for NULL inputs, TN_ERR_OOM, or TN_OK. */
TernaryError watermark_detect_text(const WatermarkConfig *wm, Tokenizer *tok,
                                   const char *text, WatermarkResult *out);

#endif /* TN_WATERMARK_TEXT_H */
