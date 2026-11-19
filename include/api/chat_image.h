#ifndef TN_API_CHAT_IMAGE_H
#define TN_API_CHAT_IMAGE_H

#include "api/api_server.h"
#include "api/chat_request.h"

/* Runs the vision pipeline for the newest user message that carries an image_url, injecting the
 * image into ctx->run_state's KV cache and replacing that message's content with the prompt text
 * left after the image part is removed. Best effort: a server without --vision/--proj, or an image
 * that cannot be decoded, is logged and skipped. Returns 1 when the pipeline ran (the KV cache was
 * rewritten, so the caller resets the prefix cache), else 0. The caller holds the engine. */
int chat_apply_image(ApiContext *ctx, ChatRequest *req);

#endif
