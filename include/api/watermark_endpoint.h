#ifndef TN_WATERMARK_ENDPOINT_H
#define TN_WATERMARK_ENDPOINT_H

/*
 * watermark_endpoint.h — Phase 33: POST /v1/watermark/detect.
 *
 * Request:  {"text": "...", "key": "passphrase"?, "gamma": 0.25?, "z_threshold": 4?}
 *           key/gamma default to the server's watermark settings.
 * Response: {"tokens_scored":N,"green_tokens":G,"green_fraction":f,
 *            "z_score":z,"z_threshold":t,"watermarked":bool}
 * Read-only on the tokenizer, so it never takes the generation mutex.
 */

#include "api/api_server.h"

/* Returns a heap JSON body (caller frees) and sets *status (200/400/500). */
char *api_watermark_detect(const ApiContext *ctx, const char *json_body, int *status);

#endif /* TN_WATERMARK_ENDPOINT_H */
