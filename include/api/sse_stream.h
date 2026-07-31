#ifndef TN_SSE_STREAM_H
#define TN_SSE_STREAM_H

/*
 * Phase 21 — OpenAI-Compatible API Layer
 * include/api/sse_stream.h
 *
 * Server-Sent Events (SSE) helpers for streaming chat completion tokens
 * over an HTTP connection using the OpenAI streaming response format.
 *
 * Wire format (each token):
 *   data: {"id":"chatcmpl-<id>","object":"chat.completion.chunk","choices":[
 *           {"index":0,"delta":{"content":"<token>"},"finish_reason":null}]}\n\n
 *
 * Final event:
 *   data: {"id":"chatcmpl-<id>","object":"chat.completion.chunk","choices":[
 *           {"index":0,"delta":{},"finish_reason":"stop"}]}\n\n
 *   data: [DONE]\n\n
 */

#include <stddef.h>

/* Write a streaming token chunk to a file descriptor.
 * token_text: decoded token string (may contain JSON-unsafe characters —
 *             they are escaped before writing).
 * id:         request ID string (e.g. "chatcmpl-001").
 * fd:         open writable file descriptor (socket).
 * Returns bytes written, or -1 on error. */
int sse_write_token(int fd, const char *id, const char *token_text);

/* Write the final [DONE] SSE event to fd.
 * Returns bytes written, or -1 on error. */
int sse_write_done(int fd, const char *id);

/* Write a non-streaming (buffered) chat completion JSON response.
 * full_text: complete generated text.
 * id:        request ID string.
 * fd:        open writable file descriptor.
 * Returns bytes written, or -1 on error. */
int sse_write_full_response(int fd, const char *id, const char *full_text);

/* Phase 22: builds the same non-streaming chat completion JSON as
 * sse_write_full_response() but returns it as a heap-allocated,
 * NUL-terminated string instead of writing it directly — so the caller can
 * compute a correct Content-Length header before sending any bytes (writing
 * headers with Content-Length first, then the body via a separate write(),
 * requires knowing the length up front). Caller must free() the result.
 * Returns NULL on allocation failure or invalid arguments. */
char *sse_format_full_response(const char *id, const char *full_text);

#endif /* TN_SSE_STREAM_H */
