#ifndef TN_CHAT_REQUEST_H
#define TN_CHAT_REQUEST_H

/*
 * Phase 21 — OpenAI-Compatible API Layer
 *
 * Data structures for chat completion requests and responses.
 * Matches the OpenAI /v1/chat/completions schema.
 */

#include <stdbool.h>
#include "core/error.h"

#define CHAT_MAX_MESSAGES   64
#define CHAT_MAX_ROLE_LEN   32
#define CHAT_MAX_CONTENT    65536   /* 64 KiB per message */
#define CHAT_MAX_MODEL_LEN  128

/* ── Single chat message ─────────────────────────────────────────────────── */
typedef struct {
    char role[CHAT_MAX_ROLE_LEN];       /* "system" | "user" | "assistant" */
    char *content;                      /* Heap-allocated, NULL-terminated  */
    /* Phase 22.2: set when `content` was the OpenAI "content parts" array
     * form (text + image_url) rather than a plain string — carries the
     * image's data: URL (heap-allocated, NULL-terminated). NULL for
     * ordinary text-only messages. */
    char *image_data_url;
} ChatMessage;

/* ── Parsed request from POST /v1/chat/completions ──────────────────────── */
typedef struct {
    char          model[CHAT_MAX_MODEL_LEN];
    ChatMessage   messages[CHAT_MAX_MESSAGES];
    int           num_messages;
    float         temperature;          /* default 1.0  */
    float         top_p;                /* default 1.0  */
    int           max_tokens;           /* default 512  */
    bool          stream;               /* default false */
} ChatRequest;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

/* Initialise a ChatRequest to safe defaults. Must be called before
 * chat_request_parse() to ensure consistent starting state. */
void chat_request_init(ChatRequest *req);

/* Release heap-allocated message content fields. Safe to call on a
 * zero-initialised or partially-parsed struct. */
void chat_request_free(ChatRequest *req);

/* Parse a JSON request body (NUL-terminated) into *req.
 * req must have been initialised with chat_request_init() first.
 * Returns TN_OK on success or TN_ERR_JSON_PARSE / TN_ERR_INVALID_ARGS. */
TernaryError chat_request_parse(const char *json_body, ChatRequest *req);

#endif /* TN_CHAT_REQUEST_H */
