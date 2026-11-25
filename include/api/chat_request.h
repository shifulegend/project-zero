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
#include "reasoning/mcts_options.h"

#define CHAT_MAX_MESSAGES 64
#define CHAT_MAX_ROLE_LEN 32
#define CHAT_MAX_CONTENT 65536 /* 64 KiB per message */
#define CHAT_MAX_MODEL_LEN 128
#define CHAT_MAX_STOP 8         /* == GEN_MAX_STOP */
#define CHAT_MAX_TEMPLATE 65536 /* per-request "chat_template" cap (2026-10-19) */

/* ChatRequest.set_mask bits: which optional fields the client sent. */
enum {
    CHAT_SET_TEMPERATURE   = 1u << 0,
    CHAT_SET_TOP_P         = 1u << 1,
    CHAT_SET_MAX_TOKENS    = 1u << 2,
    CHAT_SET_TOP_K         = 1u << 3,
    CHAT_SET_MIN_P         = 1u << 4,
    CHAT_SET_REP_PENALTY   = 1u << 5,
    CHAT_SET_PRES_PENALTY  = 1u << 6,
    CHAT_SET_FREQ_PENALTY  = 1u << 7,
    CHAT_SET_PENALTY_N     = 1u << 8,
    CHAT_SET_SEED          = 1u << 9,
    CHAT_SET_STOP          = 1u << 10,
    CHAT_SET_WATERMARK     = 1u << 11,
    CHAT_SET_WM_GAMMA      = 1u << 12,
    CHAT_SET_WM_DELTA      = 1u << 13,
    CHAT_SET_JSON_MODE     = 1u << 14,
    CHAT_SET_CACHE_PROMPT  = 1u << 15,
    CHAT_SET_BEST_OF       = 1u << 16,
    CHAT_SET_IGNORE_EOS    = 1u << 17,
    CHAT_SET_CHAT_TEMPLATE = 1u << 18,
    CHAT_SET_MCTS          = 1u << 19
};

/* ── Single chat message ─────────────────────────────────────────────────── */
typedef struct {
    char role[CHAT_MAX_ROLE_LEN]; /* "system" | "user" | "assistant" */
    char *content;                /* Heap-allocated, NULL-terminated  */
    /* Phase 22.2: set when `content` was the OpenAI "content parts" array
     * form (text + image_url) rather than a plain string — carries the
     * image's data: URL (heap-allocated, NULL-terminated). NULL for
     * ordinary text-only messages. */
    char *image_data_url;
} ChatMessage;

/* Sampling knobs a request may set. Each is honored only when its CHAT_SET_*
 * bit is set in ChatRequest.set_mask -- absent fields fall back to the
 * server-side defaults (CLI flags), see api/request_params.h. */
typedef struct {
    float temperature; /* default 1.0  */
    float top_p;       /* default 1.0  */
    int top_k;
    float min_p;
    float repetition_penalty; /* "repetition_penalty" or "repeat_penalty" */
    float presence_penalty;
    float frequency_penalty;
    int penalty_last_n; /* "penalty_last_n" or "repeat_last_n" */
    long long seed;
} ChatSampling;

/* How the reply is produced and cut off. */
typedef struct {
    int max_tokens; /* default 512  */
    bool stream;    /* default false */
    /* Phase 20: OpenAI-style `"response_format": {"type": "json_object"}`.
     * Only the basic json_object type is supported (grammar-masked to any
     * syntactically valid JSON) -- json_schema-constrained variants are out
     * of scope, see sampling/grammar.h's scope note. */
    bool json_mode; /* default false */
    /* "ignore_eos": true keeps generating to max_tokens (llama.cpp server name). */
    bool ignore_eos;
    char *stop[CHAT_MAX_STOP]; /* heap strings; "stop": "x" or ["x","y"] */
    int n_stop;
} ChatOutput;

/* Phase 33: "watermark": {"enabled": bool, "key": str, "gamma": n, "delta": n} */
typedef struct {
    bool enabled;
    char *key; /* heap; NULL = use the server's key */
    float gamma;
    float delta;
} ChatWatermark;

/* Candidate-search options: best-of-N and tree search. */
typedef struct {
    /* Phase 30: "best_of": N candidates, return the most likely one. */
    int best_of;
    /* Phase 30: "mcts": N (simulations) or {"simulations", "expand",
     * "max_depth", "step_tokens", "c", "temperature"}; 0 fields = server's. */
    MctsOptions mcts;
} ChatSearch;

/* ── Parsed request from POST /v1/chat/completions ──────────────────────── */
typedef struct {
    char model[CHAT_MAX_MODEL_LEN];
    ChatMessage messages[CHAT_MAX_MESSAGES];
    int num_messages;
    unsigned set_mask; /* which optional fields the client sent */
    ChatSampling sampling;
    ChatOutput output;
    ChatWatermark watermark;
    ChatSearch search;
    /* Phase 21.5: "cache_prompt": false forces a full prefill for this request. */
    bool cache_prompt;
    /* "chat_template": Jinja template replacing the model's for this request
     * (heap; NULL = server's). Ignored when the server runs
     * --no-request-template. */
    char *chat_template;
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

/* The 400 message for a non-OK chat_request_parse() result (no characters that need JSON escaping).
 */
const char *chat_request_error_message(TernaryError err);

#endif /* TN_CHAT_REQUEST_H */
