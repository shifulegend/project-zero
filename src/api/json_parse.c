/*
 * Phase 21 — OpenAI-Compatible API Layer
 * src/api/json_parse.c
 *
 * Purpose-built, zero-dependency JSON parser that extracts only the fields
 * needed to handle POST /v1/chat/completions requests.  Designed for
 * correctness over generality — the parser handles valid JSON produced by
 * well-behaved HTTP clients and returns TN_ERR_JSON_PARSE for anything
 * structurally wrong.
 *
 * Parsed fields
 * ─────────────
 *   .model          string
 *   .messages[]     array of {role, content} objects
 *   .temperature    number  (default 1.0)
 *   .top_p          number  (default 1.0)
 *   .max_tokens     integer (default 512)
 *   .stream         boolean (default false)
 */

#include "api/chat_request.h"
#include "core/error.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

/* ── Internal helpers ────────────────────────────────────────────────────── */

/* Skip whitespace; advance *p. */
static void skip_ws(const char **p) {
    while (**p && (unsigned char)**p <= ' ') (*p)++;
}

/* Expect and consume a literal character.  Returns 0 on mismatch. */
static int expect_char(const char **p, char c) {
    skip_ws(p);
    if (**p != c) return 0;
    (*p)++;
    return 1;
}

/*
 * Parse a JSON string starting at *p (which must point to the opening '"').
 * Writes up to (dst_cap - 1) decoded bytes into dst and NUL-terminates.
 * Advances *p past the closing '"'.
 * Returns 0 on error.
 */
static int parse_string(const char **p, char *dst, size_t dst_cap) {
    skip_ws(p);
    if (**p != '"') return 0;
    (*p)++; /* skip opening quote */

    size_t i = 0;
    while (**p && **p != '"') {
        char c = **p;
        if (c == '\\') {
            (*p)++;
            switch (**p) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'u':
                    /* Skip 4 hex digits; replace with '?' for simplicity */
                    if ((*p)[1] && (*p)[2] && (*p)[3] && (*p)[4]) (*p) += 4;
                    c = '?';
                    break;
                default:   c = **p; break;
            }
        }
        if (i < dst_cap - 1) dst[i++] = c;
        (*p)++;
    }
    if (**p != '"') return 0;
    (*p)++; /* skip closing quote */
    if (dst) dst[i] = '\0';
    return 1;
}

/*
 * Heap-allocate a copy of a JSON string (may be longer than fixed buffers).
 * Returns NULL on error.  Caller must free().
 */
static char *parse_string_heap(const char **p) {
    skip_ws(p);
    if (**p != '"') return NULL;

    /* First pass: measure length */
    const char *scan = *p + 1; /* skip opening quote */
    size_t len = 0;
    while (*scan && *scan != '"') {
        if (*scan == '\\') { scan++; if (!*scan) return NULL; }
        len++;
        scan++;
    }
    if (*scan != '"') return NULL;

    /* Second pass: fill buffer */
    char *buf = (char *)malloc(len + 1);
    if (!buf) return NULL;

    if (!parse_string(p, buf, len + 1)) {
        free(buf);
        return NULL;
    }
    return buf;
}

/* Parse a JSON number and return its double value. Advances *p. */
static double parse_number(const char **p) {
    skip_ws(p);
    char buf[64];
    size_t i = 0;
    while (**p && (isdigit((unsigned char)**p) || **p == '.' || **p == '-'
                   || **p == '+' || **p == 'e' || **p == 'E')) {
        if (i < sizeof(buf) - 1) buf[i++] = **p;
        (*p)++;
    }
    buf[i] = '\0';
    return atof(buf);
}

/* Parse a JSON boolean (true/false). Returns -1 on error, 0/1 on success. */
static int parse_bool(const char **p) {
    skip_ws(p);
    if (strncmp(*p, "true", 4) == 0)  { *p += 4; return 1; }
    if (strncmp(*p, "false", 5) == 0) { *p += 5; return 0; }
    return -1;
}

/* Skip an arbitrary JSON value (used to skip unknown keys). */
static void skip_value(const char **p) {
    skip_ws(p);
    char c = **p;
    if (c == '"') {
        char discard[4096];
        parse_string(p, discard, sizeof(discard));
    } else if (c == '{') {
        /* Skip object */
        int depth = 0;
        while (**p) {
            if (**p == '{') depth++;
            else if (**p == '}') { depth--; (*p)++; if (depth == 0) return; continue; }
            else if (**p == '"') { char d[4096]; parse_string(p, d, sizeof(d)); continue; }
            (*p)++;
        }
    } else if (c == '[') {
        int depth = 0;
        while (**p) {
            if (**p == '[') depth++;
            else if (**p == ']') { depth--; (*p)++; if (depth == 0) return; continue; }
            else if (**p == '"') { char d[4096]; parse_string(p, d, sizeof(d)); continue; }
            (*p)++;
        }
    } else {
        /* number, bool, null */
        while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
    }
}

/*
 * Phase 22.2 — parses the OpenAI "content parts" array form, used for image
 * uploads:
 *   "content": [
 *     {"type": "text", "text": "..."},
 *     {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
 *   ]
 * Populates msg->content (from the text part) and msg->image_data_url (from
 * the image_url part's nested "url" field). Assumes *p already points past
 * the "content": key, at the opening '['.
 */
static TernaryError parse_content_parts(const char **p, ChatMessage *msg) {
    if (!expect_char(p, '[')) return TN_ERR_JSON_PARSE;
    skip_ws(p);

    while (**p && **p != ']') {
        if (!expect_char(p, '{')) return TN_ERR_JSON_PARSE;

        while (**p && **p != '}') {
            skip_ws(p);
            if (**p == '}') break;

            char key[64];
            if (!parse_string(p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
            if (!expect_char(p, ':'))               return TN_ERR_JSON_PARSE;
            skip_ws(p);

            if (strcmp(key, "text") == 0) {
                char *text = parse_string_heap(p);
                if (!text) return TN_ERR_JSON_PARSE;
                free(msg->content);
                msg->content = text;
            } else if (strcmp(key, "image_url") == 0 && **p == '{') {
                (*p)++; /* enter the nested {"url": "..."} object */
                while (**p && **p != '}') {
                    skip_ws(p);
                    if (**p == '}') break;
                    char nkey[32];
                    if (!parse_string(p, nkey, sizeof(nkey))) return TN_ERR_JSON_PARSE;
                    if (!expect_char(p, ':'))                 return TN_ERR_JSON_PARSE;
                    skip_ws(p);
                    if (strcmp(nkey, "url") == 0) {
                        char *url = parse_string_heap(p);
                        if (!url) return TN_ERR_JSON_PARSE;
                        free(msg->image_data_url);
                        msg->image_data_url = url;
                    } else {
                        skip_value(p);
                    }
                    skip_ws(p);
                    if (**p == ',') (*p)++;
                }
                if (!expect_char(p, '}')) return TN_ERR_JSON_PARSE;
            } else {
                skip_value(p);
            }

            skip_ws(p);
            if (**p == ',') (*p)++;
        }
        if (!expect_char(p, '}')) return TN_ERR_JSON_PARSE;

        skip_ws(p);
        if (**p == ',') { (*p)++; skip_ws(p); }
    }
    if (!expect_char(p, ']')) return TN_ERR_JSON_PARSE;
    return TN_OK;
}

/* ── Message array parser ────────────────────────────────────────────────── */
static TernaryError parse_messages(const char **p, ChatRequest *req) {
    if (!expect_char(p, '[')) return TN_ERR_JSON_PARSE;
    skip_ws(p);

    while (**p && **p != ']') {
        if (!expect_char(p, '{')) return TN_ERR_JSON_PARSE;

        if (req->num_messages >= CHAT_MAX_MESSAGES) {
            /* Too many messages — skip remainder of array */
            while (**p && **p != ']') (*p)++;
            break;
        }

        ChatMessage *msg = &req->messages[req->num_messages];
        msg->role[0]        = '\0';
        msg->content        = NULL;
        msg->image_data_url = NULL;

        /* Parse key-value pairs inside the message object */
        while (**p && **p != '}') {
            skip_ws(p);
            if (**p == '}') break;

            char key[64];
            if (!parse_string(p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
            if (!expect_char(p, ':'))               return TN_ERR_JSON_PARSE;
            skip_ws(p);

            if (strcmp(key, "role") == 0) {
                if (!parse_string(p, msg->role, sizeof(msg->role)))
                    return TN_ERR_JSON_PARSE;
            } else if (strcmp(key, "content") == 0) {
                skip_ws(p);
                if (**p == '[') {
                    /* Phase 22.2: OpenAI "content parts" array (text + image_url) */
                    TernaryError err = parse_content_parts(p, msg);
                    if (err != TN_OK) return err;
                } else {
                    msg->content = parse_string_heap(p);
                    if (!msg->content) return TN_ERR_JSON_PARSE;
                }
            } else {
                skip_value(p);
            }

            skip_ws(p);
            if (**p == ',') { (*p)++; }
        }
        if (!expect_char(p, '}')) return TN_ERR_JSON_PARSE;
        req->num_messages++;

        skip_ws(p);
        if (**p == ',') { (*p)++; skip_ws(p); }
    }
    if (!expect_char(p, ']')) return TN_ERR_JSON_PARSE;
    return TN_OK;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void chat_request_init(ChatRequest *req) {
    if (!req) return;
    memset(req, 0, sizeof(*req));
    req->temperature = 1.0f;
    req->top_p       = 1.0f;
    req->max_tokens  = 512;
    req->stream      = 0;
    strncpy(req->model, "local", sizeof(req->model) - 1);
}

void chat_request_free(ChatRequest *req) {
    if (!req) return;
    for (int i = 0; i < req->num_messages; i++) {
        free(req->messages[i].content);
        req->messages[i].content = NULL;
        free(req->messages[i].image_data_url);
        req->messages[i].image_data_url = NULL;
    }
    req->num_messages = 0;
}

/*
 * Parse a raw JSON request body into *req.
 * req must have been initialised with chat_request_init() first.
 */
TernaryError chat_request_parse(const char *json_body, ChatRequest *req) {
    if (!json_body || !req) return TN_ERR_INVALID_ARGS;

    const char *p = json_body;
    skip_ws(&p);
    if (*p != '{') return TN_ERR_JSON_PARSE;
    p++; /* skip '{' */

    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == '}') break;

        char key[128];
        if (!parse_string(&p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
        if (!expect_char(&p, ':'))               return TN_ERR_JSON_PARSE;
        skip_ws(&p);

        if (strcmp(key, "model") == 0) {
            if (!parse_string(&p, req->model, sizeof(req->model)))
                return TN_ERR_JSON_PARSE;
        } else if (strcmp(key, "messages") == 0) {
            TernaryError err = parse_messages(&p, req);
            if (err != TN_OK) return err;
        } else if (strcmp(key, "temperature") == 0) {
            req->temperature = (float)parse_number(&p);
        } else if (strcmp(key, "top_p") == 0) {
            req->top_p = (float)parse_number(&p);
        } else if (strcmp(key, "max_tokens") == 0) {
            req->max_tokens = (int)parse_number(&p);
        } else if (strcmp(key, "stream") == 0) {
            int b = parse_bool(&p);
            if (b < 0) return TN_ERR_JSON_PARSE;
            req->stream = (b == 1);
        } else {
            skip_value(&p);
        }

        skip_ws(&p);
        if (*p == ',') { p++; }
    }
    return TN_OK;
}
