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
 *   .top_k .min_p .repetition_penalty .presence_penalty .frequency_penalty
 *   .penalty_last_n .seed .stop .watermark .response_format
 *                   optional; presence recorded in .set_mask so the server's
 *                   own defaults apply to anything the client omits
 */

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/chat_request.h"
#include "api/json_scan.h"
#include "core/error.h"
#include "core/escape.h"
#include "core/json_fields.h"
/*
 * Phase 22.2 — parses the OpenAI "content parts" array form, used for image
 * uploads:
 *   "content": [
 *     {"type": "text", "text": "..."},
 *     {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
 *   ]
 * Populates msg->content (from the text parts in order) and msg->image_data_url (from
 * the image_url part's nested "url" field). Assumes *p already points past
 * the "content": key, at the opening '['.
 */
/* The text of a message being assembled from several "text" parts. */
typedef struct {
    size_t len, cap;
} TextBuf;

/* Adds one text part (taking ownership of `text`): the first becomes the message content, later
 * ones are appended to it. */
static TernaryError append_text(ChatMessage *msg, TextBuf *tb, char *text) {
    size_t part_len = strlen(text);
    if (!msg->content) {
        msg->content = text;
        tb->len      = part_len;
        tb->cap      = tb->len + 1;
        return TN_OK;
    }
    if (part_len > SIZE_MAX - tb->len - 1) {
        free(text);
        return TN_ERR_OOM;
    }
    size_t required = tb->len + part_len + 1;
    if (required > tb->cap) {
        size_t grown    = tb->cap <= SIZE_MAX / 2 ? tb->cap * 2 : required;
        size_t capacity = required > grown ? required : grown;
        char *joined    = realloc(msg->content, capacity);
        if (!joined) {
            free(text);
            return TN_ERR_OOM;
        }
        msg->content = joined;
        tb->cap      = capacity;
    }
    memcpy(msg->content + tb->len, text, part_len + 1);
    tb->len += part_len;
    free(text);
    return TN_OK;
}

/* The nested {"url": "..."} object of an image_url part, after its '{'; a later url replaces an
 * earlier one, other keys are skipped. */
static TernaryError parse_image_url_object(const char **p, ChatMessage *msg) {
    while (**p && **p != '}') {
        json_skip_ws(p);
        if (**p == '}') break;
        char nkey[32];
        if (!json_parse_string(p, nkey, sizeof(nkey))) return TN_ERR_JSON_PARSE;
        if (!json_expect_char(p, ':')) return TN_ERR_JSON_PARSE;
        json_skip_ws(p);
        if (strcmp(nkey, "url") == 0) {
            char *url = json_parse_string_heap(p);
            if (!url) return TN_ERR_JSON_PARSE;
            free(msg->image_data_url);
            msg->image_data_url = url;
        } else {
            json_skip_value(p);
        }
        json_skip_ws(p);
        if (**p == ',') (*p)++;
    }
    return json_expect_char(p, '}') ? TN_OK : TN_ERR_JSON_PARSE;
}

/* One member of a part object, after its key and ':': "text" extends the content, "image_url" with
 * an object value sets the image, anything else is skipped. */
static TernaryError parse_part_value(const char **p, ChatMessage *msg, TextBuf *tb,
                                     const char *key) {
    if (strcmp(key, "text") == 0) {
        char *text = json_parse_string_heap(p);
        if (!text) return TN_ERR_JSON_PARSE;
        return append_text(msg, tb, text);
    }
    if (strcmp(key, "image_url") == 0 && **p == '{') {
        (*p)++; /* enter the nested {"url": "..."} object */
        return parse_image_url_object(p, msg);
    }
    json_skip_value(p);
    return TN_OK;
}

/* One part object {...}. */
static TernaryError parse_part(const char **p, ChatMessage *msg, TextBuf *tb) {
    if (!json_expect_char(p, '{')) return TN_ERR_JSON_PARSE;
    while (**p && **p != '}') {
        json_skip_ws(p);
        if (**p == '}') break;
        char key[64];
        if (!json_parse_string(p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
        if (!json_expect_char(p, ':')) return TN_ERR_JSON_PARSE;
        json_skip_ws(p);
        TernaryError err = parse_part_value(p, msg, tb, key);
        if (err != TN_OK) return err;
        json_skip_ws(p);
        if (**p == ',') (*p)++;
    }
    return json_expect_char(p, '}') ? TN_OK : TN_ERR_JSON_PARSE;
}

static TernaryError parse_content_parts(const char **p, ChatMessage *msg) {
    if (!json_expect_char(p, '[')) return TN_ERR_JSON_PARSE;
    json_skip_ws(p);

    TextBuf tb;
    tb.len = msg->content ? strlen(msg->content) : 0;
    tb.cap = msg->content ? tb.len + 1 : 0;

    while (**p && **p != ']') {
        TernaryError err = parse_part(p, msg, &tb);
        if (err != TN_OK) return err;
        json_skip_ws(p);
        if (**p == ',') {
            (*p)++;
            json_skip_ws(p);
        }
    }
    return json_expect_char(p, ']') ? TN_OK : TN_ERR_JSON_PARSE;
}

/* ── Message array parser ────────────────────────────────────────────────── */
TernaryError json_parse_messages(const char **p, ChatRequest *req) {
    if (!json_expect_char(p, '[')) return TN_ERR_JSON_PARSE;
    json_skip_ws(p);

    while (**p && **p != ']') {
        if (!json_expect_char(p, '{')) return TN_ERR_JSON_PARSE;

        if (req->num_messages >= CHAT_MAX_MESSAGES) {
            /* Too many messages — skip remainder of array */
            while (**p && **p != ']') (*p)++;
            break;
        }

        /* Include partial messages so chat_request_free can release them on error. */
        ChatMessage *msg    = &req->messages[req->num_messages++];
        msg->role[0]        = '\0';
        msg->content        = NULL;
        msg->image_data_url = NULL;

        /* Parse key-value pairs inside the message object */
        while (**p && **p != '}') {
            json_skip_ws(p);
            if (**p == '}') break;

            char key[64];
            if (!json_parse_string(p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
            if (!json_expect_char(p, ':')) return TN_ERR_JSON_PARSE;
            json_skip_ws(p);

            if (strcmp(key, "role") == 0) {
                if (!json_parse_string(p, msg->role, sizeof(msg->role))) return TN_ERR_JSON_PARSE;
            } else if (strcmp(key, "content") == 0) {
                json_skip_ws(p);
                /* A repeated "content" key replaces the earlier value (the last one
                 * wins); release the old text first, or it leaks. */
                free(msg->content);
                msg->content = NULL;
                if (**p == '[') {
                    /* Phase 22.2: OpenAI "content parts" array (text + image_url) */
                    TernaryError err = parse_content_parts(p, msg);
                    if (err != TN_OK) return err;
                } else if (strncmp(*p, "null", 4) == 0) {
                    /* OpenAI allows "content": null (e.g. assistant turns
                     * that only carried tool calls) — treat as empty. */
                    *p += 4;
                    msg->content = strdup("");
                    if (!msg->content) return TN_ERR_OOM;
                } else {
                    msg->content = json_parse_string_heap(p);
                    if (!msg->content) return TN_ERR_JSON_PARSE;
                }
            } else {
                json_skip_value(p);
            }

            json_skip_ws(p);
            if (**p == ',') {
                (*p)++;
            }
        }
        if (!json_expect_char(p, '}')) return TN_ERR_JSON_PARSE;

        json_skip_ws(p);
        if (**p == ',') {
            (*p)++;
            json_skip_ws(p);
        }
    }
    if (!json_expect_char(p, ']')) return TN_ERR_JSON_PARSE;
    return TN_OK;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void chat_request_init(ChatRequest *req) {
    if (!req) return;
    memset(req, 0, sizeof(*req));
    req->sampling.temperature = 1.0f;
    req->sampling.top_p       = 1.0f;
    req->output.max_tokens    = 512;
    req->output.stream        = 0;
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
    for (int i = 0; i < req->output.n_stop; i++) {
        free(req->output.stop[i]);
        req->output.stop[i] = NULL;
    }
    req->output.n_stop = 0;
    free(req->watermark.key);
    req->watermark.key = NULL;
    free(req->chat_template);
    req->chat_template = NULL;
}

/* ── Generic top-level field access (core/json_fields.h) ──────────────────── */

/* Positions *p at the value of top-level `key`. Returns 1 if found. */
static int find_top_level(const char *json_body, const char *key, const char **p) {
    if (!json_body || !key) return 0;
    const char *q = json_body;
    json_skip_ws(&q);
    if (*q != '{') return 0;
    q++;
    while (*q && *q != '}') {
        json_skip_ws(&q);
        if (*q == '}') break;
        char k[128];
        if (!json_parse_string(&q, k, sizeof(k))) return 0;
        if (!json_expect_char(&q, ':')) return 0;
        json_skip_ws(&q);
        if (strcmp(k, key) == 0) {
            *p = q;
            return 1;
        }
        json_skip_value(&q);
        json_skip_ws(&q);
        if (*q == ',') q++;
    }
    return 0;
}

char *json_field_string(const char *json_body, const char *key) {
    const char *p;
    if (!find_top_level(json_body, key, &p) || *p != '"') return NULL;
    return json_parse_string_heap(&p);
}

int json_field_number(const char *json_body, const char *key, double *out) {
    const char *p;
    if (!find_top_level(json_body, key, &p)) return 0;
    if (!(*p == '-' || isdigit((unsigned char)*p))) return 0;
    *out = json_parse_number(&p);
    return 1;
}

int json_field_bool(const char *json_body, const char *key, int *out) {
    const char *p;
    if (!find_top_level(json_body, key, &p)) return 0;
    int b = json_parse_bool(&p);
    if (b < 0) return 0;
    *out = b;
    return 1;
}

char *json_field_number_text(const char *json_body, const char *key) {
    const char *p;
    if (!find_top_level(json_body, key, &p)) return NULL;
    if (!(*p == '-' || isdigit((unsigned char)*p))) return NULL;
    const char *start = p;
    while (*p && (isdigit((unsigned char)*p) || strchr("+-.eE", *p))) p++;
    size_t n  = (size_t)(p - start);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, start, n);
    out[n] = '\0';
    return out;
}

int json_field_string_list(const char *json_body, const char *key, char **out, int max) {
    const char *p;
    if (!find_top_level(json_body, key, &p) || !json_expect_char(&p, '[')) return -1;
    int n = 0;
    json_skip_ws(&p);
    while (*p && *p != ']') {
        char *s = (*p == '"') ? json_parse_string_heap(&p) : NULL;
        if (!s || n >= max) {
            free(s);
            while (n > 0) free(out[--n]);
            return -1;
        }
        out[n++] = s;
        json_skip_ws(&p);
        if (*p == ',') {
            p++;
            json_skip_ws(&p);
        }
    }
    if (*p != ']') {
        while (n > 0) free(out[--n]);
        return -1;
    }
    return n;
}
