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

#include "api/chat_request.h"
#include "api/json_fields.h"
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

static int hex4(const char *s, unsigned *out) {
    unsigned v = 0;
    for (int k = 0; k < 4; k++) {
        char h = s[k];
        v <<= 4;
        if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
        else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
        else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

static int utf8_encode(unsigned cp, char *o) {
    if (cp < 0x80)    { o[0] = (char)cp; return 1; }
    if (cp < 0x800)   { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        o[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
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
                case 'u': {
                    /* \uXXXX (+ optional low surrogate) -> UTF-8. The
                     * escape is >= 6 input bytes and UTF-8 output <= 4 per
                     * code point, so parse_string_heap's length bound holds.
                     * (2026-10-18: previously every \u escape became '?'.) */
                    unsigned cp = 0;
                    if (!hex4(*p + 1, &cp)) { c = '?'; break; }
                    (*p) += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && (*p)[1] == '\\' && (*p)[2] == 'u') {
                        unsigned lo = 0;
                        if (hex4(*p + 3, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            (*p) += 6;
                        }
                    }
                    if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD; /* lone surrogate */
                    char u8[4];
                    int nb = utf8_encode(cp, u8);
                    for (int k = 0; k < nb - 1; k++)
                        if (i < dst_cap - 1) dst[i++] = u8[k];
                    c = u8[nb - 1];
                    break;
                }
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
                } else if (strncmp(*p, "null", 4) == 0) {
                    /* OpenAI allows "content": null (e.g. assistant turns
                     * that only carried tool calls) — treat as empty. */
                    *p += 4;
                    msg->content = strdup("");
                    if (!msg->content) return TN_ERR_OOM;
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
    for (int i = 0; i < req->n_stop; i++) {
        free(req->stop[i]);
        req->stop[i] = NULL;
    }
    req->n_stop = 0;
    free(req->watermark_key);
    req->watermark_key = NULL;
}

/* "stop": "x" | ["x", "y", ...] | null */
static TernaryError parse_stop(const char **p, ChatRequest *req) {
    skip_ws(p);
    if (strncmp(*p, "null", 4) == 0) { *p += 4; return TN_OK; }
    if (**p == '"') {
        char *s = parse_string_heap(p);
        if (!s) return TN_ERR_JSON_PARSE;
        if (req->n_stop < CHAT_MAX_STOP && s[0]) req->stop[req->n_stop++] = s;
        else free(s);
        req->set_mask |= CHAT_SET_STOP;
        return TN_OK;
    }
    if (!expect_char(p, '[')) return TN_ERR_JSON_PARSE;
    skip_ws(p);
    while (**p && **p != ']') {
        char *s = parse_string_heap(p);
        if (!s) return TN_ERR_JSON_PARSE;
        if (req->n_stop < CHAT_MAX_STOP && s[0]) req->stop[req->n_stop++] = s;
        else free(s);
        skip_ws(p);
        if (**p == ',') { (*p)++; skip_ws(p); }
    }
    if (!expect_char(p, ']')) return TN_ERR_JSON_PARSE;
    req->set_mask |= CHAT_SET_STOP;
    return TN_OK;
}

/* "watermark": {"enabled": bool, "key": str, "gamma": n, "delta": n} | bool */
static TernaryError parse_watermark(const char **p, ChatRequest *req) {
    skip_ws(p);
    int b = parse_bool(p);
    if (b >= 0) {
        req->watermark_enabled = (b == 1);
        req->set_mask |= CHAT_SET_WATERMARK;
        return TN_OK;
    }
    if (!expect_char(p, '{')) return TN_ERR_JSON_PARSE;
    while (**p && **p != '}') {
        skip_ws(p);
        if (**p == '}') break;
        char key[32];
        if (!parse_string(p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
        if (!expect_char(p, ':')) return TN_ERR_JSON_PARSE;
        skip_ws(p);
        if (strcmp(key, "enabled") == 0) {
            int e = parse_bool(p);
            if (e < 0) return TN_ERR_JSON_PARSE;
            req->watermark_enabled = (e == 1);
            req->set_mask |= CHAT_SET_WATERMARK;
        } else if (strcmp(key, "key") == 0) {
            char *k = parse_string_heap(p);
            if (!k) return TN_ERR_JSON_PARSE;
            free(req->watermark_key);
            req->watermark_key = k;
        } else if (strcmp(key, "gamma") == 0) {
            req->watermark_gamma = (float)parse_number(p);
            req->set_mask |= CHAT_SET_WM_GAMMA;
        } else if (strcmp(key, "delta") == 0) {
            req->watermark_delta = (float)parse_number(p);
            req->set_mask |= CHAT_SET_WM_DELTA;
        } else {
            skip_value(p);
        }
        skip_ws(p);
        if (**p == ',') (*p)++;
    }
    if (!expect_char(p, '}')) return TN_ERR_JSON_PARSE;
    return TN_OK;
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
        } else if (strncmp(p, "null", 4) == 0) {
            p += 4; /* explicit null = "use the default" for any field */
        } else if (strcmp(key, "temperature") == 0) {
            req->temperature = (float)parse_number(&p);
            req->set_mask |= CHAT_SET_TEMPERATURE;
        } else if (strcmp(key, "top_p") == 0) {
            req->top_p = (float)parse_number(&p);
            req->set_mask |= CHAT_SET_TOP_P;
        } else if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_completion_tokens") == 0) {
            req->max_tokens = (int)parse_number(&p);
            req->set_mask |= CHAT_SET_MAX_TOKENS;
        } else if (strcmp(key, "top_k") == 0) {
            req->top_k = (int)parse_number(&p);
            req->set_mask |= CHAT_SET_TOP_K;
        } else if (strcmp(key, "min_p") == 0) {
            req->min_p = (float)parse_number(&p);
            req->set_mask |= CHAT_SET_MIN_P;
        } else if (strcmp(key, "repetition_penalty") == 0 || strcmp(key, "repeat_penalty") == 0) {
            req->repetition_penalty = (float)parse_number(&p);
            req->set_mask |= CHAT_SET_REP_PENALTY;
        } else if (strcmp(key, "presence_penalty") == 0) {
            req->presence_penalty = (float)parse_number(&p);
            req->set_mask |= CHAT_SET_PRES_PENALTY;
        } else if (strcmp(key, "frequency_penalty") == 0) {
            req->frequency_penalty = (float)parse_number(&p);
            req->set_mask |= CHAT_SET_FREQ_PENALTY;
        } else if (strcmp(key, "penalty_last_n") == 0 || strcmp(key, "repeat_last_n") == 0) {
            req->penalty_last_n = (int)parse_number(&p);
            req->set_mask |= CHAT_SET_PENALTY_N;
        } else if (strcmp(key, "seed") == 0) {
            req->seed = (long long)parse_number(&p);
            req->set_mask |= CHAT_SET_SEED;
        } else if (strcmp(key, "stop") == 0) {
            TernaryError err = parse_stop(&p, req);
            if (err != TN_OK) return err;
        } else if (strcmp(key, "watermark") == 0) {
            TernaryError err = parse_watermark(&p, req);
            if (err != TN_OK) return err;
        } else if (strcmp(key, "best_of") == 0) {
            req->best_of = (int)parse_number(&p);
            req->set_mask |= CHAT_SET_BEST_OF;
        } else if (strcmp(key, "cache_prompt") == 0) {
            int b = parse_bool(&p);
            if (b < 0) return TN_ERR_JSON_PARSE;
            req->cache_prompt = (b == 1);
            req->set_mask |= CHAT_SET_CACHE_PROMPT;
        } else if (strcmp(key, "stream") == 0) {
            int b = parse_bool(&p);
            if (b < 0) return TN_ERR_JSON_PARSE;
            req->stream = (b == 1);
        } else if (strcmp(key, "response_format") == 0) {
            /* {"type": "json_object"} -- only key this project's grammar
             * engine can honor; any other response_format value (or the
             * default absence of one) leaves json_mode false. */
            skip_ws(&p);
            if (*p == '{') {
                p++;
                while (*p && *p != '}') {
                    skip_ws(&p);
                    if (*p == '}') break;
                    char rf_key[64];
                    if (!parse_string(&p, rf_key, sizeof(rf_key))) return TN_ERR_JSON_PARSE;
                    if (!expect_char(&p, ':')) return TN_ERR_JSON_PARSE;
                    skip_ws(&p);
                    if (strcmp(rf_key, "type") == 0) {
                        char type_val[32];
                        if (!parse_string(&p, type_val, sizeof(type_val))) return TN_ERR_JSON_PARSE;
                        req->json_mode = (strcmp(type_val, "json_object") == 0);
                        req->set_mask |= CHAT_SET_JSON_MODE;
                    } else {
                        skip_value(&p);
                    }
                    skip_ws(&p);
                    if (*p == ',') p++;
                }
                if (*p == '}') p++;
            } else {
                skip_value(&p);
            }
        } else {
            skip_value(&p);
        }

        skip_ws(&p);
        if (*p == ',') { p++; }
    }
    return TN_OK;
}

/* ── Generic top-level field access (api/json_fields.h) ──────────────────── */

/* Positions *p at the value of top-level `key`. Returns 1 if found. */
static int find_top_level(const char *json_body, const char *key, const char **p) {
    if (!json_body || !key) return 0;
    const char *q = json_body;
    skip_ws(&q);
    if (*q != '{') return 0;
    q++;
    while (*q && *q != '}') {
        skip_ws(&q);
        if (*q == '}') break;
        char k[128];
        if (!parse_string(&q, k, sizeof(k))) return 0;
        if (!expect_char(&q, ':')) return 0;
        skip_ws(&q);
        if (strcmp(k, key) == 0) { *p = q; return 1; }
        skip_value(&q);
        skip_ws(&q);
        if (*q == ',') q++;
    }
    return 0;
}

char *json_field_string(const char *json_body, const char *key) {
    const char *p;
    if (!find_top_level(json_body, key, &p) || *p != '"') return NULL;
    return parse_string_heap(&p);
}

int json_field_number(const char *json_body, const char *key, double *out) {
    const char *p;
    if (!find_top_level(json_body, key, &p)) return 0;
    if (!(*p == '-' || isdigit((unsigned char)*p))) return 0;
    *out = parse_number(&p);
    return 1;
}

int json_field_bool(const char *json_body, const char *key, int *out) {
    const char *p;
    if (!find_top_level(json_body, key, &p)) return 0;
    int b = parse_bool(&p);
    if (b < 0) return 0;
    *out = b;
    return 1;
}
