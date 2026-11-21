/*
 * api/chat_request_parse.c — chat_request_parse(): a request body into a ChatRequest. Plain
 * numbers and booleans go through a table (name, type, offset, set-mask bit); the fields with
 * their own shape (stop, watermark, mcts, chat_template, response_format, messages) have their own
 * small parsers. See api/chat_request.h.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "api/chat_request.h"
#include "api/json_scan.h"
#include "core/error.h"

/* "stop": "x" | ["x", "y", ...] | null */
static TernaryError parse_stop(const char **p, ChatRequest *req) {
    json_skip_ws(p);
    if (strncmp(*p, "null", 4) == 0) {
        *p += 4;
        return TN_OK;
    }
    if (**p == '"') {
        char *s = json_parse_string_heap(p);
        if (!s) return TN_ERR_JSON_PARSE;
        if (req->output.n_stop < CHAT_MAX_STOP && s[0])
            req->output.stop[req->output.n_stop++] = s;
        else
            free(s);
        req->set_mask |= CHAT_SET_STOP;
        return TN_OK;
    }
    if (!json_expect_char(p, '[')) return TN_ERR_JSON_PARSE;
    json_skip_ws(p);
    while (**p && **p != ']') {
        char *s = json_parse_string_heap(p);
        if (!s) return TN_ERR_JSON_PARSE;
        if (req->output.n_stop < CHAT_MAX_STOP && s[0])
            req->output.stop[req->output.n_stop++] = s;
        else
            free(s);
        json_skip_ws(p);
        if (**p == ',') {
            (*p)++;
            json_skip_ws(p);
        }
    }
    if (!json_expect_char(p, ']')) return TN_ERR_JSON_PARSE;
    req->set_mask |= CHAT_SET_STOP;
    return TN_OK;
}

/* "watermark": {"enabled": bool, "key": str, "gamma": n, "delta": n} | bool */
static TernaryError parse_watermark(const char **p, ChatRequest *req) {
    json_skip_ws(p);
    int b = json_parse_bool(p);
    if (b >= 0) {
        req->watermark.enabled = (b == 1);
        req->set_mask |= CHAT_SET_WATERMARK;
        return TN_OK;
    }
    if (!json_expect_char(p, '{')) return TN_ERR_JSON_PARSE;
    while (**p && **p != '}') {
        json_skip_ws(p);
        if (**p == '}') break;
        char key[32];
        if (!json_parse_string(p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
        if (!json_expect_char(p, ':')) return TN_ERR_JSON_PARSE;
        json_skip_ws(p);
        if (strcmp(key, "enabled") == 0) {
            int e = json_parse_bool(p);
            if (e < 0) return TN_ERR_JSON_PARSE;
            req->watermark.enabled = (e == 1);
            req->set_mask |= CHAT_SET_WATERMARK;
        } else if (strcmp(key, "key") == 0) {
            char *k = json_parse_string_heap(p);
            if (!k) return TN_ERR_JSON_PARSE;
            free(req->watermark.key);
            req->watermark.key = k;
        } else if (strcmp(key, "gamma") == 0) {
            req->watermark.gamma = (float)json_parse_number(p);
            req->set_mask |= CHAT_SET_WM_GAMMA;
        } else if (strcmp(key, "delta") == 0) {
            req->watermark.delta = (float)json_parse_number(p);
            req->set_mask |= CHAT_SET_WM_DELTA;
        } else {
            json_skip_value(p);
        }
        json_skip_ws(p);
        if (**p == ',') (*p)++;
    }
    if (!json_expect_char(p, '}')) return TN_ERR_JSON_PARSE;
    return TN_OK;
}

/*
 * Parse a raw JSON request body into *req.
 * req must have been initialised with chat_request_init() first.
 */
/* Deepest { / [ nesting a request may have. Real requests nest fewer than 10
 * levels; a body deeper than this is malformed for this API. */
#define JSON_MAX_DEPTH 256

/* 1 when `s` nests objects/arrays deeper than `max`. One iterative pass that
 * skips string contents (so brackets inside strings do not count). */
static int json_too_deep(const char *s, int max) {
    int depth = 0;
    for (; *s; s++) {
        if (*s == '"') {
            for (s++; *s && *s != '"'; s++)
                if (*s == '\\' && s[1]) s++;
            if (!*s) break;
        } else if (*s == '{' || *s == '[') {
            if (++depth > max) return 1;
        } else if (*s == '}' || *s == ']') {
            if (depth > 0) depth--;
        }
    }
    return 0;
}

/* The fields that are one number or one boolean, straight into the request: JSON name, how to
 * store it, where, and which CHAT_SET_* bit records that the client sent it (0 = none). */
typedef enum { FIELD_FLOAT, FIELD_INT, FIELD_LLONG, FIELD_BOOL } FieldKind;

typedef struct {
    const char *name;
    FieldKind kind;
    size_t offset;
    unsigned mask;
} ScalarField;

static const ScalarField k_scalar_fields[] = {
    {"temperature", FIELD_FLOAT, offsetof(ChatRequest, sampling.temperature), CHAT_SET_TEMPERATURE},
    {"top_p", FIELD_FLOAT, offsetof(ChatRequest, sampling.top_p), CHAT_SET_TOP_P},
    {"max_tokens", FIELD_INT, offsetof(ChatRequest, output.max_tokens), CHAT_SET_MAX_TOKENS},
    {"max_completion_tokens", FIELD_INT, offsetof(ChatRequest, output.max_tokens),
     CHAT_SET_MAX_TOKENS},
    {"top_k", FIELD_INT, offsetof(ChatRequest, sampling.top_k), CHAT_SET_TOP_K},
    {"min_p", FIELD_FLOAT, offsetof(ChatRequest, sampling.min_p), CHAT_SET_MIN_P},
    {"repetition_penalty", FIELD_FLOAT, offsetof(ChatRequest, sampling.repetition_penalty),
     CHAT_SET_REP_PENALTY},
    {"repeat_penalty", FIELD_FLOAT, offsetof(ChatRequest, sampling.repetition_penalty),
     CHAT_SET_REP_PENALTY},
    {"presence_penalty", FIELD_FLOAT, offsetof(ChatRequest, sampling.presence_penalty),
     CHAT_SET_PRES_PENALTY},
    {"frequency_penalty", FIELD_FLOAT, offsetof(ChatRequest, sampling.frequency_penalty),
     CHAT_SET_FREQ_PENALTY},
    {"penalty_last_n", FIELD_INT, offsetof(ChatRequest, sampling.penalty_last_n),
     CHAT_SET_PENALTY_N},
    {"repeat_last_n", FIELD_INT, offsetof(ChatRequest, sampling.penalty_last_n),
     CHAT_SET_PENALTY_N},
    {"seed", FIELD_LLONG, offsetof(ChatRequest, sampling.seed), CHAT_SET_SEED},
    {"best_of", FIELD_INT, offsetof(ChatRequest, search.best_of), CHAT_SET_BEST_OF},
    {"ignore_eos", FIELD_BOOL, offsetof(ChatRequest, output.ignore_eos), CHAT_SET_IGNORE_EOS},
    {"cache_prompt", FIELD_BOOL, offsetof(ChatRequest, cache_prompt), CHAT_SET_CACHE_PROMPT},
    {"stream", FIELD_BOOL, offsetof(ChatRequest, output.stream), 0},
};

/* The keys of the "mcts" object (the CHAT_SET_MCTS bit is set for the whole field). */
static const ScalarField k_mcts_fields[] = {
    {"simulations", FIELD_INT, offsetof(ChatRequest, search.mcts.simulations), 0},
    {"expand", FIELD_INT, offsetof(ChatRequest, search.mcts.expand), 0},
    {"max_depth", FIELD_INT, offsetof(ChatRequest, search.mcts.max_depth), 0},
    {"step_tokens", FIELD_INT, offsetof(ChatRequest, search.mcts.step_tokens), 0},
    {"c", FIELD_FLOAT, offsetof(ChatRequest, search.mcts.c_ucb), 0},
    {"temperature", FIELD_FLOAT, offsetof(ChatRequest, search.mcts.temperature), 0},
};

static const ScalarField *find_field(const ScalarField *table, size_t n, const char *key) {
    for (size_t i = 0; i < n; i++)
        if (strcmp(key, table[i].name) == 0) return &table[i];
    return NULL;
}

static void store_number(FieldKind kind, char *dst, double v) {
    if (kind == FIELD_FLOAT)
        *(float *)dst = (float)v;
    else if (kind == FIELD_INT)
        *(int *)dst = (int)v;
    else
        *(long long *)dst = (long long)v;
}

/* Reads the value of scalar field `f` into the request and records its set-mask bit. */
static TernaryError parse_scalar(const char **p, ChatRequest *req, const ScalarField *f) {
    char *dst = (char *)req + f->offset;
    if (f->kind == FIELD_BOOL) {
        int b = json_parse_bool(p);
        if (b < 0) return TN_ERR_JSON_PARSE;
        *(bool *)dst = (b == 1);
    } else {
        store_number(f->kind, dst, json_parse_number(p));
    }
    req->set_mask |= f->mask;
    return TN_OK;
}

/* Reads `"key":` and the whitespace after the colon; 0 on a malformed member. */
static int read_member_key(const char **p, char *key, size_t cap) {
    if (!json_parse_string(p, key, cap)) return 0;
    if (!json_expect_char(p, ':')) return 0;
    json_skip_ws(p);
    return 1;
}

/* "mcts": N (simulations) | {"simulations", "expand", "max_depth", "step_tokens", "c",
 * "temperature"}; unknown keys inside the object are skipped. */
static TernaryError parse_mcts(const char **p, ChatRequest *req) {
    json_skip_ws(p);
    if (**p != '{') {
        req->search.mcts.simulations = (int)json_parse_number(p);
        req->set_mask |= CHAT_SET_MCTS;
        return TN_OK;
    }
    (*p)++;
    while (**p && **p != '}') {
        json_skip_ws(p);
        if (**p == '}') break;
        char mk[32];
        if (!read_member_key(p, mk, sizeof(mk))) return TN_ERR_JSON_PARSE;
        const ScalarField *f = find_field(k_mcts_fields, sizeof(k_mcts_fields) / sizeof(*f), mk);
        if (f)
            parse_scalar(p, req, f);
        else
            json_skip_value(p);
        json_skip_ws(p);
        if (**p == ',') (*p)++;
    }
    if (**p != '}') return TN_ERR_JSON_PARSE;
    (*p)++;
    req->set_mask |= CHAT_SET_MCTS;
    return TN_OK;
}

/* "chat_template": string | null; "" and null mean the server's template */
static TernaryError parse_chat_template(const char **p, ChatRequest *req) {
    json_skip_ws(p);
    if (strncmp(*p, "null", 4) == 0) {
        *p += 4; /* null = use the server's template */
        return TN_OK;
    }
    char *t = json_parse_string_heap(p);
    if (!t) return TN_ERR_JSON_PARSE;
    if (strlen(t) > CHAT_MAX_TEMPLATE) {
        free(t);
        return TN_ERR_INVALID_ARGS;
    }
    free(req->chat_template);
    req->chat_template = NULL;
    if (t[0]) {
        req->chat_template = t;
        req->set_mask |= CHAT_SET_CHAT_TEMPLATE;
    } else
        free(t); /* "" = use the server's template */
    return TN_OK;
}

/* The value of "response_format"."type": "json_object" turns on the grammar engine, "text" is a
 * no-op, anything else is refused. */
static TernaryError parse_response_type(const char **p, ChatRequest *req) {
    char type_val[32];
    if (!json_parse_string(p, type_val, sizeof(type_val))) return TN_ERR_JSON_PARSE;
    if (strcmp(type_val, "json_object") != 0 && strcmp(type_val, "text") != 0)
        return TN_ERR_INVALID_ARGS;
    req->output.json_mode = (strcmp(type_val, "json_object") == 0);
    req->set_mask |= CHAT_SET_JSON_MODE;
    return TN_OK;
}

/* "response_format": {"type": ...}; a non-object value is skipped. */
static TernaryError parse_response_format(const char **p, ChatRequest *req) {
    json_skip_ws(p);
    if (**p != '{') {
        json_skip_value(p);
        return TN_OK;
    }
    (*p)++;
    while (**p && **p != '}') {
        json_skip_ws(p);
        if (**p == '}') break;
        char rf_key[64];
        if (!read_member_key(p, rf_key, sizeof(rf_key))) return TN_ERR_JSON_PARSE;
        TernaryError err = TN_OK;
        if (strcmp(rf_key, "type") == 0)
            err = parse_response_type(p, req);
        else
            json_skip_value(p);
        if (err != TN_OK) return err;
        json_skip_ws(p);
        if (**p == ',') (*p)++;
    }
    if (**p == '}') (*p)++;
    return TN_OK;
}

/* The fields with their own parser, after the scalar table. */
static const struct {
    const char *name;
    TernaryError (*parse)(const char **p, ChatRequest *req);
} k_object_fields[] = {
    {"stop", parse_stop},
    {"watermark", parse_watermark},
    {"mcts", parse_mcts},
    {"chat_template", parse_chat_template},
    {"response_format", parse_response_format},
};

/* One top-level field of the request, the key already read and the colon consumed. An explicit
 * null means "use the default" for any field except model and messages; unknown fields are
 * skipped whole. */
static TernaryError parse_request_field(const char **p, ChatRequest *req, const char *key) {
    if (strcmp(key, "model") == 0)
        return json_parse_string(p, req->model, sizeof(req->model)) ? TN_OK : TN_ERR_JSON_PARSE;
    if (strcmp(key, "messages") == 0) return json_parse_messages(p, req);
    if (strncmp(*p, "null", 4) == 0) {
        *p += 4;
        return TN_OK;
    }
    const ScalarField *f = find_field(k_scalar_fields, sizeof(k_scalar_fields) / sizeof(*f), key);
    if (f) return parse_scalar(p, req, f);
    for (size_t i = 0; i < sizeof(k_object_fields) / sizeof(k_object_fields[0]); i++)
        if (strcmp(key, k_object_fields[i].name) == 0) return k_object_fields[i].parse(p, req);
    json_skip_value(p);
    return TN_OK;
}

TernaryError chat_request_parse(const char *json_body, ChatRequest *req) {
    if (!json_body || !req) return TN_ERR_INVALID_ARGS;
    if (json_too_deep(json_body, JSON_MAX_DEPTH)) return TN_ERR_JSON_PARSE;

    const char *p = json_body;
    json_skip_ws(&p);
    if (*p != '{') return TN_ERR_JSON_PARSE;
    p++; /* skip '{' */

    while (*p && *p != '}') {
        json_skip_ws(&p);
        if (*p == '}') break;

        char key[128];
        if (!json_parse_string(&p, key, sizeof(key))) return TN_ERR_JSON_PARSE;
        if (!json_expect_char(&p, ':')) return TN_ERR_JSON_PARSE;
        json_skip_ws(&p);

        TernaryError err = parse_request_field(&p, req, key);
        if (err != TN_OK) return err;

        json_skip_ws(&p);
        if (*p == ',') {
            p++;
        }
    }
    return TN_OK;
}
