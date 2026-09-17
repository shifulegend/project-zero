/*
 * grammar_json.c — Phase 20: JSON-mode pushdown automaton.
 *
 * See grammar.h for the scope decision (JSON-specific PDA, not a general
 * BNF compiler). Implements RFC 8259 exactly enough to reject anything that
 * cannot be a prefix of valid JSON, while accepting every real prefix
 * (including ones mid-number, mid-literal, or mid-string, which are legal
 * "not yet complete" states, not errors).
 */
#include "sampling/grammar.h"

static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
static int is_digit(char c) { return c >= '0' && c <= '9'; }
static int is_hex(char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

void json_grammar_init(JsonGrammarState *g) {
    g->state = JSON_ST_VALUE;
    g->sp = 0;
}

int json_grammar_is_complete(const JsonGrammarState *g) {
    return g->state == JSON_ST_DONE;
}

/* Pushes `resume_state` (what to become once the value about to start
 * closes) and returns the state to enter now (always JSON_ST_VALUE, since
 * we're about to parse a fresh nested value). Returns JSON_ST_INVALID if
 * nesting exceeds JSON_GRAMMAR_MAX_DEPTH (a real, generous safety cap --
 * not a design limitation, see grammar.h; 256 levels is far beyond any
 * realistic tool-call/response JSON). */
static JsonState push_value(JsonGrammarState *g, JsonState resume_state) {
    if (g->sp >= JSON_GRAMMAR_MAX_DEPTH) return JSON_ST_INVALID;
    g->stack[g->sp++] = resume_state;
    return JSON_ST_VALUE;
}

/* A value (of any kind) just completed. Pops the stack to whatever context
 * should resume -- JSON_ST_DONE at top level, JSON_ST_OBJ_AFTER_VALUE or
 * JSON_ST_ARR_AFTER_VALUE when nested. */
static JsonState close_value(JsonGrammarState *g) {
    /* sp==0 means the value that just closed was the top-level value itself
     * (its "resume address" was never pushed -- only nested values get a
     * push, from the ':' and '['/',' handlers below). */
    if (g->sp == 0) return JSON_ST_DONE;
    g->sp--;
    return g->stack[g->sp];
}

JsonState json_grammar_finalize(JsonGrammarState *g) {
    switch (g->state) {
    case JSON_ST_NUM_INT:
    case JSON_ST_NUM_FRAC:
    case JSON_ST_NUM_EXP:
        g->state = close_value(g);
        break;
    default:
        break;
    }
    return g->state;
}

JsonState json_grammar_step(JsonGrammarState *g, char c) {
    if (g->state == JSON_ST_INVALID) return JSON_ST_INVALID;

    for (;;) {
        switch (g->state) {

        case JSON_ST_VALUE:
            if (is_ws(c)) return g->state; /* stay */
            if (c == '{') { g->state = JSON_ST_OBJ_OPEN; return g->state; }
            if (c == '[') { g->state = JSON_ST_ARR_OPEN; return g->state; }
            if (c == '"') { g->state = JSON_ST_STRING; return g->state; }
            if (c == '-') { g->state = JSON_ST_NUM_INT_FIRST; return g->state; }
            if (is_digit(c)) { g->state = JSON_ST_NUM_INT; return g->state; }
            if (c == 't') { g->state = JSON_ST_LIT_TRUE_T; return g->state; }
            if (c == 'f') { g->state = JSON_ST_LIT_FALSE_F; return g->state; }
            if (c == 'n') { g->state = JSON_ST_LIT_NULL_N; return g->state; }
            g->state = JSON_ST_INVALID; return g->state;

        /* ── Object ──────────────────────────────────────────────────── */
        case JSON_ST_OBJ_OPEN:
            if (is_ws(c)) return g->state;
            if (c == '}') { g->state = close_value(g); return g->state; }
            if (c == '"') { g->state = JSON_ST_OBJ_KEY; return g->state; }
            g->state = JSON_ST_INVALID; return g->state;

        case JSON_ST_OBJ_KEY_NEXT:
            if (is_ws(c)) return g->state;
            if (c == '"') { g->state = JSON_ST_OBJ_KEY; return g->state; }
            g->state = JSON_ST_INVALID; return g->state;

        case JSON_ST_OBJ_KEY:
            if (c == '"') { g->state = JSON_ST_OBJ_COLON; return g->state; }
            if (c == '\\') { g->state = JSON_ST_OBJ_KEY_ESC; return g->state; }
            if ((unsigned char)c < 0x20) { g->state = JSON_ST_INVALID; return g->state; }
            return g->state; /* any other byte is a legal string content byte */

        case JSON_ST_OBJ_KEY_ESC:
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
                c == 'n' || c == 'r' || c == 't') { g->state = JSON_ST_OBJ_KEY; return g->state; }
            if (c == 'u') { g->state = JSON_ST_OBJ_KEY_ESC_U1; return g->state; }
            g->state = JSON_ST_INVALID; return g->state;
        case JSON_ST_OBJ_KEY_ESC_U1:
            g->state = is_hex(c) ? JSON_ST_OBJ_KEY_ESC_U2 : JSON_ST_INVALID; return g->state;
        case JSON_ST_OBJ_KEY_ESC_U2:
            g->state = is_hex(c) ? JSON_ST_OBJ_KEY_ESC_U3 : JSON_ST_INVALID; return g->state;
        case JSON_ST_OBJ_KEY_ESC_U3:
            g->state = is_hex(c) ? JSON_ST_OBJ_KEY_ESC_U4 : JSON_ST_INVALID; return g->state;
        case JSON_ST_OBJ_KEY_ESC_U4:
            g->state = is_hex(c) ? JSON_ST_OBJ_KEY : JSON_ST_INVALID; return g->state;

        case JSON_ST_OBJ_COLON:
            if (is_ws(c)) return g->state;
            if (c == ':') { g->state = push_value(g, JSON_ST_OBJ_AFTER_VALUE); return g->state; }
            g->state = JSON_ST_INVALID; return g->state;

        case JSON_ST_OBJ_AFTER_VALUE:
            if (is_ws(c)) return g->state;
            if (c == ',') { g->state = JSON_ST_OBJ_KEY_NEXT; return g->state; }
            if (c == '}') { g->state = close_value(g); return g->state; }
            g->state = JSON_ST_INVALID; return g->state;

        /* ── Array ───────────────────────────────────────────────────── */
        case JSON_ST_ARR_OPEN:
            if (is_ws(c)) return g->state;
            if (c == ']') { g->state = close_value(g); return g->state; }
            g->state = push_value(g, JSON_ST_ARR_AFTER_VALUE);
            continue; /* reprocess c as the start of that value */

        case JSON_ST_ARR_AFTER_VALUE:
            if (is_ws(c)) return g->state;
            if (c == ',') { g->state = push_value(g, JSON_ST_ARR_AFTER_VALUE); return g->state; }
            if (c == ']') { g->state = close_value(g); return g->state; }
            g->state = JSON_ST_INVALID; return g->state;

        /* ── String value (not an object key) ───────────────────────── */
        case JSON_ST_STRING:
            if (c == '"') { g->state = close_value(g); return g->state; }
            if (c == '\\') { g->state = JSON_ST_STRING_ESC; return g->state; }
            if ((unsigned char)c < 0x20) { g->state = JSON_ST_INVALID; return g->state; }
            return g->state;

        case JSON_ST_STRING_ESC:
            if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
                c == 'n' || c == 'r' || c == 't') { g->state = JSON_ST_STRING; return g->state; }
            if (c == 'u') { g->state = JSON_ST_STRING_ESC_U1; return g->state; }
            g->state = JSON_ST_INVALID; return g->state;
        case JSON_ST_STRING_ESC_U1:
            g->state = is_hex(c) ? JSON_ST_STRING_ESC_U2 : JSON_ST_INVALID; return g->state;
        case JSON_ST_STRING_ESC_U2:
            g->state = is_hex(c) ? JSON_ST_STRING_ESC_U3 : JSON_ST_INVALID; return g->state;
        case JSON_ST_STRING_ESC_U3:
            g->state = is_hex(c) ? JSON_ST_STRING_ESC_U4 : JSON_ST_INVALID; return g->state;
        case JSON_ST_STRING_ESC_U4:
            g->state = is_hex(c) ? JSON_ST_STRING : JSON_ST_INVALID; return g->state;

        /* ── Numbers (RFC 8259: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?) ──
         * Numbers have no closing delimiter of their own -- a whitespace,
         * ',', '}', ']' (or end of input, which this function never sees
         * directly, handled by the caller checking json_grammar_is_complete
         * only after a value-closing transition) all mark the end. Any of
         * those close the value NOW and reprocess `c` in the resumed
         * context via `continue`. */
        case JSON_ST_NUM_INT_FIRST:
            g->state = is_digit(c) ? JSON_ST_NUM_INT : JSON_ST_INVALID; return g->state;

        case JSON_ST_NUM_INT:
            if (is_digit(c)) return g->state;
            if (c == '.') { g->state = JSON_ST_NUM_FRAC_FIRST; return g->state; }
            if (c == 'e' || c == 'E') { g->state = JSON_ST_NUM_EXP_FIRST; return g->state; }
            g->state = close_value(g);
            continue;

        case JSON_ST_NUM_FRAC_FIRST:
            g->state = is_digit(c) ? JSON_ST_NUM_FRAC : JSON_ST_INVALID; return g->state;

        case JSON_ST_NUM_FRAC:
            if (is_digit(c)) return g->state;
            if (c == 'e' || c == 'E') { g->state = JSON_ST_NUM_EXP_FIRST; return g->state; }
            g->state = close_value(g);
            continue;

        case JSON_ST_NUM_EXP_FIRST:
            if (c == '+' || c == '-') { g->state = JSON_ST_NUM_EXP_SIGN; return g->state; }
            g->state = is_digit(c) ? JSON_ST_NUM_EXP : JSON_ST_INVALID; return g->state;

        case JSON_ST_NUM_EXP_SIGN:
            g->state = is_digit(c) ? JSON_ST_NUM_EXP : JSON_ST_INVALID; return g->state;

        case JSON_ST_NUM_EXP:
            if (is_digit(c)) return g->state;
            g->state = close_value(g);
            continue;

        /* ── Literals: true / false / null ───────────────────────────── */
        case JSON_ST_LIT_TRUE_T:
            g->state = (c == 'r') ? JSON_ST_LIT_TRUE_R : JSON_ST_INVALID; return g->state;
        case JSON_ST_LIT_TRUE_R:
            g->state = (c == 'u') ? JSON_ST_LIT_TRUE_U : JSON_ST_INVALID; return g->state;
        case JSON_ST_LIT_TRUE_U:
            g->state = (c == 'e') ? close_value(g) : JSON_ST_INVALID; return g->state;

        case JSON_ST_LIT_FALSE_F:
            g->state = (c == 'a') ? JSON_ST_LIT_FALSE_A : JSON_ST_INVALID; return g->state;
        case JSON_ST_LIT_FALSE_A:
            g->state = (c == 'l') ? JSON_ST_LIT_FALSE_L : JSON_ST_INVALID; return g->state;
        case JSON_ST_LIT_FALSE_L:
            g->state = (c == 's') ? JSON_ST_LIT_FALSE_S : JSON_ST_INVALID; return g->state;
        case JSON_ST_LIT_FALSE_S:
            g->state = (c == 'e') ? close_value(g) : JSON_ST_INVALID; return g->state;

        case JSON_ST_LIT_NULL_N:
            g->state = (c == 'u') ? JSON_ST_LIT_NULL_U : JSON_ST_INVALID; return g->state;
        case JSON_ST_LIT_NULL_U:
            g->state = (c == 'l') ? JSON_ST_LIT_NULL_L : JSON_ST_INVALID; return g->state;
        case JSON_ST_LIT_NULL_L:
            g->state = (c == 'l') ? close_value(g) : JSON_ST_INVALID; return g->state;

        /* ── Done: only trailing whitespace is legal ────────────────── */
        case JSON_ST_DONE:
            g->state = is_ws(c) ? JSON_ST_DONE : JSON_ST_INVALID; return g->state;

        case JSON_ST_INVALID:
        default:
            return JSON_ST_INVALID;
        }
    }
}
