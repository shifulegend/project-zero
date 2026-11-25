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

static int is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
static int is_digit(char c) {
    return c >= '0' && c <= '9';
}
static int is_hex(char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

void json_grammar_init(JsonGrammarState *g) {
    g->state = JSON_ST_VALUE;
    g->sp    = 0;
    g->root  = 0;
}

void json_grammar_init_root(JsonGrammarState *g) {
    g->state = JSON_ST_ROOT;
    g->sp    = 0;
    g->root  = 1;
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
        case JSON_ST_NUM_INT_ZERO:
        case JSON_ST_NUM_FRAC:
        case JSON_ST_NUM_EXP:
            g->state = close_value(g);
            break;
        default:
            break;
    }
    return g->state;
}

/* One transition per state, as one switch with a short expression per case (the dispatch is a
 * single jump table; the helpers below are small and inlined). A helper sets *again when the byte
 * must be re-read in the new state: a number or value closed on a delimiter that belongs to the
 * enclosing context. */

/* Whitespace keeps the state; `a` / `b` are the two bytes the state accepts (0 = none). */
static inline int simple_escape(char c) {
    return c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' ||
           c == 't';
}

/* Inside a string: a closing quote leads to `on_quote`, a backslash to `esc`, control bytes are
 * invalid and anything else stays in the string. */
static inline JsonState in_string(JsonState self, char c, JsonState on_quote, JsonState esc) {
    if (c == '"') return on_quote;
    if (c == '\\') return esc;
    if ((unsigned char)c < 0x20) return JSON_ST_INVALID;
    return self;
}

/* After a backslash: a simple escape returns to the string, 'u' starts four hex digits. */
static inline JsonState after_backslash(char c, JsonState body, JsonState u1) {
    if (simple_escape(c)) return body;
    if (c == 'u') return u1;
    return JSON_ST_INVALID;
}

/* One hex digit of a \uXXXX escape: on to `next`, or invalid. */
static inline JsonState hex_digit(char c, JsonState next) {
    return is_hex(c) ? next : JSON_ST_INVALID;
}

/* Next expected byte of a literal (true / false / null): on to `next`, or invalid. */
static inline JsonState literal_byte(char c, char want, JsonState next) {
    return c == want ? next : JSON_ST_INVALID;
}

/* A value starts here. */
static inline JsonState start_value(char c) {
    if (is_ws(c)) return JSON_ST_VALUE; /* stay */
    if (c == '{') return JSON_ST_OBJ_OPEN;
    if (c == '[') return JSON_ST_ARR_OPEN;
    if (c == '"') return JSON_ST_STRING;
    if (c == '-') return JSON_ST_NUM_INT_FIRST;
    if (c == '0') return JSON_ST_NUM_INT_ZERO;
    if (is_digit(c)) return JSON_ST_NUM_INT;
    if (c == 't') return JSON_ST_LIT_TRUE_T;
    if (c == 'f') return JSON_ST_LIT_FALSE_F;
    if (c == 'n') return JSON_ST_LIT_NULL_N;
    return JSON_ST_INVALID;
}

/* ── Numbers (RFC 8259: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?) ──
 * Numbers have no closing delimiter of their own -- a whitespace, ',', '}', ']' (or end of input,
 * which this function never sees directly, handled by the caller checking json_grammar_is_complete
 * only after a value-closing transition) all mark the end. Any of those close the value NOW and
 * reprocess `c` in the resumed context. */

/* Sets g->state to `next` and returns it. */
static inline JsonState set(JsonGrammarState *g, JsonState next) {
    g->state = next;
    return next;
}

/* After the digits of an integer or a fraction: '.' (when `may_frac`) and an exponent continue the
 * number. Returns 1 after moving to that state; 0 means the byte ends the number. */
static inline int number_continues(JsonGrammarState *g, char c, int may_frac) {
    if (may_frac && c == '.') {
        g->state = JSON_ST_NUM_FRAC_FIRST;
        return 1;
    }
    if (c == 'e' || c == 'E') {
        g->state = JSON_ST_NUM_EXP_FIRST;
        return 1;
    }
    return 0;
}

JsonState json_grammar_step(JsonGrammarState *g, char c) {
    if (g->state == JSON_ST_INVALID) return JSON_ST_INVALID;

    for (;;) { /* `continue` re-reads c in the state that just closed */
        switch (g->state) {
            case JSON_ST_VALUE:
                return set(g, start_value(c));

            /* ── Object ──────────────────────────────────────────────── */
            case JSON_ST_OBJ_OPEN:
                if (is_ws(c)) return g->state;
                if (c == '}') return set(g, close_value(g));
                return set(g, c == '"' ? JSON_ST_OBJ_KEY : JSON_ST_INVALID);
            case JSON_ST_OBJ_KEY_NEXT:
                if (is_ws(c)) return g->state;
                return set(g, c == '"' ? JSON_ST_OBJ_KEY : JSON_ST_INVALID);
            case JSON_ST_OBJ_COLON:
                if (is_ws(c)) return g->state;
                if (c == ':') return set(g, push_value(g, JSON_ST_OBJ_AFTER_VALUE));
                return set(g, JSON_ST_INVALID);
            case JSON_ST_OBJ_AFTER_VALUE:
                if (is_ws(c)) return g->state;
                if (c == ',') return set(g, JSON_ST_OBJ_KEY_NEXT);
                if (c == '}') return set(g, close_value(g));
                return set(g, JSON_ST_INVALID);

            /* ── Array ───────────────────────────────────────────────── */
            case JSON_ST_ARR_OPEN:
                if (is_ws(c)) return g->state;
                if (c == ']') return set(g, close_value(g));
                g->state = push_value(g, JSON_ST_ARR_AFTER_VALUE);
                continue; /* reprocess c as the start of that value */
            case JSON_ST_ARR_AFTER_VALUE:
                if (is_ws(c)) return g->state;
                if (c == ',') return set(g, push_value(g, JSON_ST_ARR_AFTER_VALUE));
                if (c == ']') return set(g, close_value(g));
                return set(g, JSON_ST_INVALID);

            /* ── Object key string ───────────────────────────────────── */
            case JSON_ST_OBJ_KEY:
                return set(g, in_string(g->state, c, JSON_ST_OBJ_COLON, JSON_ST_OBJ_KEY_ESC));
            case JSON_ST_OBJ_KEY_ESC:
                return set(g, after_backslash(c, JSON_ST_OBJ_KEY, JSON_ST_OBJ_KEY_ESC_U1));
            case JSON_ST_OBJ_KEY_ESC_U1:
                return set(g, hex_digit(c, JSON_ST_OBJ_KEY_ESC_U2));
            case JSON_ST_OBJ_KEY_ESC_U2:
                return set(g, hex_digit(c, JSON_ST_OBJ_KEY_ESC_U3));
            case JSON_ST_OBJ_KEY_ESC_U3:
                return set(g, hex_digit(c, JSON_ST_OBJ_KEY_ESC_U4));
            case JSON_ST_OBJ_KEY_ESC_U4:
                return set(g, hex_digit(c, JSON_ST_OBJ_KEY));

            /* ── String value (not an object key) ────────────────────── */
            case JSON_ST_STRING:
                if (c == '"') return set(g, close_value(g)); /* pops only when the string ends */
                return set(g, in_string(g->state, c, JSON_ST_INVALID, JSON_ST_STRING_ESC));
            case JSON_ST_STRING_ESC:
                return set(g, after_backslash(c, JSON_ST_STRING, JSON_ST_STRING_ESC_U1));
            case JSON_ST_STRING_ESC_U1:
                return set(g, hex_digit(c, JSON_ST_STRING_ESC_U2));
            case JSON_ST_STRING_ESC_U2:
                return set(g, hex_digit(c, JSON_ST_STRING_ESC_U3));
            case JSON_ST_STRING_ESC_U3:
                return set(g, hex_digit(c, JSON_ST_STRING_ESC_U4));
            case JSON_ST_STRING_ESC_U4:
                return set(g, hex_digit(c, JSON_ST_STRING));

            /* ── Numbers (RFC 8259: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?) ──
             * Numbers have no closing delimiter of their own -- a whitespace, ',', '}', ']' (or
             * end of input, which this function never sees directly, handled by the caller
             * checking json_grammar_is_complete only after a value-closing transition) all mark
             * the end. Any of those close the value NOW and reprocess `c` in the resumed
             * context via `continue`. */
            case JSON_ST_NUM_INT_FIRST:
                if (c == '0') return set(g, JSON_ST_NUM_INT_ZERO);
                return set(g, is_digit(c) ? JSON_ST_NUM_INT : JSON_ST_INVALID);
            case JSON_ST_NUM_INT:
                if (is_digit(c)) return g->state;
                if (number_continues(g, c, 1)) return g->state;
                g->state = close_value(g);
                continue;
            /* RFC 8259: int = "0" / ( digit1-9 *DIGIT ) -- a leading zero must be the entire
             * integer part; "01"/"00" are not valid JSON numbers (found via TS-2.1 differential
             * fuzzing against json.loads, 2026-09-18: this PDA previously treated '0' the same as
             * any other leading digit, silently accepting "01"/"09"/etc. as DONE). */
            case JSON_ST_NUM_INT_ZERO:
                if (is_digit(c)) return set(g, JSON_ST_INVALID);
                if (number_continues(g, c, 1)) return g->state;
                g->state = close_value(g);
                continue;
            case JSON_ST_NUM_FRAC_FIRST:
                return set(g, is_digit(c) ? JSON_ST_NUM_FRAC : JSON_ST_INVALID);
            case JSON_ST_NUM_FRAC:
                if (is_digit(c)) return g->state;
                if (number_continues(g, c, 0)) return g->state;
                g->state = close_value(g);
                continue;
            case JSON_ST_NUM_EXP_FIRST:
                if (c == '+' || c == '-') return set(g, JSON_ST_NUM_EXP_SIGN);
                return set(g, is_digit(c) ? JSON_ST_NUM_EXP : JSON_ST_INVALID);
            case JSON_ST_NUM_EXP_SIGN:
                return set(g, is_digit(c) ? JSON_ST_NUM_EXP : JSON_ST_INVALID);
            case JSON_ST_NUM_EXP:
                if (is_digit(c)) return g->state;
                g->state = close_value(g);
                continue;

            /* ── Literals: true / false / null ───────────────────────── */
            case JSON_ST_LIT_TRUE_T:
                return set(g, literal_byte(c, 'r', JSON_ST_LIT_TRUE_R));
            case JSON_ST_LIT_TRUE_R:
                return set(g, literal_byte(c, 'u', JSON_ST_LIT_TRUE_U));
            case JSON_ST_LIT_TRUE_U:
                return set(g, c == 'e' ? close_value(g) : JSON_ST_INVALID);
            case JSON_ST_LIT_FALSE_F:
                return set(g, literal_byte(c, 'a', JSON_ST_LIT_FALSE_A));
            case JSON_ST_LIT_FALSE_A:
                return set(g, literal_byte(c, 'l', JSON_ST_LIT_FALSE_L));
            case JSON_ST_LIT_FALSE_L:
                return set(g, literal_byte(c, 's', JSON_ST_LIT_FALSE_S));
            case JSON_ST_LIT_FALSE_S:
                return set(g, c == 'e' ? close_value(g) : JSON_ST_INVALID);
            case JSON_ST_LIT_NULL_N:
                return set(g, literal_byte(c, 'u', JSON_ST_LIT_NULL_U));
            case JSON_ST_LIT_NULL_U:
                return set(g, literal_byte(c, 'l', JSON_ST_LIT_NULL_L));
            case JSON_ST_LIT_NULL_L:
                return set(g, c == 'l' ? close_value(g) : JSON_ST_INVALID);

            /* ── Root / done ─────────────────────────────────────────── */
            case JSON_ST_ROOT:
                if (c == '{') return set(g, JSON_ST_OBJ_OPEN);
                return set(g, c == '[' ? JSON_ST_ARR_OPEN : JSON_ST_INVALID);
            case JSON_ST_DONE:
                /* Root mode ends at the closing bracket: only EOS may follow. */
                return set(g, (!g->root && is_ws(c)) ? JSON_ST_DONE : JSON_ST_INVALID);

            case JSON_ST_INVALID:
            default:
                return JSON_ST_INVALID;
        }
    }
}
