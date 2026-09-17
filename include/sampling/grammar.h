#ifndef TN_GRAMMAR_H
#define TN_GRAMMAR_H

/*
 * grammar.h — Phase 20: Grammar-Constrained Decoding (JSON Mode)
 *
 * Scope decision: the original plan called for a generic BNF/EBNF-to-DFA
 * grammar compiler. A *flat* DFA (states + transitions, no stack) cannot
 * represent JSON's actual grammar though -- JSON permits arbitrarily deep
 * nested objects/arrays, which is a context-free (stack-needed) property,
 * not a regular one. A flat FSM would have to hard-code a maximum nesting
 * depth to fake it, which is exactly the kind of arbitrary limit this
 * project's "no hardcoding" rule exists to avoid. This implementation is
 * therefore a small pushdown automaton (an explicit depth stack of "what
 * context to return to when this value closes") specific to JSON's real
 * grammar (RFC 8259) -- correct for unbounded nesting, not a general BNF
 * grammar compiler. General custom grammars (grammar_load_from_bnf in the
 * original plan) are out of scope for this pass; JSON is the concrete,
 * needed use case (structured tool-call arguments, OpenAI-style
 * response_format=json_object).
 */

#include <stddef.h>

#define JSON_GRAMMAR_MAX_DEPTH 256

typedef enum {
    JSON_ST_VALUE = 0,       /* expecting the start of any JSON value */
    JSON_ST_OBJ_OPEN,        /* just saw '{': expect '"' (key) or '}' (empty) */
    JSON_ST_OBJ_KEY,         /* inside an object-key string */
    JSON_ST_OBJ_KEY_ESC,     /* saw '\' inside a key string */
    JSON_ST_OBJ_KEY_ESC_U1, JSON_ST_OBJ_KEY_ESC_U2,
    JSON_ST_OBJ_KEY_ESC_U3, JSON_ST_OBJ_KEY_ESC_U4,
    JSON_ST_OBJ_COLON,       /* key string closed: expect ':' */
    JSON_ST_OBJ_AFTER_VALUE, /* a value inside an object just closed: expect ',' or '}' */
    JSON_ST_OBJ_KEY_NEXT,    /* after ',' in an object: expect '"' for the next key */
    JSON_ST_ARR_OPEN,        /* just saw '[': expect a value or ']' (empty) */
    JSON_ST_ARR_AFTER_VALUE, /* a value inside an array just closed: expect ',' or ']' */
    JSON_ST_STRING,          /* inside a string VALUE (not an object key) */
    JSON_ST_STRING_ESC,
    JSON_ST_STRING_ESC_U1, JSON_ST_STRING_ESC_U2,
    JSON_ST_STRING_ESC_U3, JSON_ST_STRING_ESC_U4,
    JSON_ST_NUM_INT_FIRST,   /* just saw '-': need >=1 digit next */
    JSON_ST_NUM_INT,         /* >=1 integer digit consumed; value may end here */
    JSON_ST_NUM_FRAC_FIRST,  /* just saw '.': need >=1 digit next */
    JSON_ST_NUM_FRAC,        /* >=1 fraction digit consumed; value may end here */
    JSON_ST_NUM_EXP_FIRST,   /* just saw e/E: optional sign then need >=1 digit */
    JSON_ST_NUM_EXP_SIGN,    /* just saw exponent sign: need >=1 digit */
    JSON_ST_NUM_EXP,         /* >=1 exponent digit consumed; value may end here */
    JSON_ST_LIT_TRUE_T, JSON_ST_LIT_TRUE_R, JSON_ST_LIT_TRUE_U,   /* matched "t"/"tr"/"tru" */
    JSON_ST_LIT_FALSE_F, JSON_ST_LIT_FALSE_A,
    JSON_ST_LIT_FALSE_L, JSON_ST_LIT_FALSE_S,                    /* matched "f".."fals" */
    JSON_ST_LIT_NULL_N, JSON_ST_LIT_NULL_U, JSON_ST_LIT_NULL_L,  /* matched "n"/"nu"/"nul" */
    JSON_ST_DONE,            /* the single top-level value is complete */
    JSON_ST_INVALID          /* sink state: this input can never be valid JSON */
} JsonState;

typedef struct {
    JsonState state;
    /* Each stack entry is the JsonState to resume once the current value
     * closes (JSON_ST_DONE for the top-level value, JSON_ST_OBJ_AFTER_VALUE
     * for a value nested in an object, JSON_ST_ARR_AFTER_VALUE for a value
     * nested in an array). */
    JsonState stack[JSON_GRAMMAR_MAX_DEPTH];
    int sp; /* stack pointer; 0 = top-level, grows on '{'/'[' */
} JsonGrammarState;

/* Resets to the initial state: expecting any JSON value, empty stack. */
void json_grammar_init(JsonGrammarState *g);

/*
 * Feeds one input byte through the automaton, returning the new state.
 * JSON_ST_INVALID is a sink: once reached, every subsequent call also
 * returns JSON_ST_INVALID (the caller should treat this as "this whole
 * candidate token/continuation is illegal", not just this one byte).
 * Whitespace (space/tab/CR/LF) is accepted between structural tokens (not
 * inside strings) and is a no-op transition (state unchanged) everywhere
 * except JSON_ST_NUM_INT/_FRAC/_EXP-family states and literal-in-progress
 * states, where it correctly closes the value first (numbers/literals have
 * no closing delimiter of their own -- whitespace, ',', '}', ']', or end of
 * input all mark their end).
 */
JsonState json_grammar_step(JsonGrammarState *g, char c);

/* True if the automaton is in a state where the top-level JSON value is
 * fully formed (JSON_ST_DONE) -- i.e. generation may legally stop here. */
int json_grammar_is_complete(const JsonGrammarState *g);

/*
 * Numbers have no closing character of their own (see json_grammar_step's
 * comment) -- a bare top-level number or literal with nothing after it
 * (e.g. the complete generation is just "42") never sees the delimiter
 * that would normally trigger closing it. Call this once, when generation
 * is about to stop (checking for EOS eligibility), to force-close any
 * value that's sitting in a "may end here" state. No-op otherwise.
 */
JsonState json_grammar_finalize(JsonGrammarState *g);

#endif /* TN_GRAMMAR_H */
