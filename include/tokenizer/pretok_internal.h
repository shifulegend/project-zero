#ifndef TN_PRETOK_INTERNAL_H
#define TN_PRETOK_INTERNAL_H

/* Shared by pretokenize.c (pipeline, GPT-2 / Llama-3 / Qwen splitters) and pretok_classes.c
 * (character-class patterns). Not part of the public interface. */

#include <stddef.h>
#include <stdint.h>

#include "tokenizer/pretokenize.h"

#define OUT_OF_RANGE 0xFFFFFFFFu

typedef struct {
    uint32_t *cp; /* code points */
    size_t *off;  /* byte offset of each code point (n + 1 entries) */
    size_t n;
} Cpts;


enum { F_LETTER = 1, F_NUMBER = 2, F_SPACE = 4, F_PUNCT = 8, F_SYMBOL = 16, F_OTHER = 32 };

static inline int in(uint32_t c, uint32_t a, uint32_t b) {
    return c >= a && c <= b;
}

typedef struct {
    const Cpts *c;
    size_t ini, end; /* code-point range of the fragment being split */
    size_t prev;     /* end of the last emitted piece */
    size_t *out;     /* piece lengths in code points */
    int n, cap, overflow;
} Split;

static inline uint32_t get_cp(const Split *s, size_t pos) {
    return (pos >= s->ini && pos < s->end) ? s->c->cp[pos] : OUT_OF_RANGE;
}
static inline int get_fl(const Split *s, size_t pos) {
    return pretok_cp_class(get_cp(s, pos));
}

static inline size_t add_tok(Split *s, size_t end) {
    size_t len = end - s->prev;
    if (len > 0) {
        if (s->n < s->cap)
            s->out[s->n++] = len;
        else
            s->overflow = 1;
    }
    s->prev = end;
    return len;
}


typedef enum {
    CLS_NUMBER_ONE,
    CLS_NUMBER_RUN,
    CLS_PUNCT_RUN,
    CLS_DIGIT3,
    CLS_NEWLINE,     /* [\r\n]                                           */
    CLS_DS_LETTERS,  /* \s?[A-Za-z..Latin/Greek/Cyrillic..]+  (deepseek-llm) */
    CLS_DS_PUNCT,    /* \s?[!-/:-~！-／：-～‘-‟　-。]+            (deepseek-llm) */
    CLS_TRAIL_SPACE, /* \s+$   (end of the fragment being split)           */
    CLS_CJK,         /* [一-龥ࠀ-一가-퟿]+                                  */
    CLS_L_RUN,       /* \s?\p{L}+                       (deepseek-coder) */
    CLS_P_RUN        /* \s?\p{P}+                       (deepseek-coder) */
} ClassPat;

/* Splits the fragment s->ini..s->end around every match of pattern `p`: matches become pieces,
 * the text between them stays as its own pieces (like std::regex_search). */
void pretok_split_class(Split *s, ClassPat p);

#endif /* TN_PRETOK_INTERNAL_H */
