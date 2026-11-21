/* Character-class patterns of the pre-tokenizer: one matcher per pattern, dispatched by table. */

#include <string.h>

#include "tokenizer/pretok_internal.h"


/* The CJK / Thai / Hangul blocks the DeepSeek patterns split out as their own runs. */
static int is_cjk(uint32_t c) {
    return in(c, 0x4E00, 0x9FA5) || in(c, 0x0800, 0x4E00) || in(c, 0xAC00, 0xD7FF);
}

/* Letter set of llama.cpp's deepseek-llm pattern: Latin, Greek, Cyrillic, Armenian,
 * Georgian, Cherokee, the Latin/Greek extension blocks and fullwidth Latin (the ranges
 * of its character class, collapsed; scripts outside it stay unsplit between matches). */
static int ds_letter(uint32_t c) {
    if (c < 0x80) return ((c | 0x20) >= 'a' && (c | 0x20) <= 'z');
    if (c == 0xB5) return 1;
    return (in(c, 0xC0, 0x2AF) && c != 0xD7 && c != 0xF7) || in(c, 0x370, 0x52F) ||
           in(c, 0x531, 0x556) || in(c, 0x10A0, 0x10C5) || in(c, 0x13A0, 0x13FD) ||
           in(c, 0x1D00, 0x1FFF) || in(c, 0x2C00, 0x2DFF) || in(c, 0xA640, 0xA7FF) ||
           in(c, 0xFF21, 0xFF3A) || in(c, 0xFF41, 0xFF5A);
}

static int ds_punct(uint32_t c) {
    return in(c, 0x21, 0x2F) || in(c, 0x3A, 0x7E) || in(c, 0xFF01, 0xFF0F) ||
           in(c, 0xFF1A, 0xFF5E) || in(c, 0x2018, 0x201F) || in(c, 0x3000, 0x3002);
}

/* Run of characters accepted by `ok`, after one optional whitespace character. */
static int opt_space_run(const Split *s, size_t pos, int (*ok)(uint32_t)) {
    size_t q = pos;
    if (get_fl(s, q) & F_SPACE) q++;
    size_t b = q;
    while (get_cp(s, q) != OUT_OF_RANGE && ok(get_cp(s, q))) q++;
    return q > b ? (int)(q - pos) : 0;
}

static int cp_is_letter(uint32_t c) {
    return (pretok_cp_class(c) & F_LETTER) != 0;
}
static int cp_is_punct(uint32_t c) {
    return (pretok_cp_class(c) & F_PUNCT) != 0;
}

typedef int (*ClsMatchFn)(const Split *s, size_t pos);

static int cls_number_one(const Split *s, size_t pos) {
    return (pretok_cp_class(get_cp(s, pos)) & F_NUMBER) ? 1 : 0;
}

static int cls_number_run(const Split *s, size_t pos) {
    size_t q = pos;
    while (get_fl(s, q) & F_NUMBER) q++;
    return (int)(q - pos);
}

/* One character of [\p{P}\$\+<=>\^~\|]. */
static int is_punct_or_symbol(uint32_t d) {
    if (pretok_cp_class(d) & F_PUNCT) return 1;
    return d > 0 && d < 0x80 && strchr("$+<=>^~|", (int)d) != NULL;
}

static int cls_punct_run(const Split *s, size_t pos) {
    size_t q = pos;
    while (is_punct_or_symbol(get_cp(s, q))) q++;
    return (int)(q - pos);
}

static int cls_newline(const Split *s, size_t pos) {
    uint32_t c = get_cp(s, pos);
    return (c == '\r' || c == '\n') ? 1 : 0;
}

static int cls_ds_letters(const Split *s, size_t pos) {
    return opt_space_run(s, pos, ds_letter);
}
static int cls_ds_punct(const Split *s, size_t pos) {
    return opt_space_run(s, pos, ds_punct);
}
static int cls_l_run(const Split *s, size_t pos) {
    return opt_space_run(s, pos, cp_is_letter);
}
static int cls_p_run(const Split *s, size_t pos) {
    return opt_space_run(s, pos, cp_is_punct);
}

static int cls_cjk(const Split *s, size_t pos) {
    size_t q = pos;
    while (get_cp(s, q) != OUT_OF_RANGE && is_cjk(get_cp(s, q))) q++;
    return (int)(q - pos);
}

static int cls_trail_space(const Split *s, size_t pos) {
    size_t q = pos;
    while (get_fl(s, q) & F_SPACE) q++;
    return (q > pos && q == s->end) ? (int)(q - pos) : 0;
}

static int cls_digit3(const Split *s, size_t pos) {
    for (int i = 0; i < 3; i++) {
        uint32_t d = get_cp(s, pos + (size_t)i);
        if (d < '0' || d > '9') return 0;
    }
    return 3;
}

static const ClsMatchFn CLS_MATCHERS[] = {
    [CLS_NUMBER_ONE]  = cls_number_one,
    [CLS_NUMBER_RUN]  = cls_number_run,
    [CLS_PUNCT_RUN]   = cls_punct_run,
    [CLS_DIGIT3]      = cls_digit3,
    [CLS_NEWLINE]     = cls_newline,
    [CLS_DS_LETTERS]  = cls_ds_letters,
    [CLS_DS_PUNCT]    = cls_ds_punct,
    [CLS_TRAIL_SPACE] = cls_trail_space,
    [CLS_CJK]         = cls_cjk,
    [CLS_L_RUN]       = cls_l_run,
    [CLS_P_RUN]       = cls_p_run,
};

static int cls_match(const Split *s, size_t pos, ClassPat p) {
    return CLS_MATCHERS[p](s, pos);
}

void pretok_split_class(Split *s, ClassPat p) {
    for (size_t pos = s->ini; pos < s->end;) {
        int m = cls_match(s, pos, p);
        if (m > 0) {
            add_tok(s, pos); /* unmatched text before the match */
            pos += (size_t)m;
            add_tok(s, pos);
        } else {
            pos++;
        }
    }
    add_tok(s, s->end);
}
