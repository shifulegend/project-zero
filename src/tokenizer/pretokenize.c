#include "tokenizer/pretokenize.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Code points ───────────────────────────────────────────────────────── */

#define OUT_OF_RANGE 0xFFFFFFFFu

typedef struct {
    uint32_t *cp;      /* code points */
    size_t   *off;     /* byte offset of each code point (n + 1 entries) */
    size_t    n;
} Cpts;

static int cpts_decode(Cpts *c, const char *s, size_t len) {
    c->cp = (uint32_t *)malloc((len + 1) * sizeof(uint32_t));
    c->off = (size_t *)malloc((len + 1) * sizeof(size_t));
    c->n = 0;
    if (!c->cp || !c->off) { free(c->cp); free(c->off); return -1; }
    const unsigned char *u = (const unsigned char *)s;
    size_t i = 0;
    while (i < len) {
        uint32_t v = u[i];
        size_t k = 1;
        if (v >= 0xF0 && i + 3 < len + 0 && (u[i+1] & 0xC0) == 0x80 && (u[i+2] & 0xC0) == 0x80 && (u[i+3] & 0xC0) == 0x80) {
            v = ((v & 0x07u) << 18) | ((u[i+1] & 0x3Fu) << 12) | ((u[i+2] & 0x3Fu) << 6) | (u[i+3] & 0x3Fu); k = 4;
        } else if (v >= 0xE0 && i + 2 < len && (u[i+1] & 0xC0) == 0x80 && (u[i+2] & 0xC0) == 0x80) {
            v = ((v & 0x0Fu) << 12) | ((u[i+1] & 0x3Fu) << 6) | (u[i+2] & 0x3Fu); k = 3;
        } else if (v >= 0xC0 && i + 1 < len && (u[i+1] & 0xC0) == 0x80) {
            v = ((v & 0x1Fu) << 6) | (u[i+1] & 0x3Fu); k = 2;
        } /* else: ASCII or a stray byte, taken as one unit */
        c->off[c->n] = i;
        c->cp[c->n++] = v;
        i += k;
    }
    c->off[c->n] = len;
    return 0;
}

static void cpts_free(Cpts *c) { free(c->cp); free(c->off); }

/* ── Character classes ─────────────────────────────────────────────────── */

enum { F_LETTER = 1, F_NUMBER = 2, F_SPACE = 4, F_PUNCT = 8, F_SYMBOL = 16, F_OTHER = 32 };

static int in(uint32_t c, uint32_t a, uint32_t b) { return c >= a && c <= b; }

static int cp_flags(uint32_t c) {
    if (c == OUT_OF_RANGE) return 0;
    if (c < 0x80) {
        if (c >= '0' && c <= '9') return F_NUMBER;
        if ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') return F_LETTER;
        if (c == ' ' || (c >= 0x09 && c <= 0x0D)) return F_SPACE;
        if (strchr("$+<=>^`|~", (int)c)) return F_SYMBOL;
        if (c >= 0x21 && c <= 0x7E) return F_PUNCT;   /* remaining ASCII punctuation */
        return F_OTHER;                               /* control characters */
    }
    /* whitespace */
    if (c == 0x85 || c == 0xA0 || c == 0x1680 || in(c, 0x2000, 0x200A) || c == 0x2028 ||
        c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000) return F_SPACE;
    if (in(c, 0x80, 0x9F) || in(c, 0x200B, 0x200F) || c == 0xFEFF) return F_OTHER;
    /* numbers (Nd / Nl / No in common blocks) */
    if (c == 0xB2 || c == 0xB3 || c == 0xB9 || in(c, 0xBC, 0xBE) || in(c, 0x660, 0x669) ||
        in(c, 0x6F0, 0x6F9) || in(c, 0x966, 0x96F) || in(c, 0x2070, 0x2079) || in(c, 0x2080, 0x2089) ||
        in(c, 0x2150, 0x2189) || in(c, 0x2460, 0x249B) || in(c, 0x2776, 0x2793) || c == 0x3007 ||
        in(c, 0x3021, 0x3029) || in(c, 0xFF10, 0xFF19)) return F_NUMBER;
    /* punctuation */
    if (c == 0xA1 || c == 0xA7 || c == 0xAB || c == 0xB6 || c == 0xB7 || c == 0xBB || c == 0xBF ||
        c == 0x37E || c == 0x387 || in(c, 0x55A, 0x55F) || in(c, 0x589, 0x58A) || in(c, 0x60C, 0x60D) ||
        c == 0x61B || c == 0x61F || c == 0x6D4 || in(c, 0x964, 0x965) || in(c, 0x2010, 0x2027) ||
        in(c, 0x2030, 0x2043) || in(c, 0x2045, 0x2051) || in(c, 0x2053, 0x205E) ||
        in(c, 0x3001, 0x3003) || in(c, 0x3008, 0x3011) || in(c, 0x3014, 0x301F) || c == 0x30FB ||
        in(c, 0xFF01, 0xFF03) || in(c, 0xFF05, 0xFF0A) || in(c, 0xFF0C, 0xFF0F) || in(c, 0xFF1A, 0xFF1B) ||
        in(c, 0xFF1F, 0xFF20) || in(c, 0xFF3B, 0xFF3D) || c == 0xFF3F || c == 0xFF5B || c == 0xFF5D)
        return F_PUNCT;
    /* symbols and marks: [^\s\p{L}\p{N}] in every pattern below */
    if (in(c, 0xA2, 0xA6) || c == 0xA8 || c == 0xA9 || c == 0xAC || in(c, 0xAE, 0xB1) || c == 0xB4 ||
        c == 0xB8 || c == 0xD7 || c == 0xF7 || in(c, 0x2C2, 0x2C5) || in(c, 0x2D2, 0x2DF) ||
        in(c, 0x300, 0x36F) || in(c, 0x20A0, 0x20CF) || in(c, 0x2190, 0x23FF) || in(c, 0x2500, 0x27BF) ||
        in(c, 0x2900, 0x2BFF) || in(c, 0x1F000, 0x1FAFF) || in(c, 0xFE00, 0xFE0F) || in(c, 0xFFE0, 0xFFEE))
        return F_SYMBOL;
    return F_LETTER;
}

/* ── Splitter state ────────────────────────────────────────────────────── */

typedef struct {
    const Cpts *c;
    size_t ini, end;        /* code-point range of the fragment being split */
    size_t prev;            /* end of the last emitted piece */
    size_t *out;            /* piece lengths in code points */
    int n, cap, overflow;
} Split;

static uint32_t get_cp(const Split *s, size_t pos) {
    return (pos >= s->ini && pos < s->end) ? s->c->cp[pos] : OUT_OF_RANGE;
}
static int get_fl(const Split *s, size_t pos) { return cp_flags(get_cp(s, pos)); }

static size_t add_tok(Split *s, size_t end) {
    size_t len = end - s->prev;
    if (len > 0) {
        if (s->n < s->cap) s->out[s->n++] = len;
        else s->overflow = 1;
    }
    s->prev = end;
    return len;
}

static int lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? (int)(c | 0x20) : (int)c; }

/* Contractions 's 't 're 've 'm 'll 'd (case-insensitive when ci). Returns the
 * match length or 0. */
static size_t contraction(const Split *s, size_t pos, int ci) {
    if (get_cp(s, pos) != '\'' || pos + 1 >= s->end) return 0;
    int a = ci ? lower(get_cp(s, pos + 1)) : (int)get_cp(s, pos + 1);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (pos + 2 < s->end) {
        int b = ci ? lower(get_cp(s, pos + 2)) : (int)get_cp(s, pos + 2);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    return 0;
}

/* llama.cpp unicode_regex_split_custom_gpt2:
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S) (and \s+) */
static void split_gpt2(Split *s) {
    for (size_t pos = s->ini; pos < s->end;) {
        uint32_t cpt = get_cp(s, pos);
        int fl = get_fl(s, pos);
        size_t k = contraction(s, pos, 0);
        if (k) { pos += add_tok(s, pos + k); continue; }
        int fl2 = (cpt == ' ') ? get_fl(s, pos + 1) : fl;
        if (fl2 & F_LETTER) {
            pos += (cpt == ' ');
            while (fl2 & F_LETTER) fl2 = get_fl(s, ++pos);
            add_tok(s, pos);
            continue;
        }
        if (fl2 & F_NUMBER) {
            pos += (cpt == ' ');
            while (fl2 & F_NUMBER) fl2 = get_fl(s, ++pos);
            add_tok(s, pos);
            continue;
        }
        if (!(fl2 & (F_SPACE | F_LETTER | F_NUMBER)) && fl2) {
            pos += (cpt == ' ');
            while (!(fl2 & (F_SPACE | F_LETTER | F_NUMBER)) && fl2) fl2 = get_fl(s, ++pos);
            add_tok(s, pos);
            continue;
        }
        size_t nws = 0;
        while (get_fl(s, pos + nws) & F_SPACE) nws++;
        if (nws > 1 && get_cp(s, pos + nws) != OUT_OF_RANGE) { pos += nws - 1; add_tok(s, pos); continue; }
        if (nws > 0) { pos += nws; add_tok(s, pos); continue; }
        add_tok(s, ++pos);
    }
}

/* llama.cpp unicode_regex_split_custom_llama3 (max_digits = 3) and _qwen2
 * (max_digits = 1):
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,max}|
 *   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ */
static void split_llama3(Split *s, int max_digits) {
    for (size_t pos = s->ini; pos < s->end;) {
        uint32_t cpt = get_cp(s, pos);
        int fl = get_fl(s, pos);
        size_t k = contraction(s, pos, 1);
        if (k) { pos += add_tok(s, pos + k); continue; }
        if (!(cpt == '\r' || cpt == '\n' || (fl & F_NUMBER))) {
            if ((fl & F_LETTER) || (get_fl(s, pos + 1) & F_LETTER)) {
                pos++;
                while (get_fl(s, pos) & F_LETTER) pos++;
                add_tok(s, pos);
                continue;
            }
        }
        if (fl & F_NUMBER) {
            size_t ini = pos;
            while (get_fl(s, pos) & F_NUMBER) {
                if (++pos - ini >= (size_t)max_digits) { add_tok(s, pos); ini = pos; }
            }
            add_tok(s, pos);
            continue;
        }
        int fl2 = (cpt == ' ') ? get_fl(s, pos + 1) : fl;
        if (!(fl2 & (F_SPACE | F_LETTER | F_NUMBER)) && fl) {
            pos += (cpt == ' ');
            while (!(fl2 & (F_SPACE | F_LETTER | F_NUMBER)) && fl2) fl2 = get_fl(s, ++pos);
            uint32_t c2 = get_cp(s, pos);
            while (c2 == '\r' || c2 == '\n') c2 = get_cp(s, ++pos);
            add_tok(s, pos);
            continue;
        }
        size_t nws = 0, last_rn = 0;
        while (get_fl(s, pos + nws) & F_SPACE) {
            uint32_t c2 = get_cp(s, pos + nws);
            if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
            nws++;
        }
        if (last_rn > 0) { pos = last_rn; add_tok(s, pos); continue; }
        if (nws > 1 && get_cp(s, pos + nws) != OUT_OF_RANGE) { pos += nws - 1; add_tok(s, pos); continue; }
        if (nws > 0) { pos += nws; add_tok(s, pos); continue; }
        add_tok(s, ++pos);
    }
}

/* Character-class patterns, matched like std::regex_search: matches become
 * pieces, the text between them stays as its own pieces. */
typedef enum { CLS_NUMBER_ONE, CLS_NUMBER_RUN, CLS_PUNCT_RUN, CLS_DIGIT3 } ClassPat;

static int cls_match(const Split *s, size_t pos, ClassPat p) {
    uint32_t c = get_cp(s, pos);
    int fl = cp_flags(c);
    switch (p) {
    case CLS_NUMBER_ONE: return (fl & F_NUMBER) ? 1 : 0;
    case CLS_NUMBER_RUN: { size_t q = pos; while (get_fl(s, q) & F_NUMBER) q++; return (int)(q - pos); }
    case CLS_PUNCT_RUN: {
        /* [\p{P}\$\+<=>\^~\|]+ */
        size_t q = pos;
        for (;;) {
            uint32_t d = get_cp(s, q);
            int f = cp_flags(d);
            if ((f & F_PUNCT) || d == '$' || d == '+' || d == '<' || d == '=' || d == '>' ||
                d == '^' || d == '~' || d == '|') q++;
            else break;
        }
        return (int)(q - pos);
    }
    case CLS_DIGIT3: {
        for (int i = 0; i < 3; i++) {
            uint32_t d = get_cp(s, pos + (size_t)i);
            if (d < '0' || d > '9') return 0;
        }
        return 3;
    }
    }
    return 0;
}

static void split_class(Split *s, ClassPat p) {
    for (size_t pos = s->ini; pos < s->end;) {
        int m = cls_match(s, pos, p);
        if (m > 0) {
            add_tok(s, pos);                 /* unmatched text before the match */
            pos += (size_t)m;
            add_tok(s, pos);
        } else {
            pos++;
        }
    }
    add_tok(s, s->end);
}

/* ── Pipeline ──────────────────────────────────────────────────────────── */

typedef enum { STEP_GPT2, STEP_LLAMA3, STEP_QWEN2, STEP_NUM1, STEP_NUMRUN, STEP_PUNCT, STEP_DIG3 } Step;

/* Runs one step over the current piece list (lengths in code points). */
static int run_step(const Cpts *c, Step st, size_t **pieces, int *n_pieces) {
    size_t cap = c->n + 1;
    size_t *out = (size_t *)malloc(cap * sizeof(size_t));
    if (!out) return -1;
    int n_out = 0;
    size_t start = 0;
    for (int i = 0; i < *n_pieces; i++) {
        Split s = { c, start, start + (*pieces)[i], start, out + n_out, 0, (int)(cap - (size_t)n_out), 0 };
        switch (st) {
        case STEP_GPT2:   split_gpt2(&s); break;
        case STEP_LLAMA3: split_llama3(&s, 3); break;
        case STEP_QWEN2:  split_llama3(&s, 1); break;
        case STEP_NUM1:   split_class(&s, CLS_NUMBER_ONE); break;
        case STEP_NUMRUN: split_class(&s, CLS_NUMBER_RUN); break;
        case STEP_PUNCT:  split_class(&s, CLS_PUNCT_RUN); break;
        case STEP_DIG3:   split_class(&s, CLS_DIGIT3); break;
        }
        if (s.overflow) { free(out); return -1; }
        n_out += s.n;
        start += (*pieces)[i];
    }
    free(*pieces);
    *pieces = out;
    *n_pieces = n_out;
    return 0;
}

int pretokenize(PretokType type, const char *text, size_t len,
                size_t *starts, size_t *lens, int max_frags) {
    if (!text || !starts || !lens || max_frags <= 0) return -1;
    if (len == 0) return 0;
    if (type == PRETOK_NONE) { starts[0] = 0; lens[0] = len; return 1; }

    Cpts c;
    if (cpts_decode(&c, text, len) != 0) return -1;
    size_t *pieces = (size_t *)malloc(sizeof(size_t));
    if (!pieces) { cpts_free(&c); return -1; }
    pieces[0] = c.n;
    int n = 1;

    static const Step DEFAULT_STEPS[] = { STEP_PUNCT, STEP_GPT2, STEP_NUMRUN, STEP_DIG3 };
    static const Step GPT2_STEPS[]    = { STEP_GPT2 };
    static const Step LLAMA3_STEPS[]  = { STEP_LLAMA3 };
    static const Step QWEN2_STEPS[]   = { STEP_QWEN2 };
    static const Step SMOLLM_STEPS[]  = { STEP_NUM1, STEP_GPT2 };
    const Step *steps = DEFAULT_STEPS;
    int n_steps = 4;
    switch (type) {
    case PRETOK_GPT2:   steps = GPT2_STEPS;   n_steps = 1; break;
    case PRETOK_LLAMA3: steps = LLAMA3_STEPS; n_steps = 1; break;
    case PRETOK_QWEN2:  steps = QWEN2_STEPS;  n_steps = 1; break;
    case PRETOK_SMOLLM: steps = SMOLLM_STEPS; n_steps = 2; break;
    default: break;
    }
    int rc = 0;
    for (int i = 0; i < n_steps && rc == 0; i++) rc = run_step(&c, steps[i], &pieces, &n);

    int out = -1;
    if (rc == 0 && n <= max_frags) {
        size_t cp = 0;
        for (int i = 0; i < n; i++) {
            starts[i] = c.off[cp];
            lens[i] = c.off[cp + pieces[i]] - c.off[cp];
            cp += pieces[i];
        }
        out = n;
    }
    free(pieces);
    cpts_free(&c);
    return out;
}

PretokType pretok_type_from_name(const char *pre) {
    if (!pre || !*pre || strcmp(pre, "default") == 0) return PRETOK_DEFAULT;
    static const char *const gpt2[]   = { "gpt-2", "gpt2", "olmo", "mpt", "jais", "trillion", "granite-docling", NULL };
    static const char *const llama3[] = { "llama3", "llama-v3", "llama-bpe", "smaug-bpe", "falcon3",
                                          "falcon-h1", "pixtral", "midm-2.0", "lfm2", NULL };
    static const char *const qwen2[]  = { "qwen2", "stablelm2", "hunyuan", "deepseek-r1-qwen", NULL };
    static const char *const smollm[] = { "smollm", "codeshell", "exaone", "minerva-7b", NULL };
    for (int i = 0; gpt2[i]; i++)   if (strcmp(pre, gpt2[i]) == 0)   return PRETOK_GPT2;
    for (int i = 0; llama3[i]; i++) if (strcmp(pre, llama3[i]) == 0) return PRETOK_LLAMA3;
    for (int i = 0; qwen2[i]; i++)  if (strcmp(pre, qwen2[i]) == 0)  return PRETOK_QWEN2;
    for (int i = 0; smollm[i]; i++) if (strcmp(pre, smollm[i]) == 0) return PRETOK_SMOLLM;
    return PRETOK_DEFAULT;
}

const char *pretok_type_name(PretokType t) {
    switch (t) {
    case PRETOK_NONE:   return "none";
    case PRETOK_GPT2:   return "gpt2";
    case PRETOK_LLAMA3: return "llama3";
    case PRETOK_QWEN2:  return "qwen2";
    case PRETOK_SMOLLM: return "smollm";
    default:            return "default";
    }
}
