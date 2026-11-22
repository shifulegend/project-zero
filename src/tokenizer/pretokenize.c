#include "tokenizer/pretokenize.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer/pretok_internal.h"

/* ── Code points ───────────────────────────────────────────────────────── */

static int cpts_decode(Cpts *c, const char *s, size_t len) {
    c->cp  = (uint32_t *)malloc((len + 1) * sizeof(uint32_t));
    c->off = (size_t *)malloc((len + 1) * sizeof(size_t));
    c->n   = 0;
    if (!c->cp || !c->off) {
        free(c->cp);
        free(c->off);
        return -1;
    }
    const unsigned char *u = (const unsigned char *)s;
    size_t i               = 0;
    while (i < len) {
        uint32_t v = u[i];
        size_t k   = 1;
        if (v >= 0xF0 && i + 3 < len + 0 && (u[i + 1] & 0xC0) == 0x80 &&
            (u[i + 2] & 0xC0) == 0x80 && (u[i + 3] & 0xC0) == 0x80) {
            v = ((v & 0x07u) << 18) | ((u[i + 1] & 0x3Fu) << 12) | ((u[i + 2] & 0x3Fu) << 6) |
                (u[i + 3] & 0x3Fu);
            k = 4;
        } else if (v >= 0xE0 && i + 2 < len && (u[i + 1] & 0xC0) == 0x80 &&
                   (u[i + 2] & 0xC0) == 0x80) {
            v = ((v & 0x0Fu) << 12) | ((u[i + 1] & 0x3Fu) << 6) | (u[i + 2] & 0x3Fu);
            k = 3;
        } else if (v >= 0xC0 && i + 1 < len && (u[i + 1] & 0xC0) == 0x80) {
            v = ((v & 0x1Fu) << 6) | (u[i + 1] & 0x3Fu);
            k = 2;
        } /* else: ASCII or a stray byte, taken as one unit */
        c->off[c->n]  = i;
        c->cp[c->n++] = v;
        i += k;
    }
    c->off[c->n] = len;
    return 0;
}

static void cpts_free(Cpts *c) {
    free(c->cp);
    free(c->off);
}

/* ── Character classes ─────────────────────────────────────────────────── */

int pretok_cp_class(uint32_t c) {
    if (c == OUT_OF_RANGE) return 0;
    if (c < 0x80) {
        if (c >= '0' && c <= '9') return F_NUMBER;
        if ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') return F_LETTER;
        if (c == ' ' || (c >= 0x09 && c <= 0x0D)) return F_SPACE;
        if (strchr("$+<=>^`|~", (int)c)) return F_SYMBOL;
        if (c >= 0x21 && c <= 0x7E) return F_PUNCT; /* remaining ASCII punctuation */
        return F_OTHER;                             /* control characters */
    }
    /* whitespace */
    if (c == 0x85 || c == 0xA0 || c == 0x1680 || in(c, 0x2000, 0x200A) || c == 0x2028 ||
        c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000)
        return F_SPACE;
    if (in(c, 0x80, 0x9F) || in(c, 0x200B, 0x200F) || c == 0xFEFF) return F_OTHER;
    /* numbers (Nd / Nl / No in common blocks) */
    if (c == 0xB2 || c == 0xB3 || c == 0xB9 || in(c, 0xBC, 0xBE) || in(c, 0x660, 0x669) ||
        in(c, 0x6F0, 0x6F9) || in(c, 0x966, 0x96F) || in(c, 0x2070, 0x2079) ||
        in(c, 0x2080, 0x2089) || in(c, 0x2150, 0x2189) || in(c, 0x2460, 0x249B) ||
        in(c, 0x2776, 0x2793) || c == 0x3007 || in(c, 0x3021, 0x3029) || in(c, 0xFF10, 0xFF19))
        return F_NUMBER;
    /* punctuation */
    if (c == 0xA1 || c == 0xA7 || c == 0xAB || c == 0xB6 || c == 0xB7 || c == 0xBB || c == 0xBF ||
        c == 0x37E || c == 0x387 || in(c, 0x55A, 0x55F) || in(c, 0x589, 0x58A) ||
        in(c, 0x60C, 0x60D) || c == 0x61B || c == 0x61F || c == 0x6D4 || in(c, 0x964, 0x965) ||
        in(c, 0x2010, 0x2027) || in(c, 0x2030, 0x2043) || in(c, 0x2045, 0x2051) ||
        in(c, 0x2053, 0x205E) || in(c, 0x3001, 0x3003) || in(c, 0x3008, 0x3011) ||
        in(c, 0x3014, 0x301F) || c == 0x30FB || in(c, 0xFF01, 0xFF03) || in(c, 0xFF05, 0xFF0A) ||
        in(c, 0xFF0C, 0xFF0F) || in(c, 0xFF1A, 0xFF1B) || in(c, 0xFF1F, 0xFF20) ||
        in(c, 0xFF3B, 0xFF3D) || c == 0xFF3F || c == 0xFF5B || c == 0xFF5D)
        return F_PUNCT;
    /* symbols and marks: [^\s\p{L}\p{N}] in every pattern below */
    if (in(c, 0xA2, 0xA6) || c == 0xA8 || c == 0xA9 || c == 0xAC || in(c, 0xAE, 0xB1) ||
        c == 0xB4 || c == 0xB8 || c == 0xD7 || c == 0xF7 || in(c, 0x2C2, 0x2C5) ||
        in(c, 0x2D2, 0x2DF) || in(c, 0x300, 0x36F) || in(c, 0x20A0, 0x20CF) ||
        in(c, 0x2190, 0x23FF) || in(c, 0x2500, 0x27BF) || in(c, 0x2900, 0x2BFF) ||
        in(c, 0x1F000, 0x1FAFF) || in(c, 0xFE00, 0xFE0F) || in(c, 0xFFE0, 0xFFEE))
        return F_SYMBOL;
    return F_LETTER;
}

/* ── Splitter state ────────────────────────────────────────────────────── */

static int lower(uint32_t c) {
    return (c >= 'A' && c <= 'Z') ? (int)(c | 0x20) : (int)c;
}

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
        int fl       = get_fl(s, pos);
        size_t k     = contraction(s, pos, 0);
        if (k) {
            pos += add_tok(s, pos + k);
            continue;
        }
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
        if (nws > 1 && get_cp(s, pos + nws) != OUT_OF_RANGE) {
            pos += nws - 1;
            add_tok(s, pos);
            continue;
        }
        if (nws > 0) {
            pos += nws;
            add_tok(s, pos);
            continue;
        }
        add_tok(s, ++pos);
    }
}

/* llama.cpp unicode_regex_split_custom_llama3 (max_digits = 3) and _qwen2
 * (max_digits = 1):
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,max}|
 *   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ */
/* A word: optional non-newline/non-number lead character, then letters. Emits the piece and
 * returns the position after it, or 0 if no word starts here. */
static size_t l3_word(Split *s, size_t pos, uint32_t cpt, int fl) {
    if (cpt == '\r' || cpt == '\n' || (fl & F_NUMBER)) return 0;
    if (!((fl & F_LETTER) || (get_fl(s, pos + 1) & F_LETTER))) return 0;
    pos++;
    while (get_fl(s, pos) & F_LETTER) pos++;
    add_tok(s, pos);
    return pos;
}

/* A run of digits, in pieces of at most max_digits. */
static size_t l3_digits(Split *s, size_t pos, int fl, int max_digits) {
    if (!(fl & F_NUMBER)) return 0;
    size_t ini = pos;
    while (get_fl(s, pos) & F_NUMBER) {
        if (++pos - ini >= (size_t)max_digits) {
            add_tok(s, pos);
            ini = pos;
        }
    }
    add_tok(s, pos);
    return pos;
}

/* Punctuation or symbols (with an optional leading space) and any newlines after them. */
static size_t l3_symbols(Split *s, size_t pos, uint32_t cpt, int fl) {
    int fl2 = (cpt == ' ') ? get_fl(s, pos + 1) : fl;
    if ((fl2 & (F_SPACE | F_LETTER | F_NUMBER)) || !fl) return 0;
    pos += (cpt == ' ');
    while (!(fl2 & (F_SPACE | F_LETTER | F_NUMBER)) && fl2) fl2 = get_fl(s, ++pos);
    uint32_t c2 = get_cp(s, pos);
    while (c2 == '\r' || c2 == '\n') c2 = get_cp(s, ++pos);
    add_tok(s, pos);
    return pos;
}

/* Whitespace (or any single character nothing above took). Always advances. */
static size_t l3_space(Split *s, size_t pos) {
    size_t nws = 0, last_rn = 0;
    while (get_fl(s, pos + nws) & F_SPACE) {
        uint32_t c2 = get_cp(s, pos + nws);
        if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
        nws++;
    }
    if (last_rn > 0) {
        pos = last_rn;
        add_tok(s, pos);
        return pos;
    }
    if (nws > 1 && get_cp(s, pos + nws) != OUT_OF_RANGE) {
        pos += nws - 1;
        add_tok(s, pos);
        return pos;
    }
    if (nws > 0) {
        pos += nws;
        add_tok(s, pos);
        return pos;
    }
    add_tok(s, ++pos);
    return pos;
}

static void split_llama3(Split *s, int max_digits) {
    for (size_t pos = s->ini; pos < s->end;) {
        uint32_t cpt = get_cp(s, pos);
        int fl       = get_fl(s, pos);
        size_t k     = contraction(s, pos, 1);
        if (k) {
            pos += add_tok(s, pos + k);
            continue;
        }
        size_t next = l3_word(s, pos, cpt, fl);
        if (!next) next = l3_digits(s, pos, fl, max_digits);
        if (!next) next = l3_symbols(s, pos, cpt, fl);
        pos = next ? next : l3_space(s, pos);
    }
}

/* ── Pipeline ──────────────────────────────────────────────────────────── */

typedef enum {
    STEP_GPT2,
    STEP_LLAMA3,
    STEP_QWEN2,
    STEP_NUM1,
    STEP_NUMRUN,
    STEP_PUNCT,
    STEP_DIG3,
    STEP_NL,
    STEP_DS_LETTERS,
    STEP_DS_PUNCT,
    STEP_TRAIL_SPACE,
    STEP_CJK,
    STEP_L_RUN,
    STEP_P_RUN
} Step;

/* Runs one step over the current piece list (lengths in code points). */
static int run_step(const Cpts *c, Step st, size_t **pieces, int *n_pieces) {
    size_t cap  = c->n + 1;
    size_t *out = (size_t *)malloc(cap * sizeof(size_t));
    if (!out) return -1;
    int n_out    = 0;
    size_t start = 0;
    for (int i = 0; i < *n_pieces; i++) {
        Split s = {c,           start, start + (*pieces)[i],       start,
                   out + n_out, 0,     (int)(cap - (size_t)n_out), 0};
        switch (st) {
            case STEP_GPT2:
                split_gpt2(&s);
                break;
            case STEP_LLAMA3:
                split_llama3(&s, 3);
                break;
            case STEP_QWEN2:
                split_llama3(&s, 1);
                break;
            case STEP_NUM1:
                pretok_split_class(&s, CLS_NUMBER_ONE);
                break;
            case STEP_NUMRUN:
                pretok_split_class(&s, CLS_NUMBER_RUN);
                break;
            case STEP_PUNCT:
                pretok_split_class(&s, CLS_PUNCT_RUN);
                break;
            case STEP_DIG3:
                pretok_split_class(&s, CLS_DIGIT3);
                break;
            case STEP_NL:
                pretok_split_class(&s, CLS_NEWLINE);
                break;
            case STEP_DS_LETTERS:
                pretok_split_class(&s, CLS_DS_LETTERS);
                break;
            case STEP_DS_PUNCT:
                pretok_split_class(&s, CLS_DS_PUNCT);
                break;
            case STEP_TRAIL_SPACE:
                pretok_split_class(&s, CLS_TRAIL_SPACE);
                break;
            case STEP_CJK:
                pretok_split_class(&s, CLS_CJK);
                break;
            case STEP_L_RUN:
                pretok_split_class(&s, CLS_L_RUN);
                break;
            case STEP_P_RUN:
                pretok_split_class(&s, CLS_P_RUN);
                break;
        }
        if (s.overflow) {
            free(out);
            return -1;
        }
        n_out += s.n;
        start += (*pieces)[i];
    }
    free(*pieces);
    *pieces   = out;
    *n_pieces = n_out;
    return 0;
}

int pretokenize(PretokType type, const char *text, size_t len, size_t *starts, size_t *lens,
                int max_frags) {
    if (!text || !starts || !lens || max_frags <= 0) return -1;
    if (len == 0) return 0;
    if (type == PRETOK_NONE) {
        starts[0] = 0;
        lens[0]   = len;
        return 1;
    }

    Cpts c;
    if (cpts_decode(&c, text, len) != 0) return -1;
    size_t *pieces = (size_t *)malloc(sizeof(size_t));
    if (!pieces) {
        cpts_free(&c);
        return -1;
    }
    pieces[0] = c.n;
    int n     = 1;

    static const Step DEFAULT_STEPS[] = {STEP_PUNCT, STEP_GPT2, STEP_NUMRUN, STEP_DIG3};
    static const Step GPT2_STEPS[]    = {STEP_GPT2};
    static const Step LLAMA3_STEPS[]  = {STEP_LLAMA3};
    static const Step QWEN2_STEPS[]   = {STEP_QWEN2};
    static const Step SMOLLM_STEPS[]  = {STEP_NUM1, STEP_GPT2};
    /* llama.cpp deepseek-llm / deepseek-coder: each regex splits the pieces left by the one before */
    static const Step DEEPSEEK_LLM_STEPS[]   = {STEP_NL,          STEP_DS_LETTERS, STEP_DS_PUNCT,
                                                STEP_TRAIL_SPACE, STEP_CJK,        STEP_NUMRUN};
    static const Step DEEPSEEK_CODER_STEPS[] = {STEP_NL, STEP_L_RUN, STEP_P_RUN, STEP_CJK,
                                                STEP_NUM1};
    const Step *steps                        = DEFAULT_STEPS;
    int n_steps                              = 4;
    switch (type) {
        case PRETOK_GPT2:
            steps   = GPT2_STEPS;
            n_steps = 1;
            break;
        case PRETOK_LLAMA3:
            steps   = LLAMA3_STEPS;
            n_steps = 1;
            break;
        case PRETOK_QWEN2:
            steps   = QWEN2_STEPS;
            n_steps = 1;
            break;
        case PRETOK_SMOLLM:
            steps   = SMOLLM_STEPS;
            n_steps = 2;
            break;
        case PRETOK_DEEPSEEK_LLM:
            steps   = DEEPSEEK_LLM_STEPS;
            n_steps = 6;
            break;
        case PRETOK_DEEPSEEK_CODER:
            steps   = DEEPSEEK_CODER_STEPS;
            n_steps = 5;
            break;
        default:
            break;
    }
    int rc = 0;
    for (int i = 0; i < n_steps && rc == 0; i++) rc = run_step(&c, steps[i], &pieces, &n);

    int out = -1;
    if (rc == 0 && n <= max_frags) {
        size_t cp = 0;
        for (int i = 0; i < n; i++) {
            starts[i] = c.off[cp];
            lens[i]   = c.off[cp + pieces[i]] - c.off[cp];
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
    static const char *const gpt2[] = {"gpt-2",           "gpt2", "olmo", "mpt", "jais", "trillion",
                                       "granite-docling", NULL};
    static const char *const llama3[] = {"llama3",  "llama-v3",  "llama-bpe", "smaug-bpe",
                                         "falcon3", "falcon-h1", "pixtral",   "midm-2.0",
                                         "lfm2",    NULL};
    static const char *const qwen2[]  = {"qwen2", "stablelm2", "hunyuan", "deepseek-r1-qwen", NULL};
    static const char *const smollm[] = {"smollm", "codeshell", "exaone", "minerva-7b", NULL};
    for (int i = 0; gpt2[i]; i++)
        if (strcmp(pre, gpt2[i]) == 0) return PRETOK_GPT2;
    for (int i = 0; llama3[i]; i++)
        if (strcmp(pre, llama3[i]) == 0) return PRETOK_LLAMA3;
    for (int i = 0; qwen2[i]; i++)
        if (strcmp(pre, qwen2[i]) == 0) return PRETOK_QWEN2;
    for (int i = 0; smollm[i]; i++)
        if (strcmp(pre, smollm[i]) == 0) return PRETOK_SMOLLM;
    if (strcmp(pre, "deepseek-llm") == 0) return PRETOK_DEEPSEEK_LLM;
    if (strcmp(pre, "deepseek-coder") == 0) return PRETOK_DEEPSEEK_CODER;
    return PRETOK_DEFAULT;
}

const char *pretok_type_name(PretokType t) {
    switch (t) {
        case PRETOK_NONE:
            return "none";
        case PRETOK_GPT2:
            return "gpt2";
        case PRETOK_LLAMA3:
            return "llama3";
        case PRETOK_QWEN2:
            return "qwen2";
        case PRETOK_SMOLLM:
            return "smollm";
        case PRETOK_DEEPSEEK_LLM:
            return "deepseek-llm";
        case PRETOK_DEEPSEEK_CODER:
            return "deepseek-coder";
        default:
            return "default";
    }
}
