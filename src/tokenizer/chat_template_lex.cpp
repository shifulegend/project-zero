/*
 * src/tokenizer/chat_template_lex.cpp
 *
 * Lexer: template text to tokens. Split out of chat_template.cpp. One small function per kind of
 * token; jinja_lex() walks the text, the tags and the tokens inside them.
 */

#include "chat_template_internal.hpp"

namespace {

/* The cursor over the template and the tokens produced so far. */
struct Lexer {
    const char *p;
    const char *end;
    std::vector<Token> toks;

    void push(TK type) {
        Token t;
        t.type = type;
        toks.push_back(std::move(t));
    }
};

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

void skip_ws(Lexer &L) {
    while (L.p < L.end && is_ws(*L.p)) ++L.p;
}

/* Text up to the next tag becomes one Text token. */
void lex_text(Lexer &L) {
    const char *text_start = L.p;
    while (L.p < L.end) {
        if (L.p[0] == '{' && L.p + 1 < L.end && (L.p[1] == '{' || L.p[1] == '%' || L.p[1] == '#'))
            break;
        ++L.p;
    }
    if (L.p > text_start) {
        Token t;
        t.type = TK::Text;
        t.sval = std::string(text_start, (size_t)(L.p - text_start));
        L.toks.push_back(std::move(t));
    }
}

/* Comment {# ... #}: skipped silently. */
void skip_comment(Lexer &L) {
    L.p += 2;
    while (L.p < L.end && !(L.p[0] == '#' && L.p + 1 < L.end && L.p[1] == '}')) ++L.p;
    if (L.p < L.end) L.p += 2;
}

/* Whitespace control: {%- or {{- strips trailing whitespace from the previous text. */
void strip_previous_text(Lexer &L) {
    if (L.toks.empty() || L.toks.back().type != TK::Text) return;
    std::string &ts = L.toks.back().sval;
    while (!ts.empty() && is_ws(ts.back())) ts.pop_back();
    if (ts.empty()) L.toks.pop_back();
}

/* String literal in single or double quotes, with \n \t \r and \<char> escapes. */
void lex_string(Lexer &L) {
    char delim = *L.p++;
    std::string sv;
    while (L.p < L.end && *L.p != delim) {
        if (*L.p == '\\' && L.p + 1 < L.end) {
            ++L.p;
            switch (*L.p) {
                case 'n':
                    sv += '\n';
                    break;
                case 't':
                    sv += '\t';
                    break;
                case 'r':
                    sv += '\r';
                    break;
                default:
                    sv += *L.p;
                    break;
            }
        } else {
            sv += *L.p;
        }
        ++L.p;
    }
    if (L.p < L.end) ++L.p;
    Token t;
    t.type = TK::Str;
    t.sval = std::move(sv);
    L.toks.push_back(std::move(t));
}

void lex_int(Lexer &L) {
    long long v = 0;
    while (L.p < L.end && std::isdigit((unsigned char)*L.p)) v = v * 10 + (*L.p++ - '0');
    Token t;
    t.type = TK::Int;
    t.ival = v;
    L.toks.push_back(std::move(t));
}

/* Identifier or keyword. */
void lex_ident(Lexer &L) {
    const char *s = L.p;
    while (L.p < L.end && (std::isalnum((unsigned char)*L.p) || *L.p == '_')) ++L.p;
    Token t;
    t.type = TK::Ident;
    t.sval = std::string(s, (size_t)(L.p - s));
    L.toks.push_back(std::move(t));
}

const struct {
    char a, b;
    TK tk;
} k_two_char_ops[] = {
    {'=', '=', TK::EqEq},
    {'!', '=', TK::Neq},
    {'<', '=', TK::Le},
    {'>', '=', TK::Ge},
};

/* Two-character operators; returns whether one was lexed. */
bool lex_two_char_op(Lexer &L) {
    if (L.p + 1 >= L.end) return false;
    for (const auto &op : k_two_char_ops) {
        if (L.p[0] != op.a || L.p[1] != op.b) continue;
        L.p += 2;
        L.push(op.tk);
        return true;
    }
    return false;
}

const struct {
    char c;
    TK tk;
} k_single_char_ops[] = {
    {'<', TK::Lt},      {'>', TK::Gt},       {'=', TK::Eq},       {'+', TK::Plus},
    {'~', TK::Tilde},   {'-', TK::Minus},    {'*', TK::Star},     {'/', TK::Slash},
    {'%', TK::Percent}, {'|', TK::Pipe},     {',', TK::Comma},    {'.', TK::Dot},
    {':', TK::Colon},   {'[', TK::Lbracket}, {']', TK::Rbracket}, {'(', TK::Lparen},
    {')', TK::Rparen},  {'{', TK::Lbrace},   {'}', TK::Rbrace},
};

/* One character that is an operator or punctuation; anything else is skipped. */
void lex_single_char_op(Lexer &L) {
    char c = *L.p++;
    for (const auto &op : k_single_char_ops)
        if (op.c == c) {
            L.push(op.tk);
            return;
        }
}

/* `-` + closing delimiter: whitespace control after the tag. */
bool at_close_strip(const Lexer &L, char c1) {
    return L.p[0] == '-' && L.p + 2 < L.end && L.p[1] == c1 && L.p[2] == '}';
}

bool at_close(const Lexer &L, char c1) {
    return L.p[0] == c1 && L.p + 1 < L.end && L.p[1] == '}';
}

/* The closing delimiter of a tag (with optional - for whitespace control). The stripped
 * whitespace is consumed directly from the source cursor right here, rather than via a flag
 * applied to whatever Text token happens to be lexed next: two tags can be source-adjacent with
 * zero characters between them (e.g. `{%- endmacro -%}{{- foo }}`), and a deferred flag would
 * incorrectly strip leading whitespace off a LATER, unrelated text run instead of the (empty) gap
 * actually following this tag. See docs/ai/mistakes.md. Returns whether the tag closed here. */
bool lex_tag_close(Lexer &L, bool is_expr) {
    char c1          = is_expr ? '}' : '%';
    bool close_strip = at_close_strip(L, c1);
    if (close_strip) ++L.p;
    if (!at_close(L, c1)) return false;
    L.p += 2;
    L.push(is_expr ? TK::ExprClose : TK::StmtClose);
    if (close_strip) skip_ws(L);
    return true;
}

/* One token inside a tag: a string, integer, identifier or operator (unknown characters are
 * skipped). */
void lex_tag_token(Lexer &L) {
    char c = *L.p;
    if (c == '\'' || c == '"') return lex_string(L);
    if (std::isdigit((unsigned char)c)) return lex_int(L);
    if (std::isalpha((unsigned char)c) || c == '_') return lex_ident(L);
    if (!lex_two_char_op(L)) lex_single_char_op(L);
}

/* The tokens inside one tag, up to and including its closing delimiter. */
void lex_tag_body(Lexer &L, bool is_expr) {
    while (L.p < L.end) {
        skip_ws(L);
        if (L.p >= L.end || lex_tag_close(L, is_expr)) break;
        lex_tag_token(L);
    }
}

} /* namespace */

std::vector<Token> jinja_lex(const std::string &tmpl) {
    Lexer L;
    L.p   = tmpl.c_str();
    L.end = L.p + tmpl.size();

    while (L.p < L.end) {
        lex_text(L);
        if (L.p >= L.end) break;
        if (L.p[1] == '#') {
            skip_comment(L);
            continue;
        }
        bool is_expr = (L.p[1] == '{');
        L.p += 2;
        if (L.p < L.end && *L.p == '-') {
            ++L.p;
            strip_previous_text(L);
        }
        L.push(is_expr ? TK::ExprOpen : TK::StmtOpen);
        lex_tag_body(L, is_expr);
    }
    L.push(TK::Eof);
    return std::move(L.toks);
}
