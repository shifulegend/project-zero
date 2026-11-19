/*
 * src/tokenizer/chat_template_lex.cpp
 *
 * Lexer: template text to tokens. Split out of chat_template.cpp; the code is unchanged.
 */

#include "chat_template_internal.hpp"

std::vector<Token> jinja_lex(const std::string &tmpl) {
    std::vector<Token> toks;
    const char *p   = tmpl.c_str();
    const char *end = p + tmpl.size();

    while (p < end) {
        /* Accumulate text until next tag */
        const char *text_start = p;
        while (p < end) {
            if (p[0] == '{' && p + 1 < end && (p[1] == '{' || p[1] == '%' || p[1] == '#')) break;
            ++p;
        }

        if (p > text_start) {
            std::string txt(text_start, (size_t)(p - text_start));
            if (!txt.empty()) {
                Token t;
                t.type = TK::Text;
                t.sval = std::move(txt);
                toks.push_back(std::move(t));
            }
        }
        if (p >= end) break;

        /* Comment {# ... #} — skip silently */
        if (p[1] == '#') {
            p += 2;
            while (p < end && !(p[0] == '#' && p + 1 < end && p[1] == '}')) ++p;
            if (p < end) p += 2;
            continue;
        }

        bool is_expr = (p[1] == '{');
        p += 2;

        /* Whitespace control: {%- or {{- strips trailing WS from previous text */
        if (p < end && *p == '-') {
            ++p;
            if (!toks.empty() && toks.back().type == TK::Text) {
                std::string &ts = toks.back().sval;
                while (!ts.empty() && (ts.back() == ' ' || ts.back() == '\t' || ts.back() == '\r' ||
                                       ts.back() == '\n'))
                    ts.pop_back();
                if (ts.empty()) toks.pop_back();
            }
        }
        {
            Token t;
            t.type = is_expr ? TK::ExprOpen : TK::StmtOpen;
            toks.push_back(std::move(t));
        }

        /* Lex tokens inside the tag */
        char c1 = is_expr ? '}' : '%';
        while (p < end) {
            while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
            if (p >= end) break;

            /* Closing delimiter (with optional - for WS control). The
             * stripped whitespace is consumed directly from the source
             * cursor right here, rather than via a flag applied to
             * whatever Text token happens to be lexed next — two tags can
             * be source-adjacent with zero characters between them (e.g.
             * `{%- endmacro -%}{{- foo }}`), and a deferred flag would
             * incorrectly strip leading whitespace off a LATER, unrelated
             * text run instead of the (empty) gap actually following this
             * tag. See docs/ai/mistakes.md. */
            bool close_strip = (p[0] == '-' && p + 2 < end && p[1] == c1 && p[2] == '}');
            if (close_strip) ++p;
            if (p[0] == c1 && p + 1 < end && p[1] == '}') {
                p += 2;
                Token t;
                t.type = is_expr ? TK::ExprClose : TK::StmtClose;
                toks.push_back(std::move(t));
                if (close_strip)
                    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
                break;
            }

            /* String literal */
            if (*p == '\'' || *p == '"') {
                char delim = *p++;
                std::string sv;
                while (p < end && *p != delim) {
                    if (*p == '\\' && p + 1 < end) {
                        ++p;
                        switch (*p) {
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
                                sv += *p;
                                break;
                        }
                    } else {
                        sv += *p;
                    }
                    ++p;
                }
                if (p < end) ++p;
                Token t;
                t.type = TK::Str;
                t.sval = std::move(sv);
                toks.push_back(std::move(t));
                continue;
            }

            /* Integer */
            if (std::isdigit((unsigned char)*p)) {
                long long v = 0;
                while (p < end && std::isdigit((unsigned char)*p)) v = v * 10 + (*p++ - '0');
                Token t;
                t.type = TK::Int;
                t.ival = v;
                toks.push_back(std::move(t));
                continue;
            }

            /* Identifier / keyword */
            if (std::isalpha((unsigned char)*p) || *p == '_') {
                const char *s = p;
                while (p < end && (std::isalnum((unsigned char)*p) || *p == '_')) ++p;
                Token t;
                t.type = TK::Ident;
                t.sval = std::string(s, (size_t)(p - s));
                toks.push_back(std::move(t));
                continue;
            }

            /* Two-char operators */
            if (p + 1 < end) {
                TK tk2 = TK::Eof;
                if (p[0] == '=' && p[1] == '=')
                    tk2 = TK::EqEq;
                else if (p[0] == '!' && p[1] == '=')
                    tk2 = TK::Neq;
                else if (p[0] == '<' && p[1] == '=')
                    tk2 = TK::Le;
                else if (p[0] == '>' && p[1] == '=')
                    tk2 = TK::Ge;
                if (tk2 != TK::Eof) {
                    p += 2;
                    Token t;
                    t.type = tk2;
                    toks.push_back(std::move(t));
                    continue;
                }
            }

            /* Single-char operators */
            TK tk = TK::Eof;
            switch (*p++) {
                case '<':
                    tk = TK::Lt;
                    break;
                case '>':
                    tk = TK::Gt;
                    break;
                case '=':
                    tk = TK::Eq;
                    break;
                case '+':
                    tk = TK::Plus;
                    break;
                case '~':
                    tk = TK::Tilde;
                    break;
                case '-':
                    tk = TK::Minus;
                    break;
                case '*':
                    tk = TK::Star;
                    break;
                case '/':
                    tk = TK::Slash;
                    break;
                case '%':
                    tk = TK::Percent;
                    break;
                case '|':
                    tk = TK::Pipe;
                    break;
                case ',':
                    tk = TK::Comma;
                    break;
                case '.':
                    tk = TK::Dot;
                    break;
                case ':':
                    tk = TK::Colon;
                    break;
                case '[':
                    tk = TK::Lbracket;
                    break;
                case ']':
                    tk = TK::Rbracket;
                    break;
                case '(':
                    tk = TK::Lparen;
                    break;
                case ')':
                    tk = TK::Rparen;
                    break;
                case '{':
                    tk = TK::Lbrace;
                    break;
                case '}':
                    tk = TK::Rbrace;
                    break;
                default:
                    continue; /* skip unknown */
            }
            Token t;
            t.type = tk;
            toks.push_back(std::move(t));
        }
    }
    {
        Token t;
        t.type = TK::Eof;
        toks.push_back(std::move(t));
    }
    return toks;
}
