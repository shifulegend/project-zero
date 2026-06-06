/*
 * chat_template.cpp — C++17 Jinja2 engine for LLM chat templates.
 *
 * Architecture mirrors llama.cpp's common/jinja: lexer → parser (AST) → runtime.
 * Exposed as a plain C function via extern "C" so the C99 engine calls it
 * without any ABI friction.
 *
 * Jinja2 subset supported (covers every real-world LLM chat template):
 *   DeepSeek-V2, ChatML (Qwen/SmolLM2), Llama-3, Mistral, Phi-3, Gemma-2
 *
 * Design principles:
 *   - No external dependencies: pure C++17 + STL
 *   - CPU-first: template rendering is a one-time pre-inference cost
 *   - Dynamic: all behaviour driven by the template string from the GGUF
 */

#include "tokenizer/chat_template.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/* ======================================================================== *
 *  SECTION 1 — RUNTIME VALUE
 * ======================================================================== */

enum class VType { Undef, None, Bool, Int, Float, Str, Array, Object };

struct Value {
    VType type = VType::Undef;
    bool b = false;
    long long i = 0;
    double f = 0.0;
    std::string s;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    Value() = default;
    explicit Value(VType t)          : type(t)                {}
    explicit Value(bool b_)          : type(VType::Bool),  b(b_)      {}
    explicit Value(long long i_)     : type(VType::Int),   i(i_)      {}
    explicit Value(double f_)        : type(VType::Float), f(f_)      {}
    Value(const std::string& s_)     : type(VType::Str),   s(s_)      {} // NOLINT
    Value(std::string&& s_)          : type(VType::Str),   s(std::move(s_)) {} // NOLINT
    Value(const char* s_)            : type(VType::Str),   s(s_ ? s_ : "") {} // NOLINT

    bool truthy() const {
        switch (type) {
            case VType::Undef:  return false;
            case VType::None:   return false;
            case VType::Bool:   return b;
            case VType::Int:    return i != 0;
            case VType::Float:  return f != 0.0;
            case VType::Str:    return !s.empty();
            case VType::Array:  return !arr.empty();
            case VType::Object: return !obj.empty();
        }
        return false;
    }

    std::string to_str() const {
        switch (type) {
            case VType::Undef:  return "";
            case VType::None:   return "";
            case VType::Bool:   return b ? "True" : "False";
            case VType::Int:    return std::to_string(i);
            case VType::Float: {
                char buf[64]; snprintf(buf, sizeof(buf), "%g", f); return buf;
            }
            case VType::Str:    return s;
            case VType::Array:  return "";
            case VType::Object: return "";
        }
        return "";
    }

    long long to_int() const {
        switch (type) {
            case VType::Int:   return i;
            case VType::Bool:  return b ? 1 : 0;
            case VType::Float: return (long long)f;
            case VType::Str:
                try { return std::stoll(s); } catch (...) { return 0; }
            default: return 0;
        }
    }

    bool equals(const Value& o) const {
        if (type == VType::Str || o.type == VType::Str)
            return to_str() == o.to_str();
        if (type == VType::Bool && o.type == VType::Bool) return b == o.b;
        if (type == VType::Int  && o.type == VType::Int)  return i == o.i;
        if (type == VType::None && o.type == VType::None) return true;
        if (type == VType::Undef && o.type == VType::Undef) return true;
        return false;
    }

    int cmp(const Value& o) const {
        if (type == VType::Int && o.type == VType::Int)
            return (i < o.i) ? -1 : (i > o.i) ? 1 : 0;
        return to_str().compare(o.to_str());
    }
};

/* ======================================================================== *
 *  SECTION 2 — LEXER
 * ======================================================================== */

enum class TK {
    Eof, Text, Str, Int, Ident,
    EqEq, Neq, Lt, Le, Gt, Ge, Eq,
    Plus, Tilde, Minus, Star, Slash, Percent,
    Pipe, Comma, Dot,
    Lbracket, Rbracket, Lparen, Rparen, Lbrace, Rbrace,
    StmtOpen, StmtClose, ExprOpen, ExprClose
};

struct Token {
    TK type = TK::Eof;
    std::string sval;
    long long ival = 0;
};

static std::vector<Token> lex(const std::string& tmpl) {
    std::vector<Token> toks;
    const char* p   = tmpl.c_str();
    const char* end = p + tmpl.size();
    bool strip_next = false;   /* set by -%} or -}} — strip leading WS from next text */

    while (p < end) {
        /* Accumulate text until next tag */
        const char* text_start = p;
        while (p < end) {
            if (p[0] == '{' && p+1 < end &&
                (p[1] == '{' || p[1] == '%' || p[1] == '#')) break;
            ++p;
        }

        if (p > text_start) {
            std::string txt(text_start, (size_t)(p - text_start));
            if (strip_next) {
                size_t i = 0;
                while (i < txt.size() &&
                       (txt[i]==' '||txt[i]=='\t'||txt[i]=='\r'||txt[i]=='\n')) ++i;
                txt = txt.substr(i);
                strip_next = false;
            }
            if (!txt.empty()) {
                Token t; t.type = TK::Text; t.sval = std::move(txt);
                toks.push_back(std::move(t));
            }
        }
        if (p >= end) break;

        /* Comment {# ... #} — skip silently */
        if (p[1] == '#') {
            p += 2;
            while (p < end && !(p[0]=='#' && p+1<end && p[1]=='}')) ++p;
            if (p < end) p += 2;
            continue;
        }

        bool is_expr = (p[1] == '{');
        p += 2;

        /* Whitespace control: {%- or {{- strips trailing WS from previous text */
        if (p < end && *p == '-') {
            ++p;
            if (!toks.empty() && toks.back().type == TK::Text) {
                std::string& ts = toks.back().sval;
                while (!ts.empty() &&
                       (ts.back()==' '||ts.back()=='\t'||
                        ts.back()=='\r'||ts.back()=='\n'))
                    ts.pop_back();
                if (ts.empty()) toks.pop_back();
            }
        }
        {
            Token t; t.type = is_expr ? TK::ExprOpen : TK::StmtOpen;
            toks.push_back(std::move(t));
        }

        /* Lex tokens inside the tag */
        char c1 = is_expr ? '}' : '%';
        while (p < end) {
            while (p < end && (*p==' '||*p=='\t'||*p=='\r'||*p=='\n')) ++p;
            if (p >= end) break;

            /* Closing delimiter (with optional - for WS control) */
            bool close_strip = (p[0]=='-' && p+2<end && p[1]==c1 && p[2]=='}');
            if (close_strip) { strip_next = true; ++p; }
            if (p[0]==c1 && p+1<end && p[1]=='}') {
                p += 2;
                Token t; t.type = is_expr ? TK::ExprClose : TK::StmtClose;
                toks.push_back(std::move(t));
                break;
            }

            /* String literal */
            if (*p == '\'' || *p == '"') {
                char delim = *p++;
                std::string sv;
                while (p < end && *p != delim) {
                    if (*p == '\\' && p+1 < end) {
                        ++p;
                        switch (*p) {
                            case 'n': sv += '\n'; break;
                            case 't': sv += '\t'; break;
                            case 'r': sv += '\r'; break;
                            default:  sv += *p;   break;
                        }
                    } else { sv += *p; }
                    ++p;
                }
                if (p < end) ++p;
                Token t; t.type = TK::Str; t.sval = std::move(sv);
                toks.push_back(std::move(t));
                continue;
            }

            /* Integer */
            if (std::isdigit((unsigned char)*p)) {
                long long v = 0;
                while (p < end && std::isdigit((unsigned char)*p)) v = v*10 + (*p++ - '0');
                Token t; t.type = TK::Int; t.ival = v;
                toks.push_back(std::move(t));
                continue;
            }

            /* Identifier / keyword */
            if (std::isalpha((unsigned char)*p) || *p == '_') {
                const char* s = p;
                while (p < end && (std::isalnum((unsigned char)*p) || *p=='_')) ++p;
                Token t; t.type = TK::Ident; t.sval = std::string(s, (size_t)(p-s));
                toks.push_back(std::move(t));
                continue;
            }

            /* Two-char operators */
            if (p+1 < end) {
                TK tk2 = TK::Eof;
                if (p[0]=='='&&p[1]=='=') tk2=TK::EqEq;
                else if(p[0]=='!'&&p[1]=='=') tk2=TK::Neq;
                else if(p[0]=='<'&&p[1]=='=') tk2=TK::Le;
                else if(p[0]=='>'&&p[1]=='=') tk2=TK::Ge;
                if (tk2 != TK::Eof) {
                    p += 2;
                    Token t; t.type = tk2; toks.push_back(std::move(t));
                    continue;
                }
            }

            /* Single-char operators */
            TK tk = TK::Eof;
            switch (*p++) {
                case '<': tk = TK::Lt;       break;
                case '>': tk = TK::Gt;       break;
                case '=': tk = TK::Eq;       break;
                case '+': tk = TK::Plus;     break;
                case '~': tk = TK::Tilde;    break;
                case '-': tk = TK::Minus;    break;
                case '*': tk = TK::Star;     break;
                case '/': tk = TK::Slash;    break;
                case '%': tk = TK::Percent;  break;
                case '|': tk = TK::Pipe;     break;
                case ',': tk = TK::Comma;    break;
                case '.': tk = TK::Dot;      break;
                case '[': tk = TK::Lbracket; break;
                case ']': tk = TK::Rbracket; break;
                case '(': tk = TK::Lparen;   break;
                case ')': tk = TK::Rparen;   break;
                case '{': tk = TK::Lbrace;   break;
                case '}': tk = TK::Rbrace;   break;
                default: continue; /* skip unknown */
            }
            Token t; t.type = tk; toks.push_back(std::move(t));
        }
    }
    { Token t; t.type = TK::Eof; toks.push_back(std::move(t)); }
    return toks;
}

/* ======================================================================== *
 *  SECTION 3 — AST
 * ======================================================================== */

enum class NT {
    /* Statements */
    Block, Text, ExprOut, For, IfChain, Set, Noop,
    /* Expressions */
    LitStr, LitInt, LitBool, LitNone,
    Ident, Getitem, Getattr, Binop, Unop, Call, IsDef, Filter, Ternary
};

struct Node {
    NT type = NT::Noop;
    std::string sval;          /* string literal / ident name / op / filter / var */
    long long   ival = 0;      /* integer literal */
    bool        bval = false;  /* bool literal; also "negated" flag in IsDef */
    std::vector<std::unique_ptr<Node>> ch;

    explicit Node(NT t) : type(t) {}
};

using NodePtr = std::unique_ptr<Node>;

/* ======================================================================== *
 *  SECTION 4 — PARSER
 * ======================================================================== */

struct Parser {
    const std::vector<Token>& toks;
    size_t pos = 0;

    const Token& peek(int off = 0) const {
        size_t i = pos + (size_t)off;
        return (i < toks.size()) ? toks[i] : toks.back();
    }
    Token& peek_mut(int off = 0) {
        size_t i = pos + (size_t)off;
        return (i < toks.size()) ? const_cast<Token&>(toks[i]) : const_cast<Token&>(toks.back());
    }
    const Token& next() {
        const Token& t = peek();
        if (pos < toks.size()) ++pos;
        return t;
    }
    bool is(TK t, int off=0)  const { return peek(off).type == t; }
    bool kw(const char* s, int off=0) const {
        return is(TK::Ident, off) && peek(off).sval == s;
    }
    void skip_to_stmt_close() {
        while (!is(TK::StmtClose) && !is(TK::Eof)) next();
        if (is(TK::StmtClose)) next();
    }
    /* True if next tokens are {%  <block-terminator>  (without consuming) */
    bool at_block_end() const {
        if (!is(TK::StmtOpen)) return false;
        if (!is(TK::Ident, 1)) return false;
        const std::string& k = peek(1).sval;
        return k=="endif"||k=="endfor"||k=="else"||k=="elif"
              ||k=="endblock"||k=="endmacro"||k=="endcall"
              ||k=="endfilter"||k=="endset";
    }
};

/* Forward declarations */
static NodePtr parse_expr(Parser& p);
static NodePtr parse_block(Parser& p);

/* ---- Primary / postfix ---- */
static NodePtr parse_primary(Parser& p) {
    /* Parenthesised expression */
    if (p.is(TK::Lparen)) {
        p.next();
        auto e = parse_expr(p);
        if (p.is(TK::Rparen)) p.next();
        return e;
    }

    /* Literals */
    if (p.is(TK::Str))  { auto n=std::make_unique<Node>(NT::LitStr); n->sval=p.next().sval; return n; }
    if (p.is(TK::Int))  { auto n=std::make_unique<Node>(NT::LitInt); n->ival=p.next().ival; return n; }
    if (p.is(TK::Ident)) {
        const std::string& sv = p.peek().sval;
        if (sv=="true"||sv=="True")  { p.next(); auto n=std::make_unique<Node>(NT::LitBool); n->bval=true;  return n; }
        if (sv=="false"||sv=="False"){ p.next(); auto n=std::make_unique<Node>(NT::LitBool); n->bval=false; return n; }
        if (sv=="none"||sv=="None")  { p.next(); return std::make_unique<Node>(NT::LitNone); }

        auto n = std::make_unique<Node>(NT::Ident);
        n->sval = p.next().sval;

        /* Postfixes: [key], .attr, (args) */
        for (;;) {
            if (p.is(TK::Lbracket)) {
                p.next();
                auto key = parse_expr(p);
                if (p.is(TK::Rbracket)) p.next();
                auto gi = std::make_unique<Node>(NT::Getitem);
                gi->ch.push_back(std::move(n));
                gi->ch.push_back(std::move(key));
                n = std::move(gi);
            } else if (p.is(TK::Dot)) {
                p.next();
                std::string attr = p.is(TK::Ident) ? p.next().sval
                                 : p.is(TK::Int)   ? std::to_string(p.next().ival)
                                 : "";
                auto ga = std::make_unique<Node>(NT::Getattr);
                ga->sval = attr;
                ga->ch.push_back(std::move(n));
                n = std::move(ga);
            } else if (p.is(TK::Lparen)) {
                p.next();
                auto call = std::make_unique<Node>(NT::Call);
                call->ch.push_back(std::move(n));
                while (!p.is(TK::Rparen) && !p.is(TK::Eof)) {
                    call->ch.push_back(parse_expr(p));
                    if (p.is(TK::Comma)) p.next();
                }
                if (p.is(TK::Rparen)) p.next();
                n = std::move(call);
            } else { break; }
        }
        return n;
    }
    /* Fallback */
    return std::make_unique<Node>(NT::LitNone);
}

/* ---- Filter: primary | fname(args) ---- */
static NodePtr parse_filter(Parser& p) {
    auto n = parse_primary(p);
    while (p.is(TK::Pipe)) {
        p.next();
        std::string fname = p.is(TK::Ident) ? p.next().sval : "";
        auto f = std::make_unique<Node>(NT::Filter);
        f->sval = fname;
        f->ch.push_back(std::move(n));
        if (p.is(TK::Lparen)) {
            p.next();
            while (!p.is(TK::Rparen) && !p.is(TK::Eof)) {
                f->ch.push_back(parse_expr(p));
                if (p.is(TK::Comma)) p.next();
            }
            if (p.is(TK::Rparen)) p.next();
        }
        n = std::move(f);
    }
    return n;
}

/* ---- Unary minus ---- */
static NodePtr parse_unary(Parser& p) {
    if (p.is(TK::Minus)) {
        p.next();
        auto u = std::make_unique<Node>(NT::Unop);
        u->sval = "-";
        u->ch.push_back(parse_filter(p));
        return u;
    }
    return parse_filter(p);
}

/* ---- Multiply / divide / modulo ---- */
static NodePtr parse_mul(Parser& p) {
    auto n = parse_unary(p);
    while (p.is(TK::Star)||p.is(TK::Slash)||p.is(TK::Percent)) {
        std::string op = p.is(TK::Star)?  "*" : p.is(TK::Slash)? "/" : "%";
        p.next();
        auto b = std::make_unique<Node>(NT::Binop); b->sval=op;
        b->ch.push_back(std::move(n)); b->ch.push_back(parse_unary(p));
        n = std::move(b);
    }
    return n;
}

/* ---- Add / subtract / tilde (string concat) ---- */
static NodePtr parse_add(Parser& p) {
    auto n = parse_mul(p);
    while (p.is(TK::Plus)||p.is(TK::Minus)||p.is(TK::Tilde)) {
        std::string op = p.is(TK::Plus)?"+": p.is(TK::Minus)?"-":"~";
        p.next();
        auto b = std::make_unique<Node>(NT::Binop); b->sval=op;
        b->ch.push_back(std::move(n)); b->ch.push_back(parse_mul(p));
        n = std::move(b);
    }
    return n;
}

/* ---- Comparison + is-test + in ---- */
static NodePtr parse_compare(Parser& p) {
    auto n = parse_add(p);

    /* Comparison operators */
    if (p.is(TK::EqEq)||p.is(TK::Neq)||p.is(TK::Lt)||
        p.is(TK::Le)  ||p.is(TK::Gt) ||p.is(TK::Ge)) {
        std::string op;
        switch (p.peek().type) {
            case TK::EqEq: op="=="; break; case TK::Neq: op="!="; break;
            case TK::Lt:   op="<";  break; case TK::Le:  op="<="; break;
            case TK::Gt:   op=">";  break; case TK::Ge:  op=">="; break;
            default: break;
        }
        p.next();
        auto b = std::make_unique<Node>(NT::Binop); b->sval=op;
        b->ch.push_back(std::move(n)); b->ch.push_back(parse_add(p));
        n = std::move(b);
    }

    /* is [not] <test_name> */
    if (p.kw("is")) {
        p.next();
        bool neg = false;
        if (p.kw("not")) { neg=true; p.next(); }
        std::string test = p.is(TK::Ident) ? p.next().sval : "";
        auto t = std::make_unique<Node>(NT::IsDef);
        t->sval = test; t->bval = neg;
        t->ch.push_back(std::move(n));
        return t;
    }

    /* [not] in <iterable> */
    if (p.kw("not") && p.kw("in",1)) {
        p.next(); p.next();
        auto b = std::make_unique<Node>(NT::Binop); b->sval="not_in";
        b->ch.push_back(std::move(n)); b->ch.push_back(parse_add(p));
        return b;
    }
    if (p.kw("in")) {
        p.next();
        auto b = std::make_unique<Node>(NT::Binop); b->sval="in";
        b->ch.push_back(std::move(n)); b->ch.push_back(parse_add(p));
        return b;
    }

    return n;
}

/* ---- not ---- */
static NodePtr parse_not(Parser& p) {
    if (p.kw("not")) {
        p.next();
        auto u = std::make_unique<Node>(NT::Unop); u->sval="not";
        u->ch.push_back(parse_not(p));
        return u;
    }
    return parse_compare(p);
}

/* ---- and ---- */
static NodePtr parse_and(Parser& p) {
    auto n = parse_not(p);
    while (p.kw("and")) {
        p.next();
        auto b = std::make_unique<Node>(NT::Binop); b->sval="and";
        b->ch.push_back(std::move(n)); b->ch.push_back(parse_not(p));
        n = std::move(b);
    }
    return n;
}

/* ---- or ---- */
static NodePtr parse_or(Parser& p) {
    auto n = parse_and(p);
    while (p.kw("or")) {
        p.next();
        auto b = std::make_unique<Node>(NT::Binop); b->sval="or";
        b->ch.push_back(std::move(n)); b->ch.push_back(parse_and(p));
        n = std::move(b);
    }
    return n;
}

/* ---- Ternary: value if cond else other ---- */
static NodePtr parse_expr(Parser& p) {
    auto n = parse_or(p);
    if (p.kw("if")) {
        p.next();
        auto cond = parse_or(p);
        auto t = std::make_unique<Node>(NT::Ternary);
        t->ch.push_back(std::move(n));
        t->ch.push_back(std::move(cond));
        if (p.kw("else")) {
            p.next(); t->ch.push_back(parse_or(p));
        } else {
            t->ch.push_back(std::make_unique<Node>(NT::LitStr)); /* "" */
        }
        return t;
    }
    return n;
}

/* ---- Statement parsers (called after {%…} open and keyword consumed) ---- */

static NodePtr parse_for(Parser& p) {
    p.next(); /* consume "for" */
    std::string var = p.is(TK::Ident) ? p.next().sval : "";
    /* tuple: for k, v in ... — discard second var */
    if (p.is(TK::Comma)) { p.next(); if (p.is(TK::Ident)) p.next(); }
    if (p.kw("in")) p.next();
    auto iter = parse_expr(p);
    p.skip_to_stmt_close();
    auto body = parse_block(p);
    if (p.is(TK::StmtOpen)) { p.next(); if (p.kw("endfor")) p.next(); p.skip_to_stmt_close(); }
    auto n = std::make_unique<Node>(NT::For); n->sval = var;
    n->ch.push_back(std::move(iter));
    n->ch.push_back(std::move(body));
    return n;
}

static NodePtr parse_if(Parser& p) {
    p.next(); /* consume "if" */
    auto n = std::make_unique<Node>(NT::IfChain);
    /* if branch */
    n->ch.push_back(parse_expr(p));
    p.skip_to_stmt_close();
    n->ch.push_back(parse_block(p));
    /* elif / else */
    while (p.is(TK::StmtOpen)) {
        if (p.kw("elif",1)) {
            p.next(); p.next();
            n->ch.push_back(parse_expr(p));
            p.skip_to_stmt_close();
            n->ch.push_back(parse_block(p));
        } else if (p.kw("else",1)) {
            p.next(); p.next();
            p.skip_to_stmt_close();
            n->ch.push_back(nullptr); /* null cond = else */
            n->ch.push_back(parse_block(p));
        } else { break; }
    }
    if (p.is(TK::StmtOpen)) { p.next(); if (p.kw("endif")) p.next(); p.skip_to_stmt_close(); }
    return n;
}

static NodePtr parse_set(Parser& p) {
    p.next(); /* consume "set" */
    std::string var = p.is(TK::Ident) ? p.next().sval : "";
    if (p.is(TK::Eq)) p.next();
    auto val = parse_expr(p);
    p.skip_to_stmt_close();
    auto n = std::make_unique<Node>(NT::Set); n->sval = var;
    n->ch.push_back(std::move(val));
    return n;
}

static NodePtr parse_stmt(Parser& p) {
    if (p.is(TK::Eof)) return nullptr;

    /* Text token */
    if (p.is(TK::Text)) {
        auto n = std::make_unique<Node>(NT::Text); n->sval = p.next().sval; return n;
    }

    /* {{ expr }} */
    if (p.is(TK::ExprOpen)) {
        p.next();
        auto expr = parse_expr(p);
        if (p.is(TK::ExprClose)) p.next();
        auto n = std::make_unique<Node>(NT::ExprOut);
        n->ch.push_back(std::move(expr));
        return n;
    }

    /* {% ... %} */
    if (p.is(TK::StmtOpen)) {
        if (p.at_block_end()) return nullptr; /* signal end-of-block */
        p.next(); /* consume {% */
        if (!p.is(TK::Ident)) { p.skip_to_stmt_close(); return std::make_unique<Node>(NT::Noop); }
        const std::string& kw = p.peek().sval;
        if (kw=="for")  return parse_for(p);
        if (kw=="if")   return parse_if(p);
        if (kw=="set")  return parse_set(p);
        /* macro, block, raw, call, etc. — skip */
        p.skip_to_stmt_close();
        return std::make_unique<Node>(NT::Noop);
    }
    return nullptr;
}

static NodePtr parse_block(Parser& p) {
    auto blk = std::make_unique<Node>(NT::Block);
    for (;;) {
        if (p.is(TK::Eof) || p.at_block_end()) break;
        auto s = parse_stmt(p);
        if (!s) break;
        if (s->type != NT::Noop) blk->ch.push_back(std::move(s));
    }
    return blk;
}

/* ======================================================================== *
 *  SECTION 5 — RUNTIME
 * ======================================================================== */

class Scope {
public:
    explicit Scope(Scope* par = nullptr) : parent_(par) {}

    Value get(const std::string& name) const {
        auto it = vars_.find(name);
        if (it != vars_.end()) return it->second;
        if (parent_) return parent_->get(name);
        return Value(); /* Undef */
    }
    void set(const std::string& name, Value v) { vars_[name] = std::move(v); }
    bool has(const std::string& name) const {
        if (vars_.count(name)) return true;
        if (parent_) return parent_->has(name);
        return false;
    }

private:
    std::map<std::string, Value> vars_;
    Scope* parent_ = nullptr;
};

static Value eval(const Node* n, Scope& sc);
static void  exec(const Node* n, Scope& sc, std::string& out);

static Value eval(const Node* n, Scope& sc) {
    if (!n) return Value();
    switch (n->type) {
        case NT::LitStr:  return Value(n->sval);
        case NT::LitInt:  return Value(n->ival);
        case NT::LitBool: return Value(n->bval);
        case NT::LitNone: return Value(VType::None);

        case NT::Ident:   return sc.get(n->sval);

        case NT::Getitem: {
            Value obj = eval(n->ch[0].get(), sc);
            Value key = eval(n->ch[1].get(), sc);
            if (obj.type == VType::Object) {
                auto it = obj.obj.find(key.to_str());
                return (it != obj.obj.end()) ? it->second : Value();
            }
            if (obj.type == VType::Array && key.type == VType::Int) {
                long long idx = key.i;
                if (idx < 0) idx += (long long)obj.arr.size();
                if (idx >= 0 && (size_t)idx < obj.arr.size()) return obj.arr[(size_t)idx];
            }
            return Value();
        }

        case NT::Getattr: {
            Value obj = eval(n->ch[0].get(), sc);
            if (obj.type == VType::Object) {
                auto it = obj.obj.find(n->sval);
                if (it != obj.obj.end()) return it->second;
            }
            if (obj.type == VType::Array) {
                if (n->sval=="length"||n->sval=="size") return Value((long long)obj.arr.size());
            }
            return Value();
        }

        case NT::Binop: {
            const std::string& op = n->sval;
            /* Short-circuit boolean ops */
            if (op=="and") { Value l=eval(n->ch[0].get(),sc); return l.truthy()?eval(n->ch[1].get(),sc):l; }
            if (op=="or")  { Value l=eval(n->ch[0].get(),sc); return l.truthy()?l:eval(n->ch[1].get(),sc); }

            Value l = eval(n->ch[0].get(), sc);
            Value r = eval(n->ch[1].get(), sc);

            if (op=="+"||op=="~") return Value(l.to_str() + r.to_str());
            if (op=="-") { if(l.type==VType::Int&&r.type==VType::Int) return Value(l.i-r.i); return Value(); }
            if (op=="*") { if(l.type==VType::Int&&r.type==VType::Int) return Value(l.i*r.i); return Value(); }
            if (op=="/") { if(l.type==VType::Int&&r.type==VType::Int&&r.i) return Value(l.i/r.i); return Value(); }
            if (op=="%") { if(l.type==VType::Int&&r.type==VType::Int&&r.i) return Value(l.i%r.i); return Value(); }
            if (op=="==") return Value(l.equals(r));
            if (op=="!=") return Value(!l.equals(r));
            if (op=="<")  return Value(l.cmp(r)<0);
            if (op=="<=") return Value(l.cmp(r)<=0);
            if (op==">")  return Value(l.cmp(r)>0);
            if (op==">=") return Value(l.cmp(r)>=0);
            if (op=="in") {
                std::string ls = l.to_str();
                if (r.type==VType::Str)    return Value(r.s.find(ls)!=std::string::npos);
                if (r.type==VType::Array)  { for(auto& x:r.arr) if(x.equals(l)) return Value(true); return Value(false); }
                if (r.type==VType::Object) return Value(r.obj.count(ls)>0);
                return Value(false);
            }
            if (op=="not_in") {
                std::string ls = l.to_str();
                bool found = false;
                if (r.type==VType::Str)   found = r.s.find(ls)!=std::string::npos;
                if (r.type==VType::Array) { for(auto& x:r.arr) if(x.equals(l)) { found=true; break; } }
                return Value(!found);
            }
            return Value();
        }

        case NT::Unop: {
            Value v = eval(n->ch[0].get(), sc);
            if (n->sval=="not") return Value(!v.truthy());
            if (n->sval=="-" && v.type==VType::Int) return Value(-v.i);
            return Value();
        }

        case NT::IsDef: {
            bool result = false;
            const std::string& test = n->sval;
            if (test=="defined") {
                if (!n->ch.empty() && n->ch[0]->type==NT::Ident)
                    result = sc.has(n->ch[0]->sval);
                else if (!n->ch.empty())
                    result = (eval(n->ch[0].get(),sc).type != VType::Undef);
            } else if (test=="none"||test=="None") {
                result = !n->ch.empty() && eval(n->ch[0].get(),sc).type==VType::None;
            } else if (test=="string") {
                result = !n->ch.empty() && eval(n->ch[0].get(),sc).type==VType::Str;
            } else if (test=="integer"||test=="number") {
                VType t = n->ch.empty() ? VType::Undef : eval(n->ch[0].get(),sc).type;
                result = (t==VType::Int||t==VType::Float);
            } else if (test=="iterable") {
                VType t = n->ch.empty() ? VType::Undef : eval(n->ch[0].get(),sc).type;
                result = (t==VType::Array||t==VType::Str);
            } else if (test=="mapping") {
                result = !n->ch.empty() && eval(n->ch[0].get(),sc).type==VType::Object;
            } else if (test=="odd") {
                result = !n->ch.empty() && (eval(n->ch[0].get(),sc).to_int()%2!=0);
            } else if (test=="even") {
                result = !n->ch.empty() && (eval(n->ch[0].get(),sc).to_int()%2==0);
            }
            return Value(n->bval ? !result : result);
        }

        case NT::Filter: {
            Value v = eval(n->ch[0].get(), sc);
            const std::string& fn = n->sval;
            auto arg = [&](int i) -> Value {
                return (i+1 < (int)n->ch.size()) ? eval(n->ch[i+1].get(),sc) : Value();
            };
            if (fn=="trim") {
                std::string s = v.to_str();
                size_t a = s.find_first_not_of(" \t\r\n");
                size_t b = s.find_last_not_of(" \t\r\n");
                return (a==std::string::npos) ? Value(std::string("")) : Value(s.substr(a,b-a+1));
            }
            if (fn=="upper") { std::string s=v.to_str(); for(char& c:s) c=(char)toupper((unsigned char)c); return Value(s); }
            if (fn=="lower") { std::string s=v.to_str(); for(char& c:s) c=(char)tolower((unsigned char)c); return Value(s); }
            if (fn=="length"||fn=="count") {
                if (v.type==VType::Array)  return Value((long long)v.arr.size());
                if (v.type==VType::Object) return Value((long long)v.obj.size());
                return Value((long long)v.to_str().size());
            }
            if (fn=="default"||fn=="d") {
                bool use_default = (v.type==VType::Undef||v.type==VType::None);
                /* default(value, boolean) — second arg=true also triggers on falsy */
                if (!use_default && n->ch.size()>2 && arg(1).truthy())
                    use_default = !v.truthy();
                return use_default ? arg(0) : v;
            }
            if (fn=="replace") {
                std::string s=v.to_str(), from=arg(0).to_str(), to=arg(1).to_str();
                if (!from.empty()) {
                    size_t pos=0;
                    while((pos=s.find(from,pos))!=std::string::npos)
                        { s.replace(pos,from.size(),to); pos+=to.size(); }
                }
                return Value(s);
            }
            if (fn=="join") {
                std::string sep = arg(0).to_str();
                if (v.type==VType::Array) {
                    std::string r;
                    for (size_t i=0;i<v.arr.size();i++) { if(i) r+=sep; r+=v.arr[i].to_str(); }
                    return Value(r);
                }
                return v;
            }
            if (fn=="int")   return Value(v.to_int());
            if (fn=="string") return Value(v.to_str());
            if (fn=="tojson"||fn=="safe"||fn=="e"||fn=="escape") return v;
            if (fn=="indent") {
                /* indent(width, first=false) */
                int width = (int)arg(0).to_int(); if (width<=0) width=4;
                std::string s=v.to_str(), pad(width,' '), r;
                for (char c:s) { r+=c; if(c=='\n') r+=pad; }
                return Value(r);
            }
            /* Unknown filter — pass through */
            return v;
        }

        case NT::Call: {
            /* ch[0] = function expr, ch[1..] = args */
            std::string fname;
            if (!n->ch.empty() && n->ch[0]->type==NT::Ident) fname=n->ch[0]->sval;
            if (fname=="raise_exception") return Value(); /* no-op */
            if (fname=="namespace") { Value o; o.type=VType::Object; return o; }
            if (fname=="range") {
                long long start=0, end_v=0, step=1;
                size_t nargs = n->ch.size()-1;
                if (nargs==1) end_v = eval(n->ch[1].get(),sc).to_int();
                else if (nargs>=2) { start=eval(n->ch[1].get(),sc).to_int(); end_v=eval(n->ch[2].get(),sc).to_int(); }
                if (nargs>=3) step=eval(n->ch[3].get(),sc).to_int();
                if (step==0) return Value();
                Value a; a.type=VType::Array;
                for (long long i=start; step>0?(i<end_v):(i>end_v); i+=step) a.arr.push_back(Value(i));
                return a;
            }
            if (fname=="dict") { Value o; o.type=VType::Object; return o; }
            if (fname=="list") { Value a; a.type=VType::Array; return a; }
            return Value();
        }

        case NT::Ternary: {
            Value cond = eval(n->ch[1].get(), sc);
            return cond.truthy() ? eval(n->ch[0].get(),sc) : eval(n->ch[2].get(),sc);
        }

        default: return Value();
    }
}

static void exec(const Node* n, Scope& sc, std::string& out) {
    if (!n) return;
    switch (n->type) {

        case NT::Block:
            for (auto& c : n->ch) exec(c.get(), sc, out);
            break;

        case NT::Text:
            out += n->sval;
            break;

        case NT::ExprOut:
            out += eval(n->ch[0].get(), sc).to_str();
            break;

        case NT::For: {
            Value iter = eval(n->ch[0].get(), sc);
            if (iter.type == VType::Array) {
                size_t sz = iter.arr.size();
                for (size_t i = 0; i < sz; i++) {
                    Scope child(&sc);
                    child.set(n->sval, iter.arr[i]);
                    /* Loop variable object */
                    Value lv; lv.type = VType::Object;
                    lv.obj["index"]     = Value((long long)(i+1));
                    lv.obj["index0"]    = Value((long long)i);
                    lv.obj["first"]     = Value(i==0);
                    lv.obj["last"]      = Value(i==sz-1);
                    lv.obj["length"]    = Value((long long)sz);
                    lv.obj["revindex"]  = Value((long long)(sz-i));
                    lv.obj["revindex0"] = Value((long long)(sz-i-1));
                    child.set("loop", std::move(lv));
                    exec(n->ch[1].get(), child, out);
                }
            }
            break;
        }

        case NT::IfChain: {
            /* Children: [cond0, body0, cond1, body1, ..., ?else_body]
             * null condition ptr means "else" branch */
            bool done = false;
            size_t i = 0;
            while (i+1 < n->ch.size()) {
                if (n->ch[i] == nullptr) {          /* else */
                    if (!done) exec(n->ch[i+1].get(), sc, out);
                    done = true;
                } else {
                    Value cond = eval(n->ch[i].get(), sc);
                    if (cond.truthy() && !done) {
                        exec(n->ch[i+1].get(), sc, out);
                        done = true;
                    }
                }
                i += 2;
            }
            break;
        }

        case NT::Set: {
            Value v = eval(n->ch[0].get(), sc);
            sc.set(n->sval, std::move(v));
            break;
        }

        default: break;
    }
}

/* ======================================================================== *
 *  SECTION 6 — PUBLIC C API
 * ======================================================================== */

extern "C" {

char *chat_template_apply(const char *tmpl_cstr,
                          const char * const *roles,
                          const char * const *contents,
                          int n_messages,
                          const char *bos_token_str,
                          const char *eos_token_str,
                          int add_generation_prompt)
{
    if (!tmpl_cstr) return nullptr;

    try {
        auto toks = lex(std::string(tmpl_cstr));
        Parser p{toks, 0};
        auto ast = parse_block(p);

        Scope sc;
        sc.set("bos_token", Value(bos_token_str ? bos_token_str : ""));
        sc.set("eos_token", Value(eos_token_str ? eos_token_str : ""));
        sc.set("add_generation_prompt", Value(add_generation_prompt != 0));

        /* Build messages array */
        Value msgs; msgs.type = VType::Array;
        for (int i = 0; i < n_messages; i++) {
            Value msg; msg.type = VType::Object;
            msg.obj["role"]    = Value(roles    && roles[i]    ? roles[i]    : "");
            msg.obj["content"] = Value(contents && contents[i] ? contents[i] : "");
            msgs.arr.push_back(std::move(msg));
        }
        sc.set("messages", std::move(msgs));

        std::string out;
        exec(ast.get(), sc, out);

        char *result = (char*)malloc(out.size() + 1);
        if (result) memcpy(result, out.c_str(), out.size() + 1);
        return result;

    } catch (...) {
        return nullptr;
    }
}

} /* extern "C" */
