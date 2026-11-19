/*
 * src/tokenizer/chat_template_parse_expr.cpp
 *
 * Parser, expressions: call arguments, primaries, filters and the operator precedence chain.
 * Split out of chat_template_parse.cpp; the code is unchanged.
 */

#include "chat_template_internal.hpp"

static NodePtr parse_expr(Parser &p);

/*
 * Parse one argument inside a call `(...)` or filter `(...)` argument list.
 * Handles Python/Jinja keyword-argument syntax `name=expr` (e.g.
 * `namespace(multi_step_tool=true, last_query_index=...)`,
 * `is_system_content=false`) by wrapping it in an NT::Kwarg node (sval=name,
 * ch[0]=value expr) — eval() treats a bare Kwarg as a transparent pass-
 * through to its value (see NT::Kwarg's case), so every existing positional
 * consumer (range()/dict()/list(), builtin filters' arg(i) lookup) keeps
 * working unchanged even when handed a Kwarg node; only namespace() and
 * macro-call argument binding actually look at the name (see their eval()
 * cases). Consuming the `name =` pair here is also what guarantees forward
 * progress — `=` (TK::Eq) is not a valid expression-start token, so without
 * this, parse_expr() would fall through to parse_primary's literal-none
 * fallback without advancing the parser, spinning the caller's
 * `while (!Rparen)` loop forever while it keeps allocating AST nodes (this
 * is exactly what happened parsing Qwen 3.6's real chat_template.jinja,
 * which OOM'd the process — see docs/ai/mistakes.md).
 */
static NodePtr parse_call_arg(Parser &p) {
    if (p.is(TK::Ident) && p.is(TK::Eq, 1)) {
        auto kw  = std::make_unique<Node>(NT::Kwarg);
        kw->sval = p.next().sval;
        p.next(); /* consume '=' */
        kw->ch.push_back(parse_expr(p));
        return kw;
    }
    return parse_expr(p);
}

/* ---- Primary / postfix ---- */
static NodePtr parse_primary(Parser &p) {
    /* Parenthesised expression */
    if (p.is(TK::Lparen)) {
        p.next();
        auto e = parse_expr(p);
        if (p.is(TK::Rparen)) p.next();
        return e;
    }

    NodePtr n;

    /* Literals */
    if (p.is(TK::Str)) {
        n       = std::make_unique<Node>(NT::LitStr);
        n->sval = p.next().sval;
    } else if (p.is(TK::Int)) {
        n       = std::make_unique<Node>(NT::LitInt);
        n->ival = p.next().ival;
    } else if (p.is(TK::Ident)) {
        const std::string &sv = p.peek().sval;
        if (sv == "true" || sv == "True") {
            p.next();
            n       = std::make_unique<Node>(NT::LitBool);
            n->bval = true;
        } else if (sv == "false" || sv == "False") {
            p.next();
            n       = std::make_unique<Node>(NT::LitBool);
            n->bval = false;
        } else if (sv == "none" || sv == "None") {
            p.next();
            n = std::make_unique<Node>(NT::LitNone);
        } else {
            n       = std::make_unique<Node>(NT::Ident);
            n->sval = p.next().sval;
        }
    } else {
        /* Fallback */
        return std::make_unique<Node>(NT::LitNone);
    }

    /* Postfixes: [key], .attr, (args) — applied uniformly to any primary
     * (identifier or string/int/bool/none literal), not just identifiers,
     * so e.g. `'abcdef'.startswith('abc')` or `content.split(...)` parse
     * the same way. (eval()'s NT::Call only special-cases method calls
     * whose receiver is a Getattr — see that case for which string methods
     * are actually implemented; other literal-postfix combinations just
     * evaluate through the existing Getitem/Getattr/Call machinery.) */
    for (;;) {
        if (p.is(TK::Lbracket)) {
            p.next();
            /* Python slice syntax `[start:stop:step]` (each part
             * optional), e.g. Qwen 3.6's chat_template.jinja uses
             * `messages[::-1]` to scan history in reverse. Only enters
             * the slice path once a ':' is actually seen, so plain
             * `[key]` indexing (the common case) is unaffected. */
            NodePtr start, stop, step;
            bool is_slice = false;
            if (p.is(TK::Colon)) {
                is_slice = true;
            } else {
                start = parse_expr(p);
                if (p.is(TK::Colon)) is_slice = true;
            }
            if (is_slice) {
                p.next(); /* consume first ':' */
                if (!p.is(TK::Colon) && !p.is(TK::Rbracket)) stop = parse_expr(p);
                if (p.is(TK::Colon)) {
                    p.next();
                    if (!p.is(TK::Rbracket)) step = parse_expr(p);
                }
                if (p.is(TK::Rbracket)) p.next();
                auto gs = std::make_unique<Node>(NT::Getslice);
                gs->ch.push_back(std::move(n));
                gs->ch.push_back(std::move(start));
                gs->ch.push_back(std::move(stop));
                gs->ch.push_back(std::move(step));
                n = std::move(gs);
            } else {
                if (p.is(TK::Rbracket)) p.next();
                auto gi = std::make_unique<Node>(NT::Getitem);
                gi->ch.push_back(std::move(n));
                gi->ch.push_back(std::move(start));
                n = std::move(gi);
            }
        } else if (p.is(TK::Dot)) {
            p.next();
            std::string attr = p.is(TK::Ident) ? p.next().sval
                               : p.is(TK::Int) ? std::to_string(p.next().ival)
                                               : "";
            auto ga          = std::make_unique<Node>(NT::Getattr);
            ga->sval         = attr;
            ga->ch.push_back(std::move(n));
            n = std::move(ga);
        } else if (p.is(TK::Lparen)) {
            p.next();
            auto call = std::make_unique<Node>(NT::Call);
            call->ch.push_back(std::move(n));
            while (!p.is(TK::Rparen) && !p.is(TK::Eof)) {
                size_t before = p.pos;
                call->ch.push_back(parse_call_arg(p));
                if (p.is(TK::Comma)) p.next();
                if (p.pos == before)
                    throw std::runtime_error(
                        "chat_template: call-arg parse stalled (no forward progress)");
            }
            if (p.is(TK::Rparen)) p.next();
            n = std::move(call);
        } else {
            break;
        }
    }
    return n;
}

/* ---- Filter: primary | fname(args) ---- */
static NodePtr parse_filter(Parser &p) {
    auto n = parse_primary(p);
    while (p.is(TK::Pipe)) {
        p.next();
        std::string fname = p.is(TK::Ident) ? p.next().sval : "";
        auto f            = std::make_unique<Node>(NT::Filter);
        f->sval           = fname;
        f->ch.push_back(std::move(n));
        if (p.is(TK::Lparen)) {
            p.next();
            while (!p.is(TK::Rparen) && !p.is(TK::Eof)) {
                size_t before = p.pos;
                f->ch.push_back(parse_call_arg(p));
                if (p.is(TK::Comma)) p.next();
                if (p.pos == before)
                    throw std::runtime_error(
                        "chat_template: filter-arg parse stalled (no forward progress)");
            }
            if (p.is(TK::Rparen)) p.next();
        }
        n = std::move(f);
    }
    return n;
}

/* ---- Unary minus ---- */
static NodePtr parse_unary(Parser &p) {
    if (p.is(TK::Minus)) {
        p.next();
        auto u  = std::make_unique<Node>(NT::Unop);
        u->sval = "-";
        u->ch.push_back(parse_filter(p));
        return u;
    }
    return parse_filter(p);
}

/* ---- Multiply / divide / modulo ---- */
static NodePtr parse_mul(Parser &p) {
    auto n = parse_unary(p);
    while (p.is(TK::Star) || p.is(TK::Slash) || p.is(TK::Percent)) {
        std::string op = p.is(TK::Star) ? "*" : p.is(TK::Slash) ? "/" : "%";
        p.next();
        auto b  = std::make_unique<Node>(NT::Binop);
        b->sval = op;
        b->ch.push_back(std::move(n));
        b->ch.push_back(parse_unary(p));
        n = std::move(b);
    }
    return n;
}

/* ---- Add / subtract / tilde (string concat) ---- */
static NodePtr parse_add(Parser &p) {
    auto n = parse_mul(p);
    while (p.is(TK::Plus) || p.is(TK::Minus) || p.is(TK::Tilde)) {
        std::string op = p.is(TK::Plus) ? "+" : p.is(TK::Minus) ? "-" : "~";
        p.next();
        auto b  = std::make_unique<Node>(NT::Binop);
        b->sval = op;
        b->ch.push_back(std::move(n));
        b->ch.push_back(parse_mul(p));
        n = std::move(b);
    }
    return n;
}

/* ---- Comparison + is-test + in ---- */
static NodePtr parse_compare(Parser &p) {
    auto n = parse_add(p);

    /* Comparison operators */
    if (p.is(TK::EqEq) || p.is(TK::Neq) || p.is(TK::Lt) || p.is(TK::Le) || p.is(TK::Gt) ||
        p.is(TK::Ge)) {
        std::string op;
        switch (p.peek().type) {
            case TK::EqEq:
                op = "==";
                break;
            case TK::Neq:
                op = "!=";
                break;
            case TK::Lt:
                op = "<";
                break;
            case TK::Le:
                op = "<=";
                break;
            case TK::Gt:
                op = ">";
                break;
            case TK::Ge:
                op = ">=";
                break;
            default:
                break;
        }
        p.next();
        auto b  = std::make_unique<Node>(NT::Binop);
        b->sval = op;
        b->ch.push_back(std::move(n));
        b->ch.push_back(parse_add(p));
        n = std::move(b);
    }

    /* is [not] <test_name> */
    if (p.kw("is")) {
        p.next();
        bool neg = false;
        if (p.kw("not")) {
            neg = true;
            p.next();
        }
        std::string test = p.is(TK::Ident) ? p.next().sval : "";
        auto t           = std::make_unique<Node>(NT::IsDef);
        t->sval          = test;
        t->bval          = neg;
        t->ch.push_back(std::move(n));
        return t;
    }

    /* [not] in <iterable> */
    if (p.kw("not") && p.kw("in", 1)) {
        p.next();
        p.next();
        auto b  = std::make_unique<Node>(NT::Binop);
        b->sval = "not_in";
        b->ch.push_back(std::move(n));
        b->ch.push_back(parse_add(p));
        return b;
    }
    if (p.kw("in")) {
        p.next();
        auto b  = std::make_unique<Node>(NT::Binop);
        b->sval = "in";
        b->ch.push_back(std::move(n));
        b->ch.push_back(parse_add(p));
        return b;
    }

    return n;
}

/* ---- not ---- */
static NodePtr parse_not(Parser &p) {
    if (p.kw("not")) {
        p.next();
        auto u  = std::make_unique<Node>(NT::Unop);
        u->sval = "not";
        u->ch.push_back(parse_not(p));
        return u;
    }
    return parse_compare(p);
}

/* ---- and ---- */
static NodePtr parse_and(Parser &p) {
    auto n = parse_not(p);
    while (p.kw("and")) {
        p.next();
        auto b  = std::make_unique<Node>(NT::Binop);
        b->sval = "and";
        b->ch.push_back(std::move(n));
        b->ch.push_back(parse_not(p));
        n = std::move(b);
    }
    return n;
}

/* ---- or ---- */
static NodePtr parse_or(Parser &p) {
    auto n = parse_and(p);
    while (p.kw("or")) {
        p.next();
        auto b  = std::make_unique<Node>(NT::Binop);
        b->sval = "or";
        b->ch.push_back(std::move(n));
        b->ch.push_back(parse_and(p));
        n = std::move(b);
    }
    return n;
}

/* ---- Ternary: value if cond else other ---- */
static NodePtr parse_expr(Parser &p) {
    auto n = parse_or(p);
    if (p.kw("if")) {
        p.next();
        auto cond = parse_or(p);
        auto t    = std::make_unique<Node>(NT::Ternary);
        t->ch.push_back(std::move(n));
        t->ch.push_back(std::move(cond));
        if (p.kw("else")) {
            p.next();
            t->ch.push_back(parse_or(p));
        } else {
            t->ch.push_back(std::make_unique<Node>(NT::LitStr)); /* "" */
        }
        return t;
    }
    return n;
}

/* ---- Statement parsers (called after {%…} open and keyword consumed) ---- */

NodePtr jinja_parse_expr(Parser &p) {
    return parse_expr(p);
}
