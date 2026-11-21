/*
 * src/tokenizer/chat_template_parse.cpp
 *
 * Parser: tokens to AST. Split out of chat_template.cpp; the code is unchanged.
 */

#include "chat_template_internal.hpp"

/* Forward declarations */
static NodePtr parse_block(Parser &p);

/*
 * Skip an entire {% kw ... %} ... {% endkw %} construct (block/call/filter/
 * raw), handling nested occurrences of the same construct.
 *
 * block/call/filter/raw are not supported by this engine's runtime, so the
 * correct behaviour for their *definition* is "skip the whole thing,
 * produce no AST nodes". (macro used to be skipped here too; it is now
 * actually parsed and executed — see parse_macro() and NT::MacroDef —
 * because Qwen 3.6's real chat_template.jinja routes every message's
 * rendered content through a `render_content` macro, so skipping it meant
 * every message rendered as empty content. See docs/ai/mistakes.md.)
 * Previously the caller only did p.skip_to_stmt_close() on the *opening*
 * tag, which skipped just e.g. "{% filter upper %}" and then let
 * parse_block keep going — parsing the construct's own body (its nested
 * for/if/text) as literal top-level statements of whatever block happened
 * to be parsing at that point, and letting the body's own "{% endfilter %}"
 * terminate that enclosing block early (at_block_end() matches it). For a
 * template that defines one of these before its main per-message loop,
 * that silently truncates the whole template to "everything up to the end
 * tag", with the body's own control-flow constructs executing
 * unconditionally against an out-of-scope Scope. See docs/ai/mistakes.md.
 */
static void skip_matching_block(Parser &p, const std::string &kw) {
    std::string end_kw = "end" + kw;
    p.skip_to_stmt_close(); /* finish the opening "{% kw ... %}" tag */
    int depth        = 1;
    size_t max_iters = p.toks.size() + 1000;
    size_t iters     = 0;
    while (depth > 0 && !p.is(TK::Eof)) {
        if (++iters > max_iters)
            throw std::runtime_error("chat template macro-skip exceeded iteration bound");
        if (p.is(TK::StmtOpen) && p.is(TK::Ident, 1)) {
            const std::string &k = p.peek(1).sval;
            if (k == kw) {
                depth++;
                p.next();
                p.next();
                p.skip_to_stmt_close();
                continue;
            }
            if (k == end_kw) {
                depth--;
                p.next();
                p.next();
                p.skip_to_stmt_close();
                continue;
            }
        }
        p.next();
    }
}

static NodePtr parse_for(Parser &p) {
    p.next(); /* consume "for" */
    std::string var = p.is(TK::Ident) ? p.next().sval : "";
    /* Tuple unpacking: `for k, v in dict|items` (real chat templates use
     * exactly this for tool-call argument serialization). Both names are
     * encoded into sval as "k,v" — Ident tokens never contain a literal
     * ',', so it's an unambiguous separator at exec() time (same trick
     * parse_set uses for dotted targets). */
    if (p.is(TK::Comma)) {
        p.next();
        if (p.is(TK::Ident)) {
            var += ",";
            var += p.next().sval;
        }
    }
    if (p.kw("in")) p.next();
    auto iter = jinja_parse_expr(p);
    p.skip_to_stmt_close();
    auto body = parse_block(p);
    if (p.is(TK::StmtOpen)) {
        p.next();
        if (p.kw("endfor")) p.next();
        p.skip_to_stmt_close();
    }
    auto n  = std::make_unique<Node>(NT::For);
    n->sval = var;
    n->ch.push_back(std::move(iter));
    n->ch.push_back(std::move(body));
    return n;
}

static NodePtr parse_if(Parser &p) {
    p.next(); /* consume "if" */
    auto n = std::make_unique<Node>(NT::IfChain);
    /* if branch */
    n->ch.push_back(jinja_parse_expr(p));
    p.skip_to_stmt_close();
    n->ch.push_back(parse_block(p));
    /* elif / else */
    while (p.is(TK::StmtOpen)) {
        if (p.kw("elif", 1)) {
            p.next();
            p.next();
            n->ch.push_back(jinja_parse_expr(p));
            p.skip_to_stmt_close();
            n->ch.push_back(parse_block(p));
        } else if (p.kw("else", 1)) {
            p.next();
            p.next();
            p.skip_to_stmt_close();
            n->ch.push_back(nullptr); /* null cond = else */
            n->ch.push_back(parse_block(p));
        } else {
            break;
        }
    }
    if (p.is(TK::StmtOpen)) {
        p.next();
        if (p.kw("endif")) p.next();
        p.skip_to_stmt_close();
    }
    return n;
}

static NodePtr parse_set(Parser &p) {
    p.next(); /* consume "set" */
    /* Dotted target (`set ns.attr = ...`, used by every real chat template
     * that mutates a namespace() object, e.g. Qwen 3.6's `ns.multi_step_tool`
     * / `image_count.value`) is encoded as "var.attr1.attr2" in sval — Ident
     * tokens never contain a literal '.', so '.' is an unambiguous separator
     * at exec() time. See NT::Set's exec() case for the mutate-in-place
     * semantics this requires. */
    std::string var = p.is(TK::Ident) ? p.next().sval : "";
    while (p.is(TK::Dot)) {
        p.next();
        var += ".";
        var += p.is(TK::Ident) ? p.next().sval : "";
    }
    if (p.is(TK::Eq)) p.next();
    auto val = jinja_parse_expr(p);
    p.skip_to_stmt_close();
    auto n  = std::make_unique<Node>(NT::Set);
    n->sval = var;
    n->ch.push_back(std::move(val));
    return n;
}

/* {% macro name(param, param2=default) %} body {% endmacro %}
 * ch[0] = body Block; ch[1..] = Param descriptors, in declaration order. */
static NodePtr parse_macro(Parser &p) {
    p.next(); /* consume "macro" */
    auto def  = std::make_unique<Node>(NT::MacroDef);
    def->sval = p.is(TK::Ident) ? p.next().sval : "";
    def->ch.push_back(nullptr); /* body placeholder, filled in below */
    if (p.is(TK::Lparen)) {
        p.next();
        while (!p.is(TK::Rparen) && !p.is(TK::Eof)) {
            size_t before = p.pos;
            if (p.is(TK::Ident)) {
                auto param  = std::make_unique<Node>(NT::Param);
                param->sval = p.next().sval;
                if (p.is(TK::Eq)) {
                    p.next();
                    param->ch.push_back(jinja_parse_expr(p));
                }
                def->ch.push_back(std::move(param));
            }
            if (p.is(TK::Comma)) p.next();
            if (p.pos == before)
                throw std::runtime_error(
                    "chat_template: macro-param parse stalled (no forward progress)");
        }
        if (p.is(TK::Rparen)) p.next();
    }
    p.skip_to_stmt_close();
    def->ch[0] = parse_block(p); /* stops at {% endmacro %} via at_block_end() */
    if (p.is(TK::StmtOpen)) {
        p.next();
        if (p.kw("endmacro")) p.next();
        p.skip_to_stmt_close();
    }
    return def;
}

static NodePtr parse_stmt(Parser &p) {
    if (p.is(TK::Eof)) return nullptr;

    /* Text token */
    if (p.is(TK::Text)) {
        auto n  = std::make_unique<Node>(NT::Text);
        n->sval = p.next().sval;
        return n;
    }

    /* {{ expr }} */
    if (p.is(TK::ExprOpen)) {
        p.next();
        auto expr = jinja_parse_expr(p);
        if (p.is(TK::ExprClose)) p.next();
        auto n = std::make_unique<Node>(NT::ExprOut);
        n->ch.push_back(std::move(expr));
        return n;
    }

    /* {% ... %} */
    if (p.is(TK::StmtOpen)) {
        if (p.at_block_end()) return nullptr; /* signal end-of-block */
        p.next();                             /* consume {% */
        if (!p.is(TK::Ident)) {
            p.skip_to_stmt_close();
            return std::make_unique<Node>(NT::Noop);
        }
        const std::string &kw = p.peek().sval;
        if (kw == "for") return parse_for(p);
        if (kw == "if") return parse_if(p);
        if (kw == "set") return parse_set(p);
        if (kw == "macro") return parse_macro(p);
        if (kw == "block" || kw == "call" || kw == "filter" || kw == "raw") {
            /* Not supported by the runtime — skip the whole construct
             * (including its body) rather than just its opening tag, so
             * the body doesn't leak into the enclosing block. See
             * skip_matching_block()'s comment. */
            skip_matching_block(p, kw);
            return std::make_unique<Node>(NT::Noop);
        }
        /* Other unrecognized statement keywords — skip just the tag. */
        p.skip_to_stmt_close();
        return std::make_unique<Node>(NT::Noop);
    }
    return nullptr;
}

static NodePtr parse_block(Parser &p) {
    auto blk = std::make_unique<Node>(NT::Block);
    /* Deterministic non-termination guard: parse_stmt must strictly advance
     * p.pos every call (Parser::next() is a no-op once pos==toks.size()-1,
     * the trailing Eof token — every other token consumes at least one).
     * A parser bug that returns a Noop without advancing pos would spin
     * here forever, which for a real GGUF chat_template.jinja is a genuine
     * multi-GB memory blow-up in the caller (chat_template_apply builds a
     * result string only after this returns, but the runaway CPU spin
     * itself was observed alongside runaway RSS during verification —
     * see docs/ai/mistakes.md). Bound iterations to comfortably more than
     * any real template could need (tokens.size() would already bound a
     * *correct* parser; this is 4x that plus a floor, purely defensive)
     * and fail closed (caught by chat_template_apply, returns nullptr)
     * rather than hang/leak unboundedly. */
    size_t max_iters = p.toks.size() * 4 + 10000;
    size_t iters     = 0;
    for (;;) {
        if (p.is(TK::Eof) || p.at_block_end()) break;
        if (++iters > max_iters)
            throw std::runtime_error("chat template parser exceeded iteration bound");
        auto s = parse_stmt(p);
        if (!s) break;
        if (s->type != NT::Noop) blk->ch.push_back(std::move(s));
    }
    return blk;
}

NodePtr jinja_parse_block(Parser &p) {
    return parse_block(p);
}
