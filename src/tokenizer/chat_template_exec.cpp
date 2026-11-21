/*
 * src/tokenizer/chat_template_exec.cpp
 *
 * Runtime, statements: renders the AST nodes (text, loops, conditionals, assignments, macros) into
 * the output string. Split out of chat_template_eval.cpp; the code is unchanged.
 */

#include "chat_template_internal.hpp"

/* Deterministic runaway guard: a well-formed chat template's rendered
 * output is at most a few hundred KB even for a long conversation (this
 * engine's own seq_len ceilings are far smaller than that in tokens). This
 * bounds output growth from any interpreter bug (this file's Jinja subset
 * is intentionally incomplete — e.g. macros are skipped, not executed —
 * see skip_matching_block()'s comment) the same way run_state.c traps
 * pathological allocations deterministically rather than discovering them
 * via OOM: throws (caught by chat_template_apply, returns nullptr) instead
 * of growing without bound. See docs/ai/mistakes.md. */
#define TN_JINJA_MAX_OUTPUT_BYTES (64u * 1024u * 1024u)
static void tn_jinja_check_output_bound(const std::string &out) {
    if (out.size() > TN_JINJA_MAX_OUTPUT_BYTES)
        throw std::runtime_error("chat template output exceeded sanity bound");
}


/* The `loop` object of iteration i of sz. */
static Value make_loop_object(const Value &iter, size_t i, size_t sz) {
    Value lv;
    lv.type             = VType::Object;
    lv.obj["index"]     = Value((long long)(i + 1));
    lv.obj["index0"]    = Value((long long)i);
    lv.obj["first"]     = Value(i == 0);
    lv.obj["last"]      = Value(i == sz - 1);
    lv.obj["length"]    = Value((long long)sz);
    lv.obj["revindex"]  = Value((long long)(sz - i));
    lv.obj["revindex0"] = Value((long long)(sz - i - 1));
    if (i > 0) lv.obj["previtem"] = iter.arr[i - 1];
    if (i + 1 < sz) lv.obj["nextitem"] = iter.arr[i + 1];
    return lv;
}

/* Binds the loop target(s) for one item. With a tuple target (var2 not empty) a pair is
 * unpacked; anything else binds the whole item to var1 and leaves var2 undefined, rather than
 * crash (e.g. a plain array iterated with a tuple target). */
static void bind_loop_vars(Scope &child, const std::string &var1, const std::string &var2,
                           const Value &item) {
    if (!var2.empty() && item.type == VType::Array && item.arr.size() >= 2) {
        child.set(var1, item.arr[0]);
        child.set(var2, item.arr[1]);
        return;
    }
    child.set(var1, item);
}

static void exec_for(const Node *n, Scope &sc, std::string &out) {
    Value iter = jinja_eval(n->ch[0].get(), sc);
    if (iter.type != VType::Array) return;
    size_t sz = iter.arr.size();
    /* Tuple-unpacking target ("k,v" from parse_for, see its comment) vs. a single loop
     * variable. */
    size_t comma     = n->sval.find(',');
    std::string var1 = (comma == std::string::npos) ? n->sval : n->sval.substr(0, comma);
    std::string var2 = (comma == std::string::npos) ? "" : n->sval.substr(comma + 1);

    for (size_t i = 0; i < sz; i++) {
        Scope child(&sc);
        bind_loop_vars(child, var1, var2, iter.arr[i]);
        child.set("loop", make_loop_object(iter, i, sz));
        jinja_exec(n->ch[1].get(), child, out);
    }
}

/* Children: [cond0, body0, cond1, body1, ..., ?else_body]; a null condition means "else". */
static void exec_if_chain(const Node *n, Scope &sc, std::string &out) {
    bool done = false;
    for (size_t i = 0; i + 1 < n->ch.size(); i += 2) {
        bool take = n->ch[i] == nullptr || jinja_eval(n->ch[i].get(), sc).truthy();
        if (take && !done) jinja_exec(n->ch[i + 1].get(), sc, out);
        if (take) done = true;
    }
}

/* `base.attr1.attr2 = v` — mutate the namespace object in place via get_ref() rather than
 * sc.set(base, ...), which would silently replace the whole object with `v`. If the base var
 * doesn't exist, no-op (matches this engine's undefined-tolerant behavior elsewhere). */
static void set_attribute_path(const Node *n, Scope &sc, size_t dot, Value &&v) {
    Value *target = sc.get_ref(n->sval.substr(0, dot));
    size_t start  = dot + 1;
    while (target) {
        size_t next_dot  = n->sval.find('.', start);
        std::string attr = n->sval.substr(start, next_dot == std::string::npos ? std::string::npos
                                                                               : next_dot - start);
        if (next_dot == std::string::npos) {
            target->obj[attr] = std::move(v);
            return;
        }
        target = &target->obj[attr];
        start  = next_dot + 1;
    }
}

static void exec_set(const Node *n, Scope &sc) {
    Value v    = jinja_eval(n->ch[0].get(), sc);
    size_t dot = n->sval.find('.');
    if (dot == std::string::npos)
        sc.set(n->sval, std::move(v));
    else
        set_attribute_path(n, sc, dot, std::move(v));
}

void jinja_exec(const Node *n, Scope &sc, std::string &out) {
    if (!n) return;
    switch (n->type) {
        case NT::Block:
            for (auto &c : n->ch) jinja_exec(c.get(), sc, out);
            break;
        case NT::Text:
            out += n->sval;
            tn_jinja_check_output_bound(out);
            break;
        case NT::ExprOut:
            out += jinja_eval(n->ch[0].get(), sc).to_str();
            tn_jinja_check_output_bound(out);
            break;
        case NT::For:
            exec_for(n, sc, out);
            break;
        case NT::IfChain:
            exec_if_chain(n, sc, out);
            break;
        case NT::Set:
            exec_set(n, sc);
            break;
        case NT::MacroDef:
            sc.def_macro(n->sval, n);
            break;
        default:
            break;
    }
}
