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

        case NT::For: {
            Value iter = jinja_eval(n->ch[0].get(), sc);
            if (iter.type == VType::Array) {
                size_t sz = iter.arr.size();
                /* Tuple-unpacking target ("k,v" from parse_for, see its
                 * comment) vs. a single loop variable. */
                size_t comma = n->sval.find(',');
                std::string var1 =
                    (comma == std::string::npos) ? n->sval : n->sval.substr(0, comma);
                std::string var2 = (comma == std::string::npos) ? "" : n->sval.substr(comma + 1);

                for (size_t i = 0; i < sz; i++) {
                    Scope child(&sc);
                    if (var2.empty()) {
                        child.set(var1, iter.arr[i]);
                    } else if (iter.arr[i].type == VType::Array && iter.arr[i].arr.size() >= 2) {
                        child.set(var1, iter.arr[i].arr[0]);
                        child.set(var2, iter.arr[i].arr[1]);
                    } else {
                        /* Not actually a pair (e.g. plain array iterated
                         * with a tuple target) — bind the whole item to
                         * var1, leave var2 undefined, rather than crash. */
                        child.set(var1, iter.arr[i]);
                    }
                    /* Loop variable object */
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
                    child.set("loop", std::move(lv));
                    jinja_exec(n->ch[1].get(), child, out);
                }
            }
            break;
        }

        case NT::IfChain: {
            /* Children: [cond0, body0, cond1, body1, ..., ?else_body]
             * null condition ptr means "else" branch */
            bool done = false;
            size_t i  = 0;
            while (i + 1 < n->ch.size()) {
                if (n->ch[i] == nullptr) { /* else */
                    if (!done) jinja_exec(n->ch[i + 1].get(), sc, out);
                    done = true;
                } else {
                    Value cond = jinja_eval(n->ch[i].get(), sc);
                    if (cond.truthy() && !done) {
                        jinja_exec(n->ch[i + 1].get(), sc, out);
                        done = true;
                    }
                }
                i += 2;
            }
            break;
        }

        case NT::Set: {
            Value v    = jinja_eval(n->ch[0].get(), sc);
            size_t dot = n->sval.find('.');
            if (dot == std::string::npos) {
                sc.set(n->sval, std::move(v));
            } else {
                /* `base.attr1.attr2 = v` — mutate the namespace object in
                 * place via get_ref() rather than sc.set(base, ...), which
                 * would silently replace the whole object with `v`. If the
                 * base var doesn't exist, no-op (matches this engine's
                 * undefined-tolerant behavior elsewhere). */
                Value *target = sc.get_ref(n->sval.substr(0, dot));
                size_t start  = dot + 1;
                while (target) {
                    size_t next_dot = n->sval.find('.', start);
                    std::string attr =
                        n->sval.substr(start, next_dot == std::string::npos ? std::string::npos
                                                                            : next_dot - start);
                    if (next_dot == std::string::npos) {
                        target->obj[attr] = std::move(v);
                        break;
                    }
                    target = &target->obj[attr];
                    start  = next_dot + 1;
                }
            }
            break;
        }

        case NT::MacroDef:
            sc.def_macro(n->sval, n);
            break;

        default:
            break;
    }
}
