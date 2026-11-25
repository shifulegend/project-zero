/*
 * src/tokenizer/chat_template_builtins.cpp
 *
 * Runtime: the filters (`x|trim`), Python string methods (`s.split(..)`) and builtin functions
 * (`namespace`, `range`) the evaluator calls. Split out of chat_template_eval.cpp; the code is
 * unchanged.
 */

#include "chat_template_internal.hpp"

/* The i-th argument after the callee of a Call / Filter node, or Undef. */
static Value call_arg(const Node *n, Scope &sc, int i) {
    return (i + 1 < (int)n->ch.size()) ? jinja_eval(n->ch[i + 1].get(), sc) : Value();
}

/* ── Filters ─────────────────────────────────────────────────────────────── */

/* Text filters: trim, upper, lower, replace, join, indent. */
static Value filter_text(const Node *n, Scope &sc, const std::string &fn, Value &v) {
    if (fn == "trim") {
        std::string s = v.to_str();
        size_t a      = s.find_first_not_of(" \t\r\n");
        size_t b      = s.find_last_not_of(" \t\r\n");
        return (a == std::string::npos) ? Value(std::string("")) : Value(s.substr(a, b - a + 1));
    }
    if (fn == "upper") {
        std::string s = v.to_str();
        for (char &c : s) c = (char)toupper((unsigned char)c);
        return Value(s);
    }
    if (fn == "lower") {
        std::string s = v.to_str();
        for (char &c : s) c = (char)tolower((unsigned char)c);
        return Value(s);
    }
    if (fn == "replace") {
        std::string s = v.to_str(), from = call_arg(n, sc, 0).to_str(),
                    to = call_arg(n, sc, 1).to_str();
        if (!from.empty()) {
            size_t pos = 0;
            while ((pos = s.find(from, pos)) != std::string::npos) {
                s.replace(pos, from.size(), to);
                pos += to.size();
            }
        }
        return Value(s);
    }
    if (fn == "join") {
        std::string sep = call_arg(n, sc, 0).to_str();
        if (v.type == VType::Array) {
            std::string r;
            for (size_t i = 0; i < v.arr.size(); i++) {
                if (i) r += sep;
                r += v.arr[i].to_str();
            }
            return Value(r);
        }
        return v;
    }
    if (fn == "indent") {
        /* indent(width, first=false) */
        int width = (int)call_arg(n, sc, 0).to_int();
        if (width <= 0) width = 4;
        std::string s = v.to_str(), pad(width, ' '), r;
        for (char c : s) {
            r += c;
            if (c == '\n') r += pad;
        }
        return Value(r);
    }
    return v;
}

/* Value filters: length, default, int, string, tojson / safe / escape, items. */
static Value filter_value(const Node *n, Scope &sc, const std::string &fn, Value &v) {
    if (fn == "length" || fn == "count") {
        if (v.type == VType::Array) return Value((long long)v.arr.size());
        if (v.type == VType::Object) return Value((long long)v.obj.size());
        return Value((long long)v.to_str().size());
    }
    if (fn == "default" || fn == "d") {
        bool use_default = (v.type == VType::Undef || v.type == VType::None);
        /* default(value, boolean) — second arg=true also triggers on falsy */
        if (!use_default && n->ch.size() > 2 && call_arg(n, sc, 1).truthy())
            use_default = !v.truthy();
        return use_default ? call_arg(n, sc, 0) : v;
    }
    if (fn == "int") return Value(v.to_int());
    if (fn == "string") return Value(v.to_str());
    if (fn == "tojson" || fn == "safe" || fn == "e" || fn == "escape") return v;
    if (fn == "items") {
        /* dict.items() -> array of [key, value] pairs, consumed by
         * `{% for k, v in x|items %}` (tuple-unpacking for, see
         * NT::For's exec() case) — real chat templates use exactly
         * this for tool-call argument serialization. Object keys
         * iterate in std::map's sorted order (this engine's Value
         * has no ordered-map type anywhere), matching this file's
         * existing general object-iteration behavior elsewhere. */
        Value out_arr;
        out_arr.type = VType::Array;
        if (v.type == VType::Object) {
            for (auto &kv : v.obj) {
                Value pair;
                pair.type = VType::Array;
                pair.arr.push_back(Value(kv.first));
                pair.arr.push_back(kv.second);
                out_arr.arr.push_back(std::move(pair));
            }
        }
        return out_arr;
    }
    /* Unknown filter — pass through */
    return v;
}

Value jinja_filter(const Node *n, Scope &sc) {
    Value v               = jinja_eval(n->ch[0].get(), sc);
    const std::string &fn = n->sval;
    if (fn == "trim" || fn == "upper" || fn == "lower" || fn == "replace" || fn == "join" ||
        fn == "indent")
        return filter_text(n, sc, fn, v);
    return filter_value(n, sc, fn, v);
}

/* ── Python string methods ───────────────────────────────────────────────── */

/* `content.split(...)`, `.rstrip(...)`, `.startswith(...)` etc.: the callee is a Getattr rather
 * than a plain Ident. Qwen 3.6's chat_template.jinja uses these to split an assistant message's
 * `<think>...</think>` reasoning block out of its visible content on every multi-turn render
 * (real Jinja/Python str semantics, not a filter). */
Value jinja_string_method(const Node *n, Scope &sc) {
    const Node *ga            = n->ch[0].get();
    Value recv                = jinja_eval(ga->ch[0].get(), sc);
    const std::string &method = ga->sval;
    std::string s             = recv.to_str();
    if (method == "split") {
        std::string sep = call_arg(n, sc, 0).to_str();
        Value out;
        out.type = VType::Array;
        if (sep.empty()) {
            out.arr.push_back(Value(s));
            return out;
        }
        size_t pos = 0;
        for (;;) {
            size_t next = s.find(sep, pos);
            if (next == std::string::npos) {
                out.arr.push_back(Value(s.substr(pos)));
                break;
            }
            out.arr.push_back(Value(s.substr(pos, next - pos)));
            pos = next + sep.size();
        }
        return out;
    }
    if (method == "strip" || method == "rstrip" || method == "lstrip") {
        std::string chars =
            (n->ch.size() > 1) ? call_arg(n, sc, 0).to_str() : std::string(" \t\r\n");
        if (method == "rstrip") {
            size_t b = s.find_last_not_of(chars);
            return Value(b == std::string::npos ? std::string("") : s.substr(0, b + 1));
        }
        if (method == "lstrip") {
            size_t a = s.find_first_not_of(chars);
            return Value(a == std::string::npos ? std::string("") : s.substr(a));
        }
        size_t a = s.find_first_not_of(chars), b = s.find_last_not_of(chars);
        return Value((a == std::string::npos) ? std::string("") : s.substr(a, b - a + 1));
    }
    if (method == "startswith") {
        std::string pre = call_arg(n, sc, 0).to_str();
        return Value(s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0);
    }
    if (method == "endswith") {
        std::string suf = call_arg(n, sc, 0).to_str();
        return Value(s.size() >= suf.size() &&
                     s.compare(s.size() - suf.size(), suf.size(), suf) == 0);
    }
    /* Unknown method — fall through to Undef, matching this
     * engine's general unknown-callee behavior. */
    return Value();
}

/* ── Builtin functions ───────────────────────────────────────────────────── */

/* namespace(k=v, ...) seeds the returned object from its keyword args (e.g. Qwen 3.6's
 * `namespace(multi_step_tool=true, last_query_index=...)`); positional args aren't valid
 * namespace() syntax in Jinja, so only Kwarg children are consulted. */
Value jinja_call_namespace(const Node *n, Scope &sc) {
    /* namespace(k=v, ...) seeds the returned object from its
     * keyword args (e.g. Qwen 3.6's
     * `namespace(multi_step_tool=true, last_query_index=...)`)
     * — positional args aren't valid namespace() syntax in
     * Jinja, so only Kwarg children are consulted. */
    Value o;
    o.type = VType::Object;
    for (size_t i = 1; i < n->ch.size(); i++)
        if (n->ch[i]->type == NT::Kwarg)
            o.obj[n->ch[i]->sval] = jinja_eval(n->ch[i]->ch[0].get(), sc);
    return o;
}

Value jinja_call_range(const Node *n, Scope &sc) {
    long long start = 0, end_v = 0, step = 1;
    size_t nargs = n->ch.size() - 1;
    if (nargs == 1)
        end_v = jinja_eval(n->ch[1].get(), sc).to_int();
    else if (nargs >= 2) {
        start = jinja_eval(n->ch[1].get(), sc).to_int();
        end_v = jinja_eval(n->ch[2].get(), sc).to_int();
    }
    if (nargs >= 3) step = jinja_eval(n->ch[3].get(), sc).to_int();
    if (step == 0) return Value();
    Value a;
    a.type = VType::Array;
    for (long long i = start; step > 0 ? (i < end_v) : (i > end_v); i += step)
        a.arr.push_back(Value(i));
    return a;
}
