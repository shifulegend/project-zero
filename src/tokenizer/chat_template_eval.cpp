/*
 * src/tokenizer/chat_template_eval.cpp
 *
 * Runtime: evaluates and renders the AST. Split out of chat_template.cpp; the code is unchanged.
 * One function per node type; the filters, string methods and builtin functions live in
 * chat_template_builtins.cpp.
 */

#include "chat_template_internal.hpp"

static Value eval(const Node *n, Scope &sc);

static Value eval_getitem(const Node *n, Scope &sc) {
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

/* Python slice `obj[start:stop:step]`, any part optional. Arrays only (the one real-world use
 * seen, Qwen 3.6's `messages[::-1]`, is array reversal) — mirrors CPython's clamping /
 * negative-index rules (PySlice_GetIndicesEx). */
static Value eval_getslice(const Node *n, Scope &sc) {
    /* Python slice `obj[start:stop:step]`, any part optional.
     * Arrays only (the one real-world use seen, Qwen 3.6's
     * `messages[::-1]`, is array reversal) — mirrors CPython's
     * clamping/negative-index rules (PySlice_GetIndicesEx). */
    Value obj = eval(n->ch[0].get(), sc);
    if (obj.type != VType::Array) return Value();
    long long len  = (long long)obj.arr.size();
    long long step = n->ch[3] ? eval(n->ch[3].get(), sc).to_int() : 1;
    if (step == 0) step = 1;
    long long start, stop;
    if (step > 0) {
        start = n->ch[1] ? eval(n->ch[1].get(), sc).to_int() : 0;
        stop  = n->ch[2] ? eval(n->ch[2].get(), sc).to_int() : len;
        start = (start < 0) ? std::max<long long>(0, start + len) : std::min(start, len);
        stop  = (stop < 0) ? std::max<long long>(0, stop + len) : std::min(stop, len);
    } else {
        start = n->ch[1] ? eval(n->ch[1].get(), sc).to_int() : len - 1;
        stop  = n->ch[2] ? eval(n->ch[2].get(), sc).to_int() : -1; /* -1 = "before index 0" */
        start = (start < 0) ? std::max<long long>(-1, start + len) : std::min(start, len - 1);
        if (n->ch[2])
            stop = (stop < 0) ? std::max<long long>(-1, stop + len) : std::min(stop, len - 1);
    }
    Value out;
    out.type = VType::Array;
    for (long long i = start; step > 0 ? (i < stop) : (i > stop); i += step)
        out.arr.push_back(obj.arr[(size_t)i]);
    return out;
}

static Value eval_getattr(const Node *n, Scope &sc) {
    Value obj = eval(n->ch[0].get(), sc);
    if (obj.type == VType::Object) {
        auto it = obj.obj.find(n->sval);
        if (it != obj.obj.end()) return it->second;
    }
    if (obj.type == VType::Array) {
        if (n->sval == "length" || n->sval == "size") return Value((long long)obj.arr.size());
    }
    return Value();
}

/* Arithmetic, comparison and membership on two evaluated operands. */
static Value binop_apply(const std::string &op, Value &l, Value &r) {

    if (op == "+" || op == "~") return Value(l.to_str() + r.to_str());
    if (op == "-") {
        if (l.type == VType::Int && r.type == VType::Int) return Value(l.i - r.i);
        return Value();
    }
    if (op == "*") {
        if (l.type == VType::Int && r.type == VType::Int) return Value(l.i * r.i);
        return Value();
    }
    if (op == "/") {
        if (l.type == VType::Int && r.type == VType::Int && r.i) return Value(l.i / r.i);
        return Value();
    }
    if (op == "%") {
        if (l.type == VType::Int && r.type == VType::Int && r.i) return Value(l.i % r.i);
        return Value();
    }
    if (op == "==") return Value(l.equals(r));
    if (op == "!=") return Value(!l.equals(r));
    if (op == "<") return Value(l.cmp(r) < 0);
    if (op == "<=") return Value(l.cmp(r) <= 0);
    if (op == ">") return Value(l.cmp(r) > 0);
    if (op == ">=") return Value(l.cmp(r) >= 0);
    if (op == "in") {
        std::string ls = l.to_str();
        if (r.type == VType::Str) return Value(r.s.find(ls) != std::string::npos);
        if (r.type == VType::Array) {
            for (auto &x : r.arr)
                if (x.equals(l)) return Value(true);
            return Value(false);
        }
        if (r.type == VType::Object) return Value(r.obj.count(ls) > 0);
        return Value(false);
    }
    if (op == "not_in") {
        std::string ls = l.to_str();
        bool found     = false;
        if (r.type == VType::Str) found = r.s.find(ls) != std::string::npos;
        if (r.type == VType::Array) {
            for (auto &x : r.arr)
                if (x.equals(l)) {
                    found = true;
                    break;
                }
        }
        return Value(!found);
    }
    return Value();
}

static Value eval_binop(const Node *n, Scope &sc) {
    const std::string &op = n->sval;
    /* Short-circuit boolean ops */
    if (op == "and") {
        Value l = eval(n->ch[0].get(), sc);
        return l.truthy() ? eval(n->ch[1].get(), sc) : l;
    }
    if (op == "or") {
        Value l = eval(n->ch[0].get(), sc);
        return l.truthy() ? l : eval(n->ch[1].get(), sc);
    }

    Value l = eval(n->ch[0].get(), sc);
    Value r = eval(n->ch[1].get(), sc);
    return binop_apply(op, l, r);
}

static Value eval_unop(const Node *n, Scope &sc) {
    Value v = eval(n->ch[0].get(), sc);
    if (n->sval == "not") return Value(!v.truthy());
    if (n->sval == "-" && v.type == VType::Int) return Value(-v.i);
    return Value();
}

/* `x is defined / none / string / ...`, optionally negated (`is not`). */
static Value eval_isdef(const Node *n, Scope &sc) {
    bool result             = false;
    const std::string &test = n->sval;
    if (test == "defined") {
        if (!n->ch.empty() && n->ch[0]->type == NT::Ident)
            result = sc.has(n->ch[0]->sval);
        else if (!n->ch.empty())
            result = (eval(n->ch[0].get(), sc).type != VType::Undef);
    } else if (test == "none" || test == "None") {
        result = !n->ch.empty() && eval(n->ch[0].get(), sc).type == VType::None;
    } else if (test == "string") {
        result = !n->ch.empty() && eval(n->ch[0].get(), sc).type == VType::Str;
    } else if (test == "integer" || test == "number") {
        VType t = n->ch.empty() ? VType::Undef : eval(n->ch[0].get(), sc).type;
        result  = (t == VType::Int || t == VType::Float);
    } else if (test == "iterable") {
        VType t = n->ch.empty() ? VType::Undef : eval(n->ch[0].get(), sc).type;
        result  = (t == VType::Array || t == VType::Str);
    } else if (test == "mapping") {
        result = !n->ch.empty() && eval(n->ch[0].get(), sc).type == VType::Object;
    } else if (test == "odd") {
        result = !n->ch.empty() && (eval(n->ch[0].get(), sc).to_int() % 2 != 0);
    } else if (test == "even") {
        result = !n->ch.empty() && (eval(n->ch[0].get(), sc).to_int() % 2 == 0);
    }
    return Value(n->bval ? !result : result);
}

/* Macros execute in their *definition* scope (root), not the call site's — matches real Jinja and
 * keeps this simple (no closures to capture). ch[0]=body, ch[1..]=Param descriptors. Bind Kwarg
 * call args to their named param first, then fill remaining params positionally in declaration
 * order from the leftover (non-Kwarg) call args, then defaults for anything still unbound. */
static Value eval_macro_call(const Node *n, Scope &sc, const Node *def) {
    /* Macros execute in their *definition* scope (root), not
     * the call site's — matches real Jinja and keeps this
     * simple (no closures to capture). ch[0]=body,
     * ch[1..]=Param descriptors. Bind Kwarg call args to their
     * named param first, then fill remaining params
     * positionally in declaration order from the leftover
     * (non-Kwarg) call args, then defaults for anything still
     * unbound. */
    Scope msc(&sc.root());
    std::vector<std::string> bound_names;
    std::vector<size_t> positional_arg_idx;
    for (size_t i = 1; i < n->ch.size(); i++) {
        if (n->ch[i]->type == NT::Kwarg) {
            msc.set(n->ch[i]->sval, eval(n->ch[i]->ch[0].get(), sc));
            bound_names.push_back(n->ch[i]->sval);
        } else {
            positional_arg_idx.push_back(i);
        }
    }
    size_t pos_i = 0;
    for (size_t i = 1; i < def->ch.size(); i++) {
        const Node *param = def->ch[i].get();
        if (std::find(bound_names.begin(), bound_names.end(), param->sval) != bound_names.end())
            continue;
        Value pv = (pos_i < positional_arg_idx.size())
                       ? eval(n->ch[positional_arg_idx[pos_i++]].get(), sc)
                       : (!param->ch.empty() ? eval(param->ch[0].get(), msc) : Value());
        msc.set(param->sval, std::move(pv));
    }
    std::string macro_out;
    jinja_exec(def->ch[0].get(), msc, macro_out);
    return Value(std::move(macro_out));
}

static Value eval_call(const Node *n, Scope &sc) {
    /* ch[0] = function expr, ch[1..] = args */
    if (!n->ch.empty() && n->ch[0]->type == NT::Getattr) return jinja_string_method(n, sc);

    std::string fname;
    if (!n->ch.empty() && n->ch[0]->type == NT::Ident) fname = n->ch[0]->sval;
    if (fname == "raise_exception") return Value(); /* no-op */
    if (fname == "namespace") return jinja_call_namespace(n, sc);
    if (fname == "range") return jinja_call_range(n, sc);
    if (fname == "dict") {
        Value o;
        o.type = VType::Object;
        return o;
    }
    if (fname == "list") {
        Value a;
        a.type = VType::Array;
        return a;
    }
    if (const Node *def = sc.find_macro(fname)) return eval_macro_call(n, sc, def);
    return Value();
}

static Value eval(const Node *n, Scope &sc) {
    if (!n) return Value();
    switch (n->type) {
        case NT::LitStr:
            return Value(n->sval);
        case NT::LitInt:
            return Value(n->ival);
        case NT::LitBool:
            return Value(n->bval);
        case NT::LitNone:
            return Value(VType::None);
        case NT::Ident:
            return sc.get(n->sval);
        case NT::Getitem:
            return eval_getitem(n, sc);
        case NT::Getslice:
            return eval_getslice(n, sc);
        case NT::Getattr:
            return eval_getattr(n, sc);
        case NT::Binop:
            return eval_binop(n, sc);
        case NT::Unop:
            return eval_unop(n, sc);
        case NT::IsDef:
            return eval_isdef(n, sc);
        case NT::Filter:
            return jinja_filter(n, sc);
        case NT::Call:
            return eval_call(n, sc);
        case NT::Ternary: {
            Value cond = eval(n->ch[1].get(), sc);
            return cond.truthy() ? eval(n->ch[0].get(), sc) : eval(n->ch[2].get(), sc);
        }
        /* Transparent pass-through: only namespace() and macro-call argument binding (see
         * eval_call) actually inspect a Kwarg's name; every other positional consumer
         * (range()/dict()/list(), builtin filters' arg(i)) just wants the value. */
        case NT::Kwarg:
            return eval(n->ch[0].get(), sc);
        default:
            return Value();
    }
}

Value jinja_eval(const Node *n, Scope &sc) {
    return eval(n, sc);
}
