/*
 * src/tokenizer/chat_template_eval.cpp
 *
 * Runtime: evaluates and renders the AST. Split out of chat_template.cpp; the code is unchanged.
 */

#include "chat_template_internal.hpp"

static Value eval(const Node *n, Scope &sc);


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

        case NT::Getslice: {
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
                stop =
                    n->ch[2] ? eval(n->ch[2].get(), sc).to_int() : -1; /* -1 = "before index 0" */
                start =
                    (start < 0) ? std::max<long long>(-1, start + len) : std::min(start, len - 1);
                if (n->ch[2])
                    stop =
                        (stop < 0) ? std::max<long long>(-1, stop + len) : std::min(stop, len - 1);
            }
            Value out;
            out.type = VType::Array;
            for (long long i = start; step > 0 ? (i < stop) : (i > stop); i += step)
                out.arr.push_back(obj.arr[(size_t)i]);
            return out;
        }

        case NT::Getattr: {
            Value obj = eval(n->ch[0].get(), sc);
            if (obj.type == VType::Object) {
                auto it = obj.obj.find(n->sval);
                if (it != obj.obj.end()) return it->second;
            }
            if (obj.type == VType::Array) {
                if (n->sval == "length" || n->sval == "size")
                    return Value((long long)obj.arr.size());
            }
            return Value();
        }

        case NT::Binop: {
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

        case NT::Unop: {
            Value v = eval(n->ch[0].get(), sc);
            if (n->sval == "not") return Value(!v.truthy());
            if (n->sval == "-" && v.type == VType::Int) return Value(-v.i);
            return Value();
        }

        case NT::IsDef: {
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

        case NT::Filter: {
            Value v               = eval(n->ch[0].get(), sc);
            const std::string &fn = n->sval;
            auto arg              = [&](int i) -> Value {
                return (i + 1 < (int)n->ch.size()) ? eval(n->ch[i + 1].get(), sc) : Value();
            };
            if (fn == "trim") {
                std::string s = v.to_str();
                size_t a      = s.find_first_not_of(" \t\r\n");
                size_t b      = s.find_last_not_of(" \t\r\n");
                return (a == std::string::npos) ? Value(std::string(""))
                                                : Value(s.substr(a, b - a + 1));
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
            if (fn == "length" || fn == "count") {
                if (v.type == VType::Array) return Value((long long)v.arr.size());
                if (v.type == VType::Object) return Value((long long)v.obj.size());
                return Value((long long)v.to_str().size());
            }
            if (fn == "default" || fn == "d") {
                bool use_default = (v.type == VType::Undef || v.type == VType::None);
                /* default(value, boolean) — second arg=true also triggers on falsy */
                if (!use_default && n->ch.size() > 2 && arg(1).truthy()) use_default = !v.truthy();
                return use_default ? arg(0) : v;
            }
            if (fn == "replace") {
                std::string s = v.to_str(), from = arg(0).to_str(), to = arg(1).to_str();
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
                std::string sep = arg(0).to_str();
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
            if (fn == "indent") {
                /* indent(width, first=false) */
                int width = (int)arg(0).to_int();
                if (width <= 0) width = 4;
                std::string s = v.to_str(), pad(width, ' '), r;
                for (char c : s) {
                    r += c;
                    if (c == '\n') r += pad;
                }
                return Value(r);
            }
            /* Unknown filter — pass through */
            return v;
        }

        case NT::Call: {
            /* ch[0] = function expr, ch[1..] = args */

            /* Python string method calls (`content.split(...)`,
             * `.rstrip(...)`, `.startswith(...)`, etc.) — the callee is a
             * Getattr rather than a plain Ident. Qwen 3.6's
             * chat_template.jinja uses these to split an assistant
             * message's `<think>...</think>` reasoning block out of its
             * visible content on every multi-turn render (real Jinja/
             * Python str semantics, not a filter). */
            if (!n->ch.empty() && n->ch[0]->type == NT::Getattr) {
                const Node *ga            = n->ch[0].get();
                Value recv                = eval(ga->ch[0].get(), sc);
                const std::string &method = ga->sval;
                std::string s             = recv.to_str();
                auto marg                 = [&](int i) -> Value {
                    return (i + 1 < (int)n->ch.size()) ? eval(n->ch[i + 1].get(), sc) : Value();
                };
                if (method == "split") {
                    std::string sep = marg(0).to_str();
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
                        (n->ch.size() > 1) ? marg(0).to_str() : std::string(" \t\r\n");
                    if (method == "rstrip") {
                        size_t b = s.find_last_not_of(chars);
                        return Value(b == std::string::npos ? std::string("") : s.substr(0, b + 1));
                    }
                    if (method == "lstrip") {
                        size_t a = s.find_first_not_of(chars);
                        return Value(a == std::string::npos ? std::string("") : s.substr(a));
                    }
                    size_t a = s.find_first_not_of(chars), b = s.find_last_not_of(chars);
                    return Value((a == std::string::npos) ? std::string("")
                                                          : s.substr(a, b - a + 1));
                }
                if (method == "startswith") {
                    std::string pre = marg(0).to_str();
                    return Value(s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0);
                }
                if (method == "endswith") {
                    std::string suf = marg(0).to_str();
                    return Value(s.size() >= suf.size() &&
                                 s.compare(s.size() - suf.size(), suf.size(), suf) == 0);
                }
                /* Unknown method — fall through to Undef, matching this
                 * engine's general unknown-callee behavior. */
                return Value();
            }

            std::string fname;
            if (!n->ch.empty() && n->ch[0]->type == NT::Ident) fname = n->ch[0]->sval;
            if (fname == "raise_exception") return Value(); /* no-op */
            if (fname == "namespace") {
                /* namespace(k=v, ...) seeds the returned object from its
                 * keyword args (e.g. Qwen 3.6's
                 * `namespace(multi_step_tool=true, last_query_index=...)`)
                 * — positional args aren't valid namespace() syntax in
                 * Jinja, so only Kwarg children are consulted. */
                Value o;
                o.type = VType::Object;
                for (size_t i = 1; i < n->ch.size(); i++)
                    if (n->ch[i]->type == NT::Kwarg)
                        o.obj[n->ch[i]->sval] = eval(n->ch[i]->ch[0].get(), sc);
                return o;
            }
            if (fname == "range") {
                long long start = 0, end_v = 0, step = 1;
                size_t nargs = n->ch.size() - 1;
                if (nargs == 1)
                    end_v = eval(n->ch[1].get(), sc).to_int();
                else if (nargs >= 2) {
                    start = eval(n->ch[1].get(), sc).to_int();
                    end_v = eval(n->ch[2].get(), sc).to_int();
                }
                if (nargs >= 3) step = eval(n->ch[3].get(), sc).to_int();
                if (step == 0) return Value();
                Value a;
                a.type = VType::Array;
                for (long long i = start; step > 0 ? (i < end_v) : (i > end_v); i += step)
                    a.arr.push_back(Value(i));
                return a;
            }
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

            if (const Node *def = sc.find_macro(fname)) {
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
                    if (std::find(bound_names.begin(), bound_names.end(), param->sval) !=
                        bound_names.end())
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
            return Value();
        }

        case NT::Ternary: {
            Value cond = eval(n->ch[1].get(), sc);
            return cond.truthy() ? eval(n->ch[0].get(), sc) : eval(n->ch[2].get(), sc);
        }

        /* Transparent pass-through: only namespace() and macro-call
         * argument binding (see NT::Call) actually inspect a Kwarg's name;
         * every other positional consumer (range()/dict()/list(), builtin
         * filters' arg(i)) just wants the value. */
        case NT::Kwarg:
            return eval(n->ch[0].get(), sc);

        default:
            return Value();
    }
}

Value jinja_eval(const Node *n, Scope &sc) {
    return eval(n, sc);
}
