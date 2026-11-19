#ifndef TN_TOKENIZER_CHAT_TEMPLATE_INTERNAL_HPP
#define TN_TOKENIZER_CHAT_TEMPLATE_INTERNAL_HPP

/* Private to the chat-template engine: the Jinja value, token, AST and scope types shared by the
 * lexer (chat_template_lex.cpp), parser (chat_template_parse.cpp), evaluator
 * (chat_template_eval.cpp) and the C API (chat_template.cpp). Split out of chat_template.cpp; the
 * definitions are unchanged. */

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
    VType type  = VType::Undef;
    bool b      = false;
    long long i = 0;
    double f    = 0.0;
    std::string s;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    Value() = default;
    explicit Value(VType t) : type(t) {}
    explicit Value(bool b_) : type(VType::Bool), b(b_) {}
    explicit Value(long long i_) : type(VType::Int), i(i_) {}
    explicit Value(double f_) : type(VType::Float), f(f_) {}
    // The three string constructors are implicit on purpose: template values are built from literals.
    Value(const std::string &s_) : type(VType::Str), s(s_) {} // NOLINT(google-explicit-constructor)
    Value(std::string &&s_)
        : type(VType::Str), s(std::move(s_)) {} // NOLINT(google-explicit-constructor)
    Value(const char *s_)
        : type(VType::Str), s(s_ ? s_ : "") {} // NOLINT(google-explicit-constructor)

    bool truthy() const {
        switch (type) {
            case VType::Undef:
                return false;
            case VType::None:
                return false;
            case VType::Bool:
                return b;
            case VType::Int:
                return i != 0;
            case VType::Float:
                return f != 0.0;
            case VType::Str:
                return !s.empty();
            case VType::Array:
                return !arr.empty();
            case VType::Object:
                return !obj.empty();
        }
        return false;
    }

    std::string to_str() const {
        switch (type) {
            case VType::Undef:
                return "";
            case VType::None:
                return "";
            case VType::Bool:
                return b ? "True" : "False";
            case VType::Int:
                return std::to_string(i);
            case VType::Float: {
                char buf[64];
                snprintf(buf, sizeof(buf), "%g", f);
                return buf;
            }
            case VType::Str:
                return s;
            case VType::Array:
                return "";
            case VType::Object:
                return "";
        }
        return "";
    }

    long long to_int() const {
        switch (type) {
            case VType::Int:
                return i;
            case VType::Bool:
                return b ? 1 : 0;
            case VType::Float:
                return (long long)f;
            case VType::Str:
                try {
                    return std::stoll(s);
                } catch (...) {
                    return 0;
                }
            default:
                return 0;
        }
    }

    bool equals(const Value &o) const {
        if (type == VType::Str || o.type == VType::Str) return to_str() == o.to_str();
        if (type == VType::Bool && o.type == VType::Bool) return b == o.b;
        if (type == VType::Int && o.type == VType::Int) return i == o.i;
        if (type == VType::None && o.type == VType::None) return true;
        if (type == VType::Undef && o.type == VType::Undef) return true;
        return false;
    }

    int cmp(const Value &o) const {
        if (type == VType::Int && o.type == VType::Int) return (i < o.i) ? -1 : (i > o.i) ? 1 : 0;
        return to_str().compare(o.to_str());
    }
};

/* ======================================================================== *
 *  SECTION 2 — LEXER
 * ======================================================================== */

enum class TK {
    Eof,
    Text,
    Str,
    Int,
    Ident,
    EqEq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Plus,
    Tilde,
    Minus,
    Star,
    Slash,
    Percent,
    Pipe,
    Comma,
    Dot,
    Colon,
    Lbracket,
    Rbracket,
    Lparen,
    Rparen,
    Lbrace,
    Rbrace,
    StmtOpen,
    StmtClose,
    ExprOpen,
    ExprClose
};

struct Token {
    TK type = TK::Eof;
    std::string sval;
    long long ival = 0;
};

/* ======================================================================== *
 *  SECTION 3 — AST
 * ======================================================================== */

enum class NT {
    /* Statements */
    Block,
    Text,
    ExprOut,
    For,
    IfChain,
    Set,
    Noop,
    MacroDef,
    /* Expressions */
    LitStr,
    LitInt,
    LitBool,
    LitNone,
    Ident,
    Getitem,
    Getslice,
    Getattr,
    Binop,
    Unop,
    Call,
    IsDef,
    Filter,
    Ternary,
    Kwarg,
    /* Macro-parameter descriptor: sval=name, ch=[] (required) or
     * ch[0]=default-value expr. Only ever appears inside a MacroDef's
     * ch[1..]; never fed through eval() as a normal expression. */
    Param
};

struct Node {
    NT type = NT::Noop;
    std::string sval;       /* string literal / ident name / op / filter / var */
    long long ival = 0;     /* integer literal */
    bool bval      = false; /* bool literal; also "negated" flag in IsDef */
    std::vector<std::unique_ptr<Node>> ch;

    explicit Node(NT t) : type(t) {}
};

using NodePtr = std::unique_ptr<Node>;

/* ======================================================================== *
 *  SECTION 4 — PARSER
 * ======================================================================== */

struct Parser {
    const std::vector<Token> &toks;
    size_t pos = 0;

    const Token &peek(int off = 0) const {
        size_t i = pos + (size_t)off;
        return (i < toks.size()) ? toks[i] : toks.back();
    }
    Token &peek_mut(int off = 0) {
        size_t i = pos + (size_t)off;
        return (i < toks.size()) ? const_cast<Token &>(toks[i]) : const_cast<Token &>(toks.back());
    }
    const Token &next() {
        const Token &t = peek();
        if (pos < toks.size()) ++pos;
        return t;
    }
    bool is(TK t, int off = 0) const { return peek(off).type == t; }
    bool kw(const char *s, int off = 0) const { return is(TK::Ident, off) && peek(off).sval == s; }
    void skip_to_stmt_close() {
        while (!is(TK::StmtClose) && !is(TK::Eof)) next();
        if (is(TK::StmtClose)) next();
    }
    /* True if next tokens are {%  <block-terminator>  (without consuming) */
    bool at_block_end() const {
        if (!is(TK::StmtOpen)) return false;
        if (!is(TK::Ident, 1)) return false;
        const std::string &k = peek(1).sval;
        return k == "endif" || k == "endfor" || k == "else" || k == "elif" || k == "endblock" ||
               k == "endmacro" || k == "endcall" || k == "endfilter" || k == "endset";
    }
};

/* ======================================================================== *
 *  SECTION 5 — RUNTIME
 * ======================================================================== */

class Scope {
  public:
    explicit Scope(Scope *par = nullptr) : parent_(par) {}

    Value get(const std::string &name) const {
        auto it = vars_.find(name);
        if (it != vars_.end()) return it->second;
        if (parent_) return parent_->get(name);
        return Value(); /* Undef */
    }
    void set(const std::string &name, Value v) { vars_[name] = std::move(v); }
    bool has(const std::string &name) const {
        if (vars_.count(name)) return true;
        if (parent_) return parent_->has(name);
        return false;
    }
    /* Pointer to the live storage backing `name`, walking up the parent
     * chain (nullptr if unset anywhere). Needed for dotted-target
     * assignment (`set ns.attr = ...`) to mutate the namespace object in
     * place rather than through get()'s by-value copy. */
    Value *get_ref(const std::string &name) {
        auto it = vars_.find(name);
        if (it != vars_.end()) return &it->second;
        if (parent_) return parent_->get_ref(name);
        return nullptr;
    }
    Scope &root() { return parent_ ? parent_->root() : *this; }
    const Scope &root() const { return parent_ ? parent_->root() : *this; }

    /* Macros are registered (and looked up) at the root scope: real Jinja
     * macros execute in their definition scope, not the call site's, and
     * every macro this engine has seen is defined once at template
     * top-level and called from anywhere — root-scoped storage matches
     * that without threading a separate registry through every eval/exec
     * call site. */
    void def_macro(const std::string &name, const Node *def) { root().macros_[name] = def; }
    const Node *find_macro(const std::string &name) const {
        auto it = root().macros_.find(name);
        return it != root().macros_.end() ? it->second : nullptr;
    }

  private:
    std::map<std::string, Value> vars_;
    std::map<std::string, const Node *> macros_;
    Scope *parent_ = nullptr;
};


/* Entry points of the three stages (each defined in its own file). */
std::vector<Token> jinja_lex(const std::string &tmpl);
NodePtr jinja_parse_block(Parser &p);
NodePtr jinja_parse_expr(Parser &p);
Value jinja_eval(const Node *n, Scope &sc);
void jinja_exec(const Node *n, Scope &sc, std::string &out);

#endif
