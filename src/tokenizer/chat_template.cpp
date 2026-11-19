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

#include "chat_template_internal.hpp"

/* ======================================================================== *
 *  SECTION 6 — PUBLIC C API
 * ======================================================================== */

extern "C" {

char *chat_template_apply(const char *tmpl_cstr, const char *const *roles,
                          const char *const *contents, int n_messages, const char *bos_token_str,
                          const char *eos_token_str, int add_generation_prompt) {
    if (!tmpl_cstr) return nullptr;

    try {
        auto toks = jinja_lex(std::string(tmpl_cstr));
        Parser p{toks, 0};
        auto ast = jinja_parse_block(p);

        Scope sc;
        sc.set("bos_token", Value(bos_token_str ? bos_token_str : ""));
        sc.set("eos_token", Value(eos_token_str ? eos_token_str : ""));
        sc.set("add_generation_prompt", Value(add_generation_prompt != 0));

        /* Build messages array */
        Value msgs;
        msgs.type = VType::Array;
        for (int i = 0; i < n_messages; i++) {
            Value msg;
            msg.type           = VType::Object;
            msg.obj["role"]    = Value(roles && roles[i] ? roles[i] : "");
            msg.obj["content"] = Value(contents && contents[i] ? contents[i] : "");
            msgs.arr.push_back(std::move(msg));
        }
        sc.set("messages", std::move(msgs));

        std::string out;
        jinja_exec(ast.get(), sc, out);

        char *result = (char *)malloc(out.size() + 1);
        if (result) memcpy(result, out.c_str(), out.size() + 1);
        return result;

    } catch (...) {
        return nullptr;
    }
}

} /* extern "C" */
