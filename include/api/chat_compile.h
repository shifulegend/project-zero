#ifndef TN_CHAT_COMPILE_H
#define TN_CHAT_COMPILE_H

/*
 * Phase 21 — OpenAI-Compatible API Layer
 * include/api/chat_compile.h
 *
 * Compiles an array of chat messages into a single formatted prompt string.
 */

#include "api/chat_request.h"
#include "core/error.h"
#include <stddef.h>

/* Template format to use when the tokenizer has no embedded Jinja2 template */
typedef enum {
    CHAT_TMPL_CHATML   = 0,   /* <|im_start|>role\ncontent<|im_end|>           */
    CHAT_TMPL_LLAMA3   = 1,   /* <|start_header_id|>role<|end_header_id|>...   */
    CHAT_TMPL_DEEPSEEK = 2,   /* DeepSeek "User: / Assistant:" style           */
    CHAT_TMPL_MISTRAL  = 3,   /* [INST] ... [/INST]                            */
    CHAT_TMPL_RAW      = 4    /* role: content\n (plain fallback)               */
} ChatTemplateType;

/*
 * Heuristically detect the best template from a model name string.
 * Falls back to CHAT_TMPL_CHATML when unknown.
 */
ChatTemplateType chat_template_detect(const char *model_name);

/*
 * Compile all messages in *req into a single NUL-terminated prompt stored
 * in out[0..cap-1].  Truncates silently when cap is too small.
 *
 * Returns TN_OK on success, TN_ERR_OOM if the buffer is too small to fit
 * any output, TN_ERR_INVALID_ARGS on NULL arguments.
 */
TernaryError chat_compile_prompt(const ChatRequest *req,
                                 ChatTemplateType tmpl,
                                 char *out, size_t cap);

#endif /* TN_CHAT_COMPILE_H */
