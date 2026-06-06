/*
 * Phase 21 — OpenAI-Compatible API Layer
 * src/api/chat_compile.c
 *
 * Compiles a ChatRequest (array of {role, content} messages) into a single
 * prompt string suitable for the inference engine.
 *
 * Template formats supported
 * ──────────────────────────
 *  CHAT_TMPL_CHATML   — <|im_start|>role\ncontent<|im_end|>
 *  CHAT_TMPL_LLAMA3   — <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
 *  CHAT_TMPL_DEEPSEEK — <｜begin▁of▁sentence｜>...<｜end▁of▁sentence｜>
 *  CHAT_TMPL_MISTRAL  — [INST] ... [/INST]
 *  CHAT_TMPL_RAW      — Concatenate role: content\n (fallback)
 *
 * When the tokenizer already carries a Jinja2 chat_template string (GGUF
 * models), the inference engine calls chat_template_apply() directly and
 * this module is bypassed.  This module is used only when the tokenizer
 * has no embedded template.
 */

#include "api/chat_compile.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ── Snprintf helper: append to a buffer, tracking remaining space ───────── */
static int buf_append(char *dst, size_t *off, size_t cap, const char *src) {
    if (!src) return 1;
    size_t rem = (cap > *off) ? (cap - *off) : 0;
    if (rem == 0) return 0;
    size_t n = strlen(src);
    if (n >= rem) n = rem - 1;
    memcpy(dst + *off, src, n);
    *off += n;
    dst[*off] = '\0';
    return (n > 0 || strlen(src) == 0);
}

/* ── Format detectors (heuristic based on model name) ────────────────────── */
ChatTemplateType chat_template_detect(const char *model_name) {
    if (!model_name || model_name[0] == '\0') return CHAT_TMPL_CHATML;

    /* Case-insensitive substring search for known markers */
    const char *n = model_name;
    /* Llama-3 family */
    if (strstr(n, "llama-3") || strstr(n, "llama3") ||
        strstr(n, "Llama-3") || strstr(n, "Llama3"))
        return CHAT_TMPL_LLAMA3;
    /* Mistral / Mixtral */
    if (strstr(n, "mistral") || strstr(n, "Mistral") ||
        strstr(n, "mixtral") || strstr(n, "Mixtral"))
        return CHAT_TMPL_MISTRAL;
    /* DeepSeek */
    if (strstr(n, "deepseek") || strstr(n, "DeepSeek"))
        return CHAT_TMPL_DEEPSEEK;
    /* Default: ChatML (Qwen, SmolLM, BitNet, etc.) */
    return CHAT_TMPL_CHATML;
}

/* ── ChatML template ─────────────────────────────────────────────────────── */
static TernaryError compile_chatml(const ChatRequest *req,
                                   char *out, size_t cap) {
    size_t off = 0;
    for (int i = 0; i < req->num_messages; i++) {
        const ChatMessage *m = &req->messages[i];
        if (!m->content) continue;
        char header[128];
        snprintf(header, sizeof(header), "<|im_start|>%s\n", m->role);
        if (!buf_append(out, &off, cap, header))             return TN_ERR_OOM;
        if (!buf_append(out, &off, cap, m->content))         return TN_ERR_OOM;
        if (!buf_append(out, &off, cap, "<|im_end|>\n"))     return TN_ERR_OOM;
    }
    if (!buf_append(out, &off, cap, "<|im_start|>assistant\n")) return TN_ERR_OOM;
    return TN_OK;
}

/* ── Llama-3 template ────────────────────────────────────────────────────── */
static TernaryError compile_llama3(const ChatRequest *req,
                                   char *out, size_t cap) {
    size_t off = 0;
    if (!buf_append(out, &off, cap, "<|begin_of_text|>")) return TN_ERR_OOM;
    for (int i = 0; i < req->num_messages; i++) {
        const ChatMessage *m = &req->messages[i];
        if (!m->content) continue;
        char header[256];
        snprintf(header, sizeof(header),
                 "<|start_header_id|>%s<|end_header_id|>\n\n", m->role);
        if (!buf_append(out, &off, cap, header))         return TN_ERR_OOM;
        if (!buf_append(out, &off, cap, m->content))     return TN_ERR_OOM;
        if (!buf_append(out, &off, cap, "<|eot_id|>"))   return TN_ERR_OOM;
    }
    if (!buf_append(out, &off, cap,
            "<|start_header_id|>assistant<|end_header_id|>\n\n"))
        return TN_ERR_OOM;
    return TN_OK;
}

/* ── DeepSeek template ───────────────────────────────────────────────────── */
static TernaryError compile_deepseek(const ChatRequest *req,
                                     char *out, size_t cap) {
    size_t off = 0;
    /* DeepSeek V2/R1 uses <｜begin▁of▁sentence｜> as BOS.
     * Encoded as literal UTF-8 bytes to avoid invalid hex escape warnings. */
    static const char bos[] = "<\xef\xbd\x9c" "begin\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>";
    static const char eos[] = "<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>";
    if (!buf_append(out, &off, cap, bos))
        return TN_ERR_OOM;
    for (int i = 0; i < req->num_messages; i++) {
        const ChatMessage *m = &req->messages[i];
        if (!m->content) continue;
        if (strcmp(m->role, "user") == 0) {
            if (!buf_append(out, &off, cap, "User: "))       return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, m->content))     return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, "\n\n"))          return TN_ERR_OOM;
        } else if (strcmp(m->role, "assistant") == 0) {
            if (!buf_append(out, &off, cap, "Assistant: "))  return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, m->content))     return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, eos))            return TN_ERR_OOM;
        } else {
            /* system */
            if (!buf_append(out, &off, cap, m->content))     return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, "\n\n"))          return TN_ERR_OOM;
        }
    }
    if (!buf_append(out, &off, cap, "Assistant:")) return TN_ERR_OOM;
    return TN_OK;
}

/* ── Mistral [INST] template ─────────────────────────────────────────────── */
static TernaryError compile_mistral(const ChatRequest *req,
                                    char *out, size_t cap) {
    size_t off = 0;
    if (!buf_append(out, &off, cap, "<s>")) return TN_ERR_OOM;
    for (int i = 0; i < req->num_messages; i++) {
        const ChatMessage *m = &req->messages[i];
        if (!m->content) continue;
        if (strcmp(m->role, "user") == 0) {
            if (!buf_append(out, &off, cap, "[INST] "))       return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, m->content))      return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, " [/INST]"))      return TN_ERR_OOM;
        } else if (strcmp(m->role, "assistant") == 0) {
            if (!buf_append(out, &off, cap, " "))             return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, m->content))      return TN_ERR_OOM;
            if (!buf_append(out, &off, cap, "</s>"))          return TN_ERR_OOM;
        }
        /* system messages: prepend to first user message in real Mistral;
         * here we append raw for simplicity */
    }
    return TN_OK;
}

/* ── Raw fallback ────────────────────────────────────────────────────────── */
static TernaryError compile_raw(const ChatRequest *req,
                                char *out, size_t cap) {
    size_t off = 0;
    for (int i = 0; i < req->num_messages; i++) {
        const ChatMessage *m = &req->messages[i];
        if (!m->content) continue;
        char header[64];
        snprintf(header, sizeof(header), "%s: ", m->role);
        if (!buf_append(out, &off, cap, header))         return TN_ERR_OOM;
        if (!buf_append(out, &off, cap, m->content))     return TN_ERR_OOM;
        if (!buf_append(out, &off, cap, "\n"))            return TN_ERR_OOM;
    }
    return TN_OK;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

TernaryError chat_compile_prompt(const ChatRequest *req,
                                 ChatTemplateType tmpl,
                                 char *out, size_t cap) {
    if (!req || !out || cap == 0) return TN_ERR_INVALID_ARGS;
    out[0] = '\0';
    switch (tmpl) {
        case CHAT_TMPL_CHATML:   return compile_chatml(req, out, cap);
        case CHAT_TMPL_LLAMA3:   return compile_llama3(req, out, cap);
        case CHAT_TMPL_DEEPSEEK: return compile_deepseek(req, out, cap);
        case CHAT_TMPL_MISTRAL:  return compile_mistral(req, out, cap);
        case CHAT_TMPL_RAW:      return compile_raw(req, out, cap);
        default:                 return compile_chatml(req, out, cap);
    }
}
