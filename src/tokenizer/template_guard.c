/*
 * template_guard.c — see include/tokenizer/template_guard.h.
 */
#include "tokenizer/template_guard.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "core/log.h"

/* Copy of docs/templates/bitnet-b1.58.jinja (the model card's
 * "User: ...<|eot_id|>Assistant: " format, with add_generation_prompt).
 * 2026-11-16: embedded so installed users, who do not have the docs/ tree,
 * get a working BitNet without a flag. tests/test_template_guard.c fails when
 * this text and the docs file differ. */
static const char BITNET_B158_TEMPLATE[] =
    "{%- for message in messages -%}\n"
    "{%- if loop.first -%}{{ bos_token }}{%- endif -%}\n"
    "{%- if message['role'] == 'system' -%}{{ 'System: ' + message['content'] | trim + "
    "'<|eot_id|>' }}\n"
    "{%- elif message['role'] == 'user' -%}{{ 'User: ' + message['content'] | trim + '<|eot_id|>' "
    "}}\n"
    "{%- elif message['role'] == 'assistant' -%}{{ 'Assistant: ' + message['content'] | trim + "
    "'<|eot_id|>' }}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}{{ 'Assistant: ' }}{%- endif -%}\n";

const char *tn_template_bitnet_b158(void) {
    return BITNET_B158_TEMPLATE;
}

const char *tn_template_known_fix(const char *arch, const char *gguf_template) {
    if (!arch || !gguf_template) return NULL;
    /* BitNet b1.58: the GGUF template writes eos_token right after its
     * "BITNETAssistant:" generation prefix (and never reads
     * add_generation_prompt). Both the marker and an eos_token reference after
     * it must be present, so a user's own or an already-fixed template is left
     * alone. */
    if (strncmp(arch, "bitnet", 6) == 0) {
        const char *marker = strstr(gguf_template, "BITNETAssistant:");
        if (marker && strstr(marker, "eos_token")) return BITNET_B158_TEMPLATE;
    }
    return NULL;
}

int tn_prompt_ends_with_token(const char *rendered, const char *token) {
    if (!rendered || !token || !token[0]) return 0;
    size_t n = strlen(rendered), k = strlen(token);
    while (n > 0 && (rendered[n - 1] == ' ' || rendered[n - 1] == '\n' || rendered[n - 1] == '\t' ||
                     rendered[n - 1] == '\r'))
        n--;
    return n >= k && memcmp(rendered + n - k, token, k) == 0;
}

void tn_template_warn_eos_ending_once(void) {
    static atomic_flag warned = ATOMIC_FLAG_INIT;
    if (atomic_flag_test_and_set(&warned)) return;
    TN_LOG_WRN("warn",
               "[warn] the chat template renders a prompt that ends with the end-of-sequence "
               "token, so the model will probably generate nothing. Check the template "
               "(--chat-template-file replaces it).\n");
}
