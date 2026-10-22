#include "transformer/generate.h"
#include "transformer/gen_session.h"
#include "transformer/forward.h"
#include <stdio.h>
#include <string.h>

/* ── stdout callback used by the public generate() wrapper ─────────────── */
static int stdout_token_callback(const char *piece, void *userdata) {
    (void)userdata;
    printf("%s", piece);
    fflush(stdout);
    return 0;
}

/* ── Core implementation: all output goes through callback ──────────────── */
/* 2026-10-22: the per-request state and logic live in transformer/gen_session
 * so the API server's parallel scheduler runs the same code; this is the
 * single-sequence driver. */
void generate_ex(const Config *cfg, const TransformerWeights *w,
                 RunState *s, const MoEConfig *mc,
                 Tokenizer *tok, ThreadPool *tp, const char *prompt,
                 const GenerationParams *gp,
                 TokenCallback callback, void *userdata) {
    GenSession g;
    if (gen_session_begin(&g, cfg, w, s, mc, tok, tp, prompt, gp, callback, userdata) == 0) {
        int token, pos;
        while (gen_session_input(&g, &token, &pos))
            gen_session_consume(&g, transformer_forward(token, pos, cfg, w, s, mc, tp));
    }
    gen_session_end(&g);
}

/* ── Legacy positional wrappers ─────────────────────────────────────────── */
static void legacy_params(GenerationParams *gp, int max_tokens, float temperature,
                          float top_p, bool json_mode) {
    gen_params_default(gp);
    gp->max_tokens = max_tokens;
    gp->sampler.temperature = temperature;
    gp->sampler.top_p = top_p;
    gp->json_mode = json_mode;
}

void generate_with_callback(const Config *cfg, const TransformerWeights *w,
                             RunState *s, const MoEConfig *mc,
                             Tokenizer *tok, ThreadPool *tp, const char *prompt,
                             int max_tokens, float temperature, float top_p,
                             bool json_mode,
                             TokenCallback callback, void *userdata) {
    GenerationParams gp;
    legacy_params(&gp, max_tokens, temperature, top_p, json_mode);
    generate_ex(cfg, w, s, mc, tok, tp, prompt, &gp, callback, userdata);
}

void generate_params(const Config *cfg, const TransformerWeights *w, RunState *s,
                     const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                     const char *prompt, const GenerationParams *gp) {
    generate_ex(cfg, w, s, mc, tok, tp, prompt, gp, stdout_token_callback, NULL);
    printf("\n");
}

/* ── Public wrapper: generate() prints to stdout ────────────────────────── */
void generate(const Config *cfg, const TransformerWeights *w, RunState *s,
              const MoEConfig *mc,
              Tokenizer *tok, ThreadPool *tp, const char *prompt,
              int max_tokens, float temperature, float top_p, bool json_mode) {
    GenerationParams gp;
    legacy_params(&gp, max_tokens, temperature, top_p, json_mode);
    generate_params(cfg, w, s, mc, tok, tp, prompt, &gp);
}
