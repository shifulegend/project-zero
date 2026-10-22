/*
 * fsm_real_vocab_check.c — TS-2.2: token-masking correctness against a
 * real, full-size tokenizer vocab (not the 9-token synthetic fixture in
 * tests/test_fsm.c). Loads the embedded tokenizer from a real GGUF model
 * file (argv[1]) and drives fsm_compute_token_mask()/fsm_advance() through
 * several realistic JSON documents, checking every case from the test plan:
 *
 *   - no-deadlock invariant: at every step, >=1 token is unmasked
 *   - EOS gating: masked while incomplete, unmasked exactly when
 *     json_grammar_is_complete() after trial-finalize
 *   - the real next token of a valid document is never masked
 *     (byte-level BPE / raw-byte <0xHH> tokens included, since these are
 *     real tokens from the real vocab, not synthesized)
 *   - fsm_compute_token_mask never mutates fsm->json (scratch copy) --
 *     asserted via byte-for-byte state comparison before/after
 *
 * Usage: tools/fsm_real_vocab_check <model.gguf>
 * Not part of `make test` (needs a real downloaded model) -- run manually.
 */
#include "sampling/fsm.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_gguf.h"
#include "core/gguf_reader.h"
#include "memory/mapped_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

static void check(int cond, const char *msg) {
    if (cond) { g_pass++; }
    else { g_fail++; printf("  [FAIL] %s\n", msg); }
}

/* Realistic JSON documents spanning object/array nesting, all number forms,
 * strings with escapes and unicode, and the three literals. */
static const char *DOCS[] = {
    "{}",
    "[]",
    "null",
    "true",
    "false",
    "0",
    "-0",
    "42",
    "-17.5",
    "6.02e23",
    "\"hello world\"",
    "\"line1\\nline2\\ttab\"",
    "\"unicode: \\u00e9\\u2705\"",
    "[1,2,3,4,5]",
    "{\"name\":\"Alice\",\"age\":30,\"active\":true}",
    "{\"a\":[1,2,{\"b\":\"nested\"}],\"c\":null,\"d\":-3.14e-2}",
    "{\"deep\":{\"deeper\":{\"deepest\":[1,[2,[3,[4,5]]]]}}}",
    "[{\"x\":1},{\"y\":2},{\"z\":3}]",
};
#define N_DOCS (sizeof(DOCS) / sizeof(DOCS[0]))

static void check_one_document(Tokenizer *t, const char *doc) {
    int tokens[512];
    int n = tokenizer_encode(t, doc, strlen(doc), tokens, 512);
    if (n <= 0) {
        printf("  [SKIP] tokenizer_encode failed for: %s\n", doc);
        return;
    }

    FSMState fsm;
    fsm_init(&fsm);
    int prev = -1;
    float *logits = malloc((size_t)t->vocab_size * sizeof(float));

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < t->vocab_size; k++) logits[k] = 0.0f;

        /* Non-destructiveness: snapshot fsm->json before, compare after. */
        JsonGrammarState before = fsm.json;
        fsm_compute_token_mask(&fsm, t, prev, logits, t->vocab_size);
        JsonGrammarState after = fsm.json;
        check(memcmp(&before, &after, sizeof(JsonGrammarState)) == 0,
              "fsm_compute_token_mask does not mutate fsm->json (scratch-copy invariant)");

        /* No-deadlock invariant: at least one unmasked token. */
        int any_unmasked = 0;
        for (int k = 0; k < t->vocab_size; k++) {
            if (logits[k] > -1e29f) { any_unmasked = 1; break; }
        }
        check(any_unmasked, "at least one token is left unmasked (no-deadlock invariant)");

        /* The real next token of this valid document must never be masked
         * -- if it were, real generation could never produce this document. */
        int next_tok = tokens[i];
        check(logits[next_tok] > -1e29f,
              "the real next token of a valid document is not masked");

        /* EOS gating: exactly the plan's stated invariant -- unmasked IFF
         * a trial-finalize of the CURRENT state (before feeding token i)
         * would already be complete. This is NOT "always masked until the
         * last token": a top-level number is closeable at every digit
         * boundary (e.g. mid-way through "-17.5", the "-17" fed so far is
         * itself already valid, complete JSON -- EOS legitimately becomes
         * unmasked there, matching RFC 8259, not a bug). Recompute the
         * expectation independently (trial-finalize a fresh scratch copy)
         * rather than assuming a fixed shape, so this doesn't just
         * re-encode the same assumption the implementation makes. */
        if (t->eos_token_id >= 0) {
            JsonGrammarState scratch = fsm.json;
            json_grammar_finalize(&scratch);
            int expected_complete = json_grammar_is_complete(&scratch);
            int eos_unmasked = logits[t->eos_token_id] > -1e29f;
            check((expected_complete && eos_unmasked) || (!expected_complete && !eos_unmasked),
                  "EOS unmasked iff a trial-finalize of the current state is already complete");
        }

        fsm_advance(&fsm, t, prev, next_tok);
        prev = next_tok;
    }

    /* After the whole document, the grammar must be finalizable to
     * complete, and EOS must now be unmasked. Numbers have no closing
     * delimiter of their own (see grammar.h) -- a bare top-level number
     * legitimately sits in a "may end here" state (JSON_ST_NUM_INT et al)
     * until json_grammar_finalize() is called, exactly once, when checking
     * EOS eligibility; fsm_compute_token_mask does this internally on a
     * scratch copy (already verified non-destructive above), so call it
     * here too before asserting is_complete on our own copy of the state --
     * checking is_complete() without a prior finalize (as this test
     * originally did) is not how any real caller uses it. */
    json_grammar_finalize(&fsm.json);
    check(json_grammar_is_complete(&fsm.json), "grammar is complete after the full document + finalize");
    for (int k = 0; k < t->vocab_size; k++) logits[k] = 0.0f;
    fsm_compute_token_mask(&fsm, t, prev, logits, t->vocab_size);
    if (t->eos_token_id >= 0) {
        check(logits[t->eos_token_id] > -1e29f, "EOS unmasked once the document is grammar-complete");
    }
    for (int e = 0; e < t->n_eos; e++) {
        check(logits[t->eos_list[e]] > -1e29f, "every eos_list entry unmasked once complete");
    }

    free(logits);
}

/* Multi-byte / byte-level BPE token sanity: scan the real vocab for pieces
 * that decode to raw high bytes or multi-byte UTF-8 (common in GPT-2-style
 * byte-level BPE vocabs) and confirm masking them never crashes and is
 * self-consistent (running the same input twice gives the same mask). */
static void check_multibyte_tokens_no_crash(Tokenizer *t) {
    FSMState fsm;
    fsm_init(&fsm);
    fsm_advance(&fsm, t, -1, 0); /* arbitrary token 0, just to get prev_token != -1 */

    float *logits1 = malloc((size_t)t->vocab_size * sizeof(float));
    float *logits2 = malloc((size_t)t->vocab_size * sizeof(float));
    for (int k = 0; k < t->vocab_size; k++) { logits1[k] = 0.0f; logits2[k] = 0.0f; }

    fsm_compute_token_mask(&fsm, t, 0, logits1, t->vocab_size);
    fsm_compute_token_mask(&fsm, t, 0, logits2, t->vocab_size);
    int consistent = 1;
    for (int k = 0; k < t->vocab_size; k++) {
        if ((logits1[k] < -1e29f) != (logits2[k] < -1e29f)) { consistent = 0; break; }
    }
    check(consistent, "masking the whole real vocab (incl. multi-byte/raw-byte tokens) twice gives the same result");
    free(logits1);
    free(logits2);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    MappedFile mf;
    if (mapped_file_open(&mf, argv[1]) != TN_OK) {
        fprintf(stderr, "failed to open %s\n", argv[1]);
        return 1;
    }
    GGUFHeader hdr;
    if (gguf_read_header(&hdr, mf.data, mf.size) != TN_OK) {
        fprintf(stderr, "failed to parse GGUF header\n");
        return 1;
    }
    Tokenizer t;
    memset(&t, 0, sizeof(t));
    if (tokenizer_load_from_gguf(&t, &hdr) != TN_OK) {
        fprintf(stderr, "failed to load tokenizer from GGUF\n");
        return 1;
    }

    printf("=== TS-2.2: real vocab (%d tokens) from %s ===\n", t.vocab_size, argv[1]);

    for (size_t d = 0; d < N_DOCS; d++) {
        check_one_document(&t, DOCS[d]);
    }
    check_multibyte_tokens_no_crash(&t);

    printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    tokenizer_free(&t);
    gguf_header_free(&hdr);
    return g_fail == 0 ? 0 : 1;
}
