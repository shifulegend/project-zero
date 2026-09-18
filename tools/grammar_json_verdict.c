/*
 * grammar_json_verdict.c — batch verdict harness for TS-2.1 differential
 * fuzzing of the JSON grammar PDA (src/sampling/grammar_json.c) against a
 * reference parser (Python's json.loads, driven by tools/fuzz_grammar_json.py).
 *
 * Reads one base64-encoded candidate byte string per line from stdin (base64
 * so arbitrary bytes, including embedded NUL/newline, survive line-oriented
 * I/O intact). For each candidate, runs it through the real production
 * json_grammar_step()/json_grammar_finalize() and prints one tab-separated
 * result line to stdout:
 *
 *   <verdict>\t<invalid_offset>\t<original_base64>
 *
 * verdict is one of DONE (grammar-complete after finalize), INVALID
 * (json_grammar_step ever returned JSON_ST_INVALID), INCOMPLETE (neither).
 * invalid_offset is the 0-based byte index INVALID was first reached at, or
 * -1 if never.
 */
#include "sampling/grammar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -1 = not a valid base64 character; built once by b64_table_init() below
 * since a designated-initializer array literal can't set an aggregate
 * default other than 0, which would be indistinguishable from the valid
 * value 0 assigned to 'A'. */
static signed char b64_dec[256];

static void b64_table_init(void) {
    static const char *alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 256; i++) b64_dec[i] = -1;
    for (int i = 0; alphabet[i]; i++) b64_dec[(unsigned char)alphabet[i]] = (signed char)i;
}

/* Decodes base64 `in` (NUL-terminated, no embedded whitespace) into `out`
 * (caller-provided buffer of at least strlen(in) bytes). Returns decoded
 * length, or -1 on malformed input. */
static long b64_decode(const char *in, unsigned char *out, size_t out_cap) {
    size_t len = strlen(in);
    while (len > 0 && (in[len - 1] == '\n' || in[len - 1] == '\r')) len--;
    size_t pad = 0;
    if (len >= 1 && in[len - 1] == '=') pad++;
    if (len >= 2 && in[len - 2] == '=') pad++;
    long o = 0;
    unsigned int buf = 0;
    int bits = 0;
    for (size_t i = 0; i < len - pad; i++) {
        signed char v = b64_dec[(unsigned char)in[i]];
        if (v < 0) return -1;
        buf = (buf << 6) | (unsigned int)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if ((size_t)o >= out_cap) return -1;
            out[o++] = (unsigned char)((buf >> bits) & 0xFF);
        }
    }
    return o;
}

#define MAX_LINE 65536
#define MAX_DECODED 49152

int main(void) {
    b64_table_init();
    char line[MAX_LINE];
    unsigned char decoded[MAX_DECODED];

    while (fgets(line, sizeof(line), stdin)) {
        size_t linelen = strlen(line);
        if (linelen == 0) continue;
        /* strip trailing newline for the echoed base64 field */
        char b64_copy[MAX_LINE];
        memcpy(b64_copy, line, linelen + 1);
        while (linelen > 0 && (b64_copy[linelen - 1] == '\n' || b64_copy[linelen - 1] == '\r')) {
            b64_copy[--linelen] = '\0';
        }
        /* Note: an empty line IS a valid candidate (the empty byte string --
         * important for prefix-truncation-at-offset-0 tests) and must still
         * produce one output line to keep the caller's candidate<->result
         * pairing aligned; do not skip it. */

        long n = b64_decode(line, decoded, sizeof(decoded));
        if (n < 0) {
            printf("DECODE_ERROR\t-1\t%s\n", b64_copy);
            continue;
        }

        JsonGrammarState g;
        json_grammar_init(&g);
        long invalid_at = -1;
        for (long i = 0; i < n; i++) {
            JsonState st = json_grammar_step(&g, (char)decoded[i]);
            if (st == JSON_ST_INVALID && invalid_at < 0) {
                invalid_at = i;
                break; /* sink state: no point feeding more bytes */
            }
        }

        const char *verdict;
        if (invalid_at >= 0) {
            verdict = "INVALID";
        } else {
            json_grammar_finalize(&g);
            verdict = json_grammar_is_complete(&g) ? "DONE" : "INCOMPLETE";
        }
        printf("%s\t%ld\t%s\n", verdict, invalid_at, b64_copy);
    }
    return 0;
}
