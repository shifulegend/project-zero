/*
 * test_args_golden.c -- a golden table for parse_args(): for every flag it accepts, parse the flag
 * alone and with a range of sample values, and compare the result code, everything printed and a
 * digest of the whole CliArgs struct with tests/fixtures/args_golden.txt. A refactor of the flag
 * parser must leave the table untouched; a deliberate behaviour change regenerates it with
 *     ARGS_GOLDEN_UPDATE=1 build/tests/test_args_golden
 * The digest covers every byte of the struct. Words that point into the argument strings or into
 * the program image are replaced by the text they point to, and pointers into the struct itself by
 * their offset, so it does not depend on addresses.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cli/args.h"

extern char __executable_start;
extern char _end;

#define ARENA_BYTES 4096
#define MAX_CASES 6000
#define GOLDEN "tests/fixtures/args_golden.txt"

static char *g_arena;
static size_t g_arena_used;

static uint64_t mix(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int in_arena(uintptr_t v) {
    return v >= (uintptr_t)g_arena && v < (uintptr_t)g_arena + ARENA_BYTES;
}

static int in_image(uintptr_t v) {
    return v >= (uintptr_t)&__executable_start && v < (uintptr_t)&_end;
}

#define ARGS_FIELDS(X)                                                                             \
    X(eval, act_quant)                                                                             \
    X(server, api_key)                                                                             \
    X(perf, attn_kernel)                                                                           \
    X(perf, attn_parallel_min_ctx)                                                                 \
    X(perf, attn_tile)                                                                             \
    X(sampling, best_of)                                                                           \
    X(sampling, best_of_max)                                                                       \
    X(perf, calibrate)                                                                             \
    X(sampling, chat_format)                                                                       \
    X(input, chat_template_file)                                                                   \
    X(update, check_update)                                                                        \
    X(perf, classifier_override)                                                                   \
    X(server, client_timeout_s)                                                                    \
    X(output, color_mode)                                                                          \
    X(server, cors_enabled)                                                                        \
    X(server, cors_origins)                                                                        \
    X(perf, ctx_size)                                                                              \
    X(input, draft_model_path)                                                                     \
    X(eval, dump_tensors_path)                                                                     \
    X(search, enable_reasoning)                                                                    \
    X(sampling, frequency_penalty)                                                                 \
    X(sampling, ignore_eos)                                                                        \
    X(input, image_path)                                                                           \
    X(sampling, json_mode)                                                                         \
    X(perf, kv_offload_dir)                                                                        \
    X(perf, kv_ram_fraction)                                                                       \
    X(perf, kv_type)                                                                               \
    X(output, log_json)                                                                            \
    X(output, log_level)                                                                           \
    X(input, lora_path)                                                                            \
    X(server, max_connections)                                                                     \
    X(server, max_prompt_bytes)                                                                    \
    X(sampling, max_tokens)                                                                        \
    X(sampling, max_tokens_limit)                                                                  \
    X(search, mcts)                                                                                \
    X(search, mcts_delim_buf)                                                                      \
    X(search, mcts_max)                                                                            \
    X(input, memory_db_path)                                                                       \
    X(server, metrics_enabled)                                                                     \
    X(sampling, min_p)                                                                             \
    X(input, model_path)                                                                           \
    X(sampling, n_stop)                                                                            \
    X(perf, no_calibrate)                                                                          \
    X(server, no_config_write)                                                                     \
    X(server, no_request_template)                                                                 \
    X(update, no_update_check)                                                                     \
    X(server, num_cors_origins)                                                                    \
    X(perf, num_threads)                                                                           \
    X(server, parallel)                                                                            \
    X(server, parallel_batch)                                                                      \
    X(eval, perplexity_file)                                                                       \
    X(dist, plan_layers)                                                                           \
    X(eval, ppl_chunks)                                                                            \
    X(eval, ppl_ctx)                                                                               \
    X(sampling, presence_penalty)                                                                  \
    X(search, prm_bad)                                                                             \
    X(search, prm_format)                                                                          \
    X(search, prm_good)                                                                            \
    X(search, prm_path)                                                                            \
    X(input, proj_path)                                                                            \
    X(input, prompt)                                                                               \
    X(perf, prompt_cache)                                                                          \
    X(perf, prompt_cache_mb)                                                                       \
    X(perf, prompt_cache_min_tokens)                                                               \
    X(eval, quality_tolerance)                                                                     \
    X(server, queue_size)                                                                          \
    X(server, queue_timeout_s)                                                                     \
    X(server, rate_limit)                                                                          \
    X(sampling, raw_prompt)                                                                        \
    X(dist, remote_layers)                                                                         \
    X(sampling, repeat_last_n)                                                                     \
    X(sampling, repeat_penalty)                                                                    \
    X(perf, rope_layout)                                                                           \
    X(dist, rpc_bind_all)                                                                          \
    X(dist, rpc_worker_port)                                                                       \
    X(sampling, seed)                                                                              \
    X(eval, self_check)                                                                            \
    X(eval, self_check_tokens)                                                                     \
    X(server, server_mode)                                                                         \
    X(server, server_port)                                                                         \
    X(output, show_version)                                                                        \
    X(perf, simd_override)                                                                         \
    X(search, spec_length)                                                                         \
    X(server, static_dir)                                                                          \
    X(sampling, stop)                                                                              \
    X(input, system_prompt)                                                                        \
    X(sampling, temperature)                                                                       \
    X(input, tokenizer_path)                                                                       \
    X(sampling, top_k)                                                                             \
    X(sampling, top_p)                                                                             \
    X(server, ui_logo)                                                                             \
    X(server, ui_title)                                                                            \
    X(update, update_api)                                                                          \
    X(update, update_channel)                                                                      \
    X(update, update_interval_h)                                                                   \
    X(update, update_repo)                                                                         \
    X(output, verbose)                                                                             \
    X(input, vision_path)                                                                          \
    X(watermark, watermark_delta)                                                                  \
    X(watermark, watermark_detect)                                                                 \
    X(watermark, watermark_gamma)                                                                  \
    X(watermark, watermark_key)                                                                    \
    X(server, web_ui_mode)                                                                         \
    X(dist, worker_layers)

/* The digest walks the fields in name order, so it does not depend on how CliArgs is laid out (the
 * struct is grouped by topic; ARGS_FLAT_LAYOUT builds this test against the old flat struct, which
 * is how the grouping was proven equal). Pointer-looking words are replaced by the (bounded) string
 * they point to; the one pointer into the struct itself (the MCTS delimiter buffer) by a marker. */
#ifdef ARGS_FLAT_LAYOUT
#define FIELD(a, g, f) ((a)->f)
#define WM_ENABLED(a) ((a)->watermark)
#else
#define FIELD(a, g, f) ((a)->g.f)
#define WM_ENABLED(a) ((a)->watermark.enabled) /* the one field renamed by the grouping */
#endif

static uint64_t digest_bytes(uint64_t h, const void *p, size_t n, uintptr_t self) {
    const unsigned char *b = (const unsigned char *)p;
    size_t i               = 0;
    while (i < n) {
        if (n - i >= sizeof(uintptr_t) && (uintptr_t)(b + i) % sizeof(uintptr_t) == 0) {
            uintptr_t v;
            memcpy(&v, b + i, sizeof v);
            if (v == self) {
                h = mix(h, "S", 1);
                i += sizeof(uintptr_t);
                continue;
            }
            if (in_arena(v) || in_image(v)) {
                const char *s = (const char *)v;
                size_t len    = 0;
                while (len < 200 && s[len]) len++;
                h = mix(h, "P", 1);
                h = mix(h, s, len);
                i += sizeof(uintptr_t);
                continue;
            }
        }
        h = mix(h, b + i, 1);
        i++;
    }
    return h;
}

static uint64_t digest(const CliArgs *a) {
    uint64_t h     = 1469598103934665603ULL;
    uintptr_t self = (uintptr_t)FIELD(a, search, mcts_delim_buf);
#define DIGEST_FIELD(g, f)                                                                         \
    h = digest_bytes(mix(h, #f, sizeof #f), &FIELD(a, g, f), sizeof FIELD(a, g, f), self);
    ARGS_FIELDS(DIGEST_FIELD)
#undef DIGEST_FIELD
    h = digest_bytes(mix(h, "watermark", 9), &WM_ENABLED(a), sizeof WM_ENABLED(a), self);
    return h;
}

static char *arena_str(const char *s) {
    size_t n = strlen(s) + 1;
    if (g_arena_used + n > ARENA_BYTES) abort();
    char *p = g_arena + g_arena_used;
    memcpy(p, s, n);
    g_arena_used += n;
    return p;
}

/* Runs parse_args on argv, capturing stdout and stderr; returns "rc digest outputhash". */
static void run_case(char **words, int n, char *out, size_t cap) {
    char *argv[12];
    g_arena_used = 0;
    argv[0]      = arena_str("prog");
    for (int i = 0; i < n; i++) argv[i + 1] = arena_str(words[i]);
    CliArgs a;
    memset(&a, 0, sizeof a);

    fflush(stdout);
    fflush(stderr);
    int so = dup(1), se = dup(2);
    FILE *tmp = tmpfile();
    dup2(fileno(tmp), 1);
    dup2(fileno(tmp), 2);
    TernaryError rc = parse_args(&a, n + 1, argv);
    fflush(stdout);
    fflush(stderr);
    dup2(so, 1);
    dup2(se, 2);
    close(so);
    close(se);
    uint64_t oh = 1469598103934665603ULL;
    rewind(tmp);
    char buf[512];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, tmp)) > 0) oh = mix(oh, buf, r);
    fclose(tmp);
    snprintf(out, cap, "%d %016llx %016llx", (int)rc, (unsigned long long)digest(&a),
             (unsigned long long)oh);
}

/* Every long flag parse_args matches, read from the source files of the parser (args.c and the
 * flag groups) and sorted, so a new flag joins the table and a move between files changes nothing. */
static int cmp_flags(const void *a, const void *b) {
    return strcmp((const char *)a, (const char *)b);
}

static int read_flags(char flags[][40], int max) {
    static const char *files[] = {"src/cli/args.c",
                                  "src/cli/args_flags_generation.c",
                                  "src/cli/args_flags_memory.c",
                                  "src/cli/args_flags_serving.c",
                                  "src/cli/args_flags_limits.c",
                                  "src/cli/args_flags_search.c",
                                  "src/cli/args_flags_behavior.c",
                                  "src/cli/args_flags_extras.c"};
    int n                      = 0;
    int found                  = 0;
    for (size_t fi = 0; fi < sizeof files / sizeof files[0]; fi++) {
        FILE *f = fopen(files[fi], "r");
        if (!f) continue;
        found = 1;
        char line[512];
        while (fgets(line, sizeof line, f)) {
            const char *p = line;
            /* a flag is either a strcmp(argv[i], "--x") branch or a {"--x", ...} table entry */
            for (;;) {
                const char *s1 = strstr(p, "strcmp(argv[i], \"--");
                const char *s2 = strstr(p, "{\"--");
                if (!s1 && !s2) break;
                int table     = s1 == NULL || (s2 != NULL && s2 < s1);
                p             = table ? s2 + strlen("{\"") : s1 + strlen("strcmp(argv[i], \"");
                const char *e = strchr(p, '"');
                if (!e || e - p >= 40) break;
                char name[40];
                memcpy(name, p, (size_t)(e - p));
                name[e - p] = '\0';
                int dup     = 0;
                for (int i = 0; i < n; i++) dup |= strcmp(flags[i], name) == 0;
                if (!dup && n < max) strcpy(flags[n++], name);
                p = e;
            }
        }
        fclose(f);
    }
    if (!found) return -1;
    qsort(flags, (size_t)n, 40, cmp_flags);
    return n;
}

int main(void) {
    g_arena = (char *)malloc(ARENA_BYTES);
    static char flags[300][40];
    int nf = read_flags(flags, 300);
    if (nf <= 0) {
        printf("SKIP: run from the repository root (src/cli/args*.c not found)\n");
        return 0;
    }
    static const char *values[] = {
        "2",        "0",           "0.5", "-1",   "on",   "off",  "auto",   "avx2",
        "i8",       "i4",          "f32", "int8", "int4", "bf16", "stable", "prerelease",
        "file.txt", "99999999999", "abc", "1e3",  "x:1",  "8080"};
    enum { NV = sizeof values / sizeof values[0] };

    static char lines[MAX_CASES][200];
    int nl = 0;
    for (int i = 0; i < nf && nl < MAX_CASES - 2 * NV - 8; i++) {
        char *w1[1] = {flags[i]};
        char res[140];
        run_case(w1, 1, res, sizeof res);
        snprintf(lines[nl++], sizeof lines[0], "%s | %s", flags[i], res);
        for (int v = 0; v < NV; v++) {
            char *w2[2] = {flags[i], (char *)values[v]};
            run_case(w2, 2, res, sizeof res);
            snprintf(lines[nl++], sizeof lines[0], "%s %s | %s", flags[i], values[v], res);
            /* with a model, so a valid flag reaches the success path instead of "--model required" */
            char *w4[4] = {"--model", "m.gguf", flags[i], (char *)values[v]};
            run_case(w4, 4, res, sizeof res);
            snprintf(lines[nl++], sizeof lines[0], "M %s %s | %s", flags[i], values[v], res);
        }
    }
    /* a few multi-flag sequences: later flags override, errors stop the parse */
    char *seq1[6] = {"--model", "m.gguf", "--threads", "4", "--temperature", "0.3"};
    char *seq2[4] = {"--server", "--port", "9000", "--cors"};
    char *seq3[4] = {"--rate-limit", "abc", "--threads", "4"};
    char *seq4[5] = {"--top-k", "7", "--top-k", "9", "--seed"};
    char res[140];
    run_case(seq1, 6, res, sizeof res);
    snprintf(lines[nl++], sizeof lines[0], "SEQ1 | %s", res);
    run_case(seq2, 4, res, sizeof res);
    snprintf(lines[nl++], sizeof lines[0], "SEQ2 | %s", res);
    run_case(seq3, 4, res, sizeof res);
    snprintf(lines[nl++], sizeof lines[0], "SEQ3 | %s", res);
    run_case(seq4, 5, res, sizeof res);
    snprintf(lines[nl++], sizeof lines[0], "SEQ4 | %s", res);

    if (getenv("ARGS_GOLDEN_UPDATE")) {
        FILE *f = fopen(GOLDEN, "w");
        if (!f) {
            perror(GOLDEN);
            return 1;
        }
        for (int i = 0; i < nl; i++) fprintf(f, "%s\n", lines[i]);
        fclose(f);
        printf("wrote %d cases to %s\n", nl, GOLDEN);
        return 0;
    }
    FILE *f = fopen(GOLDEN, "r");
    if (!f) {
        printf("FAIL: %s is missing (generate it with ARGS_GOLDEN_UPDATE=1)\n", GOLDEN);
        return 1;
    }
    int bad = 0, seen = 0;
    char exp[200];
    while (fgets(exp, sizeof exp, f)) {
        exp[strcspn(exp, "\n")] = '\0';
        if (seen >= nl || strcmp(exp, lines[seen]) != 0) {
            if (bad < 5)
                printf("FAIL: case %d\n  want: %s\n  got:  %s\n", seen, exp,
                       seen < nl ? lines[seen] : "(none)");
            bad++;
        }
        seen++;
    }
    fclose(f);
    if (seen != nl) {
        printf("FAIL: the golden file has %d cases, the parser table %d\n", seen, nl);
        bad++;
    }
    printf("=== %s: %d cases ===\n", bad ? "FAILED" : "args golden table passed", nl);
    return bad ? 1 : 0;
}
