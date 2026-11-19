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

/* Digest of the struct: pointer-looking words are replaced by the (bounded) string they point to. */
static uint64_t digest(const CliArgs *a) {
    uint64_t h             = 1469598103934665603ULL;
    const unsigned char *b = (const unsigned char *)a;
    size_t i               = 0;
    while (i < sizeof(*a)) {
        if (i % sizeof(uintptr_t) == 0 && i + sizeof(uintptr_t) <= sizeof(*a)) {
            uintptr_t v;
            memcpy(&v, b + i, sizeof v);
            if (v >= (uintptr_t)a &&
                v < (uintptr_t)a + sizeof(*a)) { /* points into the struct itself */
                uintptr_t off = v - (uintptr_t)a;
                h             = mix(h, "S", 1);
                h             = mix(h, &off, sizeof off);
                i += sizeof(uintptr_t);
                continue;
            }
            if (in_arena(v) || in_image(v)) {
                const char *s = (const char *)v;
                size_t n      = 0;
                while (n < 200 && s[n]) n++;
                h = mix(h, "P", 1);
                h = mix(h, s, n);
                i += sizeof(uintptr_t);
                continue;
            }
        }
        h = mix(h, b + i, 1);
        i++;
    }
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
            while ((p = strstr(p, "strcmp(argv[i], \"--")) != NULL) {
                p += strlen("strcmp(argv[i], \"");
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
