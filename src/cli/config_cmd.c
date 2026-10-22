/*
 * cli/config_cmd.c — see cli/config_cmd.h.
 */
#include "cli/config_cmd.h"
#include "core/platform.h"

#include <stdlib.h>
#include <string.h>
#if TN_POSIX
#include <dirent.h>
#include <unistd.h>
#endif

static void warn_stderr(const char *m) { fprintf(stderr, "[config] %s\n", m); }

static int load_file(PzConfig *c, char *path, size_t pathn) {
    if (pz_config_init(c) != 0) { fprintf(stderr, "[config] out of memory\n"); return -1; }
    if (pz_config_path(path, pathn) != 0) {
        fprintf(stderr, "[config] no config location (set PZ_CONFIG or HOME)\n");
        return -1;
    }
    if (pz_config_load(c, path, warn_stderr) < 0) {
        fprintf(stderr, "[config] cannot read %s\n", path);
        return -1;
    }
    return 0;
}

static void print_value(const PzCfgEntry *e) {
    if (e->key->secret) { printf("****"); return; }
    for (int j = 0; j < e->n; j++) printf(j ? ", %s" : "%s", e->v[j]);
}

static void usage(void) {
    printf("Usage: pz config [list [--all] | get <key> | set <key> <value> | add <key> <value> |\n"
           "                  unset <key> | path]\n"
           "With no arguments, starts the interactive setup (needs a terminal).\n"
           "Precedence: command-line flag > PZ_<KEY> environment > config file > default.\n");
}

static int cmd_list(const PzConfig *c, int all) {
    int n = 0;
    const PzCfgKey *s = pz_config_schema(&n);
    const char *group = NULL;
    int shown = 0;
    for (int i = 0; i < n; i++) {
        const PzCfgEntry *e = pz_config_get(c, s[i].key);
        if (!all && !e) continue;
        if (!group || strcmp(group, s[i].group) != 0) { group = s[i].group; printf("[%s]\n", group); }
        printf("  %-18s = ", s[i].key);
        if (e) print_value(e); else printf("(default)");
        if (all) {
            printf("\n  %-18s   %s", "", s[i].help);
            if (s[i].choices) printf(" [%s]", s[i].choices);
        }
        printf("\n");
        shown++;
    }
    if (!shown) printf("No saved settings. Run `pz config` to set up, or `pz config list --all`.\n");
    return 0;
}

#if TN_POSIX
static int scan_models(const char *const *dirs, char found[][1024], int maxn) {
    int n = 0;
    for (int d = 0; dirs && dirs[d] && n < maxn; d++) {
        DIR *dp = opendir(dirs[d]);
        if (!dp) continue;
        struct dirent *de;
        while ((de = readdir(dp)) && n < maxn) {
            size_t l = strlen(de->d_name);
            if (l > 5 && strcmp(de->d_name + l - 5, ".gguf") == 0)
                snprintf(found[n++], 1024, "%s/%s", dirs[d], de->d_name);
        }
        closedir(dp);
    }
    return n;
}
#endif

/* Reads one line; returns 0 on EOF. Trailing newline stripped. */
static int read_line(FILE *in, char *buf, size_t n) {
    if (!fgets(buf, (int)n, in)) return 0;
    buf[strcspn(buf, "\r\n")] = '\0';
    return 1;
}

/* Asks for one setting; Enter keeps the current value, "-" clears it.
 * Re-asks until the value validates. Returns 0 on EOF. */
static int ask(PzConfig *c, FILE *in, FILE *out, const char *key, const char *prompt) {
    const PzCfgKey *k = pz_config_find_key(key);
    for (;;) {
        const PzCfgEntry *e = pz_config_get(c, key);
        fprintf(out, "%s", prompt);
        if (k->choices) fprintf(out, " [%s]", k->choices);
        fprintf(out, " (%s%s): ", e ? "current: " : "Enter for default", e ? (k->secret ? "****" : e->v[0]) : "");
        fflush(out);
        char line[1024];
        if (!read_line(in, line, sizeof(line))) return 0;
        if (line[0] == '\0') return 1;
        if (strcmp(line, "-") == 0) { pz_config_unset(c, key); return 1; }
        char err[256];
        if (pz_config_set(c, key, line, err, sizeof(err)) == 0) return 1;
        fprintf(out, "  %s\n", err);
    }
}

int pz_config_wizard(PzConfig *c, FILE *in, FILE *out, const char *const *model_dirs) {
    if (!c || !in || !out) return -1;
    fprintf(out, "Project Zero setup. Press Enter to keep a value, \"-\" to clear it.\n"
                 "Nothing here is required: hardware is detected automatically.\n\n");
    /* 1. model */
    char line[1024];
#if TN_POSIX
    static char found[32][1024];
    int nf = scan_models(model_dirs, found, 32);
    if (nf > 0) {
        fprintf(out, "Models found:\n");
        for (int i = 0; i < nf; i++) fprintf(out, "  %d) %s\n", i + 1, found[i]);
        fprintf(out, "Default model: number, path, or Enter to keep: ");
        fflush(out);
        if (!read_line(in, line, sizeof(line))) return 0;
        char *end = NULL;
        long pick = strtol(line, &end, 10);
        char err[256];
        if (line[0] && end && *end == '\0' && pick >= 1 && pick <= nf) pz_config_set(c, "model", found[pick - 1], err, sizeof(err));
        else if (strcmp(line, "-") == 0) pz_config_unset(c, "model");
        else if (line[0]) pz_config_set(c, "model", line, err, sizeof(err));
    } else
#else
    (void)model_dirs;
#endif
    if (!ask(c, in, out, "model", "Default model file (.gguf)")) return 0;
    /* 2. hardware and memory */
#if TN_POSIX && defined(_SC_NPROCESSORS_ONLN)
    fprintf(out, "This machine has %ld CPU threads; 0 lets Project Zero choose.\n", sysconf(_SC_NPROCESSORS_ONLN));
#endif
    if (!ask(c, in, out, "threads", "Threads")) return 0;
    if (!ask(c, in, out, "kv-type", "KV cache (i8/i4 use less memory for long contexts)")) return 0;
    /* 3. server and updates */
    if (!ask(c, in, out, "port", "Server port for `pz --server`")) return 0;
    if (!ask(c, in, out, "update-channel", "Update notices channel")) return 0;
    fprintf(out, "\nSave these settings? [Y/n]: ");
    fflush(out);
    if (!read_line(in, line, sizeof(line))) return 0;
    return (line[0] == '\0' || line[0] == 'y' || line[0] == 'Y') ? 1 : 0;
}

int pz_config_command(int argc, char **argv) {
    char path[1024];
    PzConfig c;
    const char *sub = argc > 1 ? argv[1] : NULL;
    if (sub && (strcmp(sub, "-h") == 0 || strcmp(sub, "--help") == 0 || strcmp(sub, "help") == 0)) {
        usage();
        return 0;
    }
    if (sub && strcmp(sub, "path") == 0) {
        if (pz_config_path(path, sizeof(path)) != 0) { fprintf(stderr, "[config] no config location\n"); return 1; }
        printf("%s\n", path);
        return 0;
    }
    if (load_file(&c, path, sizeof(path)) != 0) { pz_config_free(&c); return 1; }
    int rc = 0, dirty = 0;
    char err[256];
    if (!sub || strcmp(sub, "wizard") == 0) {
#if TN_POSIX
        if (!isatty(0)) { usage(); pz_config_free(&c); return 1; }
#endif
        char home_models[1024] = "", data_models[1024] = "";
        const char *h = getenv("HOME"), *xd = getenv("XDG_DATA_HOME");
        if (h) snprintf(home_models, sizeof(home_models), "%s/models", h);
        if (xd && *xd) snprintf(data_models, sizeof(data_models), "%s/project-zero/models", xd);
        else if (h) snprintf(data_models, sizeof(data_models), "%s/.local/share/project-zero/models", h);
        const char *dirs[] = { "models", home_models, data_models, NULL };
        int w = pz_config_wizard(&c, stdin, stdout, dirs);
        if (w == 1) dirty = 1; else printf("Nothing saved.\n");
    } else if (strcmp(sub, "list") == 0) {
        rc = cmd_list(&c, argc > 2 && strcmp(argv[2], "--all") == 0);
    } else if (strcmp(sub, "get") == 0 && argc == 3) {
        const PzCfgEntry *e = pz_config_get(&c, argv[2]);
        if (!pz_config_find_key(argv[2])) { fprintf(stderr, "unknown setting \"%s\"\n", argv[2]); rc = 1; }
        else if (!e) rc = 1;   /* unset: empty output, exit 1 (like git config) */
        else { for (int j = 0; j < e->n; j++) printf("%s\n", e->v[j]); }
    } else if ((strcmp(sub, "set") == 0 || strcmp(sub, "add") == 0) && argc == 4) {
        int r = sub[0] == 's' ? pz_config_set(&c, argv[2], argv[3], err, sizeof(err))
                              : pz_config_add(&c, argv[2], argv[3], err, sizeof(err));
        if (r != 0) { fprintf(stderr, "%s\n", err); rc = 1; } else dirty = 1;
    } else if (strcmp(sub, "unset") == 0 && argc == 3) {
        if (!pz_config_find_key(argv[2])) { fprintf(stderr, "unknown setting \"%s\"\n", argv[2]); rc = 1; }
        else { pz_config_unset(&c, argv[2]); dirty = 1; }
    } else {
        usage();
        rc = 1;
    }
    if (dirty) {
        if (pz_config_save(&c, path) != 0) { fprintf(stderr, "[config] cannot write %s\n", path); rc = 1; }
        else printf("Saved %s\n", path);
    }
    pz_config_free(&c);
    return rc;
}

int pz_config_prepare_argv(int argc, char **argv, char ***out_argv) {
    static PzConfig cfg;      /* values referenced by the returned argv: process lifetime */
    *out_argv = argv;
    int skip = 0;
    const char *ne = getenv("PZ_NO_CONFIG");
    if (ne && *ne && strcmp(ne, "0") != 0) skip = 1;
    int keep = 0;
    char **filtered = (char **)malloc(((size_t)argc + 1) * sizeof(char *));
    if (!filtered) return argc;
    for (int i = 0; i < argc; i++) {
        if (i > 0 && strcmp(argv[i], "--no-config") == 0) { skip = 1; continue; }
        filtered[keep++] = argv[i];
    }
    filtered[keep] = NULL;
    if (skip) { *out_argv = filtered; return keep; }
    char path[1024];
    if (pz_config_init(&cfg) != 0) { *out_argv = filtered; return keep; }
    if (pz_config_path(path, sizeof(path)) == 0) pz_config_load(&cfg, path, warn_stderr);
    pz_config_apply_env(&cfg, warn_stderr);
    char **merged = NULL;
    int n = pz_config_merge_argv(&cfg, keep, filtered, &merged);
    if (n < 0) { *out_argv = filtered; return keep; }
    free(filtered);
    *out_argv = merged;
    return n;
}
