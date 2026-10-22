/*
 * cli/config_cmd.c — see cli/config_cmd.h.
 */
#include "cli/config_cmd.h"
#include "core/platform.h"

#include <stdlib.h>
#include <string.h>
#include "kv_cache/kv_strategy.h"
#if TN_POSIX
#include <dirent.h>
#include <spawn.h>
#include <unistd.h>
extern char **environ;
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

/* ── Guided setup: plain-language steps (2026-10-22) ─────────────────────
 * One row per question: what it means, example choices with their effect,
 * and whether/how it can be changed later. Wording is aimed at people who
 * have never configured an AI model; keep it that way when editing. */
typedef struct {
    const char *key;       /* config key ("model" handled specially) */
    const char *title;
    const char *what;
    const char *examples;
    const char *later;
} SetupStep;

static const SetupStep STEPS[] = {
    { "model", "Choose a model",
      "A model is the AI \"brain\" file that Project Zero runs. Bigger models give better\n"
      "  answers but need more memory and answer more slowly.",
      "Pick a number below: a model already on this computer, or one to download\n"
      "  (it is checked to be complete and genuine before use). You can also type the path\n"
      "  of any .gguf model file you already have.",
      "Yes, any time: run `pz config` again, or `pz config set model <file>`.\n"
      "  You can also use a different model for one run with --model <file>." },
    { "threads", "How much of your processor to use",
      "How many processor cores Project Zero may use while it writes an answer.",
      "0 = let Project Zero choose (recommended).\n"
      "  More = faster answers, but your computer is busier. For example, on a 4-core laptop,\n"
      "  2 leaves room for your other apps.",
      "Yes: `pz config set threads <number>`, or --threads for one run." },
    { "kv-type", "Memory for long conversations",
      "While it answers, Project Zero keeps the conversation in memory. This choice trades\n"
      "  memory for precision; it matters most for long chats.",
      "f32 = most precise, uses the most memory (the default).\n"
      "  i8  = half the memory, practically the same answers.\n"
      "  i4  = a quarter of the memory, for very long chats or small computers; answers may\n"
      "        differ slightly.",
      "Yes, from the next start: `pz config set kv-type <f32|i8|i4>`." },
    { "port", "Where the chat page lives",
      "`pz --server` opens a chat page in your web browser at http://127.0.0.1:<port>.\n"
      "  The port is just a number that tells the browser where to find it. Only this\n"
      "  computer can open the page.",
      "8080 = the default. Choose another number (for example 8090) only if another program\n"
      "  already uses 8080.",
      "Yes: `pz config set port <number>`, or --port for one run." },
    { "update-channel", "Update notices",
      "Project Zero can tell you when a newer version is available. It never downloads or\n"
      "  installs anything by itself.",
      "auto       = finished releases, or test releases if you are running one (default).\n"
      "  stable     = only finished releases.\n"
      "  prerelease = also test versions with the newest features.",
      "Yes: `pz config set update-channel <choice>`. To turn notices off:\n"
      "  `pz config set no-update-check true`." },
};
#define N_STEPS ((int)(sizeof(STEPS) / sizeof(STEPS[0])))

static void explain(FILE *out, int i) {
    fprintf(out, "\n── Step %d of %d: %s ──\n", i + 1, N_STEPS, STEPS[i].title);
    fprintf(out, "  What this is: %s\n", STEPS[i].what);
    fprintf(out, "  Your options: %s\n", STEPS[i].examples);
    fprintf(out, "  Can you change it later? %s\n", STEPS[i].later);
}

static void prompt_line(FILE *out, const PzCfgEntry *e, const PzCfgKey *k, const char *suggest) {
    if (e) fprintf(out, "Your choice (Enter = keep %s, ? = explain again): ", k->secret ? "****" : e->v[0]);
    else if (suggest) fprintf(out, "Your choice (Enter = %s, ? = explain again): ", suggest);
    else fprintf(out, "Your choice (Enter = use the default, ? = explain again): ");
    fflush(out);
}

/* One plain setting step. Returns 0 on EOF. */
static int ask_step(PzConfig *c, FILE *in, FILE *out, int i) {
    const char *key = STEPS[i].key;
    const PzCfgKey *k = pz_config_find_key(key);
    explain(out, i);
    if (strcmp(key, "threads") == 0) {
#if TN_POSIX && defined(_SC_NPROCESSORS_ONLN)
        fprintf(out, "  (This computer has %ld processor threads.)\n", sysconf(_SC_NPROCESSORS_ONLN));
#endif
    }
    for (;;) {
        prompt_line(out, pz_config_get(c, key), k, NULL);
        char line[1024];
        if (!read_line(in, line, sizeof(line))) return 0;
        if (line[0] == '\0') return 1;
        if (strcmp(line, "?") == 0) { explain(out, i); continue; }
        if (strcmp(line, "-") == 0) { pz_config_unset(c, key); return 1; }
        char err[256];
        if (pz_config_set(c, key, line, err, sizeof(err)) == 0) return 1;
        fprintf(out, "  That didn't work: %s\n", err);
    }
}

static int yes(const char *line, int dflt) {
    if (!line[0]) return dflt;
    return line[0] == 'y' || line[0] == 'Y';
}

/* Step 1: model. Lists found files, then starter downloads; Enter keeps the
 * current model or takes the suggestion. Returns 0 on EOF. */
static int ask_model(PzConfig *c, FILE *in, FILE *out, PzWizardOpts *o) {
#if TN_POSIX
    static char found[32][1024];
    int nf = scan_models(o->model_dirs, found, 32);
#else
    static char found[1][1024];
    int nf = 0;
#endif
    int nc = o->catalog ? o->n_catalog : 0;
    int rec = nc > 0 ? pz_model_recommend(o->catalog, nc, o->free_ram_gb > 0 ? o->free_ram_gb : 1e9) : -1;
    explain(out, 0);
    for (;;) {
        if (nf > 0) {
            fprintf(out, "  Already on this computer:\n");
            for (int i = 0; i < nf; i++) fprintf(out, "    %d) %s\n", i + 1, found[i]);
        }
        if (nc > 0) {
            fprintf(out, "  Download a starter model%s:\n",
                    o->free_ram_gb > 0 ? "" : " (free memory unknown)");
            for (int i = 0; i < nc; i++)
                fprintf(out, "    %d) %s, %.1f GB download, needs about %.0f GB free memory%s\n"
                             "       %s\n",
                        nf + i + 1, o->catalog[i].name, (double)o->catalog[i].bytes / 1e9,
                        o->catalog[i].min_ram_gb < 1 ? 1.0 : o->catalog[i].min_ram_gb,
                        i == rec ? "  <- suggested for this computer" : "", o->catalog[i].about);
        }
        const PzCfgEntry *cur = pz_config_get(c, "model");
        char sug[160] = "";
        if (!cur && rec >= 0) snprintf(sug, sizeof(sug), "download %s", o->catalog[rec].name);
        prompt_line(out, cur, pz_config_find_key("model"), sug[0] ? sug : NULL);
        char line[1024];
        if (!read_line(in, line, sizeof(line))) return 0;
        if (strcmp(line, "?") == 0) { explain(out, 0); continue; }
        if (strcmp(line, "-") == 0) { pz_config_unset(c, "model"); return 1; }
        int pick = -1;
        if (!line[0]) {
            if (cur || rec < 0) return 1;
            pick = nf + rec + 1;
        } else {
            char *end = NULL;
            long v = strtol(line, &end, 10);
            if (end && *end == '\0' && v >= 1 && v <= nf + nc) pick = (int)v;
        }
        char err[256];
        if (pick >= 1 && pick <= nf) { pz_config_set(c, "model", found[pick - 1], err, sizeof(err)); return 1; }
        if (pick > nf) {
            const PzStarterModel *m = &o->catalog[pick - nf - 1];
            fprintf(out, "  This downloads %.1f GB from %s.\n  Continue? [Y/n]: ",
                    (double)m->bytes / 1e9, "huggingface.co");
            fflush(out);
            if (!read_line(in, line, sizeof(line))) return 0;
            if (!yes(line, 1)) continue;
            char path[1200];
            int (*dl)(const PzStarterModel *, const char *, char *, size_t, FILE *) =
                o->download ? o->download : pz_model_download;
            if (!o->download_dir || dl(m, o->download_dir, path, sizeof(path), out) != 0) {
                fprintf(out, "  No model was set. Pick another option or type a path.\n");
                continue;
            }
            pz_config_set(c, "model", path, err, sizeof(err));
            return 1;
        }
        if (pz_config_set(c, "model", line, err, sizeof(err)) == 0) return 1;
        fprintf(out, "  That didn't work: %s\n", err);
    }
}

static void summary(const PzConfig *c, FILE *out) {
    static const char *const label[N_STEPS] = { "Model", "Processor threads", "Conversation memory",
                                                "Chat page port", "Update notices" };
    fprintf(out, "\n── Summary ──\n");
    for (int i = 0; i < N_STEPS; i++) {
        const PzCfgEntry *e = pz_config_get(c, STEPS[i].key);
        fprintf(out, "  %-20s %s\n", label[i], e ? e->v[0] : "(default)");
    }
    fprintf(out, "\n  Nothing here is permanent: every choice can be changed later with `pz config`,\n"
                 "  and any single run can override it with a command-line option.\n"
                 "  Setup does not move pz itself (reinstall with install.sh --prefix to do that) and\n"
                 "  does not delete models you downloaded (remove the files to free space).\n");
}

int pz_config_wizard_ex(PzConfig *c, FILE *in, FILE *out, PzWizardOpts *o) {
    if (!c || !in || !out || !o) return -1;
    o->want_chat = 0;
    fprintf(out, "Welcome to Project Zero setup. It takes about a minute.\n"
                 "Each step explains what it means. Press Enter to keep the current or suggested choice,\n"
                 "type ? to see the explanation again, or - to clear a setting.\n"
                 "Nothing here is required: Project Zero detects your hardware by itself.\n");
    if (!ask_model(c, in, out, o)) return 0;
    for (int i = 1; i < N_STEPS; i++)
        if (!ask_step(c, in, out, i)) return 0;
    summary(c, out);
    fprintf(out, "\nSave these settings? [Y/n]: ");
    fflush(out);
    char line[64];
    if (!read_line(in, line, sizeof(line))) return 0;
    if (!yes(line, 1)) return 0;
    if (o->offer_chat && pz_config_get(c, "model")) {
        fprintf(out, "Open the chat page in your browser now? [Y/n]: ");
        fflush(out);
        if (read_line(in, line, sizeof(line)) && yes(line, 1)) o->want_chat = 1;
    }
    return 1;
}

int pz_config_wizard(PzConfig *c, FILE *in, FILE *out, const char *const *model_dirs) {
    PzWizardOpts o;
    memset(&o, 0, sizeof(o));
    o.model_dirs = model_dirs;
    return pz_config_wizard_ex(c, in, out, &o);
}

/* Starts `pz --server` in place of this process (the saved config supplies
 * the model and port) and asks the desktop to open the page once the server
 * has had a moment to load. Returns only if the server could not start. */
static void start_chat_page(int port) {
    char url[64];
    snprintf(url, sizeof(url), "http://127.0.0.1:%d/", port);
    printf("\nStarting the chat page at %s\n"
           "Loading the model can take a few seconds; leave this window open and press Ctrl+C to stop.\n"
           "Next time, start it with: pz --server\n\n", url);
    fflush(stdout);
#if TN_POSIX
    const char *opener = NULL;
#if TN_APPLE
    opener = "open";
#else
    if ((getenv("DISPLAY") && *getenv("DISPLAY")) || (getenv("WAYLAND_DISPLAY") && *getenv("WAYLAND_DISPLAY")))
        opener = "xdg-open";
#endif
    if (opener) {
        /* The url holds only digits we formatted ourselves; no user text reaches the shell. */
        char cmd[160];
        snprintf(cmd, sizeof(cmd), "sleep 4; %s %s >/dev/null 2>&1", opener, url);
        char *const sh[] = { "sh", "-c", cmd, NULL };
        pid_t pid;
        posix_spawnp(&pid, "sh", NULL, NULL, sh, environ);
    }
    char *const av[] = { "pz", "--server", NULL };
    execv("/proc/self/exe", av);
    execvp("pz", av);
    fprintf(stderr, "Could not start the chat page automatically; run: pz --server\n");
#else
    (void)port;
#endif
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
    int rc = 0, dirty = 0, launch_chat = 0;
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
        static PzStarterModel cat[PZ_CATALOG_MAX];
        const char *cat_src = NULL;
        PzWizardOpts o;
        memset(&o, 0, sizeof(o));
        o.model_dirs = dirs;
        o.catalog = cat;
        o.n_catalog = pz_model_catalog(cat, PZ_CATALOG_MAX, &cat_src);
        o.download_dir = data_models[0] ? data_models : NULL;
        tn_i64 free_b = tn_get_free_ram();
        o.free_ram_gb = free_b > 0 ? (double)free_b / 1e9 : 0;
        o.offer_chat = 1;
        int w = pz_config_wizard_ex(&c, stdin, stdout, &o);
        if (w == 1) { dirty = 1; launch_chat = o.want_chat; }
        else printf("Nothing saved.\n");
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
        if (pz_config_save(&c, path) != 0) { fprintf(stderr, "[config] cannot write %s\n", path); rc = 1; launch_chat = 0; }
        else printf("Saved %s\n", path);
    }
    if (launch_chat) {
        const PzCfgEntry *pe = pz_config_get(&c, "port");
        int port = pe ? atoi(pe->v[0]) : 8080;
        pz_config_free(&c);
        start_chat_page(port > 0 && port < 65536 ? port : 8080);   /* returns only on failure */
        return 1;
    }
    pz_config_free(&c);
    return rc;
}

int pz_config_prepare_argv(int argc, char **argv, char ***out_argv) {
    static PzConfig cfg;      /* values referenced by the returned argv: process lifetime */
    /* The returned array is owned here and freed on the next call; holding it
     * in a static keeps it reachable, so LeakSanitizer stays quiet on every
     * exit path of main() (2026-10-22: the fuzzer flagged it as a leak). */
    static char **owned;
    free(owned);
    owned = NULL;
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
    *out_argv = owned = filtered;
    char path[1024];
    pz_config_publish(NULL, NULL, 0);
    pz_config_free(&cfg);     /* no-op on first use (static: zeroed) */
    if (pz_config_init(&cfg) != 0) return keep;
    int have_path = pz_config_path(path, sizeof(path)) == 0;
    /* Provenance for GET /v1/config (Phase 40.3). With --no-config only the
     * user's flags count, but the file location is still published so the
     * web UI can save to it. */
    if (!skip) {
        if (have_path) pz_config_load(&cfg, path, warn_stderr);
        pz_config_apply_env(&cfg, warn_stderr);
    }
    pz_config_mark_cli(&cfg, keep, filtered);
    pz_config_publish(&cfg, have_path ? path : NULL, !skip);
    if (skip) return keep;
    char **merged = NULL;
    int n = pz_config_merge_argv(&cfg, keep, filtered, &merged);
    if (n < 0) return keep;
    free(filtered);
    *out_argv = owned = merged;
    return n;
}
