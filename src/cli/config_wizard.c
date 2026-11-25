/*
 * src/cli/config_wizard.c
 *
 * The interactive `pz config` setup wizard (declared in cli/config_cmd.h): model discovery, the
 * guided questions and the summary. Split out of config_cmd.c; logic unchanged.
 */

#include <stdlib.h>
#include <string.h>

#include "cli/config_cmd.h"
#include "core/log.h"
#include "core/platform.h"
#include "core/sys_probe.h"
#include "kv_cache/kv_strategy.h"
#if TN_POSIX
#include <dirent.h>
#include <unistd.h>
#endif

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
    const char *key; /* config key ("model" handled specially) */
    const char *title;
    const char *what;
    const char *examples;
    const char *later;
} SetupStep;

static const SetupStep STEPS[] = {
    {"model", "Choose a model",
     "A model is the AI \"brain\" file that Project Zero runs. Bigger models give better\n"
     "  answers but need more memory and answer more slowly.",
     "Pick a number below: a model already on this computer, or one to download\n"
     "  (it is checked to be complete and genuine before use). You can also type the path\n"
     "  of any .gguf model file you already have.",
     "Yes, any time: run `pz config` again, or `pz config set model <file>`.\n"
     "  You can also use a different model for one run with --model <file>."},
    {"threads", "How much of your processor to use",
     "How many processor cores Project Zero may use while it writes an answer.",
     "0 = let Project Zero choose (recommended).\n"
     "  More = faster answers, but your computer is busier. For example, on a 4-core laptop,\n"
     "  2 leaves room for your other apps.",
     "Yes: `pz config set threads <number>`, or --threads for one run."},
    {"kv-type", "Memory for long conversations",
     "While it answers, Project Zero keeps the conversation in memory. This choice trades\n"
     "  memory for precision; it matters most for long chats.",
     "f32 = most precise, uses the most memory (the default).\n"
     "  i8  = half the memory, practically the same answers.\n"
     "  i4  = a quarter of the memory, for very long chats or small computers; answers may\n"
     "        differ slightly.",
     "Yes, from the next start: `pz config set kv-type <f32|i8|i4>`."},
    {"port", "Where the chat page lives",
     "`pz --server` opens a chat page in your web browser at http://127.0.0.1:<port>.\n"
     "  The port is just a number that tells the browser where to find it. Only this\n"
     "  computer can open the page.",
     "8080 = the default. Choose another number (for example 8090) only if another program\n"
     "  already uses 8080.",
     "Yes: `pz config set port <number>`, or --port for one run."},
    {"update-channel", "Update notices",
     "Project Zero can tell you when a newer version is available. It never downloads or\n"
     "  installs anything by itself.",
     "auto       = finished releases, or test releases if you are running one (default).\n"
     "  stable     = only finished releases.\n"
     "  prerelease = also test versions with the newest features.",
     "Yes: `pz config set update-channel <choice>`. To turn notices off:\n"
     "  `pz config set no-update-check true`."},
};
#define N_STEPS ((int)(sizeof(STEPS) / sizeof(STEPS[0])))

static void explain(FILE *out, int i) {
    fprintf(out, "\n── Step %d of %d: %s ──\n", i + 1, N_STEPS, STEPS[i].title);
    fprintf(out, "  What this is: %s\n", STEPS[i].what);
    fprintf(out, "  Your options: %s\n", STEPS[i].examples);
    fprintf(out, "  Can you change it later? %s\n", STEPS[i].later);
}

static void prompt_line(FILE *out, const PzCfgEntry *e, const PzCfgKey *k, const char *suggest) {
    if (e)
        fprintf(out,
                "Your choice (Enter = keep %s, ? = explain again): ", k->secret ? "****" : e->v[0]);
    else if (suggest)
        fprintf(out, "Your choice (Enter = %s, ? = explain again): ", suggest);
    else
        fprintf(out, "Your choice (Enter = use the default, ? = explain again): ");
    fflush(out);
}

/* One plain setting step. Returns 0 on EOF. */
static int ask_step(PzConfig *c, FILE *in, FILE *out, int i) {
    const char *key   = STEPS[i].key;
    const PzCfgKey *k = pz_config_find_key(key);
    explain(out, i);
    if (strcmp(key, "threads") == 0) {
#if TN_POSIX && defined(_SC_NPROCESSORS_ONLN)
        fprintf(out, "  (This computer has %ld processor threads.)\n",
                sysconf(_SC_NPROCESSORS_ONLN));
#endif
    }
    for (;;) {
        prompt_line(out, pz_config_get(c, key), k, NULL);
        char line[1024];
        if (!read_line(in, line, sizeof(line))) return 0;
        if (line[0] == '\0') return 1;
        if (strcmp(line, "?") == 0) {
            explain(out, i);
            continue;
        }
        if (strcmp(line, "-") == 0) {
            pz_config_unset(c, key);
            return 1;
        }
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
/* What the model question offers: models found on disk, then the starter catalog. */
typedef struct {
    PzWizardOpts *o;
    char (*found)[1024];
    int nf;  /* models found on disk */
    int nc;  /* starter models offered */
    int rec; /* suggested starter model, -1 for none */
} ModelMenu;

static void show_menu(FILE *out, const ModelMenu *m) {
    const PzWizardOpts *o = m->o;
    if (m->nf > 0) {
        fprintf(out, "  Already on this computer:\n");
        for (int i = 0; i < m->nf; i++) fprintf(out, "    %d) %s\n", i + 1, m->found[i]);
    }
    if (m->nc == 0) return;
    fprintf(out, "  Download a starter model%s:\n",
            o->free_ram_gb > 0 ? "" : " (free memory unknown)");
    for (int i = 0; i < m->nc; i++)
        fprintf(out,
                "    %d) %s, %.1f GB download, needs about %.0f GB free memory%s\n"
                "       %s\n",
                m->nf + i + 1, o->catalog[i].name, (double)o->catalog[i].bytes / 1e9,
                o->catalog[i].min_ram_gb < 1 ? 1.0 : o->catalog[i].min_ram_gb,
                i == m->rec ? "  <- suggested for this computer" : "", o->catalog[i].about);
}

/* The menu number a typed answer names, -1 when it is not one. */
static int menu_pick(const ModelMenu *m, const char *line) {
    char *end = NULL;
    long v    = strtol(line, &end, 10);
    if (!end || *end != '\0') return -1;
    return (v >= 1 && v <= m->nf + m->nc) ? (int)v : -1;
}

enum { MODEL_EOF = 0, MODEL_DONE = 1, MODEL_AGAIN = 2 };

/* A starter model was picked: confirm, download and set it. MODEL_AGAIN returns to the menu. */
static int fetch_starter(PzConfig *c, FILE *in, FILE *out, const ModelMenu *m, int pick) {
    const PzStarterModel *sm = &m->o->catalog[pick - m->nf - 1];
    char line[1024], err[256];
    fprintf(out, "  This downloads %.1f GB from %s.\n  Continue? [Y/n]: ", (double)sm->bytes / 1e9,
            "huggingface.co");
    fflush(out);
    if (!read_line(in, line, sizeof(line))) return MODEL_EOF;
    if (!yes(line, 1)) return MODEL_AGAIN;
    char path[1200];
    int (*dl)(const PzStarterModel *, const char *, char *, size_t, FILE *) =
        m->o->download ? m->o->download : pz_model_download;
    if (!m->o->download_dir || dl(sm, m->o->download_dir, path, sizeof(path), out) != 0) {
        fprintf(out, "  No model was set. Pick another option or type a path.\n");
        return MODEL_AGAIN;
    }
    pz_config_set(c, "model", path, err, sizeof(err));
    return MODEL_DONE;
}

/* The answer to the model question: a menu number, a path, Enter (keep or take the suggestion),
 * "?" (explain again) or "-" (clear). */
static int answer_model(PzConfig *c, FILE *in, FILE *out, const ModelMenu *m, const char *line,
                        const PzCfgEntry *cur) {
    char err[256];
    if (strcmp(line, "?") == 0) {
        explain(out, 0);
        return MODEL_AGAIN;
    }
    if (strcmp(line, "-") == 0) {
        pz_config_unset(c, "model");
        return MODEL_DONE;
    }
    int pick = -1;
    if (!line[0]) {
        if (cur || m->rec < 0) return MODEL_DONE;
        pick = m->nf + m->rec + 1;
    } else {
        pick = menu_pick(m, line);
    }
    if (pick >= 1 && pick <= m->nf) {
        pz_config_set(c, "model", m->found[pick - 1], err, sizeof(err));
        return MODEL_DONE;
    }
    if (pick > m->nf) return fetch_starter(c, in, out, m, pick);
    if (pz_config_set(c, "model", line, err, sizeof(err)) == 0) return MODEL_DONE;
    fprintf(out, "  That didn't work: %s\n", err);
    return MODEL_AGAIN;
}

static ModelMenu menu_for(PzWizardOpts *o, char (*found)[1024], int nf) {
    ModelMenu m;
    m.o     = o;
    m.found = found;
    m.nf    = nf;
    m.nc    = o->catalog ? o->n_catalog : 0;
    m.rec   = -1;
    if (m.nc > 0)
        m.rec = pz_model_recommend(o->catalog, m.nc, o->free_ram_gb > 0 ? o->free_ram_gb : 1e9);
    return m;
}

static int ask_model(PzConfig *c, FILE *in, FILE *out, PzWizardOpts *o) {
#if TN_POSIX
    char found[32][1024];
    int nf = scan_models(o->model_dirs, found, 32);
#else
    char found[1][1024];
    int nf = 0;
#endif
    ModelMenu m = menu_for(o, found, nf);
    explain(out, 0);
    for (;;) {
        show_menu(out, &m);
        const PzCfgEntry *cur = pz_config_get(c, "model");
        char sug[160]         = "";
        if (!cur && m.rec >= 0) snprintf(sug, sizeof(sug), "download %s", o->catalog[m.rec].name);
        prompt_line(out, cur, pz_config_find_key("model"), sug[0] ? sug : NULL);
        char line[1024];
        if (!read_line(in, line, sizeof(line))) return MODEL_EOF;
        int r = answer_model(c, in, out, &m, line, cur);
        if (r != MODEL_AGAIN) return r;
    }
}

static void summary(const PzConfig *c, FILE *out) {
    static const char *const label[N_STEPS] = {"Model", "Processor threads", "Conversation memory",
                                               "Chat page port", "Update notices"};
    fprintf(out, "\n── Summary ──\n");
    for (int i = 0; i < N_STEPS; i++) {
        const PzCfgEntry *e = pz_config_get(c, STEPS[i].key);
        fprintf(out, "  %-20s %s\n", label[i], e ? e->v[0] : "(default)");
    }
    fprintf(out,
            "\n  Nothing here is permanent: every choice can be changed later with `pz config`,\n"
            "  and any single run can override it with a command-line option.\n"
            "  Setup does not move pz itself (reinstall with install.sh --prefix to do that) and\n"
            "  does not delete models you downloaded (remove the files to free space).\n");
}

/* The closing questions: save, then (when offered and a model is set) open the chat page. Returns 1
 * to save, 0 when cancelled. */
static int confirm_save(PzConfig *c, FILE *in, FILE *out, PzWizardOpts *o) {
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

int pz_config_wizard_ex(PzConfig *c, FILE *in, FILE *out, PzWizardOpts *o) {
    if (!c || !in) return -1;
    if (!out || !o) return -1;
    o->want_chat = 0;
    fprintf(
        out,
        "Welcome to Project Zero setup. It takes about a minute.\n"
        "Each step explains what it means. Press Enter to keep the current or suggested choice,\n"
        "type ? to see the explanation again, or - to clear a setting.\n"
        "Nothing here is required: Project Zero detects your hardware by itself.\n");
    if (!ask_model(c, in, out, o)) return 0;
    for (int i = 1; i < N_STEPS; i++)
        if (!ask_step(c, in, out, i)) return 0;
    return confirm_save(c, in, out, o);
}

int pz_config_wizard(PzConfig *c, FILE *in, FILE *out, const char *const *model_dirs) {
    PzWizardOpts o;
    memset(&o, 0, sizeof(o));
    o.model_dirs = model_dirs;
    return pz_config_wizard_ex(c, in, out, &o);
}
