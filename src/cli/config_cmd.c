/*
 * cli/config_cmd.c — see cli/config_cmd.h.
 */
#include "cli/config_cmd.h"

#include <stdlib.h>
#include <string.h>

#include "core/log.h"
#include "core/platform.h"
#include "core/sys_probe.h"
#include "kv_cache/kv_strategy.h"
#if TN_POSIX
#include <dirent.h>
#include <spawn.h>
#include <unistd.h>
extern char **environ;
#endif

static void warn_stderr(const char *m) {
    TN_LOG_WRN("config", "[config] %s\n", m);
}

static int load_file(PzConfig *c, char *path, size_t pathn) {
    if (pz_config_init(c) != 0) {
        TN_LOG_ERR("config", "[config] out of memory\n");
        return -1;
    }
    if (pz_config_path(path, pathn) != 0) {
        TN_LOG_INF("config", "[config] no config location (set PZ_CONFIG or HOME)\n");
        return -1;
    }
    if (pz_config_load(c, path, warn_stderr) < 0) {
        TN_LOG_ERR("config", "[config] cannot read %s\n", path);
        return -1;
    }
    return 0;
}

static void print_value(const PzCfgEntry *e) {
    if (e->key->secret) {
        printf("****");
        return;
    }
    for (int j = 0; j < e->n; j++) printf(j ? ", %s" : "%s", e->v[j]);
}

static void usage(void) {
    printf("Usage: pz config [list [--all] | get <key> | set <key> <value> | add <key> <value> |\n"
           "                  unset <key> | path]\n"
           "With no arguments, starts the interactive setup (needs a terminal).\n"
           "Precedence: command-line flag > PZ_<KEY> environment > config file > default.\n");
}

static int cmd_list(const PzConfig *c, int all) {
    int n             = 0;
    const PzCfgKey *s = pz_config_schema(&n);
    const char *group = NULL;
    int shown         = 0;
    for (int i = 0; i < n; i++) {
        const PzCfgEntry *e = pz_config_get(c, s[i].key);
        if (!all && !e) continue;
        if (!group || strcmp(group, s[i].group) != 0) {
            group = s[i].group;
            printf("[%s]\n", group);
        }
        printf("  %-18s = ", s[i].key);
        if (e)
            print_value(e);
        else
            printf("(default)");
        if (all) {
            printf("\n  %-18s   %s", "", s[i].help);
            if (s[i].choices) printf(" [%s]", s[i].choices);
        }
        printf("\n");
        shown++;
    }
    if (!shown)
        printf("No saved settings. Run `pz config` to set up, or `pz config list --all`.\n");
    return 0;
}

/* Starts `pz --server` in place of this process (the saved config supplies
 * the model and port) and asks the desktop to open the page once the server
 * has had a moment to load. Returns only if the server could not start. */
static void start_chat_page(int port) {
    char url[64];
    snprintf(url, sizeof(url), "http://127.0.0.1:%d/", port);
    printf("\nStarting the chat page at %s\n"
           "Loading the model can take a few seconds; leave this window open and press Ctrl+C to "
           "stop.\n"
           "Next time, start it with: pz --server\n\n",
           url);
    fflush(stdout);
#if TN_POSIX
    const char *opener = NULL;
#if TN_APPLE
    opener = "open";
#else
    if ((getenv("DISPLAY") && *getenv("DISPLAY")) ||
        (getenv("WAYLAND_DISPLAY") && *getenv("WAYLAND_DISPLAY")))
        opener = "xdg-open";
#endif
    if (opener) {
        /* The url holds only digits we formatted ourselves; no user text reaches the shell. */
        char cmd[160];
        snprintf(cmd, sizeof(cmd), "sleep 4; %s %s >/dev/null 2>&1", opener, url);
        char *const sh[] = {"sh", "-c", cmd, NULL};
        pid_t pid;
        posix_spawnp(&pid, "sh", NULL, NULL, sh, environ);
    }
    char *const av[] = {"pz", "--server", NULL};
    execv("/proc/self/exe", av);
    execvp("pz", av);
    TN_LOG_ERR("config_cmd", "Could not start the chat page automatically; run: pz --server\n");
#else
    (void)port;
#endif
}

/* `pz config` with no arguments (or `wizard`): the guided setup, offered only on a terminal. */
static int run_wizard(PzConfig *c, int *dirty, int *launch_chat) {
#if TN_POSIX
    if (!isatty(0)) {
        usage();
        return 1;
    }
#endif
    char home_models[1024] = "", data_models[1024] = "";
    const char *h = getenv("HOME"), *xd = getenv("XDG_DATA_HOME");
    if (h) snprintf(home_models, sizeof(home_models), "%s/models", h);
    if (xd && *xd)
        snprintf(data_models, sizeof(data_models), "%s/project-zero/models", xd);
    else if (h)
        snprintf(data_models, sizeof(data_models), "%s/.local/share/project-zero/models", h);
    const char *dirs[] = {"models", home_models, data_models, NULL};
    PzStarterModel cat[PZ_CATALOG_MAX];
    const char *cat_src = NULL;
    PzWizardOpts o;
    memset(&o, 0, sizeof(o));
    o.model_dirs   = dirs;
    o.catalog      = cat;
    o.n_catalog    = pz_model_catalog(cat, PZ_CATALOG_MAX, &cat_src);
    o.download_dir = data_models[0] ? data_models : NULL;
    tn_i64 free_b  = tn_get_free_ram();
    o.free_ram_gb  = free_b > 0 ? (double)free_b / 1e9 : 0;
    o.offer_chat   = 1;
    if (pz_config_wizard_ex(c, stdin, stdout, &o) == 1) {
        *dirty       = 1;
        *launch_chat = o.want_chat;
    } else
        printf("Nothing saved.\n");
    return 0;
}

/* `pz config get <key>`: the value(s), one per line; exit 1 when unset (like git config). */
static int cmd_get(const PzConfig *c, const char *key) {
    const PzCfgEntry *e = pz_config_get(c, key);
    if (!pz_config_find_key(key)) {
        TN_LOG_ERR("config_cmd", "unknown setting \"%s\"\n", key);
        return 1;
    }
    if (!e) return 1;
    for (int j = 0; j < e->n; j++) printf("%s\n", e->v[j]);
    return 0;
}

/* `pz config set|add <key> <value>`. */
static int cmd_set(PzConfig *c, int replace, const char *key, const char *value, int *dirty) {
    char err[256];
    int r = replace ? pz_config_set(c, key, value, err, sizeof(err))
                    : pz_config_add(c, key, value, err, sizeof(err));
    if (r != 0) {
        TN_LOG_INF("config_cmd", "%s\n", err);
        return 1;
    }
    *dirty = 1;
    return 0;
}

/* `pz config unset <key>`. */
static int cmd_unset(PzConfig *c, const char *key, int *dirty) {
    if (!pz_config_find_key(key)) {
        TN_LOG_ERR("config_cmd", "unknown setting \"%s\"\n", key);
        return 1;
    }
    pz_config_unset(c, key);
    *dirty = 1;
    return 0;
}

typedef struct {
    PzConfig *c;
    int argc;
    char **argv;
    int *dirty;
    int *launch_chat;
} Subcommand;

static int sub_wizard(const Subcommand *s) {
    return run_wizard(s->c, s->dirty, s->launch_chat);
}
static int sub_list(const Subcommand *s) {
    return cmd_list(s->c, s->argc > 2 && strcmp(s->argv[2], "--all") == 0);
}
static int sub_get(const Subcommand *s) {
    return cmd_get(s->c, s->argv[2]);
}
static int sub_set(const Subcommand *s) {
    return cmd_set(s->c, 1, s->argv[2], s->argv[3], s->dirty);
}
static int sub_add(const Subcommand *s) {
    return cmd_set(s->c, 0, s->argv[2], s->argv[3], s->dirty);
}
static int sub_unset(const Subcommand *s) {
    return cmd_unset(s->c, s->argv[2], s->dirty);
}

static const struct {
    const char *name;
    int argc; /* exact argument count (including `pz config <name>`), 0 = any */
    int (*run)(const Subcommand *s);
} k_subcommands[] = {
    {"wizard", 0, sub_wizard}, {"list", 0, sub_list}, {"get", 3, sub_get},
    {"set", 4, sub_set},       {"add", 4, sub_add},   {"unset", 3, sub_unset},
};

/* Runs the subcommand against the loaded config; returns its exit code. */
static int run_subcommand(PzConfig *c, int argc, char **argv, int *dirty, int *launch_chat) {
    Subcommand s = {c, argc, argv, dirty, launch_chat};
    if (argc < 2) return sub_wizard(&s);
    for (size_t k = 0; k < sizeof(k_subcommands) / sizeof(k_subcommands[0]); k++)
        if (strcmp(argv[1], k_subcommands[k].name) == 0 &&
            (k_subcommands[k].argc == 0 || k_subcommands[k].argc == argc))
            return k_subcommands[k].run(&s);
    usage();
    return 1;
}

/* `pz config help` and `pz config path` need no config file. Returns 1 (and the exit code in *rc)
 * when it handled the command. */
static int run_without_config(const char *sub, int *rc) {
    char path[1024];
    if (sub && (strcmp(sub, "-h") == 0 || strcmp(sub, "--help") == 0 || strcmp(sub, "help") == 0)) {
        usage();
        *rc = 0;
        return 1;
    }
    if (!sub || strcmp(sub, "path") != 0) return 0;
    if (pz_config_path(path, sizeof(path)) != 0) {
        TN_LOG_INF("config", "[config] no config location\n");
        *rc = 1;
    } else {
        printf("%s\n", path);
        *rc = 0;
    }
    return 1;
}

int pz_config_command(int argc, char **argv) {
    char path[1024];
    PzConfig c;
    int rc;
    if (run_without_config(argc > 1 ? argv[1] : NULL, &rc)) return rc;
    if (load_file(&c, path, sizeof(path)) != 0) {
        pz_config_free(&c);
        return 1;
    }
    int dirty = 0, launch_chat = 0;
    rc = run_subcommand(&c, argc, argv, &dirty, &launch_chat);
    if (dirty) {
        if (pz_config_save(&c, path) != 0) {
            TN_LOG_ERR("config", "[config] cannot write %s\n", path);
            rc          = 1;
            launch_chat = 0;
        } else
            printf("Saved %s\n", path);
    }
    if (launch_chat) {
        const PzCfgEntry *pe = pz_config_get(&c, "port");
        int port             = pe ? atoi(pe->v[0]) : 8080;
        pz_config_free(&c);
        start_chat_page(port > 0 && port < 65536 ? port : 8080); /* returns only on failure */
        return 1;
    }
    pz_config_free(&c);
    return rc;
}

/* --no-config anywhere in argv is dropped and skips the file and environment. */
static int strip_no_config(int argc, char **argv, char **filtered, int *skip) {
    int keep = 0;
    for (int i = 0; i < argc; i++) {
        if (i > 0 && strcmp(argv[i], "--no-config") == 0)
            *skip = 1;
        else
            filtered[keep++] = argv[i];
    }
    filtered[keep] = NULL;
    return keep;
}

void pz_launch_config_free(PzLaunchConfig *lc) {
    free(lc->owned);
    pz_config_free(&lc->cfg);
    memset(lc, 0, sizeof(*lc));
}

int pz_config_prepare_argv(PzLaunchConfig *lc, int argc, char **argv, char ***out_argv) {
    pz_launch_config_free(lc);
    *out_argv       = argv;
    int skip        = tn_sys_env_flag(NULL, "PZ_NO_CONFIG");
    char **filtered = (char **)malloc(((size_t)argc + 1) * sizeof(char *));
    if (!filtered) return argc;
    int keep  = strip_no_config(argc, argv, filtered, &skip);
    *out_argv = lc->owned = filtered;
    if (pz_config_init(&lc->cfg) != 0) return keep;
    int have_path = pz_config_path(lc->path, sizeof(lc->path)) == 0;
    /* Provenance for GET /v1/config: --no-config keeps only the flags, but the file location
     * stays known so the web UI can still save to it. */
    if (!skip) {
        if (have_path) pz_config_load(&lc->cfg, lc->path, warn_stderr);
        pz_config_apply_env(&lc->cfg, warn_stderr);
    }
    pz_config_mark_cli(&lc->cfg, keep, filtered);
    lc->provenance = (PzProvenance){&lc->cfg, have_path ? lc->path : NULL, !skip};
    if (skip) return keep;
    char **merged = NULL;
    int n         = pz_config_merge_argv(&lc->cfg, keep, filtered, &merged);
    if (n < 0) return keep;
    free(filtered);
    *out_argv = lc->owned = merged;
    return n;
}
