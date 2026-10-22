#ifndef PZ_CLI_CONFIG_CMD_H
#define PZ_CLI_CONFIG_CMD_H

/*
 * cli/config_cmd — `pz config` (Phase 40.2, 2026-10-21).
 *
 *   pz config                 interactive setup wizard (when stdin is a terminal)
 *   pz config list [--all]    saved settings (--all: every setting, with help)
 *   pz config get <key>
 *   pz config set <key> <value>      (a list setting is replaced)
 *   pz config add <key> <value>      (append to a list setting: stop, cors-origin)
 *   pz config unset <key>
 *   pz config path            where the file lives
 *
 * The file is the one config/pz_config.h describes; every value is validated
 * against the schema before it is written.
 */

#include <stdio.h>
#include "config/pz_config.h"

/* argv[0] is "config". Returns the process exit code. */
int pz_config_command(int argc, char **argv);

#include "cli/model_catalog.h"

/* Guided setup options (2026-10-22: written for people who are not
 * technical -- every step says what it means, shows example choices and
 * their effect, and whether it can be changed later). */
typedef struct {
    const char *const *model_dirs;   /* where to look for .gguf files (NULL-terminated; may be NULL) */
    const PzStarterModel *catalog;   /* starter models offered for download (may be NULL) */
    int    n_catalog;
    const char *download_dir;        /* where starter models are saved */
    double free_ram_gb;              /* for the suggestion; <= 0 = unknown */
    /* NULL = pz_model_download (tests inject a fake). */
    int  (*download)(const PzStarterModel *m, const char *dir, char *path, size_t pathn, FILE *out);
    int    offer_chat;               /* ask "open the chat page now?" after saving */
    int    want_chat;                /* out: the user said yes */
} PzWizardOpts;

/* Interactive guided setup over `in`/`out` (testable without a terminal).
 * Updates c; returns 1 when the user chose to save, 0 when cancelled, -1 on
 * error. At any question "?" repeats the explanation, Enter keeps the
 * current value or takes the suggestion, "-" clears the setting. */
int pz_config_wizard_ex(PzConfig *c, FILE *in, FILE *out, PzWizardOpts *opts);

/* Same, offering only the .gguf files found in model_dirs (no downloads,
 * no chat-page offer). */
int pz_config_wizard(PzConfig *c, FILE *in, FILE *out, const char *const *model_dirs);

/* Loads the config file + PZ_* environment and returns argv with those
 * settings placed before the user's flags (see pz_config_merge_argv). With
 * --no-config anywhere in argv, or PZ_NO_CONFIG=1, nothing is loaded and the
 * flag is removed. The returned argv and its strings are owned by this module
 * and stay valid until the next call (do not free them). Returns the new argc
 * (argv unchanged on error). */
int pz_config_prepare_argv(int argc, char **argv, char ***out_argv);

#endif /* PZ_CLI_CONFIG_CMD_H */
