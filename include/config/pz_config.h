#ifndef PZ_CONFIG_H
#define PZ_CONFIG_H

/*
 * config/pz_config — persistent configuration (Phase 40.1, 2026-10-21).
 *
 * One precedence order everywhere: CLI flag > environment > config file >
 * built-in default. The engine's flag parser stays the single source of
 * validation: settings from the file and from PZ_* environment variables are
 * turned into ordinary flag tokens that are placed *before* the user's own
 * argv, so a flag given on the command line wins (value flags are last-wins).
 *
 * File: $PZ_CONFIG, else $XDG_CONFIG_HOME/project-zero/config.toml, else
 * ~/.config/project-zero/config.toml. Format: a flat TOML subset —
 *     key = "string" | 123 | 1.5 | true | false | ["a", "b"]
 * one per line, `#` comments. Keys are the CLI flag names without "--"
 * (`threads`, `kv-type`, ...). Environment: PZ_<KEY> with '-' -> '_' and
 * upper case (PZ_THREADS, PZ_KV_TYPE); a list value from the environment is
 * one item.
 */

#include <stddef.h>

typedef enum {
    PZ_CFG_STRING = 0,
    PZ_CFG_INT,
    PZ_CFG_FLOAT,
    PZ_CFG_BOOL,      /* true -> the flag is passed; false -> omitted */
    PZ_CFG_LIST       /* repeatable flag: one token pair per item */
} PzCfgType;

typedef struct {
    const char *key;        /* flag name without "--" */
    PzCfgType   type;
    const char *group;      /* "model", "generation", "server", ... (list/wizard grouping) */
    const char *help;
    const char *choices;    /* "|"-separated allowed values, or NULL */
    double      min, max;   /* numeric range when min < max */
    int         secret;     /* value masked by `pz config list` */
} PzCfgKey;

#define PZ_CFG_MAX_VALUE 65536   /* bytes per value (a system prompt can be long) */
#define PZ_CFG_MAX_ITEMS 16      /* items per list key */

/* Where a setting's effective value comes from (Phase 40.3, 2026-10-22). */
typedef enum {
    PZ_SRC_DEFAULT = 0,
    PZ_SRC_FILE,
    PZ_SRC_ENV,
    PZ_SRC_CLI
} PzCfgSource;

typedef struct {
    const PzCfgKey *key;
    int   set;                     /* value present */
    int   src;                     /* PzCfgSource of the value (CLI: flag given, value not held) */
    int   n;                       /* items (1 for scalars) */
    char *v[PZ_CFG_MAX_ITEMS];     /* malloc'd raw values (strings unquoted) */
} PzCfgEntry;

typedef struct {
    PzCfgEntry *entries;    /* one per schema key, schema order */
    int         n;
} PzConfig;

/* Schema (static table). */
const PzCfgKey *pz_config_schema(int *n);
const PzCfgKey *pz_config_find_key(const char *key);

/* Default file path. Returns 0, or -1 when no location can be derived. */
int pz_config_path(char *buf, size_t n);

int  pz_config_init(PzConfig *c);           /* all unset */
void pz_config_free(PzConfig *c);

/* Validates one value for a key (type, choices, range). Returns 0, or -1 and
 * writes a reason to err. */
int pz_config_validate(const PzCfgKey *k, const char *value, char *err, size_t errn);

/* Sets / appends / unsets a value after validation. LIST keys: set replaces
 * the list with one item, add appends. Returns 0 or -1 (err filled). */
int pz_config_set(PzConfig *c, const char *key, const char *value, char *err, size_t errn);
int pz_config_add(PzConfig *c, const char *key, const char *value, char *err, size_t errn);
int pz_config_unset(PzConfig *c, const char *key);
const PzCfgEntry *pz_config_get(const PzConfig *c, const char *key);

/* Parses TOML-subset text into c (on top of what it holds). Unknown keys and
 * bad lines are reported to `warn` (may be NULL) and skipped; returns the
 * number of values loaded. */
int pz_config_parse(PzConfig *c, const char *text, void (*warn)(const char *msg));
/* Reads path (missing file = 0 values, not an error). Returns values loaded or -1. */
int pz_config_load(PzConfig *c, const char *path, void (*warn)(const char *msg));

/* Serialises set entries (schema order). Returns malloc'd text. */
char *pz_config_format(const PzConfig *c);
/* Atomic write (tmp + rename), parent dirs created, mode 0600 (may hold an
 * API key). Returns 0 or -1. */
int pz_config_save(const PzConfig *c, const char *path);

/* Overlays PZ_* environment variables (getenv). Returns values applied. */
int pz_config_apply_env(PzConfig *c, void (*warn)(const char *msg));

/* Builds argv = { argv[0], <config tokens...>, argv[1..argc-1] } with config
 * tokens from c. *out_argv is malloc'd (strings point into c / argv), free
 * with free(). Returns the new argc, or -1. */
int pz_config_merge_argv(const PzConfig *c, int argc, char **argv, char ***out_argv);

/* "default" | "file" | "env" | "cli". */
const char *pz_config_source_name(int src);

/* Marks every schema key given as a flag in argv[1..] (the user's own flags)
 * as PZ_SRC_CLI. Returns the number of keys marked. */
int pz_config_mark_cli(PzConfig *c, int argc, char **argv);

/* Process-wide record of the configuration the engine started with, for
 * reporting (GET /v1/config). `c` must outlive all readers; NULL clears it.
 * path may be NULL (no config location); file_used is 0 when the file and
 * environment were skipped (--no-config / PZ_NO_CONFIG). */
void pz_config_publish(const PzConfig *c, const char *path, int file_used);
/* Source of `key` in the published config ("default" when none / unknown). */
const char *pz_config_active_source(const char *key);
/* Path of the published config file, or NULL. */
const char *pz_config_active_path(void);
/* 1 when the published config read the file and environment. */
int pz_config_active_file_used(void);

#endif /* PZ_CONFIG_H */
