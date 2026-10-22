#ifndef PZ_CLI_MODEL_CATALOG_H
#define PZ_CLI_MODEL_CATALOG_H

/*
 * cli/model_catalog — starter models the setup wizard can download
 * (Phase 40, 2026-10-22). Not hardcoded into the flow: the built-in list is
 * a default, and $PZ_MODEL_CATALOG names a file that replaces it, one model
 * per line:
 *     id | name | size_bytes | min_free_ram_gb | sha256 | https-url | description
 * ('#' starts a comment). Every download is verified against its SHA-256
 * before it is used.
 */

#include <stdio.h>

typedef struct {
    char      id[32];
    char      name[80];
    long long bytes;
    double    min_ram_gb;
    char      sha256[65];
    char      url[512];
    char      about[320];
    char      file[560];       /* "<id>-<basename of url>" (fits id + url) */
} PzStarterModel;

#define PZ_CATALOG_MAX 16

/* Fills out[] (at most max); returns the count (0 = none). *source is set to
 * "built-in" or the catalog file path. */
int pz_model_catalog(PzStarterModel *out, int max, const char **source);

/* Index of the largest model that fits in free_gb (the smallest when none
 * fits), or -1 when n == 0. */
int pz_model_recommend(const PzStarterModel *m, int n, double free_gb);

/* Where downloaded models go: $XDG_DATA_HOME/project-zero/models, else
 * ~/.local/share/project-zero/models. Returns 0, or -1 when neither exists. */
int pz_models_dir(char *buf, size_t n);

/* Downloads m into dir (created if needed) with curl or wget, showing their
 * progress on the terminal, verifies the SHA-256, and writes the final path.
 * An existing file with the right checksum is reused. Returns 0, or -1 with a
 * plain-language reason printed to `out`. */
int pz_model_download(const PzStarterModel *m, const char *dir, char *path, size_t pathn, FILE *out);

#endif /* PZ_CLI_MODEL_CATALOG_H */
