/*
 * cli/model_catalog.c — see cli/model_catalog.h.
 */
#include "cli/model_catalog.h"
#include "core/platform.h"
#include "core/sha256.h"

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#if TN_POSIX
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

/* Default list (2026-10-22): sizes and SHA-256 from the Hugging Face file
 * listings. Ordered small to large; descriptions are for non-specialists. */
static const PzStarterModel g_builtin[] = {
    { "tiny", "SmolLM2 135M (tiny)", 144811360LL, 0.5,
      "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83",
      "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
      "Very fast, simple answers. Good for trying Project Zero on any computer.", "" },
    { "balanced", "SmolLM2 1.7B (balanced)", 1055609824LL, 2.0,
      "77665ea4815999596525c636fbeb56ba8b080b46ae85efef4f0d986a139834d7",
      "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
      "Good everyday answers at a comfortable speed.", "" },
    { "ternary", "Falcon3 3B 1.58-bit (best)", 2221465632LL, 3.0,
      "2ee9723dc1abcc53f231ef1637bdd7c1ec1dbaf132c2c59873100cfb48b41455",
      "https://huggingface.co/tiiuae/Falcon3-3B-Instruct-1.58bit-GGUF/resolve/main/ggml-model-i2_s.gguf",
      "The best answers of the three. A 1.58-bit (\"ternary\") model, the kind Project Zero is built to run fast.", "" },
};

static void set_file(PzStarterModel *m) {
    const char *b = strrchr(m->url, '/');
    b = b ? b + 1 : m->url;
    /* Several repos name their file generically (ggml-model-i2_s.gguf): prefix the id. */
    snprintf(m->file, sizeof(m->file), "%s-%s", m->id, b);
}

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
    return s;
}

static int valid(const PzStarterModel *m) {
    if (!m->id[0] || !m->name[0] || m->bytes <= 0 || strncmp(m->url, "https://", 8) != 0) return 0;
    if (strlen(m->sha256) != 64) return 0;
    for (int i = 0; i < 64; i++) if (!isxdigit((unsigned char)m->sha256[i])) return 0;
    for (const char *p = m->id; *p; p++)
        if (!isalnum((unsigned char)*p) && *p != '-' && *p != '_') return 0;
    return 1;
}

int pz_model_catalog(PzStarterModel *out, int max, const char **source) {
    const char *path = getenv("PZ_MODEL_CATALOG");
    if (source) *source = "built-in";
    int n = 0;
    if (path && *path) {
        FILE *f = fopen(path, "r");
        if (f) {
            if (source) *source = path;
            char line[1600];
            while (n < max && fgets(line, sizeof(line), f)) {
                char *hash = strchr(line, '#');
                if (hash) *hash = '\0';
                char *field[7];
                int k = 0;
                char *p = line;
                for (; k < 7; k++) {
                    field[k] = p;
                    char *bar = strchr(p, '|');
                    if (!bar) { k++; break; }
                    *bar = '\0';
                    p = bar + 1;
                }
                if (k < 7) continue;
                PzStarterModel m;
                memset(&m, 0, sizeof(m));
                snprintf(m.id, sizeof(m.id), "%s", trim(field[0]));
                snprintf(m.name, sizeof(m.name), "%s", trim(field[1]));
                m.bytes = strtoll(trim(field[2]), NULL, 10);
                m.min_ram_gb = strtod(trim(field[3]), NULL);
                snprintf(m.sha256, sizeof(m.sha256), "%s", trim(field[4]));
                for (char *q = m.sha256; *q; q++) *q = (char)tolower((unsigned char)*q);
                snprintf(m.url, sizeof(m.url), "%s", trim(field[5]));
                snprintf(m.about, sizeof(m.about), "%s", trim(field[6]));
                if (!valid(&m)) continue;
                set_file(&m);
                out[n++] = m;
            }
            fclose(f);
            return n;
        }
    }
    for (size_t i = 0; i < sizeof(g_builtin) / sizeof(g_builtin[0]) && n < max; i++) {
        out[n] = g_builtin[i];
        set_file(&out[n]);
        n++;
    }
    return n;
}

int pz_model_recommend(const PzStarterModel *m, int n, double free_gb) {
    if (n <= 0) return -1;
    int best = -1, smallest = 0;
    for (int i = 0; i < n; i++) {
        if (m[i].bytes < m[smallest].bytes) smallest = i;
        if (m[i].min_ram_gb <= free_gb && (best < 0 || m[i].bytes > m[best].bytes)) best = i;
    }
    return best >= 0 ? best : smallest;
}

int pz_models_dir(char *buf, size_t n) {
    const char *x = getenv("XDG_DATA_HOME"), *h = getenv("HOME");
    int w;
    if (x && *x) w = snprintf(buf, n, "%s/project-zero/models", x);
    else if (h && *h) w = snprintf(buf, n, "%s/.local/share/project-zero/models", h);
    else return -1;
    return (w < 0 || (size_t)w >= n) ? -1 : 0;
}

#if TN_POSIX
static int mkdirs(const char *dir) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", dir);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
        *p = '/';
    }
    return (mkdir(tmp, 0755) == 0 || errno == EEXIST) ? 0 : -1;
}

/* Runs argv (no shell) with the terminal attached; returns the exit code. */
static int run(char *const argv[]) {
    pid_t pid;
    if (posix_spawnp(&pid, argv[0], NULL, NULL, argv, environ) != 0) return -1;
    int st = 0;
    while (waitpid(pid, &st, 0) < 0) if (errno != EINTR) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}
#endif

int pz_model_download(const PzStarterModel *m, const char *dir, char *path, size_t pathn, FILE *out) {
#if TN_POSIX
    char hex[65];
    if (!m || !dir || mkdirs(dir) != 0) {
        fprintf(out, "  Could not create the folder %s.\n", dir ? dir : "(none)");
        return -1;
    }
    snprintf(path, pathn, "%s/%s", dir, m->file);
    if (sha256_file_hex(path, hex) == 0 && strcmp(hex, m->sha256) == 0) {
        fprintf(out, "  Already downloaded and verified: %s\n", path);
        return 0;
    }
    char part[1100];
    snprintf(part, sizeof(part), "%s.part", path);
    fprintf(out, "  Downloading %s (%.1f GB) to %s ...\n", m->name, (double)m->bytes / 1e9, dir);
    fflush(out);
    char *curl[] = { "curl", "-fL", "--retry", "3", "--proto", "=https", "--proto-redir", "=https",
                     "--progress-bar", "-o", part, (char *)m->url, NULL };
    char *wget[] = { "wget", "-q", "--show-progress", "--https-only", "-O", part, (char *)m->url, NULL };
    int rc = run(curl);
    if (rc == -1) rc = run(wget);
    if (rc != 0) {
        remove(part);
        fprintf(out, "  The download failed (no internet connection, or neither curl nor wget is installed).\n"
                     "  You can download any .gguf model yourself and enter its path instead.\n");
        return -1;
    }
    fprintf(out, "  Checking the download is complete and genuine (SHA-256)...\n");
    if (sha256_file_hex(part, hex) != 0 || strcmp(hex, m->sha256) != 0) {
        remove(part);
        fprintf(out, "  The file does not match its expected fingerprint, so it was deleted.\n"
                     "  Try again later, or download a model yourself.\n");
        return -1;
    }
    if (rename(part, path) != 0) {
        remove(part);
        fprintf(out, "  Could not save the file to %s.\n", path);
        return -1;
    }
    fprintf(out, "  Done: %s\n", path);
    return 0;
#else
    (void)m; (void)dir; (void)path; (void)pathn;
    fprintf(out, "  Downloading is not supported on this platform; enter a model path instead.\n");
    return -1;
#endif
}
