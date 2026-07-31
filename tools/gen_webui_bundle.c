/*
 * tools/gen_webui_bundle.c — Phase 22.2
 *
 * Walks a built webui/dist/ directory and emits a C translation unit
 * (src/api/webui_bundle_generated.c) embedding every file's bytes plus a
 * {path, data, len, mime_type} manifest (see include/api/webui_bundle.h).
 *
 * Usage: gen_webui_bundle <dist_dir> <output.c>
 *
 * Run via `make webui-bundle` (never part of the default build) whenever
 * webui/src changes. The generated output IS committed to git — that is
 * the "prebuilt bundle" fallback that keeps ordinary `make release`/
 * `cmake --build` Node-free.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#define MAX_FILES 256
#define MAX_PATH_LEN 512

typedef struct {
    char fs_path[MAX_PATH_LEN];   /* real filesystem path under dist_dir */
    char url_path[MAX_PATH_LEN];  /* served path, e.g. "/", "/assets/x.js" */
} FileEntry;

static FileEntry g_files[MAX_FILES];
static int g_num_files = 0;

static const char *mime_for(const char *url_path) {
    if (strcmp(url_path, "/") == 0) return "text/html"; /* index.html, special-cased to "/" */
    const char *dot = strrchr(url_path, '.');
    if (!dot) return "application/octet-stream";
    if (strcmp(dot, ".html") == 0) return "text/html";
    if (strcmp(dot, ".js") == 0)   return "application/javascript";
    if (strcmp(dot, ".css") == 0)  return "text/css";
    if (strcmp(dot, ".json") == 0) return "application/json";
    if (strcmp(dot, ".svg") == 0)  return "image/svg+xml";
    if (strcmp(dot, ".png") == 0)  return "image/png";
    if (strcmp(dot, ".ico") == 0)  return "image/x-icon";
    if (strcmp(dot, ".woff2") == 0) return "font/woff2";
    return "application/octet-stream";
}

/* Recursively collects files under dir_fs_path (real path), assigning each
 * a url_path relative to the dist root (dist_root_len bytes of prefix are
 * stripped). dist/index.html is special-cased to url_path "/". */
static void walk(const char *dir_fs_path, size_t dist_root_len) {
    DIR *d = opendir(dir_fs_path);
    if (!d) { fprintf(stderr, "gen_webui_bundle: cannot open %s\n", dir_fs_path); exit(1); }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;

        char child[MAX_PATH_LEN];
        snprintf(child, sizeof(child), "%s/%s", dir_fs_path, ent->d_name);

        struct stat st;
        if (stat(child, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            walk(child, dist_root_len);
            continue;
        }
        if (!S_ISREG(st.st_mode)) continue;

        if (g_num_files >= MAX_FILES) {
            fprintf(stderr, "gen_webui_bundle: too many files (max %d)\n", MAX_FILES);
            exit(1);
        }

        FileEntry *fe = &g_files[g_num_files++];
        snprintf(fe->fs_path, sizeof(fe->fs_path), "%s", child);

        const char *rel = child + dist_root_len;
        if (strcmp(rel, "index.html") == 0) {
            snprintf(fe->url_path, sizeof(fe->url_path), "/");
        } else {
            snprintf(fe->url_path, sizeof(fe->url_path), "/%s", rel);
        }
    }
    closedir(d);
}

static int emit_file_bytes(FILE *out, const char *fs_path, const char *c_ident, size_t *out_len) {
    FILE *in = fopen(fs_path, "rb");
    if (!in) { fprintf(stderr, "gen_webui_bundle: cannot open %s\n", fs_path); return -1; }

    fprintf(out, "static const unsigned char %s[] = {\n", c_ident);
    unsigned char buf[4096];
    size_t n, total = 0, col = 0;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
        for (size_t i = 0; i < n; i++) {
            fprintf(out, "0x%02x,", buf[i]);
            if (++col == 20) { fputc('\n', out); col = 0; }
        }
        total += n;
    }
    if (col != 0) fputc('\n', out);
    fprintf(out, "};\n");
    fclose(in);
    *out_len = total;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <dist_dir> <output.c>\n", argv[0]);
        return 1;
    }
    const char *dist_dir = argv[1];
    const char *output_path = argv[2];

    struct stat st;
    if (stat(dist_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
        fprintf(stderr, "gen_webui_bundle: %s is not a directory (build the webui first: "
                        "npm --prefix webui ci && npm --prefix webui run build)\n", dist_dir);
        return 1;
    }

    size_t dist_root_len = strlen(dist_dir) + 1; /* +1 for the trailing '/' */
    walk(dist_dir, dist_root_len);

    if (g_num_files == 0) {
        fprintf(stderr, "gen_webui_bundle: no files found under %s\n", dist_dir);
        return 1;
    }

    FILE *out = fopen(output_path, "w");
    if (!out) { fprintf(stderr, "gen_webui_bundle: cannot open %s for writing\n", output_path); return 1; }

    fprintf(out,
        "/*\n"
        " * GENERATED FILE — do not hand-edit.\n"
        " * Produced by tools/gen_webui_bundle.c from webui/dist/ (Phase 22.2).\n"
        " * To regenerate: make webui-bundle\n"
        " */\n\n"
        "#include \"api/webui_bundle.h\"\n\n");

    for (int i = 0; i < g_num_files; i++) {
        char ident[64];
        snprintf(ident, sizeof(ident), "TN_WEBUI_ASSET_%d", i);
        size_t len = 0;
        if (emit_file_bytes(out, g_files[i].fs_path, ident, &len) != 0) {
            fclose(out);
            return 1;
        }
        fprintf(out, "\n");
    }

    fprintf(out, "const WebuiAsset g_webui_assets[] = {\n");
    for (int i = 0; i < g_num_files; i++) {
        fprintf(out, "    { \"%s\", TN_WEBUI_ASSET_%d, sizeof(TN_WEBUI_ASSET_%d), \"%s\" },\n",
                g_files[i].url_path, i, i, mime_for(g_files[i].url_path));
    }
    fprintf(out, "};\n");
    fprintf(out, "const size_t g_webui_assets_count = sizeof(g_webui_assets) / sizeof(g_webui_assets[0]);\n");

    fclose(out);
    fprintf(stderr, "gen_webui_bundle: wrote %d asset(s) to %s\n", g_num_files, output_path);
    return 0;
}
