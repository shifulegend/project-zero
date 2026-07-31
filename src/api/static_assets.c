/*
 * Phase 22.2 — src/api/static_assets.c
 * See include/api/static_assets.h for responsibility.
 */

#include "api/static_assets.h"
#include "api/webui_bundle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *mime_for_path(const char *path) {
    if (strcmp(path, "/") == 0) return "text/html";
    const char *dot = strrchr(path, '.');
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

static int is_asset_path(const char *path) {
    return strncmp(path, "/assets/", 8) == 0;
}

static void write_headers(int fd, int status, const char *mime_type, size_t body_len,
                           const char *extra_headers) {
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "%s"
        "\r\n",
        status, status == 200 ? "OK" : "Not Found", mime_type, body_len,
        extra_headers ? extra_headers : "");
    if (n > 0) { ssize_t r = write(fd, hdr, (size_t)n); (void)r; }
}

static void serve_404(int fd, const char *extra_headers) {
    const char *body = "Not found";
    write_headers(fd, 404, "text/plain", strlen(body), extra_headers);
    ssize_t r = write(fd, body, strlen(body)); (void)r;
}

/* ── Disk mode (--static-dir) ─────────────────────────────────────────────── */

static int serve_from_disk(int fd, const char *static_dir, const char *path,
                            const char *extra_headers) {
    char fs_path[1024];
    if (strcmp(path, "/") == 0) {
        snprintf(fs_path, sizeof(fs_path), "%s/index.html", static_dir);
    } else {
        snprintf(fs_path, sizeof(fs_path), "%s%s", static_dir, path);
    }

    FILE *f = fopen(fs_path, "rb");
    if (!f && !is_asset_path(path)) {
        /* SPA fallback: unknown top-level route -> index.html */
        snprintf(fs_path, sizeof(fs_path), "%s/index.html", static_dir);
        f = fopen(fs_path, "rb");
    }
    if (!f) {
        serve_404(fd, extra_headers);
        return 404;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) { fclose(f); serve_404(fd, extra_headers); return 404; }

    char *buf = (char *)malloc((size_t)size);
    if (!buf) { fclose(f); serve_404(fd, extra_headers); return 404; }
    size_t nread = fread(buf, 1, (size_t)size, f);
    fclose(f);

    write_headers(fd, 200, mime_for_path(fs_path), nread, extra_headers);
    ssize_t r = write(fd, buf, nread); (void)r;
    free(buf);
    return 200;
}

/* ── Embedded-bundle mode (default) ──────────────────────────────────────── */

static const WebuiAsset *find_asset(const WebuiAsset *assets, size_t count, const char *path) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(assets[i].path, path) == 0) return &assets[i];
    }
    return NULL;
}

const WebuiAsset *static_assets_find_in(const WebuiAsset *assets, size_t count, const char *path) {
    const WebuiAsset *asset = find_asset(assets, count, path);
    if (!asset && !is_asset_path(path)) {
        asset = find_asset(assets, count, "/"); /* SPA fallback for unknown top-level routes */
    }
    return asset;
}

static int serve_from_bundle(int fd, const char *path, const char *extra_headers) {
    const WebuiAsset *asset = static_assets_find_in(g_webui_assets, g_webui_assets_count, path);
    if (!asset) {
        serve_404(fd, extra_headers);
        return 404;
    }
    write_headers(fd, 200, asset->mime_type, asset->len, extra_headers);
    ssize_t r = write(fd, asset->data, asset->len); (void)r;
    return 200;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

int static_assets_serve(int fd, const char *path, const ServerConfig *cfg,
                         const char *extra_headers) {
    if (cfg && cfg->static_dir) {
        return serve_from_disk(fd, cfg->static_dir, path, extra_headers);
    }
    return serve_from_bundle(fd, path, extra_headers);
}
