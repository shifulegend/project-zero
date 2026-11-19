/*
 * src/api/chat_image.c
 *
 * Image upload for a chat request: decodes the newest user message's image_url data URL and runs
 * the vision pipeline into the engine's KV cache, mirroring the CLI's --image handling.
 * Split out of http_server.c; the partial-write loop is net_write_all().
 */

#include "api/chat_image.h"
#include "api/data_url.h"
#include "api/net_io.h"
#include "multimodal/vision_pipeline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HTTP_STRIPPED_PROMPT_BUF 8192 /* user prompt with the image part removed */

int chat_apply_image(ApiContext *ctx, ChatRequest *req) {
    int had_image = 0;
    for (int i = req->num_messages - 1; i >= 0; i--) {
        if (strcmp(req->messages[i].role, "user") != 0 || !req->messages[i].image_data_url)
            continue;

        if (!ctx->server_config.vision_path || !ctx->server_config.proj_path) {
            fprintf(stderr, "[vision] image_url provided but server has no --vision/--proj "
                            "configured; ignoring image\n");
            break;
        }

        unsigned char *img_bytes = NULL;
        size_t img_len           = 0;
        if (data_url_decode(req->messages[i].image_data_url, &img_bytes, &img_len) != TN_OK) {
            fprintf(stderr, "[vision] failed to decode image_url data\n");
            break;
        }

        char tmp_path[] = "/tmp/pz-vision-upload-XXXXXX";
        int tmp_fd      = mkstemp(tmp_path);
        if (tmp_fd < 0) {
            fprintf(stderr, "[vision] failed to create temp file for uploaded image\n");
            free(img_bytes);
            break;
        }
        int wrote_all =
            net_write_all(tmp_fd, img_bytes, img_len) == 0; /* loops on short writes and EINTR */
        close(tmp_fd);
        free(img_bytes);
        if (!wrote_all) {
            fprintf(stderr, "[vision] failed to write temp image file\n");
            unlink(tmp_path);
            break;
        }

        /* Per-request buffer (not static): the result is copied with strdup() below, so it
         * does not need to outlive this block, and two requests never share it. */
        char stripped_prompt_buf[HTTP_STRIPPED_PROMPT_BUF];
        const char *stripped_prompt = req->messages[i].content;
        /* cfg/weights are declared const in ApiContext (text generation
     * never mutates them); the vision pipeline's KV-cache injection
     * needs non-const pointers even though it doesn't rewrite either —
     * same underlying objects main.c's CLI path passes non-const. */
        had_image = 1;
        vision_pipeline_run(tmp_path, ctx->server_config.vision_path, ctx->server_config.proj_path,
                            (Config *)ctx->cfg, (TransformerWeights *)ctx->weights, ctx->run_state,
                            ctx->moe_cfg, ctx->tok, ctx->tp, req->messages[i].content,
                            stripped_prompt_buf, sizeof(stripped_prompt_buf), &stripped_prompt);
        unlink(tmp_path);

        if (stripped_prompt != req->messages[i].content) {
            char *new_content = strdup(stripped_prompt);
            if (new_content) {
                free(req->messages[i].content);
                req->messages[i].content = new_content;
            }
        }
        break;
    }
    return had_image;
}
