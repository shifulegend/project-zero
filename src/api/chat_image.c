/*
 * src/api/chat_image.c
 *
 * Image upload for a chat request: decodes the newest user message's image_url data URL and runs
 * the vision pipeline into the engine's KV cache, mirroring the CLI's --image handling.
 * Split out of http_server.c; the partial-write loop is net_write_all().
 */

#include "api/chat_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "api/data_url.h"
#include "api/net_io.h"
#include "core/log.h"
#include "multimodal/vision_pipeline.h"

/* Decodes a data: URL into a fresh temp file; the path is written to tmp_path (which must hold
 * "/tmp/pz-vision-upload-XXXXXX"). Returns 1 on success, 0 after logging why not (the file is
 * removed again on a failed write). */
static int write_image_temp_file(const char *data_url, char *tmp_path) {
    unsigned char *img_bytes = NULL;
    size_t img_len           = 0;
    if (data_url_decode(data_url, &img_bytes, &img_len) != TN_OK) {
        TN_LOG_ERR("vision", "[vision] failed to decode image_url data\n");
        return 0;
    }
    int tmp_fd = mkstemp(tmp_path);
    if (tmp_fd < 0) {
        TN_LOG_ERR("vision", "[vision] failed to create temp file for uploaded image\n");
        free(img_bytes);
        return 0;
    }
    int wrote_all = net_write_all(tmp_fd, img_bytes, img_len) == 0; /* loops on short writes */
    close(tmp_fd);
    free(img_bytes);
    if (!wrote_all) {
        TN_LOG_ERR("vision", "[vision] failed to write temp image file\n");
        unlink(tmp_path);
    }
    return wrote_all;
}

/* Runs the vision pipeline on the image file for message i and replaces the message's text by
 * the prompt with the image markers stripped. The buffer follows the message length (the stripped
 * text is never longer than the original), so a long prompt is never cut and two requests never
 * share it; the request body limit already bounds the size. Returns 1 when the pipeline ran. */
static int run_vision_for_message(ApiContext *ctx, ChatRequest *req, int i, const char *tmp_path) {
    size_t stripped_cap         = strlen(req->messages[i].content) + 1;
    char *stripped_prompt_buf   = malloc(stripped_cap);
    const char *stripped_prompt = req->messages[i].content;
    if (!stripped_prompt_buf) {
        TN_LOG_ERR("vision", "[vision] out of memory for the prompt buffer\n");
        return 0;
    }
    /* cfg/weights are declared const in ApiContext (text generation never mutates them); the
     * vision pipeline's KV-cache injection needs non-const pointers even though it doesn't
     * rewrite either -- same underlying objects main.c's CLI path passes non-const. */
    vision_pipeline_run(tmp_path, ctx->server_config.vision.vision_path,
                        ctx->server_config.vision.proj_path, (Config *)ctx->cfg,
                        (TransformerWeights *)ctx->weights, ctx->run_state, ctx->moe_cfg, ctx->tok,
                        ctx->tp, req->messages[i].content, stripped_prompt_buf, stripped_cap,
                        &stripped_prompt);
    if (stripped_prompt != req->messages[i].content) {
        char *new_content = strdup(stripped_prompt);
        if (new_content) {
            free(req->messages[i].content);
            req->messages[i].content = new_content;
        }
    }
    free(stripped_prompt_buf);
    return 1;
}

int chat_apply_image(ApiContext *ctx, ChatRequest *req) {
    for (int i = req->num_messages - 1; i >= 0; i--) {
        if (strcmp(req->messages[i].role, "user") != 0 || !req->messages[i].image_data_url)
            continue;

        if (!ctx->server_config.vision.vision_path || !ctx->server_config.vision.proj_path) {
            TN_LOG_INF("vision", "[vision] image_url provided but server has no --vision/--proj "
                                 "configured; ignoring image\n");
            return 0;
        }
        char tmp_path[] = "/tmp/pz-vision-upload-XXXXXX";
        if (!write_image_temp_file(req->messages[i].image_data_url, tmp_path)) return 0;
        int ran = run_vision_for_message(ctx, req, i, tmp_path);
        unlink(tmp_path);
        return ran;
    }
    return 0;
}
