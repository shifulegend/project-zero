/* Output side of a chat request: token streaming / accumulation and the best-of / tree-search
 * result objects appended to a finished reply. */

#include "api/chat_stream.h"

#include <stdlib.h>
#include <string.h>

#include "api/json_write.h"
#include "api/sse_stream.h"


/* Phase 30: adds "best_of": {"n", "chosen", "scores": [...], "completion_tokens_total"}
 * to a finished response object (consumes and returns the heap string). */
char *chat_append_best_of_json(char *json, const BestOfResult *r) {
    size_t len = strlen(json);
    if (len < 2 || json[len - 1] != '}') return json;
    JsonBuf jb;
    jb_init(&jb);
    json[len - 1] = '\0';
    jb_raw(&jb, json);
    jb_raw(&jb, ",\"best_of\":{\"n\":");
    jb_int(&jb, r->n);
    jb_raw(&jb, ",\"chosen\":");
    jb_int(&jb, r->chosen);
    jb_raw(&jb, ",\"completion_tokens_total\":");
    jb_int(&jb, r->tokens_total);
    jb_raw(&jb, ",\"scores\":[");
    for (int i = 0; i < r->n && i < BEST_OF_N_MAX; i++) {
        if (i) jb_raw(&jb, ",");
        jb_num(&jb, r->scores[i]);
    }
    jb_raw(&jb, "]}}");
    char *out = jb_finish(&jb);
    if (!out) {
        json[len - 1] = '}';
        return json;
    }
    free(json);
    return out;
}

/* Phase 30: adds "mcts": {"simulations", "nodes", "steps", "path_value",
 * "scorer", "search_tokens"} to a finished response object. */
char *chat_append_mcts_json(char *json, const MctsResult *r) {
    size_t len = strlen(json);
    if (len < 2 || json[len - 1] != '}') return json;
    JsonBuf jb;
    jb_init(&jb);
    json[len - 1] = '\0';
    jb_raw(&jb, json);
    jb_raw(&jb, ",\"mcts\":{\"simulations\":");
    jb_int(&jb, r->simulations);
    jb_raw(&jb, ",\"nodes\":");
    jb_int(&jb, r->nodes);
    jb_raw(&jb, ",\"steps\":");
    jb_int(&jb, r->steps);
    jb_raw(&jb, ",\"path_value\":");
    jb_num(&jb, r->path_value);
    jb_raw(&jb, ",\"scorer\":");
    jb_str(&jb, r->used_prm ? "prm" : "self");
    jb_raw(&jb, ",\"search_tokens\":");
    jb_int(&jb, r->tokens_total);
    jb_raw(&jb, "}}");
    char *out = jb_finish(&jb);
    if (!out) {
        json[len - 1] = '}';
        return json;
    }
    free(json);
    return out;
}

/* Phase 30: lets best-of-N abandon its candidates when the client cancels. */
int chat_best_of_abort_check(void *userdata) {
    StreamState *st = (StreamState *)userdata;
    return cancel_registry_check(st->cancel, st->public_id);
}

int chat_streaming_token_callback(const char *piece, void *userdata) {
    StreamState *st = (StreamState *)userdata;
    if (!piece) return 0;

    if (st->is_streaming) {
        /* 2026-10-22: a failed write means the client left — stop instead of
         * generating the rest of the reply for nobody (and holding a slot). */
        if (sse_write_token(st->conn_fd, st->id, piece) < 0) return 1;
    } else {
        /* Accumulate into dynamic buffer */
        size_t plen = strlen(piece);
        if (st->accum_len + plen + 1 > st->accum_cap) {
            size_t new_cap = (st->accum_cap == 0) ? 4096 : st->accum_cap * 2;
            while (new_cap < st->accum_len + plen + 1) new_cap *= 2;
            char *new_buf = (char *)realloc(st->accum_buf, new_cap);
            if (!new_buf) return 1; /* OOM — stop generating */
            st->accum_buf = new_buf;
            st->accum_cap = new_cap;
        }
        memcpy(st->accum_buf + st->accum_len, piece, plen);
        st->accum_len += plen;
        st->accum_buf[st->accum_len] = '\0';
    }
    st->tokens_emitted++;

    return cancel_registry_check(st->cancel, st->public_id);
}
