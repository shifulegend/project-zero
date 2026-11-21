/* chat_request_parse: every request body below must parse to exactly what it parsed to before the
 * parser became table-driven. Each body's outcome (status, set_mask, every field the parser can
 * set, the messages, stop strings, template and watermark key) is serialised and hashed; the hash
 * list was recorded from the previous parser. `--print` lists the current hashes. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/chat_request.h"
#include "test_harness.h"

static const char *k_bodies[] = {
    "{}",
    "{\"model\":\"m1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
    "{\"messages\":[{\"role\":\"system\",\"content\":\"s\"},{\"role\":\"user\",\"content\":\"u\"},{"
    "\"role\":\"assistant\",\"content\":\"a\"}]}",
    "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"look\"},{"
    "\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}]}]}",
    "{\"temperature\":0.25}",
    "{\"temperature\":null,\"top_p\":null}",
    "{\"top_p\":0.5}",
    "{\"max_tokens\":77}",
    "{\"max_completion_tokens\":88}",
    "{\"top_k\":12}",
    "{\"min_p\":0.05}",
    "{\"repetition_penalty\":1.1}",
    "{\"repeat_penalty\":1.2}",
    "{\"presence_penalty\":0.3}",
    "{\"frequency_penalty\":0.4}",
    "{\"penalty_last_n\":64}",
    "{\"repeat_last_n\":32}",
    "{\"seed\":123456789012}",
    "{\"seed\":-5}",
    "{\"best_of\":3}",
    "{\"best_of\":null}",
    "{\"stream\":true}",
    "{\"stream\":false}",
    "{\"stream\":1}",
    "{\"ignore_eos\":true}",
    "{\"ignore_eos\":\"yes\"}",
    "{\"cache_prompt\":false}",
    "{\"cache_prompt\":true,\"stream\":true}",
    "{\"stop\":\"END\"}",
    "{\"stop\":[\"a\",\"b\",\"c\"]}",
    "{\"stop\":null}",
    "{\"stop\":\"\"}",
    "{\"stop\":[\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\",\"10\"]}",
    "{\"stop\":[\"a\",",
    "{\"watermark\":true}",
    "{\"watermark\":false}",
    "{\"watermark\":{\"enabled\":true,\"key\":\"k1\",\"gamma\":0.3,\"delta\":2.5}}",
    "{\"watermark\":{\"key\":\"k1\",\"key\":\"k2\",\"extra\":[1,2,{\"x\":3}]}}",
    "{\"watermark\":{\"enabled\":maybe}}",
    "{\"mcts\":8}",
    "{\"mcts\":{\"simulations\":4,\"expand\":3,\"max_depth\":5,\"step_tokens\":16,\"c\":1.5,"
    "\"temperature\":0.7}}",
    "{\"mcts\":{\"unknown\":{\"a\":1},\"simulations\":2}}",
    "{\"mcts\":{\"simulations\":2",
    "{\"chat_template\":\"{{ messages }}\"}",
    "{\"chat_template\":null}",
    "{\"chat_template\":\"\"}",
    "{\"chat_template\":\"x\",\"chat_template\":\"y\"}",
    "{\"response_format\":{\"type\":\"json_object\"}}",
    "{\"response_format\":{\"type\":\"text\"}}",
    "{\"response_format\":{\"type\":\"json_schema\"}}",
    "{\"response_format\":\"json_object\"}",
    "{\"response_format\":{\"other\":[1,2],\"type\":\"json_object\"}}",
    "{\"response_format\":{\"type\":\"json_object\"",
    "{\"unknown\":{\"deep\":[1,2,{\"x\":\"}\"}]},\"temperature\":0.9}",
    "{\"model\":\"quoted \\\"name\\\" \\u0041\",\"temperature\":0.1,\"top_p\":0.2,\"top_k\":3}",
    "{\"temperature\":\"hot\"}",
    "{\"temperature\":1e2,\"seed\":1e3}",
    "{\"messages\":\"nope\"}",
    "{\"messages\":[{\"role\":\"user\"}]}",
    "{\"messages\":[",
    "{\"temperature\":0.5,",
    "{\"temperature\":0.5}   trailing",
    "[1,2,3]",
    "not json",
    "",
    "   {  \"temperature\" : 0.5 , \"stream\" : true  }  ",
};
#define N_BODIES (sizeof(k_bodies) / sizeof(k_bodies[0]))

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
static uint64_t fnv_str(uint64_t h, const char *s) {
    return s ? fnv(h, s, strlen(s) + 1) : fnv(h, "\\0NULL", 5);
}

static uint64_t outcome_hash(const char *body) {
    ChatRequest req;
    chat_request_init(&req);
    int status = (int)chat_request_parse(body, &req);
    uint64_t h = 1469598103934665603ull;
    h          = fnv(h, &status, sizeof(status));
    h          = fnv(h, &req.set_mask, sizeof(req.set_mask));
    h          = fnv_str(h, req.model);
    h          = fnv(h, &req.num_messages, sizeof(int));
    for (int i = 0; i < req.num_messages; i++) {
        h = fnv_str(h, req.messages[i].role);
        h = fnv_str(h, req.messages[i].content);
        h = fnv_str(h, req.messages[i].image_data_url);
    }
    h = fnv(h, &req.sampling, sizeof(req.sampling));
    h = fnv(h, &req.output.max_tokens, sizeof(int));
    h = fnv(h, &req.output.stream, sizeof(bool));
    h = fnv(h, &req.output.json_mode, sizeof(bool));
    h = fnv(h, &req.output.ignore_eos, sizeof(bool));
    h = fnv(h, &req.output.n_stop, sizeof(int));
    for (int i = 0; i < req.output.n_stop; i++) h = fnv_str(h, req.output.stop[i]);
    h = fnv(h, &req.watermark.enabled, sizeof(bool));
    h = fnv_str(h, req.watermark.key);
    h = fnv(h, &req.watermark.gamma, sizeof(float));
    h = fnv(h, &req.watermark.delta, sizeof(float));
    h = fnv(h, &req.search.best_of, sizeof(int));
    h = fnv(h, &req.search.mcts, sizeof(req.search.mcts));
    h = fnv(h, &req.cache_prompt, sizeof(bool));
    h = fnv_str(h, req.chat_template);
    chat_request_free(&req);
    return h;
}

static const uint64_t k_golden[N_BODIES] = {
    0x97e45cb545a8da34ull, 0xfeb8218c357827a7ull, 0x657b8cdc84d30267ull, 0x5f474ba1fac497e2ull,
    0xde561f24edcd66ceull, 0x97e45cb545a8da34ull, 0x603e9535de8f359aull, 0x5e1483309bed2a29ull,
    0x81b90a474efd0982ull, 0xff754beb646b44d8ull, 0x58d00b99f59f4652ull, 0x3bbfbc5355c40a20ull,
    0x5a1b30b7a4dd400bull, 0x872c901a51a1b46aull, 0xb16134eedc29a681ull, 0x7a5bb3f1fd89f025ull,
    0x27c10c306cedcd05ull, 0x7fccffd65a1b5db1ull, 0x4ab762595163162eull, 0x9beaaa80ff58bd20ull,
    0x97e45cb545a8da34ull, 0xe18576de7e028b3full, 0x97e45cb545a8da34ull, 0xc5af974e53626f41ull,
    0xcc95d1f2b7bbfebdull, 0xc5af974e53626f41ull, 0xae86bd90af3c5fb4ull, 0xa67b3e9d662f6872ull,
    0x9ad4ca737a02d3eaull, 0xd402a1ba0ac3f4abull, 0x97e45cb545a8da34ull, 0x40406e2b6471a9f0ull,
    0x718346f72649d930ull, 0x2f9d52ce10d1e179ull, 0xfafd7b70e603c1cdull, 0x523fe5e2104c8b5cull,
    0x9fbe595ea9b3dcd4ull, 0xa111963bb039add0ull, 0xc5af974e53626f41ull, 0x413d2565fb4912e4ull,
    0x12812baebdc9a94full, 0x0bb13232800b23f6ull, 0x38f454606047e75bull, 0x4f4ca3a9dc4f7b23ull,
    0x97e45cb545a8da34ull, 0x97e45cb545a8da34ull, 0x85be460a00d7e5c0ull, 0x3a3bac35bd9b28bdull,
    0xd9d9b2687cbff5f4ull, 0x7a409270f524e93dull, 0x97e45cb545a8da34ull, 0x3a3bac35bd9b28bdull,
    0x3a3bac35bd9b28bdull, 0xbaacb34448575d71ull, 0x52d79424b9f0b37aull, 0x844fd5f1486c5a39ull,
    0xca7e77223293dd57ull, 0xc5af974e53626f41ull, 0x236c22ae3ea01cb8ull, 0xc5af974e53626f41ull,
    0x345c65f9d8a9491full, 0x345c65f9d8a9491full, 0xc5af974e53626f41ull, 0xc5af974e53626f41ull,
    0xc5af974e53626f41ull, 0xdc47b5fdff6f0c94ull,
};

static void test_all_bodies(void) {
    for (size_t i = 0; i < N_BODIES; i++) {
        uint64_t h = outcome_hash(k_bodies[i]);
        if (h != k_golden[i]) printf("  body %zu differs: %016llx\n", i, (unsigned long long)h);
        TEST_ASSERT(h == k_golden[i], "request parses exactly as before");
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--print") == 0) {
        for (size_t i = 0; i < N_BODIES; i++)
            printf("%zu %016llx\n", i, (unsigned long long)outcome_hash(k_bodies[i]));
        return 0;
    }
    RUN_TEST(test_all_bodies);
    TEST_SUMMARY();
}
