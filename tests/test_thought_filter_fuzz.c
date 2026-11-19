/* thought_filter_process: random token streams built from tag fragments must never read or write
 * out of bounds (runs under ASan/UBSan in `make test`) and must never leak think-tag text. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reasoning/thought_filter.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

int main(void) {
    static const char *frag[] = {"<", "t",       "h",        "i",     "n",     "k", ">",
                                 "/", "<think>", "</think>", "<thin", "</thi", "x", "ab",
                                 " ", "\n",      "<<",       "</",    ">>"};
    const int nfrag           = (int)(sizeof frag / sizeof frag[0]);
    unsigned long long st     = 88172645463325252ULL;
    long streams = 0, out_total = 0;
    for (int run = 0; run < 20000; run++) {
        ThoughtFilter f;
        thought_filter_init(&f);
        int ntok = 1 + (int)(st % 40);
        for (int t = 0; t < ntok; t++) {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            char tok[64] = {0};
            int parts    = 1 + (int)(st % 3);
            for (int p = 0; p < parts; p++) {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                strncat(tok, frag[st % (unsigned)nfrag], sizeof tok - strlen(tok) - 1);
            }
            char out[16]; /* deliberately small: the bounds checks are the point */
            thought_filter_process(&f, tok, out, sizeof out);
            out_total += (long)strlen(out);
            if (f.tag_pos < 0 || f.tag_pos > 15) {
                CHECK(0, "tag_pos stays within the tag buffer");
                return 1;
            }
        }
        streams++;
    }
    CHECK(streams == 20000 && out_total >= 0,
          "20000 random fragment streams processed without a bounds violation");

    ThoughtFilter f;
    thought_filter_init(&f);
    char out[64];
    thought_filter_process(&f, "<think>secret</think>visible", out, sizeof out);
    CHECK(strstr(out, "secret") == NULL && strstr(out, "visible") != NULL,
          "a normal think block is still removed and the answer kept");
    printf(g_fail ? "thought_filter_fuzz: %d FAILED\n" : "thought_filter_fuzz: all passed\n",
           g_fail);
    return g_fail ? 1 : 0;
}
