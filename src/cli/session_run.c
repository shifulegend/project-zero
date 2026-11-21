/* The execution modes of a CLI run: --self-check, --perplexity, --watermark-detect, a one-shot
 * --prompt (plain, speculative, best-of or tree search), the REPL, and --server (session_server.c).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli/repl.h"
#include "cli/session.h"
#include "core/log.h"
#include "core/text_file.h"
#include "eval/perplexity.h"
#include "eval/selfcheck.h"
#include "reasoning/best_of.h"
#include "reasoning/mcts.h"
#include "sampling/watermark_text.h"
#include "speculative/spec_decode.h"
#include "transformer/generate.h"
#include "update/update_check.h"

#define UPDATE_NOTICE_BUF 512 /* one-line update notice */

/* stdout sink for one-shot CLI generation paths that take a TokenCallback. */
static int stdout_piece_callback(const char *piece, void *userdata) {
    (void)userdata;
    fputs(piece, stdout);
    fflush(stdout);
    return 0;
}

/* --perplexity: tokenize the file (BOS-prefixed, no chat template), score it
 * like llama-perplexity and print running + final PPL. */
static void ppl_progress(int chunk, const PplResult *r, void *user) {
    (void)user;
    printf("[%d]%.4f,", chunk + 1, r->ppl);
    fflush(stdout);
}

static int run_perplexity(const CliArgs *args, const Config *p, const TransformerWeights *w,
                          RunState *s, const MoEConfig *mc, Tokenizer *t, ThreadPool *tp) {
    size_t rd  = 0;
    char *text = tn_read_text_file(args->eval.perplexity_file, (size_t)64 << 20, &rd);
    if (!text) {
        TN_LOG_ERR("session_run", "Error: cannot read %s (must be 1 B..64 MiB)\n",
                   args->eval.perplexity_file);
        return 1;
    }
    int *toks = (int *)malloc((rd + 2) * sizeof(int));
    if (!toks) {
        free(text);
        return 1;
    }
    int rc  = 0;
    int bos = tokenizer_bos_to_prepend(t, p->bos_token_id);
    int n   = 0;
    if (bos >= 0) toks[n++] = bos;
    int k = tokenizer_encode(t, text, rd, toks + n, (int)rd + 1);
    if (k < 0) {
        rc = 1;
        TN_LOG_ERR("session_run", "Error: could not tokenize %s\n", args->eval.perplexity_file);
    } else {
        n += k;
        PplOptions opt = {args->eval.ppl_ctx, args->eval.ppl_chunks, bos};
        PplResult r;
        printf("perplexity: %d tokens, n_ctx %d, %d chunk(s)\n", n,
               opt.n_ctx < s->max_seq_len ? opt.n_ctx : s->max_seq_len,
               args->eval.ppl_chunks > 0 && args->eval.ppl_chunks < n / opt.n_ctx
                   ? args->eval.ppl_chunks
                   : n / opt.n_ctx);
        TernaryError e = perplexity_eval(toks, n, &opt, p, w, s, mc, tp, ppl_progress, NULL, &r);
        if (e != TN_OK) {
            rc = 1;
            TN_LOG_ERR("session_run", "\nError: perplexity needs at least --ppl-ctx (%d) tokens\n",
                       opt.n_ctx);
        } else {
            printf("\nFinal estimate: PPL = %.4f +/- %.5f  (%d tokens scored, %d chunks, %.1f s)\n",
                   r.ppl, r.ppl_err, r.n_scored, r.n_chunks, r.seconds);
        }
    }
    free(text);
    free(toks);
    return rc;
}

/* --self-check: the accuracy probe on demand; exit status 1 over the tolerance. */
static void run_selfcheck(PzSession *ses) {
    SelfCheckOptions so = {ses->args.eval.quality_tolerance, ses->args.eval.self_check_tokens};
    SelfCheckResult sr;
    TernaryError se = selfcheck_run(&so, &ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp, &sr);
    if (se != TN_OK) {
        TN_LOG_ERR("session_run", "Error: self-check failed (%s)\n", tn_error_str(se));
        ses->exit_code = 1;
    } else {
        char line[SELFCHECK_LINE_BUF];
        selfcheck_format(&sr, line, sizeof(line));
        printf("%s\n", line);
        if (sr.verdict == SC_FAIL) ses->exit_code = 1;
    }
}

/* --watermark-detect: tokenize the text with this model's tokenizer and score it. */
static void run_watermark_detect(PzSession *ses) {
    /* Phase 33: offline watermark detection — tokenize the text with this
     * model's tokenizer and score it with the configured key/gamma. */
    GenerationParams gp;
    args_fill_gen_params(&ses->args, &gp);
    WatermarkResult wr;
    if (watermark_detect_text(&gp.watermark, &ses->t, ses->args.watermark.watermark_detect, &wr) !=
        TN_OK) {
        ses->exit_code = 1;
        TN_LOG_ERR("session_run", "Error: could not tokenize --watermark-detect text.\n");
    } else {
        printf("watermark: tokens_scored=%d green=%d green_fraction=%.3f z=%.2f -> %s\n",
               wr.n_scored, wr.n_green, wr.green_fraction, wr.z_score,
               wr.z_score >= WATERMARK_DETECT_Z_THRESHOLD ? "WATERMARKED" : "not detected");
    }
}

/* --prompt: one generation, speculative / tree search / best-of-N / plain. */
static void run_one_shot(PzSession *ses) {
    printf("\n");
    if (ses->draft_active) {
        speculative_generate(&ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp,
                             ses->args.input.prompt, ses->args.sampling.max_tokens,
                             ses->args.sampling.temperature, ses->args.sampling.top_p,
                             ses->args.sampling.json_mode, &ses->draft,
                             ses->args.search.spec_length, ses->args.sampling.seed);
    } else {
        GenerationParams gp;
        args_fill_gen_params(&ses->args, &gp);
        if (ses->args.search.mcts.simulations > 0) {
            MctsResult mr;
            TernaryError me = generate_mcts(
                &ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp, ses->args.input.prompt,
                ses->args.input.prompt, &gp, &ses->args.search.mcts,
                ses->prm.loaded ? &ses->prm : NULL, stdout_piece_callback, NULL, NULL, &mr);
            printf("\n");
            TN_LOG_INF("mcts",
                       "[mcts] %d simulations, %d nodes, %d-step path (value %.3f, %s), %d search "
                       "tokens%s\n",
                       mr.simulations, mr.nodes, mr.steps, (double)mr.path_value,
                       mr.used_prm ? "PRM" : "self-scored", mr.tokens_total,
                       me == TN_OK ? "" : " (failed)");
        } else if (ses->args.sampling.best_of > 1) {
            BestOfResult br;
            generate_best_of(&ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp,
                             ses->args.input.prompt, &gp, ses->args.sampling.best_of,
                             BEST_OF_SCORE_MEAN_LOGPROB, stdout_piece_callback, NULL, NULL, &br);
            printf("\n");
            TN_LOG_INF("best_of", "[best-of] %d candidates, chose #%d (mean log-prob %.3f)\n", br.n,
                       br.chosen, br.chosen >= 0 ? br.scores[br.chosen] : 0.0);
        } else {
            generate_params(&ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp,
                            ses->args.input.prompt, &gp);
        }
    }
    printf("\n");
}

void session_execute(PzSession *ses) {
    if (ses->args.eval.self_check) {
        run_selfcheck(ses);
    } else if (ses->args.eval.perplexity_file) {
        ses->exit_code =
            run_perplexity(&ses->args, &ses->p, &ses->w, ses->s, &ses->mc, &ses->t, ses->tp);
    } else if (ses->args.watermark.watermark_detect) {
        run_watermark_detect(ses);
    } else if (ses->args.server.server_mode) {
        session_run_server(ses);
    } else if (ses->args.input.prompt) {
        run_one_shot(ses);
    } else {
        run_repl(&ses->p, &ses->w, &ses->mc, NULL, NULL, NULL, ses->s, &ses->t, ses->tp, &ses->args,
                 ses->rag_ok ? &ses->rag : NULL, ses->draft_active ? &ses->draft : NULL);
    }
}

/* After the run: close the layer plan and print the update notice (from the cached result; an
 * in-flight check is not waited for). */
void session_finish(PzSession *ses) {
    if (ses->dist_active) dist_plan_close(&ses->dist);

    /* Phase 38: one line after the output, from the cached/background result
     * (an in-flight check is not waited for — it lands in the cache for next time). */
    if (ses->update.notice) {
        UpdateInfo ui;
        char notice[UPDATE_NOTICE_BUF];
        update_service_get(&ses->update.svc, &ui);
        update_format_notice(&ui, notice, sizeof(notice));
        if (notice[0]) TN_LOG_INF("update", "\n%s", notice);
    }
}
