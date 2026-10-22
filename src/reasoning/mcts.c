/*
 * reasoning/mcts.c — Phase 30.2/30.3 step-level MCTS (see reasoning/mcts.h).
 */
#include "reasoning/mcts.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *step;          /* step text, without the delimiter */
    int    parent;        /* -1 for the root */
    int   *children;
    int    n_children;
    int    visits;
    double value_sum;
    int    terminal;      /* model ended the turn, or max depth / context reached */
    int    ended;         /* the model itself ended the turn (EOS) */
    int    depth;
    int    tokens;
} Node;

typedef struct {
    Node *v;
    int   n, cap;
} Tree;

typedef struct { char *buf; size_t len, cap; } Sbuf;

static int sbuf_add(Sbuf *b, const char *s, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t nc = (b->cap ? b->cap * 2 : 256);
        while (nc < b->len + n + 1) nc *= 2;
        char *p = (char *)realloc(b->buf, nc);
        if (!p) return -1;
        b->buf = p;
        b->cap = nc;
    }
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
    return 0;
}

static int collect_cb(const char *piece, void *ud) {
    Sbuf *b = (Sbuf *)ud;
    return sbuf_add(b, piece, strlen(piece)) != 0;
}

double mcts_ucb(double value_sum, int visits, int parent_visits, double c) {
    if (visits <= 0) return INFINITY;
    double pv = parent_visits > 1 ? (double)parent_visits : 1.0;
    return value_sum / visits + c * sqrt(log(pv) / visits);
}

float mcts_self_score(const GenerationStats *st) {
    if (!st || st->completion_tokens <= 0) return 1.0f;   /* ended right away */
    double m = st->sum_logprob / st->completion_tokens;
    if (m > 0.0) m = 0.0;
    return (float)exp(m);
}

static int tree_add(Tree *t, int parent, char *step) {
    if (t->n == t->cap) {
        int nc = t->cap ? t->cap * 2 : 32;
        Node *p = (Node *)realloc(t->v, (size_t)nc * sizeof(Node));
        if (!p) return -1;
        t->v = p;
        t->cap = nc;
    }
    Node *nd = &t->v[t->n];
    memset(nd, 0, sizeof(*nd));
    nd->step = step;
    nd->parent = parent;
    if (parent >= 0) {
        Node *pp = &t->v[parent];
        int *c = (int *)realloc(pp->children, (size_t)(pp->n_children + 1) * sizeof(int));
        if (!c) return -1;
        pp->children = c;
        pp->children[pp->n_children++] = t->n;
        nd->depth = pp->depth + 1;
    }
    return t->n++;
}

static void tree_free(Tree *t) {
    for (int i = 0; i < t->n; i++) { free(t->v[i].step); free(t->v[i].children); }
    free(t->v);
}

/* Steps from the root to `id`, each followed by the delimiter unless it is the
 * last one and `trailing` is 0. Returns a malloc'd string. */
static char *path_text(const Tree *t, int id, const char *delim, int trailing) {
    int chain[1024], n = 0;
    for (int x = id; x > 0 && n < 1024; x = t->v[x].parent) chain[n++] = x;
    Sbuf b = {0};
    if (sbuf_add(&b, "", 0) != 0) return NULL;
    size_t dl = strlen(delim);
    for (int i = n - 1; i >= 0; i--) {
        const char *st = t->v[chain[i]].step ? t->v[chain[i]].step : "";
        if (sbuf_add(&b, st, strlen(st)) != 0) { free(b.buf); return NULL; }
        if ((i > 0 || trailing) && sbuf_add(&b, delim, dl) != 0) { free(b.buf); return NULL; }
    }
    return b.buf;
}

static void backup(Tree *t, int id, double r) {
    for (int x = id; x >= 0; x = t->v[x].parent) {
        t->v[x].visits++;
        t->v[x].value_sum += r;
    }
}

TernaryError generate_mcts(const Config *cfg, const TransformerWeights *w, RunState *s,
                           const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                           const char *prompt, const char *question,
                           const GenerationParams *base, const MctsOptions *opt_in,
                           PrmModel *prm, TokenCallback callback, void *userdata,
                           MctsAbortFn abort_fn, MctsResult *out) {
    if (!cfg || !w || !s || !tok || !prompt || !base || !opt_in) return TN_ERR_INVALID_ARGS;
    MctsOptions opt = *opt_in;
    mcts_options_defaults(&opt);
    MctsResult res;
    memset(&res, 0, sizeof(res));
    res.used_prm = prm && prm->loaded;

    Tree t = {0};
    if (tree_add(&t, -1, NULL) != 0) return TN_ERR_OOM;
    unsigned long long seed0 = base->seed >= 0 ? (unsigned long long)base->seed : 0x5eedULL;
    int expansions = 0;
    int prompt_tokens = 0;

    for (int sim = 0; sim < opt.simulations; sim++) {
        if (abort_fn && abort_fn(userdata)) break;
        /* Selection */
        int node = 0;
        while (!t.v[node].terminal && t.v[node].n_children >= opt.expand) {
            int best = t.v[node].children[0];
            double bs = -INFINITY;
            for (int k = 0; k < t.v[node].n_children; k++) {
                int c = t.v[node].children[k];
                double u = mcts_ucb(t.v[c].value_sum, t.v[c].visits, t.v[node].visits, opt.c_ucb);
                if (u > bs) { bs = u; best = c; }
            }
            node = best;
        }
        res.simulations++;
        if (t.v[node].terminal) {          /* re-evaluate a finished path */
            backup(&t, node, t.v[node].visits ? t.v[node].value_sum / t.v[node].visits : 0.0);
            continue;
        }
        /* Expansion: one sampled step continuing the selected path, within
         * what is left of the caller's max_tokens along this path. */
        int used = 0;
        for (int x = node; x > 0; x = t.v[x].parent) used += t.v[x].tokens;
        int budget = base->max_tokens - used;
        if (budget <= 0) {
            t.v[node].terminal = 1;
            backup(&t, node, t.v[node].visits ? t.v[node].value_sum / t.v[node].visits : 0.0);
            continue;
        }
        char *prefix = path_text(&t, node, opt.step_delim, /*trailing=*/node > 0);
        if (!prefix) break;
        GenerationParams gp = *base;
        GenerationStats st;
        memset(&st, 0, sizeof(st));
        gp.max_tokens = opt.step_tokens < budget ? opt.step_tokens : budget;
        gp.sampler.temperature = opt.temperature;
        gp.seed = (long long)(seed0 + (unsigned long long)(++expansions));
        gp.stop[0] = opt.step_delim;
        gp.n_stop = 1;
        gp.assistant_prefix = prefix;
        gp.compute_logprobs = !res.used_prm;
        gp.stats = &st;
        Sbuf text = {0};
        s->current_pos = 0;
        generate_ex(cfg, w, s, mc, tok, tp, prompt, &gp, collect_cb, &text);
        res.tokens_total += st.completion_tokens;
        if (node == 0 && st.prompt_tokens > 0) prompt_tokens = st.prompt_tokens;
        if (st.finish_reason == GEN_FINISH_ERROR) {
            free(prefix); free(text.buf);
            t.v[node].terminal = 1;       /* context full: nothing more below this node */
            backup(&t, node, 0.0);
            continue;
        }
        char *step = text.buf ? text.buf : (char *)calloc(1, 1);
        int child = tree_add(&t, node, step);
        if (child < 0) { free(prefix); free(step); break; }
        Node *cn = &t.v[child];
        cn->tokens = st.completion_tokens;
        cn->ended = st.hit_eos;
        cn->terminal = st.hit_eos || cn->depth >= opt.max_depth || used + cn->tokens >= base->max_tokens;
        /* Evaluation */
        double r;
        if (res.used_prm) {
            char *steps = path_text(&t, child, opt.step_delim, 0);
            float p = steps ? prm_score(prm, question ? question : prompt, steps) : -1.0f;
            free(steps);
            r = p >= 0.0f ? p : 0.0;
        } else {
            r = mcts_self_score(&st);
        }
        free(prefix);
        backup(&t, child, r);
    }
    res.nodes = t.n - 1;

    /* Final path: most visited child at each level (ties: higher mean value). */
    int node = 0;
    while (t.v[node].n_children > 0) {
        int best = t.v[node].children[0];
        for (int k = 1; k < t.v[node].n_children; k++) {
            int c = t.v[node].children[k];
            const Node *a = &t.v[c], *b = &t.v[best];
            if (a->visits > b->visits ||
                (a->visits == b->visits && a->value_sum / (a->visits ? a->visits : 1) >
                                           b->value_sum / (b->visits ? b->visits : 1)))
                best = c;
        }
        node = best;
        res.steps++;
    }
    if (node > 0) res.path_value = (float)(t.v[node].value_sum / (t.v[node].visits ? t.v[node].visits : 1));

    TernaryError rc = TN_OK;
    int path_tokens = 0;
    for (int x = node; x > 0; x = t.v[x].parent) path_tokens += t.v[x].tokens;
    char *shown = path_text(&t, node, opt.step_delim, /*trailing=*/0);
    int stop_stream = 0;
    if (shown && shown[0] && callback) stop_stream = callback(shown, userdata);
    /* Complete a path the model did not end itself, with the caller's own
     * settings (sampler, stop strings); with no searched path this is plain
     * generation. */
    res.path_tokens = path_tokens;
    int remaining = base->max_tokens - path_tokens;
    int completed = 0;
    if (!stop_stream && remaining > 0 && !(node > 0 && t.v[node].ended)) {
        completed = 1;
        char *prefix = path_text(&t, node, opt.step_delim, /*trailing=*/node > 0);
        if (!prefix) {
            rc = TN_ERR_OOM;
        } else {
            if (callback && node > 0) callback(opt.step_delim, userdata);
            GenerationParams gp = *base;
            gp.max_tokens = remaining;
            gp.assistant_prefix = prefix;
            s->current_pos = 0;
            generate_ex(cfg, w, s, mc, tok, tp, prompt, &gp, callback, userdata);
            free(prefix);
        }
    }
    /* The caller's stats describe the whole answer: searched steps + completion. */
    if (base->stats) {
        if (completed) {
            base->stats->completion_tokens += path_tokens;
        } else {
            base->stats->completion_tokens = path_tokens;
            base->stats->prompt_tokens = prompt_tokens;
            base->stats->finish_reason = stop_stream ? GEN_FINISH_CANCELLED : GEN_FINISH_STOP;
            base->stats->hit_eos = node > 0 && t.v[node].ended;
        }
    }
    free(shown);
    tree_free(&t);
    if (out) *out = res;
    return rc;
}
