/*
 * reasoning/mcts.c — Phase 30.2/30.3 step-level MCTS (see reasoning/mcts.h).
 */
#include "reasoning/mcts.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *step; /* step text, without the delimiter */
    int parent; /* -1 for the root */
    int *children;
    int n_children;
    int visits;
    double value_sum;
    int terminal; /* model ended the turn, or max depth / context reached */
    int ended;    /* the model itself ended the turn (EOS) */
    int depth;
    int tokens;
} Node;

typedef struct {
    Node *v;
    int n, cap;
} Tree;

typedef struct {
    char *buf;
    size_t len, cap;
} Sbuf;

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
    if (!st || st->completion_tokens <= 0) return 1.0f; /* ended right away */
    double m = st->sum_logprob / st->completion_tokens;
    if (m > 0.0) m = 0.0;
    return (float)exp(m);
}

static int tree_add(Tree *t, int parent, char *step) {
    if (t->n == t->cap) {
        int nc  = t->cap ? t->cap * 2 : 32;
        Node *p = (Node *)realloc(t->v, (size_t)nc * sizeof(Node));
        if (!p) return -1;
        t->v   = p;
        t->cap = nc;
    }
    Node *nd = &t->v[t->n];
    memset(nd, 0, sizeof(*nd));
    nd->step   = step;
    nd->parent = parent;
    if (parent >= 0) {
        Node *pp = &t->v[parent];
        int *c   = (int *)realloc(pp->children, (size_t)(pp->n_children + 1) * sizeof(int));
        if (!c) return -1;
        pp->children                   = c;
        pp->children[pp->n_children++] = t->n;
        nd->depth                      = pp->depth + 1;
    }
    return t->n++;
}

static void tree_free(Tree *t) {
    for (int i = 0; i < t->n; i++) {
        free(t->v[i].step);
        free(t->v[i].children);
    }
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
        if (sbuf_add(&b, st, strlen(st)) != 0) {
            free(b.buf);
            return NULL;
        }
        if ((i > 0 || trailing) && sbuf_add(&b, delim, dl) != 0) {
            free(b.buf);
            return NULL;
        }
    }
    return b.buf;
}

static void backup(Tree *t, int id, double r) {
    for (int x = id; x >= 0; x = t->v[x].parent) {
        t->v[x].visits++;
        t->v[x].value_sum += r;
    }
}

/* Everything one search needs, so the steps below take one argument. */
typedef struct {
    const Config *cfg;
    const TransformerWeights *w;
    RunState *s;
    const MoEConfig *mc;
    Tokenizer *tok;
    ThreadPool *tp;
    const char *prompt;
    const char *question;
    const GenerationParams *base;
    MctsOptions opt;
    PrmModel *prm;
    Tree t;
    MctsResult res;
    unsigned long long seed0;
    int expansions;
    int prompt_tokens;
} Search;

/* Mean value of a node's visits so far (0 when unvisited). */
static double node_mean(const Tree *t, int id) {
    return t->v[id].visits ? t->v[id].value_sum / t->v[id].visits : 0.0;
}

/* Selection: from the root, take the child with the best UCB value while a node has its full
 * set of children and is not finished. */
static int select_leaf(const Search *S) {
    const Tree *t = &S->t;
    int node      = 0;
    while (!t->v[node].terminal && t->v[node].n_children >= S->opt.expand) {
        int best  = t->v[node].children[0];
        double bs = -INFINITY;
        for (int k = 0; k < t->v[node].n_children; k++) {
            int c    = t->v[node].children[k];
            double u = mcts_ucb(t->v[c].value_sum, t->v[c].visits, t->v[node].visits, S->opt.c_ucb);
            if (u > bs) {
                bs   = u;
                best = c;
            }
        }
        node = best;
    }
    return node;
}

/* Evaluation of a new step: the PRM's score of the path so far, or the model's own confidence. */
static double evaluate_step(Search *S, int child, const GenerationStats *st) {
    if (!S->res.used_prm) return mcts_self_score(st);
    char *steps = path_text(&S->t, child, S->opt.step_delim, 0);
    float p     = steps ? prm_score(S->prm, S->question ? S->question : S->prompt, steps) : -1.0f;
    free(steps);
    return p >= 0.0f ? p : 0.0;
}

/* Expansion: one sampled step continuing the selected path, within what is left of the caller's
 * max_tokens along it (`budget`). Returns 0 when the search must stop (out of memory). */
static int expand_node(Search *S, int node, int used, int budget) {
    Tree *t      = &S->t;
    char *prefix = path_text(t, node, S->opt.step_delim, /*trailing=*/node > 0);
    if (!prefix) return 0;
    GenerationParams gp = *S->base;
    GenerationStats st;
    memset(&st, 0, sizeof(st));
    gp.max_tokens          = S->opt.step_tokens < budget ? S->opt.step_tokens : budget;
    gp.sampler.temperature = S->opt.temperature;
    gp.seed                = (long long)(S->seed0 + (unsigned long long)(++S->expansions));
    gp.stop[0]             = S->opt.step_delim;
    gp.n_stop              = 1;
    gp.assistant_prefix    = prefix;
    gp.compute_logprobs    = !S->res.used_prm;
    gp.stats               = &st;
    Sbuf text              = {0};
    S->s->current_pos      = 0;
    generate_ex(S->cfg, S->w, S->s, S->mc, S->tok, S->tp, S->prompt, &gp, collect_cb, &text);
    S->res.tokens_total += st.completion_tokens;
    if (node == 0 && st.prompt_tokens > 0) S->prompt_tokens = st.prompt_tokens;
    if (st.finish_reason == GEN_FINISH_ERROR) {
        free(prefix);
        free(text.buf);
        t->v[node].terminal = 1; /* context full: nothing more below this node */
        backup(t, node, 0.0);
        return 1;
    }
    char *step = text.buf ? text.buf : (char *)calloc(1, 1);
    int child  = tree_add(t, node, step);
    if (child < 0) {
        free(prefix);
        free(step);
        return 0;
    }
    Node *cn   = &t->v[child];
    cn->tokens = st.completion_tokens;
    cn->ended  = st.hit_eos;
    cn->terminal =
        st.hit_eos || cn->depth >= S->opt.max_depth || used + cn->tokens >= S->base->max_tokens;
    double r = evaluate_step(S, child, &st);
    free(prefix);
    backup(t, child, r);
    return 1;
}

/* One simulation: select, then either re-evaluate a finished path or expand it. Returns 0 when
 * the search must stop. */
static int simulate_once(Search *S) {
    Tree *t  = &S->t;
    int node = select_leaf(S);
    S->res.simulations++;
    if (t->v[node].terminal) { /* re-evaluate a finished path */
        backup(t, node, node_mean(t, node));
        return 1;
    }
    int used = 0;
    for (int x = node; x > 0; x = t->v[x].parent) used += t->v[x].tokens;
    int budget = S->base->max_tokens - used;
    if (budget <= 0) {
        t->v[node].terminal = 1;
        backup(t, node, node_mean(t, node));
        return 1;
    }
    return expand_node(S, node, used, budget);
}

/* The final path: the most visited child at each level (ties: higher mean value). Counts the
 * steps taken and returns the last node. */
static int best_path(const Tree *t, int *steps) {
    int node = 0;
    while (t->v[node].n_children > 0) {
        int best = t->v[node].children[0];
        for (int k = 1; k < t->v[node].n_children; k++) {
            int c         = t->v[node].children[k];
            const Node *a = &t->v[c], *b = &t->v[best];
            if (a->visits > b->visits ||
                (a->visits == b->visits && a->value_sum / (a->visits ? a->visits : 1) >
                                               b->value_sum / (b->visits ? b->visits : 1)))
                best = c;
        }
        node = best;
        (*steps)++;
    }
    return node;
}

/* Completes the searched path with the caller's own settings (sampler, stop strings); with no
 * searched path (node 0) this is plain generation. */
static TernaryError complete_path(Search *S, int node, int remaining, TokenCallback callback,
                                  void *userdata) {
    char *prefix = path_text(&S->t, node, S->opt.step_delim, /*trailing=*/node > 0);
    if (!prefix) return TN_ERR_OOM;
    if (callback && node > 0) callback(S->opt.step_delim, userdata);
    GenerationParams gp = *S->base;
    gp.max_tokens       = remaining;
    gp.assistant_prefix = prefix;
    S->s->current_pos   = 0;
    generate_ex(S->cfg, S->w, S->s, S->mc, S->tok, S->tp, S->prompt, &gp, callback, userdata);
    free(prefix);
    return TN_OK;
}

/* The caller's stats describe the whole answer: searched steps + completion. */
static void report_stats(const Search *S, int node, int completed, int path_tokens,
                         int stop_stream) {
    GenerationStats *stats = S->base->stats;
    if (!stats) return;
    if (completed) {
        stats->completion_tokens += path_tokens;
        return;
    }
    stats->completion_tokens = path_tokens;
    stats->prompt_tokens     = S->prompt_tokens;
    stats->finish_reason     = stop_stream ? GEN_FINISH_CANCELLED : GEN_FINISH_STOP;
    stats->hit_eos           = node > 0 && S->t.v[node].ended;
}

/* Records the path's value and token total in the result; returns the tokens. */
static int summarize_path(Search *S, int node) {
    const Tree *t = &S->t;
    if (node > 0)
        S->res.path_value =
            (float)(t->v[node].value_sum / (t->v[node].visits ? t->v[node].visits : 1));
    int path_tokens = 0;
    for (int x = node; x > 0; x = t->v[x].parent) path_tokens += t->v[x].tokens;
    S->res.path_tokens = path_tokens;
    return path_tokens;
}

/* The model's own end of text ends the answer; so does a consumer that stopped the stream or a
 * path that already used up max_tokens. */
static int needs_completion(const Search *S, int node, int stop_stream, int remaining) {
    return !stop_stream && remaining > 0 && !(node > 0 && S->t.v[node].ended);
}

/* Streams the searched path, completes it when the model did not end it, fills the caller's stats
 * and the result. */
static TernaryError finish_search(Search *S, int node, TokenCallback callback, void *userdata,
                                  MctsResult *out) {
    int path_tokens = summarize_path(S, node);
    char *shown     = path_text(&S->t, node, S->opt.step_delim, /*trailing=*/0);
    int stop_stream = 0;
    if (shown && shown[0] && callback) stop_stream = callback(shown, userdata);
    int remaining   = S->base->max_tokens - path_tokens;
    int completed   = needs_completion(S, node, stop_stream, remaining);
    TernaryError rc = completed ? complete_path(S, node, remaining, callback, userdata) : TN_OK;
    report_stats(S, node, completed, path_tokens, stop_stream);
    free(shown);
    if (out) *out = S->res;
    return rc;
}

TernaryError generate_mcts(const Config *cfg, const TransformerWeights *w, RunState *s,
                           const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp, const char *prompt,
                           const char *question, const GenerationParams *base,
                           const MctsOptions *opt_in, PrmModel *prm, TokenCallback callback,
                           void *userdata, MctsAbortFn abort_fn, MctsResult *out) {
    if (!cfg || !w || !s || !tok || !prompt || !base || !opt_in) return TN_ERR_INVALID_ARGS;
    Search S;
    memset(&S, 0, sizeof(S));
    S.cfg      = cfg;
    S.w        = w;
    S.s        = s;
    S.mc       = mc;
    S.tok      = tok;
    S.tp       = tp;
    S.prompt   = prompt;
    S.question = question;
    S.base     = base;
    S.opt      = *opt_in;
    S.prm      = prm;
    mcts_options_defaults(&S.opt);
    S.res.used_prm = prm && prm->loaded;
    if (tree_add(&S.t, -1, NULL) != 0) return TN_ERR_OOM;
    S.seed0 = base->seed >= 0 ? (unsigned long long)base->seed : 0x5eedULL;

    for (int sim = 0; sim < S.opt.simulations; sim++) {
        if (abort_fn && abort_fn(userdata)) break;
        if (!simulate_once(&S)) break;
    }
    S.res.nodes     = S.t.n - 1;
    int steps       = 0;
    int node        = best_path(&S.t, &steps);
    S.res.steps     = steps;
    TernaryError rc = finish_search(&S, node, callback, userdata, out);
    tree_free(&S.t);
    return rc;
}
