# Project Zero — GitHub Growth Strategy (star-conversion playbook)

> Distribution plan for an **infrastructure** project (a CPU LLM engine), not a
> consumer app. The thesis: people star *claims backed by proof*, so every growth
> move is a benchmark narrative, not generic promotion. This doc is the actionable
> companion to the README rewrite (claim → benchmark table → visual proof → demo →
> credibility). Numbers come from `docs/PERFORMANCE_CEILING_REPORT.md` and
> `.claude/BENCHMARK_SUMMARY.md` — keep them in sync if benchmarks change.

## 0. The one move that matters first
Reposition the README as a *claim + proof machine* (done): honest headline claim,
above-the-fold benchmark table, visual proof (`docs/benchmark_terminal.png` +
OpenBenchmarking screenshots), a one-command `make demo`, and exposed audit/QA
reports. Everything below amplifies that; none of it works without it.

---

## 1. GitHub-native growth loops (highest leverage, lowest cost)
GitHub promotes active repos, and Discussions are durable / discoverable. Seed them
with **benchmark-driven** threads (the empty Discussions tab is a missed signal).

> ⚙️ **Note:** these Discussions could not be auto-created from this environment (the
> available GitHub tooling exposes Issues/PRs, not the Discussions GraphQL API).
> Create them manually — copy-paste-ready bodies below.

### Discussion 1 — "Project Zero vs llama.cpp — post your CPU results" (category: Show and tell / Polls)
```
Project Zero is a pure-C CPU LLM engine. On an Intel Xeon (2.10 GHz, AVX-512 VNNI)
the same binary hits 40.4 tok/s on BitNet b1.58-2B-4T and 142 tok/s on SmolLM2-135M.

Run it on YOUR CPU and drop your numbers here:
    git clone https://github.com/shifulegend/project-zero && cd project-zero && make demo

Please include: CPU model, core/thread count, RAM type, the `[gen] … tok/s` line,
and the auto-detected SIMD backend. I'll keep a leaderboard table in this thread.
```

### Discussion 2 — "Can we beat ARM Mac performance?" (category: Ideas)
```
project-zero auto-dispatches to NEON/dotprod on Apple Silicon but has only been
tuned on x86 (AVX2..VNNI). Looking for M1/M2/M3/M4 numbers + perf-counter traces.
Where's the bottleneck on ARM — memory bandwidth, the ternary unpack, or thread
scheduling? Post `make demo` output + `sysctl -n machdep.cpu.brand_string`.
```

### Discussion 3 — "Worst-case / performance-regression thread" (category: General)
```
Honesty thread. Where does project-zero LOSE today?
Known: DeepSeek-V2 MoE is ~7× behind llama.cpp (expert-scatter penalty), and dense
models lose at T=1–2 (BLAS edge). Post the configs where it underperforms on your
hardware so we can prioritize — regressions and ugly numbers welcome here.
```

> Wire these into the README: the "post your results" links already point at
> `/discussions`. Keep a live leaderboard table in Discussion 1.

---

## 2. Piggyback on bigger repos (most practical reach)
The audience already lives in the `llama.cpp` / `bitnet.cpp` ecosystems. **Contribute
a benchmark, don't advertise.** Draft (post to a relevant llama.cpp discussion, not a
drive-by issue):

```
Title: Benchmarked a from-scratch pure-C CPU engine vs llama.cpp on the same box

I tested project-zero (single C binary, no BLAS) against llama.cpp on an Intel
Xeon / i5. Methodology + raw numbers: <link to BENCHMARK_REPORT.md>. Headline: it's
+50% at T=4 on SmolLM2-135M dense and 1.8× over bitnet.cpp on BitNet, but ~7× behind
llama.cpp on DeepSeek MoE (expert scatter). Sharing the ternary-kernel approach in
case it's useful upstream — feedback on the MoE gap especially welcome.
```
Rules of engagement: lead with reproducible numbers, link methodology, disclose the
losses, never spam. One good benchmark post > ten promo links.

> ⚠️ Outward-facing: post these yourself (or explicitly authorize it). This repo's
> automation will not post to external repositories.

---

## 3. Leverage OpenBenchmarking + the hardware niche
Already live: [Xeon result](https://openbenchmarking.org/result/2606063-SHIF-PROJECT91)
and [i5 result](https://openbenchmarking.org/result/2606062-SHIF-PROJECT21) (screenshots
in `docs/`). Keep uploading new hardware profiles as community results land — each one
is independent, third-party-hosted credibility and is itself discoverable.

---

## 4. Credibility triggers (already exposed in the README)
Experienced devs star projects that look *serious*. The README now links the
independent code audit, QA strategy, full QA report (3,367 assertions), security
findings, performance-ceiling analysis, and regression verification. Keep these
current — a stale audit hurts more than no audit.

---

## 5. What will NOT work (don't waste effort)
- ❌ **Product Hunt** — audience mismatch (infra, not a consumer app).
- ❌ **Generic "AI tools" lists** — you're infrastructure, not an app.
- ❌ **SEO blog posts** — low conversion for systems tooling.
- ❌ **Influencer outreach** — wrong niche.

---

## 6. Realistic expectations (no hype)
- README + demo fixed → **0 → 20–50 stars**.
- Benchmark narratives + ecosystem posts → **50 → 200**.
- Sustained community-benchmark loop → **200+**.

Infra projects grow slower than viral apps but more durably. The compounding asset is
the benchmark leaderboard: every contributed result is both proof and reach.
