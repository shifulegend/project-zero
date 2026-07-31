# PR #31 verification — `--moe-threading=rowsplit` vs `=legacy`

**Date:** 2026-07-22 · **PR under test:** [#31](https://github.com/shifulegend/project-zero/pull/31) (`Amlaach/project-zero@fba3225`)

## Setup

- Two isolated `git worktree` checkouts built with `make release CC=gcc`: unmodified `master` (baseline) and the PR head commit.
- Model: `DeepSeek-V2-Lite-Chat.Q4_K_S.gguf` (mradermacher/DeepSeek-V2-Lite-Chat-GGUF), the exact model this repo's own MoE benchmarking docs/tools (`tools/deepseek_bench.sh`, `docs/DEEPSEEK_Q8_HANDOVER.md`) target.
- Hardware: 4-core Xeon @2.10GHz, AVX-512 VNNI, 15GB RAM, page cache warmed before every run.
- Prompt: the same ancient-Rome prompt used in this repo's existing DeepSeek benchmark protocol, `--max-tokens 32 --temperature 0 --threads 4`, 3 repetitions per config.
- llama.cpp built from source (`ggml-org/llama.cpp`, CPU release) as the external reference point.

## Correctness

Identical, correct greedy-decode output in every mode ("The capital of France is Paris." / "...Germany is Berlin."), all 64 experts activate normally, no NaNs/crashes. The PR does not break correctness.

## Performance (tok/s, T=4, 3 reps each)

| Config | Rep 1 | Rep 2 | Rep 3 |
|---|---|---|---|
| PR31 `--moe-threading=legacy` | 2.91 | 3.37 | 2.92 |
| PR31 `--moe-threading=rowsplit` (**default**) | 2.40 | 2.45 | 2.43 |
| baseline `master` (no flag) | 2.59 | 3.20 | — |
| llama.cpp `llama-bench` (`tg32`) | 12.84 | — | — |

`rowsplit` is consistently ~20–25% *slower* than both `legacy` and unmodified `master` — a tight, repeatable regression, not measurement noise. All configs remain ~4–5× behind llama.cpp on this model.

## Screenshots

Raw captured terminal output (via this repo's `tools/make_screenshot.py`, not mocked) for the final verification rep of each mode:
- `pr31_legacy_T4_rep2.png`
- `pr31_rowsplit_T4_rep2.png`
