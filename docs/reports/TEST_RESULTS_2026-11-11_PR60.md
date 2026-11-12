# Test Results — PR #60 (chat content parts) and follow-ups — 2026-11-11

> Scope: the fix from PR #60 (carried to `master` as `1018faa`, author takkkt), and two follow-up
> fixes found while testing it. **No screenshots exist for this work**: a screenshot run was started
> on a cloud machine that restarted before it could push, so only text results are recorded here.

## What was tested and how

| Source | What it is |
|---|---|
| **Run 1** (cloud machine, gcc/clang, three models) | Full test suite and edge cases on the PR head vs `master`. Results are from that run's written report, not re-run. |
| **Independent parser harness** (this server) | The JSON parser built alone with ASan/UBSan, `master` vs PR, 16 inputs. Run and read directly. |
| **CI on GitHub** | The 12 required checks on each merged PR (build, sanitizers, TSan, coverage, golden output, installer). |

## Result: the fix works

- **Bug reproduced on `master`:** the same prompt as one string vs two text parts gave 82 vs 41 prompt tokens (only the last part survived). With the PR both gave 82; greedy replies were identical (run 1).
- **Harness (read directly):** two parts → `master` keeps `CD`, PR gives `AB CD`; 1,000 parts of 100 bytes → 100 bytes on `master`, 100,000 on the PR; text/image/text plus a trailing empty part → **empty on `master`**, joined on the PR. Escapes, unicode and emoji correct; no memory error on any input.
- **New test:** `test_json_parse_content_parts` fails on `master` and passes on the PR (run 1, gcc); clang ASan/UBSan suite passes.
- **Other checks on the PR (run 1):** `make test` gcc and clang, release and debug builds, TSan (no races), coverage 51.2% of lines, CMake build, changelog check, installer test (24 passed), golden output (26), JSON mode (17), prompt cache, concurrency, parallel serving, prompt-injection (11/11), batch scheduler (13/13), Mamba reference check.
- **Three models:** the same prompt as one string vs parts gave identical token counts and replies on SmolLM2 135M, SmolLM2 1.7B and Falcon3 3B; a prompt-injection string in the second part was neutralised the same way as in one string.
- **Edge cases (run 1, ASan build):** empty, image-only, 1,000-part, oversized (413 over the limit), wrong-type, truncated JSON, multi-message and plain-string requests all gave a clean 200 or 4xx. Memory stayed flat over 100 requests (1,276 → 1,284 MB). Speed: 72.7 tok/s on `master` vs 70.8 with the PR (run-to-run noise).
- **CI:** all 12 checks passed on #64 before it was merged.

## Findings that were not part of the PR

| Finding | Status |
|---|---|
| A repeated `content` key leaked the earlier text (2 sites in `json_parse.c`; on `master` and the PR) | Fixed in #63 (`7ba31f9`) with a regression test that fails on the old parser under LeakSanitizer |
| Sequential requests sometimes got `429` because the generation slot was released after the response was sent (run 1: 51 of 300 on `master` in its harness; 45% for a streaming client that stops at `[DONE]` on a 2-core server, measured separately) | Fixed in #65 (`fefa37e`); see `benchmark_results/slot_release_race_2026-11-11/` |
| A repeated `content` key also changed behaviour with the PR (the texts joined) | Resolved by #63: the last key wins, as on `master` |
| `CHAT_MAX_CONTENT` (64 KiB) is not enforced for any content form; the 8 MiB body limit applies | Open, unchanged by the PR |
| `tests/security_fuzzer.py` and `tests/monkey_tester.py` do not exercise the chat parser | Noted only |

## Not done

- Valgrind covered only part of the edge cases in run 1.
- The clang run on `master` (to show the new test fails there under clang) was not completed in run 1.
- A second run with screenshots and per-test transcripts was started but could not be saved.
