---
name: Performance Regression
about: A recent change made the engine slower
title: '[PERF] Regression: '
labels: performance, regression
assignees: ''
---

## What Got Slower

| Metric | Before | After | Delta |
|---|---|---|---|
| tok/s | | | |
| Commit before | | — | — |
| Commit after | — | | — |

## Model & Config Used
```bash
# Exact command
./adaptive_ai_engine --model ... --threads ... --classifier ... --prompt "..."
```

## Environment
| | Value |
|---|---|
| CPU | |
| OS | |
| Compiler | |
| SIMD detected | <!-- check engine startup header --> |

## Bisect Result (if known)
<!-- Which commit introduced the regression? `git bisect` output -->
