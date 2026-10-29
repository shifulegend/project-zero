# SmolLM2-135M: Project Zero vs llama.cpp — 2026-10-22

Intel Xeon @ 2.10 GHz, 4 vCPU. SmolLM2-135M-Instruct F16, 300 tokens, temperature 0, best of 3,
09:06–09:12 UTC. Superseded by [sweep_h2h_2026-10-23](../sweep_h2h_2026-10-23/) on the same machine.

| Threads | Project Zero | llama.cpp |
|---|---|---|
| 1 | 28.98 | 25.75 |
| 2 | 57.08 | 48.52 |
| 3 | 80.16 | 76.41 |
| 4 | 85.76 | 88.85 |

The screenshots in `screenshots/` are single runs from 17:36–17:38 UTC, so their numbers differ
from the table.
