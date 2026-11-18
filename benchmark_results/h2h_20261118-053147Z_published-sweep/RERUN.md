# Re-run of every pz cell

All `pz` and `pz-int4` cells (timing, gate verdicts, screenshots) were measured again on 2026-11-18 with pz
`v0.2.0-23-g0148ad0`, a clean build of master after #75 (the activation-quantiser rounding fix).

Why every pz cell, not only BitNet: the first measurement used a binary whose version string ended in `-dirty`
(`v0.2.0-21-g864eeea-dirty`), i.e. a build of uncommitted local state that cannot be reproduced from any commit.
It ran the int4 LM head at 26.5 tok/s on BitNet; clean builds of both 864eeea and 0148ad0 give 19.5 to 21.2 tok/s,
so those numbers were not reproducible. The first measurement also carried the truncating quantiser
(BitNet perplexity 34.39, INVALID; docs/rca/2026-11-18-ternary-output-correctness.md). The other engines are separate
binaries and were not re-measured.
