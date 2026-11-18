# Re-run of the pz BitNet cells

The `bitnet-2b` cells of `pz` and `pz-int4` (timing, gate verdicts, screenshots) were measured again on 2026-11-18 with
pz `v0.2.0-23-g0148ad0` (master after #75, the activation-quantiser rounding fix). The first measurement, with
`v0.2.0-21-g864eeea-dirty`, gave perplexity 34.39 (INVALID, +9.76 % against the reference) because of the truncating
quantiser (docs/rca/2026-11-18-ternary-output-correctness.md). Every other cell of this folder was measured with the
earlier binary: their code paths do not use that quantiser (SmolLM2, Mamba) and the other engines are separate binaries.
