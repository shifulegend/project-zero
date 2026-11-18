# Re-run of the pz cells, and the released bitnet.cpp cells

- Every `pz` and `pz-int4` cell (timing, gate verdicts, screenshots) was measured again on 2026-11-18 with a clean
  build of the final code (`git describe` without `-dirty`; the version is in system_info.txt). The first measurement
  used `v0.2.0-21-g864eeea-dirty`, a build of uncommitted state that cannot be reproduced from any commit (it gave the
  int4 head 26.5 tok/s where clean builds of that commit give about 20), and it still carried the truncating activation
  quantiser (BitNet perplexity 34.39; #75) and the per-block-reduction LM-head kernels (#76).
- The released bitnet.cpp (`bitnetcpp`) computes the BitNet feed-forward block with SiLU instead of squared-ReLU and
  fails the correctness gate (docs/rca/2026-11-18-ternary-output-correctness.md). It is no longer timed or screenshotted;
  its verdict stays in gate.json. The modified build `bitnetcpp-fix` is the BitNet comparison.
