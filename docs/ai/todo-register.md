# TODO register

Every `TODO` / `FIXME` in the code carries an ID in parentheses, `TODO(PZ-TODO-<n>)` or
`TODO(QA-ISS-<n>)`, and that ID is listed here with what is missing and why it is deferred.
`tools/check_todos.py` (CI) fails on a TODO without an ID or with an ID that is not listed.
Remove the entry and the comment together when the work is done.

| ID | Where | What is missing | Why deferred |
|---|---|---|---|
| PZ-TODO-1 | `src/runtime/model_load.c`, `src/cli/main.c` (native MoE branch) | The weight-memory estimate bills all experts of an MoE model, not only the routed ones; subtract inactive-expert bytes via `MoEConfig`. | The estimate is conservative (it over-counts), so nothing breaks; accuracy only matters for tight-RAM MoE loads. |
| PZ-TODO-2 | `src/core/calibration.c` | The memory-bandwidth base used by calibration is not model-aware. | Needs a calibration redesign: the cache is keyed on hardware, not on the model. |
| QA-ISS-005 | `include/math/simd_dispatch.h` | SIMD availability is decided at compile time; a binary built with an ISA flag crashes (SIGILL) on a CPU without it. | Needs runtime CPUID selection of kernels; the portable `make dist` build already avoids it for releases. |
