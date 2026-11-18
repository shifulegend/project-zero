# Build comparison (native, portable; reference native)

UTC start: 2026-11-18T05:22:42Z  |  host: Intel(R) Xeon(R) Processor @ 2.10GHz  |  pz commit: 864eeea

| model | variant | threads | runs | best tok/s | mean tok/s | stdev | mean / reference |
|---|---|---|---|---|---|---|---|
| bitnet-2b | native | 4 | 5 | 23.69 | 22.81 | 0.69 | 100% |
| bitnet-2b | portable | 4 | 5 | 22.59 | 22.28 | 0.31 | 98% |
| smollm2-135m | native | 4 | 5 | 120.16 | 113.05 | 4.67 | 100% |
| smollm2-135m | portable | 4 | 5 | 105.28 | 100.29 | 5.41 | 89% |


## Per-step time, bitnet-2b (64 tokens, total ms)

| step | native (ms) | portable (ms) | portable / native |
|---|---|---|---|
| Tokenization | 0 | 0 | 0.97x |
| BOS injection / chat template | 0 | 0 | 0.35x |
| Token embedding | 0 | 0 | 1.20x |
| Pre-attention RMSNorm | 5 | 5 | 1.04x |
| Q projection | 308 | 305 | 0.99x |
| YaRN RoPE | 36 | 37 | 1.03x |
| KV cache write | 7 | 6 | 0.86x |
| Attention score / softmax | 98 | 96 | 0.98x |
| Post-attention output / residual | 199 | 192 | 0.96x |
| Pre-FFN RMSNorm | 4 | 4 | 0.98x |
| Dense FFN | 1411 | 1426 | 1.01x |
| Final RMSNorm | 0 | 0 | 1.10x |
| LM head | 1984 | 2198 | 1.11x |
| Greedy sampling | 17 | 17 | 1.01x |
| EOS check | 0 | 0 | 0.59x |


## check_dist_isa.sh

| binary | result |
|---|---|
| portable | ok |


## Screenshots

| model | binary | threads | screenshot |
|---|---|---|---|
| bitnet-2b | native | 4 | benchmark_results/build_compare_20261118-052242Z_native-vs-portable/screenshots/20261118-052640Z_bitnet-2b__native_t4.png |
| bitnet-2b | portable | 4 | benchmark_results/build_compare_20261118-052242Z_native-vs-portable/screenshots/20261118-052646Z_bitnet-2b__portable_t4.png |
| smollm2-135m | native | 4 | benchmark_results/build_compare_20261118-052242Z_native-vs-portable/screenshots/20261118-052652Z_smollm2-135m__native_t4.png |
| smollm2-135m | portable | 4 | benchmark_results/build_compare_20261118-052242Z_native-vs-portable/screenshots/20261118-052655Z_smollm2-135m__portable_t4.png |

