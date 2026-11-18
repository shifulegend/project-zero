# Build comparison (old, new; reference new)

UTC start: 2026-11-18T15:35:55Z  |  host: Intel(R) Xeon(R) Processor @ 2.10GHz  |  pz commit: 09fb6ef

| model | variant | threads | runs | best tok/s | mean tok/s | stdev | mean / reference | gate |
|---|---|---|---|---|---|---|---|---|
| bitnet-2b | new | 4 | 5 | 21.48 | 20.59 | 0.96 | 100% | REF |
| bitnet-2b | old | 4 | 5 | n/a | n/a | n/a | n/a | INVALID |
| smollm2-135m | new | 4 | 5 | 90.81 | 83.91 | 4.82 | 100% | REF |
| smollm2-135m | old | 4 | 5 | 87.72 | 83.16 | 4.26 | 99% | PASS |


## Correctness gate

Mode strict; tolerance 1.0 % (WARN up to 2.0x); text `gate_text.txt` (sha256 8512a15e3523), 8 chunks of 512.

| model | variant | ppl | reference ppl | delta | smoke | verdict |
|---|---|---|---|---|---|---|
| bitnet-2b | new | 31.3668 | - | +0.00 % | ok | REF |
| bitnet-2b | old | 34.3926 | 31.3668 | +9.65 % | ok | INVALID |
| smollm2-135m | new | 8.0252 | - | +0.00 % | ok | REF |
| smollm2-135m | old | 8.0252 | 8.0252 | +0.00 % | ok | PASS |



## Per-step time, bitnet-2b (64 tokens, total ms)

| step | old (ms) | new (ms) | old / new |
|---|---|---|---|
| Tokenization | 0 | 0 | 0.94x |
| BOS injection / chat template | 0 | 0 | 1.50x |
| Token embedding | 0 | 0 | 0.82x |
| Pre-attention RMSNorm | 6 | 7 | 0.95x |
| Q projection | 393 | 474 | 0.83x |
| YaRN RoPE | 52 | 39 | 1.33x |
| KV cache write | 7 | 7 | 0.97x |
| Attention score / softmax | 122 | 119 | 1.02x |
| Post-attention output / residual | 228 | 259 | 0.88x |
| Pre-FFN RMSNorm | 4 | 5 | 0.91x |
| Dense FFN | 1561 | 1704 | 0.92x |
| Final RMSNorm | 0 | 0 | 1.20x |
| LM head | 2325 | 2446 | 0.95x |
| Greedy sampling | 18 | 19 | 0.98x |
| EOS check | 0 | 0 | 1.34x |

