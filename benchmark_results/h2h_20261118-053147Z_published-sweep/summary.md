# Head-to-head decode benchmark

UTC start: 2026-11-18T18:31:20Z  |  host: Intel(R) Xeon(R) Processor @ 2.10GHz  |  pz commit: de7952e

| model | variant | threads | runs | best tok/s | mean tok/s | stdev | mean / reference | gate |
|---|---|---|---|---|---|---|---|---|
| bitnet-2b | bitnetcpp-fix | 1 | 3 | 6.12 | 6.08 | 0.06 | 93% | REF |
| bitnet-2b | pz | 1 | 3 | 6.61 | 6.55 | 0.10 | 100% | PASS |
| bitnet-2b | pz-int4 | 1 | 3 | 8.51 | 8.40 | 0.13 | 128% | APPROX |
| bitnet-2b | bitnetcpp-fix | 2 | 3 | 10.97 | 10.84 | 0.11 | 90% | REF |
| bitnet-2b | pz | 2 | 3 | 12.18 | 11.99 | 0.16 | 100% | PASS |
| bitnet-2b | pz-int4 | 2 | 3 | 15.60 | 15.49 | 0.20 | 129% | APPROX |
| bitnet-2b | bitnetcpp-fix | 3 | 3 | 15.76 | 15.66 | 0.12 | 94% | REF |
| bitnet-2b | pz | 3 | 3 | 16.87 | 16.65 | 0.27 | 100% | PASS |
| bitnet-2b | pz-int4 | 3 | 3 | 21.40 | 20.82 | 0.75 | 125% | APPROX |
| bitnet-2b | bitnetcpp-fix | 4 | 3 | 20.21 | 19.42 | 0.72 | 95% | REF |
| bitnet-2b | pz | 4 | 3 | 20.84 | 20.55 | 0.27 | 100% | PASS |
| bitnet-2b | pz-int4 | 4 | 3 | 26.74 | 25.28 | 1.37 | 123% | APPROX |
| bitnet-2b-iq2bn | ik | 1 | 3 | 6.26 | 6.11 | 0.14 | n/a | PASS |
| bitnet-2b-iq2bn | ik | 2 | 3 | 11.34 | 11.25 | 0.08 | n/a | PASS |
| bitnet-2b-iq2bn | ik | 3 | 3 | 16.12 | 16.05 | 0.07 | n/a | PASS |
| bitnet-2b-iq2bn | ik | 4 | 3 | 20.81 | 20.59 | 0.28 | n/a | PASS |
| mamba-130m | llamacpp | 1 | 3 | 26.94 | 26.77 | 0.14 | 86% | REF |
| mamba-130m | pz | 1 | 3 | 31.94 | 31.26 | 0.59 | 100% | PASS |
| mamba-130m | llamacpp | 2 | 3 | 51.53 | 50.44 | 1.64 | 103% | REF |
| mamba-130m | pz | 2 | 3 | 50.07 | 48.87 | 1.04 | 100% | PASS |
| mamba-130m | llamacpp | 3 | 3 | 74.89 | 71.46 | 3.30 | 114% | REF |
| mamba-130m | pz | 3 | 3 | 65.80 | 62.49 | 2.98 | 100% | PASS |
| mamba-130m | llamacpp | 4 | 3 | 96.74 | 93.95 | 2.58 | 140% | REF |
| mamba-130m | pz | 4 | 3 | 71.59 | 66.96 | 4.21 | 100% | PASS |
| smollm2-1.7b-q4km | ik | 1 | 3 | 6.33 | 6.13 | 0.20 | 95% | PASS |
| smollm2-1.7b-q4km | llamacpp | 1 | 3 | 5.15 | 4.99 | 0.17 | 78% | REF |
| smollm2-1.7b-q4km | pz | 1 | 3 | 6.45 | 6.43 | 0.03 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 1 | 3 | 6.92 | 6.81 | 0.18 | 106% | APPROX |
| smollm2-1.7b-q4km | ik | 2 | 3 | 12.08 | 11.57 | 0.49 | 101% | PASS |
| smollm2-1.7b-q4km | llamacpp | 2 | 3 | 9.16 | 8.90 | 0.25 | 78% | REF |
| smollm2-1.7b-q4km | pz | 2 | 3 | 11.92 | 11.43 | 0.51 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 2 | 3 | 12.50 | 11.99 | 0.55 | 105% | APPROX |
| smollm2-1.7b-q4km | ik | 3 | 3 | 14.71 | 14.51 | 0.21 | 93% | PASS |
| smollm2-1.7b-q4km | llamacpp | 3 | 3 | 13.23 | 13.03 | 0.24 | 83% | REF |
| smollm2-1.7b-q4km | pz | 3 | 3 | 16.08 | 15.67 | 0.63 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 3 | 3 | 17.07 | 16.58 | 0.48 | 106% | APPROX |
| smollm2-1.7b-q4km | ik | 4 | 3 | 22.34 | 21.87 | 0.43 | 118% | PASS |
| smollm2-1.7b-q4km | llamacpp | 4 | 3 | 17.63 | 17.13 | 0.47 | 92% | REF |
| smollm2-1.7b-q4km | pz | 4 | 3 | 18.77 | 18.54 | 0.21 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 4 | 3 | 20.31 | 18.96 | 1.17 | 102% | APPROX |
| smollm2-135m | ik | 1 | 3 | 28.76 | 27.85 | 1.50 | 84% | PASS |
| smollm2-135m | llamacpp | 1 | 3 | 23.89 | 22.91 | 0.87 | 69% | REF |
| smollm2-135m | pz | 1 | 3 | 33.46 | 33.06 | 0.48 | 100% | PASS |
| smollm2-135m | pz-int4 | 1 | 3 | 36.62 | 36.34 | 0.25 | 110% | APPROX |
| smollm2-135m | ik | 2 | 3 | 50.18 | 48.42 | 1.65 | 86% | PASS |
| smollm2-135m | llamacpp | 2 | 3 | 39.47 | 39.17 | 0.43 | 69% | REF |
| smollm2-135m | pz | 2 | 3 | 59.16 | 56.58 | 3.24 | 100% | PASS |
| smollm2-135m | pz-int4 | 2 | 3 | 66.09 | 63.96 | 2.47 | 113% | APPROX |
| smollm2-135m | ik | 3 | 3 | 72.61 | 70.02 | 2.50 | 92% | PASS |
| smollm2-135m | llamacpp | 3 | 3 | 62.53 | 59.26 | 3.41 | 78% | REF |
| smollm2-135m | pz | 3 | 3 | 80.62 | 75.90 | 4.26 | 100% | PASS |
| smollm2-135m | pz-int4 | 3 | 3 | 89.82 | 89.33 | 0.62 | 118% | APPROX |
| smollm2-135m | ik | 4 | 3 | 92.74 | 87.58 | 6.57 | 92% | PASS |
| smollm2-135m | llamacpp | 4 | 3 | 70.96 | 69.67 | 1.33 | 73% | REF |
| smollm2-135m | pz | 4 | 3 | 99.36 | 95.65 | 3.21 | 100% | PASS |
| smollm2-135m | pz-int4 | 4 | 3 | 105.34 | 100.80 | 5.87 | 105% | APPROX |


## Correctness gate

Mode strict; tolerance 1.0 % (WARN up to 2.0x); text `gate_text.txt` (sha256 8512a15e3523), 8 chunks of 512.

| model | variant | ppl | reference ppl | delta | smoke | verdict | timed |
|---|---|---|---|---|---|---|---|
| bitnet-2b-iq2bn | ik | 31.6011 | 31.3344 | +0.85 % | ok | PASS | yes |
| bitnet-2b | bitnetcpp | 48.6683 | - | - | WRONG | INVALID | no (fails the gate) |
| bitnet-2b | bitnetcpp-fix (modified) | 31.3344 | - | +0.00 % | ok | REF | yes |
| bitnet-2b | pz | 31.3668 | 31.3344 | +0.10 % | ok | PASS | yes |
| bitnet-2b | pz-int4 | 32.7205 | 31.3344 | +4.42 % | ok | APPROX | yes |
| mamba-130m | llamacpp | 12.5667 | - | +0.00 % | ok | REF | yes |
| mamba-130m | pz | 12.5801 | 12.5667 | +0.11 % | ok | PASS | yes |
| mamba-130m | pz-int4 | 84.8102 | - | - | WRONG | INVALID | no (fails the gate) |
| smollm2-1.7b-q4km | ik | 5.8999 | 5.9495 | -0.83 % | ok | PASS | yes |
| smollm2-1.7b-q4km | llamacpp | 5.9495 | - | +0.00 % | ok | REF | yes |
| smollm2-1.7b-q4km | pz | 5.9221 | 5.9495 | -0.46 % | ok | PASS | yes |
| smollm2-1.7b-q4km | pz-int4 | 6.3295 | 5.9495 | +6.39 % | ok | APPROX | yes |
| smollm2-135m | ik | 8.0256 | 8.0257 | -0.00 % | ok | PASS | yes |
| smollm2-135m | llamacpp | 8.0257 | - | +0.00 % | ok | REF | yes |
| smollm2-135m | pz | 8.0252 | 8.0257 | -0.01 % | ok | PASS | yes |
| smollm2-135m | pz-int4 | 9.3956 | 8.0257 | +17.07 % | ok | APPROX | yes |



## Screenshots

| model | variant | threads | screenshot |
|---|---|---|---|
| smollm2-135m | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194158Z_smollm2-135m__pz_t1.png |
| smollm2-135m | pz-int4 | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194217Z_smollm2-135m__pz-int4_t1.png |
| smollm2-135m | llamacpp | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145223Z_smollm2-135m__llamacpp_t1.png |
| smollm2-135m | ik | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145249Z_smollm2-135m__ik_t1.png |
| smollm2-1.7b-q4km | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194235Z_smollm2-1.7b-q4km__pz_t1.png |
| smollm2-1.7b-q4km | pz-int4 | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194403Z_smollm2-1.7b-q4km__pz-int4_t1.png |
| smollm2-1.7b-q4km | llamacpp | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145617Z_smollm2-1.7b-q4km__llamacpp_t1.png |
| smollm2-1.7b-q4km | ik | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145814Z_smollm2-1.7b-q4km__ik_t1.png |
| bitnet-2b | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194526Z_bitnet-2b__pz_t1.png |
| bitnet-2b | pz-int4 | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194648Z_bitnet-2b__pz-int4_t1.png |
| bitnet-2b | bitnetcpp-fix | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150422Z_bitnet-2b__bitnetcpp-fix_t1.png |
| bitnet-2b-iq2bn | ik | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150552Z_bitnet-2b-iq2bn__ik_t1.png |
| mamba-130m | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194801Z_mamba-130m__pz_t1.png |
| mamba-130m | llamacpp | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150753Z_mamba-130m__llamacpp_t1.png |
| smollm2-135m | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194819Z_smollm2-135m__pz_t2.png |
| smollm2-135m | pz-int4 | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194831Z_smollm2-135m__pz-int4_t2.png |
| smollm2-135m | llamacpp | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150839Z_smollm2-135m__llamacpp_t2.png |
| smollm2-135m | ik | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150856Z_smollm2-135m__ik_t2.png |
| smollm2-1.7b-q4km | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194842Z_smollm2-1.7b-q4km__pz_t2.png |
| smollm2-1.7b-q4km | pz-int4 | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-194931Z_smollm2-1.7b-q4km__pz-int4_t2.png |
| smollm2-1.7b-q4km | llamacpp | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151055Z_smollm2-1.7b-q4km__llamacpp_t2.png |
| smollm2-1.7b-q4km | ik | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151200Z_smollm2-1.7b-q4km__ik_t2.png |
| bitnet-2b | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195021Z_bitnet-2b__pz_t2.png |
| bitnet-2b | pz-int4 | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195108Z_bitnet-2b__pz-int4_t2.png |
| bitnet-2b | bitnetcpp-fix | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151526Z_bitnet-2b__bitnetcpp-fix_t2.png |
| bitnet-2b-iq2bn | ik | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151614Z_bitnet-2b-iq2bn__ik_t2.png |
| mamba-130m | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195152Z_mamba-130m__pz_t2.png |
| mamba-130m | llamacpp | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151735Z_mamba-130m__llamacpp_t2.png |
| smollm2-135m | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195205Z_smollm2-135m__pz_t3.png |
| smollm2-135m | pz-int4 | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195213Z_smollm2-135m__pz-int4_t3.png |
| smollm2-135m | llamacpp | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151809Z_smollm2-135m__llamacpp_t3.png |
| smollm2-135m | ik | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151822Z_smollm2-135m__ik_t3.png |
| smollm2-1.7b-q4km | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195222Z_smollm2-1.7b-q4km__pz_t3.png |
| smollm2-1.7b-q4km | pz-int4 | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195301Z_smollm2-1.7b-q4km__pz-int4_t3.png |
| smollm2-1.7b-q4km | llamacpp | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151953Z_smollm2-1.7b-q4km__llamacpp_t3.png |
| smollm2-1.7b-q4km | ik | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152043Z_smollm2-1.7b-q4km__ik_t3.png |
| bitnet-2b | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195338Z_bitnet-2b__pz_t3.png |
| bitnet-2b | pz-int4 | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195413Z_bitnet-2b__pz-int4_t3.png |
| bitnet-2b | bitnetcpp-fix | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152327Z_bitnet-2b__bitnetcpp-fix_t3.png |
| bitnet-2b-iq2bn | ik | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152403Z_bitnet-2b-iq2bn__ik_t3.png |
| mamba-130m | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195449Z_mamba-130m__pz_t3.png |
| mamba-130m | llamacpp | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152509Z_mamba-130m__llamacpp_t3.png |
| smollm2-135m | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195500Z_smollm2-135m__pz_t4.png |
| smollm2-135m | pz-int4 | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195509Z_smollm2-135m__pz-int4_t4.png |
| smollm2-135m | llamacpp | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152535Z_smollm2-135m__llamacpp_t4.png |
| smollm2-135m | ik | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152545Z_smollm2-135m__ik_t4.png |
| smollm2-1.7b-q4km | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195516Z_smollm2-1.7b-q4km__pz_t4.png |
| smollm2-1.7b-q4km | pz-int4 | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195550Z_smollm2-1.7b-q4km__pz-int4_t4.png |
| smollm2-1.7b-q4km | llamacpp | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152704Z_smollm2-1.7b-q4km__llamacpp_t4.png |
| smollm2-1.7b-q4km | ik | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152743Z_smollm2-1.7b-q4km__ik_t4.png |
| bitnet-2b | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195622Z_bitnet-2b__pz_t4.png |
| bitnet-2b | pz-int4 | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195649Z_bitnet-2b__pz-int4_t4.png |
| bitnet-2b | bitnetcpp-fix | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152951Z_bitnet-2b__bitnetcpp-fix_t4.png |
| bitnet-2b-iq2bn | ik | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-153019Z_bitnet-2b-iq2bn__ik_t4.png |
| mamba-130m | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-195720Z_mamba-130m__pz_t4.png |
| mamba-130m | llamacpp | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-153117Z_mamba-130m__llamacpp_t4.png |


## Chart

![throughput vs threads](throughput_vs_threads.png)
