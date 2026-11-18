# Head-to-head decode benchmark

UTC start: 2026-11-18T05:15:37Z  |  host: Intel(R) Xeon(R) Processor @ 2.10GHz  |  pz commit: 864eeea

| model | variant | threads | runs | best tok/s | mean tok/s | stdev | mean / reference |
|---|---|---|---|---|---|---|---|
| bitnet-2b | bitnetcpp | 4 | 3 | 22.96 | 22.72 | 0.30 | 92% |
| bitnet-2b | pz | 4 | 3 | 25.19 | 24.73 | 0.56 | 100% |
| bitnet-2b-iq2bn | ik | 4 | 3 | 17.29 | 16.65 | 0.59 | - |
| smollm2-1.7b-q4km | ik | 4 | 3 | 26.26 | 25.74 | 0.66 | 118% |
| smollm2-1.7b-q4km | llamacpp | 4 | 3 | 20.53 | 20.33 | 0.34 | 93% |
| smollm2-1.7b-q4km | pz | 4 | 3 | 22.87 | 21.89 | 0.88 | 100% |
| smollm2-135m | ik | 4 | 3 | 95.54 | 91.53 | 3.57 | 79% |
| smollm2-135m | llamacpp | 4 | 3 | 104.20 | 97.95 | 5.92 | 84% |
| smollm2-135m | pz | 4 | 3 | 120.34 | 115.92 | 7.09 | 100% |


## Screenshots

| model | variant | threads | screenshot |
|---|---|---|---|
| smollm2-135m | pz | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052157Z_smollm2-135m__pz_t4.png |
| smollm2-135m | llamacpp | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052159Z_smollm2-135m__llamacpp_t4.png |
| smollm2-135m | ik | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052202Z_smollm2-135m__ik_t4.png |
| smollm2-1.7b-q4km | pz | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052204Z_smollm2-1.7b-q4km__pz_t4.png |
| smollm2-1.7b-q4km | llamacpp | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052211Z_smollm2-1.7b-q4km__llamacpp_t4.png |
| smollm2-1.7b-q4km | ik | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052217Z_smollm2-1.7b-q4km__ik_t4.png |
| bitnet-2b | pz | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052223Z_bitnet-2b__pz_t4.png |
| bitnet-2b | bitnetcpp | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052230Z_bitnet-2b__bitnetcpp_t4.png |
| bitnet-2b-iq2bn | ik | 4 | benchmark_results/h2h_20261118-051537Z_four-engines/screenshots/20261118-052235Z_bitnet-2b-iq2bn__ik_t4.png |

