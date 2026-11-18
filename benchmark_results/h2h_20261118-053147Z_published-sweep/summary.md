# Head-to-head decode benchmark

UTC start: 2026-11-18T14:40:04Z  |  host: Intel(R) Xeon(R) Processor @ 2.10GHz  |  pz commit: 78689ec

| model | variant | threads | runs | best tok/s | mean tok/s | stdev | mean / reference | gate |
|---|---|---|---|---|---|---|---|---|
| bitnet-2b | bitnetcpp | 1 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | bitnetcpp-fix | 1 | 3 | 6.12 | 6.08 | 0.06 | 93% | REF |
| bitnet-2b | pz | 1 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | pz-int4 | 1 | 3 | 7.76 | 7.60 | 0.20 | 117% | APPROX |
| bitnet-2b | bitnetcpp | 2 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | bitnetcpp-fix | 2 | 3 | 10.97 | 10.84 | 0.11 | 89% | REF |
| bitnet-2b | pz | 2 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | pz-int4 | 2 | 3 | 14.49 | 14.27 | 0.37 | 117% | APPROX |
| bitnet-2b | bitnetcpp | 3 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | bitnetcpp-fix | 3 | 3 | 15.76 | 15.66 | 0.12 | 92% | REF |
| bitnet-2b | pz | 3 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | pz-int4 | 3 | 3 | 21.24 | 20.53 | 0.63 | 120% | APPROX |
| bitnet-2b | bitnetcpp | 4 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | bitnetcpp-fix | 4 | 3 | 20.21 | 19.42 | 0.72 | 87% | REF |
| bitnet-2b | pz | 4 | 3 | n/a | n/a | n/a | n/a | INVALID |
| bitnet-2b | pz-int4 | 4 | 3 | 26.57 | 26.46 | 0.10 | 119% | APPROX |
| bitnet-2b-iq2bn | ik | 1 | 3 | 6.26 | 6.11 | 0.14 | n/a | PASS |
| bitnet-2b-iq2bn | ik | 2 | 3 | 11.34 | 11.25 | 0.08 | n/a | PASS |
| bitnet-2b-iq2bn | ik | 3 | 3 | 16.12 | 16.05 | 0.07 | n/a | PASS |
| bitnet-2b-iq2bn | ik | 4 | 3 | 20.81 | 20.59 | 0.28 | n/a | PASS |
| mamba-130m | llamacpp | 1 | 3 | 26.94 | 26.77 | 0.14 | 90% | REF |
| mamba-130m | pz | 1 | 3 | 30.29 | 29.87 | 0.72 | 100% | PASS |
| mamba-130m | pz-int4 | 1 | 3 | n/a | n/a | n/a | n/a | INVALID |
| mamba-130m | llamacpp | 2 | 3 | 51.53 | 50.44 | 1.64 | 103% | REF |
| mamba-130m | pz | 2 | 3 | 50.18 | 48.99 | 1.06 | 100% | PASS |
| mamba-130m | pz-int4 | 2 | 3 | n/a | n/a | n/a | n/a | INVALID |
| mamba-130m | llamacpp | 3 | 3 | 74.89 | 71.46 | 3.30 | 113% | REF |
| mamba-130m | pz | 3 | 3 | 66.33 | 63.31 | 2.62 | 100% | PASS |
| mamba-130m | pz-int4 | 3 | 3 | n/a | n/a | n/a | n/a | INVALID |
| mamba-130m | llamacpp | 4 | 3 | 96.74 | 93.95 | 2.58 | 125% | REF |
| mamba-130m | pz | 4 | 3 | 77.29 | 74.90 | 3.96 | 100% | PASS |
| mamba-130m | pz-int4 | 4 | 3 | n/a | n/a | n/a | n/a | INVALID |
| smollm2-1.7b-q4km | ik | 1 | 3 | 6.33 | 6.13 | 0.20 | 123% | PASS |
| smollm2-1.7b-q4km | llamacpp | 1 | 3 | 5.15 | 4.99 | 0.17 | 100% | REF |
| smollm2-1.7b-q4km | pz | 1 | 3 | 5.15 | 4.97 | 0.16 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 1 | 3 | 5.33 | 5.16 | 0.15 | 104% | APPROX |
| smollm2-1.7b-q4km | ik | 2 | 3 | 12.08 | 11.57 | 0.49 | 121% | PASS |
| smollm2-1.7b-q4km | llamacpp | 2 | 3 | 9.16 | 8.90 | 0.25 | 93% | REF |
| smollm2-1.7b-q4km | pz | 2 | 3 | 9.96 | 9.54 | 0.37 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 2 | 3 | 10.15 | 9.85 | 0.31 | 103% | APPROX |
| smollm2-1.7b-q4km | ik | 3 | 3 | 14.71 | 14.51 | 0.21 | 105% | PASS |
| smollm2-1.7b-q4km | llamacpp | 3 | 3 | 13.23 | 13.03 | 0.24 | 94% | REF |
| smollm2-1.7b-q4km | pz | 3 | 3 | 14.01 | 13.80 | 0.26 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 3 | 3 | 14.77 | 14.31 | 0.49 | 104% | APPROX |
| smollm2-1.7b-q4km | ik | 4 | 3 | 22.34 | 21.87 | 0.43 | 123% | PASS |
| smollm2-1.7b-q4km | llamacpp | 4 | 3 | 17.63 | 17.13 | 0.47 | 97% | REF |
| smollm2-1.7b-q4km | pz | 4 | 3 | 18.71 | 17.72 | 0.85 | 100% | PASS |
| smollm2-1.7b-q4km | pz-int4 | 4 | 3 | 18.85 | 17.82 | 0.89 | 101% | APPROX |
| smollm2-135m | ik | 1 | 3 | 28.76 | 27.85 | 1.50 | 89% | PASS |
| smollm2-135m | llamacpp | 1 | 3 | 23.89 | 22.91 | 0.87 | 73% | REF |
| smollm2-135m | pz | 1 | 3 | 35.14 | 31.38 | 3.28 | 100% | PASS |
| smollm2-135m | pz-int4 | 1 | 3 | 34.34 | 32.61 | 1.95 | 104% | APPROX |
| smollm2-135m | ik | 2 | 3 | 50.18 | 48.42 | 1.65 | 83% | PASS |
| smollm2-135m | llamacpp | 2 | 3 | 39.47 | 39.17 | 0.43 | 67% | REF |
| smollm2-135m | pz | 2 | 3 | 61.88 | 58.54 | 3.80 | 100% | PASS |
| smollm2-135m | pz-int4 | 2 | 3 | 61.13 | 57.83 | 5.57 | 99% | APPROX |
| smollm2-135m | ik | 3 | 3 | 72.61 | 70.02 | 2.50 | 84% | PASS |
| smollm2-135m | llamacpp | 3 | 3 | 62.53 | 59.26 | 3.41 | 71% | REF |
| smollm2-135m | pz | 3 | 3 | 88.90 | 83.16 | 6.58 | 100% | PASS |
| smollm2-135m | pz-int4 | 3 | 3 | 96.56 | 89.98 | 5.88 | 108% | APPROX |
| smollm2-135m | ik | 4 | 3 | 92.74 | 87.58 | 6.57 | 85% | PASS |
| smollm2-135m | llamacpp | 4 | 3 | 70.96 | 69.67 | 1.33 | 68% | REF |
| smollm2-135m | pz | 4 | 3 | 103.03 | 102.49 | 0.54 | 100% | PASS |
| smollm2-135m | pz-int4 | 4 | 3 | 127.40 | 114.87 | 10.90 | 112% | APPROX |


## Correctness gate

Mode strict; tolerance 1.0 % (WARN up to 2.0x); text `gate_text.txt` (sha256 8512a15e3523), 8 chunks of 512.

| model | variant | ppl | reference ppl | delta | smoke | verdict |
|---|---|---|---|---|---|---|
| bitnet-2b-iq2bn | ik | 31.6011 | 31.3344 | +0.85 % | ok | PASS |
| bitnet-2b | bitnetcpp | 48.6683 | - | - | WRONG | INVALID |
| bitnet-2b | bitnetcpp-fix (modified) | 31.3344 | - | +0.00 % | ok | REF |
| bitnet-2b | pz | 34.3926 | 31.3344 | +9.76 % | ok | INVALID |
| bitnet-2b | pz-int4 | 35.6045 | 31.3344 | +13.63 % | ok | APPROX |
| mamba-130m | llamacpp | 12.5667 | - | +0.00 % | ok | REF |
| mamba-130m | pz | 12.5801 | 12.5667 | +0.11 % | ok | PASS |
| mamba-130m | pz-int4 | 84.8103 | - | - | WRONG | INVALID |
| smollm2-1.7b-q4km | ik | 5.8999 | 5.9495 | -0.83 % | ok | PASS |
| smollm2-1.7b-q4km | llamacpp | 5.9495 | - | +0.00 % | ok | REF |
| smollm2-1.7b-q4km | pz | 5.9221 | 5.9495 | -0.46 % | ok | PASS |
| smollm2-1.7b-q4km | pz-int4 | 6.3295 | 5.9495 | +6.39 % | ok | APPROX |
| smollm2-135m | ik | 8.0256 | 8.0257 | -0.00 % | ok | PASS |
| smollm2-135m | llamacpp | 8.0257 | - | +0.00 % | ok | REF |
| smollm2-135m | pz | 8.0252 | 8.0257 | -0.01 % | ok | PASS |
| smollm2-135m | pz-int4 | 9.3956 | 8.0257 | +17.07 % | ok | APPROX |



## Screenshots

| model | variant | threads | screenshot |
|---|---|---|---|
| smollm2-135m | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145145Z_smollm2-135m__pz_t1.png |
| smollm2-135m | pz-int4 | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145204Z_smollm2-135m__pz-int4_t1.png |
| smollm2-135m | llamacpp | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145223Z_smollm2-135m__llamacpp_t1.png |
| smollm2-135m | ik | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145249Z_smollm2-135m__ik_t1.png |
| smollm2-1.7b-q4km | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145312Z_smollm2-1.7b-q4km__pz_t1.png |
| smollm2-1.7b-q4km | pz-int4 | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145444Z_smollm2-1.7b-q4km__pz-int4_t1.png |
| smollm2-1.7b-q4km | llamacpp | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145617Z_smollm2-1.7b-q4km__llamacpp_t1.png |
| smollm2-1.7b-q4km | ik | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145814Z_smollm2-1.7b-q4km__ik_t1.png |
| bitnet-2b | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-145942Z_bitnet-2b__pz_t1.png |
| bitnet-2b | pz-int4 | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150110Z_bitnet-2b__pz-int4_t1.png |
| bitnet-2b | bitnetcpp | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150254Z_bitnet-2b__bitnetcpp_t1.png |
| bitnet-2b | bitnetcpp-fix | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150422Z_bitnet-2b__bitnetcpp-fix_t1.png |
| bitnet-2b-iq2bn | ik | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150552Z_bitnet-2b-iq2bn__ik_t1.png |
| mamba-130m | pz | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150713Z_mamba-130m__pz_t1.png |
| mamba-130m | pz-int4 | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150732Z_mamba-130m__pz-int4_t1.png |
| mamba-130m | llamacpp | 1 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150753Z_mamba-130m__llamacpp_t1.png |
| smollm2-135m | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150814Z_smollm2-135m__pz_t2.png |
| smollm2-135m | pz-int4 | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150826Z_smollm2-135m__pz-int4_t2.png |
| smollm2-135m | llamacpp | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150839Z_smollm2-135m__llamacpp_t2.png |
| smollm2-135m | ik | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150856Z_smollm2-135m__ik_t2.png |
| smollm2-1.7b-q4km | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-150910Z_smollm2-1.7b-q4km__pz_t2.png |
| smollm2-1.7b-q4km | pz-int4 | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151002Z_smollm2-1.7b-q4km__pz-int4_t2.png |
| smollm2-1.7b-q4km | llamacpp | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151055Z_smollm2-1.7b-q4km__llamacpp_t2.png |
| smollm2-1.7b-q4km | ik | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151200Z_smollm2-1.7b-q4km__ik_t2.png |
| bitnet-2b | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151248Z_bitnet-2b__pz_t2.png |
| bitnet-2b | pz-int4 | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151338Z_bitnet-2b__pz-int4_t2.png |
| bitnet-2b | bitnetcpp | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151438Z_bitnet-2b__bitnetcpp_t2.png |
| bitnet-2b | bitnetcpp-fix | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151526Z_bitnet-2b__bitnetcpp-fix_t2.png |
| bitnet-2b-iq2bn | ik | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151614Z_bitnet-2b-iq2bn__ik_t2.png |
| mamba-130m | pz | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151707Z_mamba-130m__pz_t2.png |
| mamba-130m | pz-int4 | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151721Z_mamba-130m__pz-int4_t2.png |
| mamba-130m | llamacpp | 2 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151735Z_mamba-130m__llamacpp_t2.png |
| smollm2-135m | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151749Z_smollm2-135m__pz_t3.png |
| smollm2-135m | pz-int4 | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151759Z_smollm2-135m__pz-int4_t3.png |
| smollm2-135m | llamacpp | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151809Z_smollm2-135m__llamacpp_t3.png |
| smollm2-135m | ik | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151822Z_smollm2-135m__ik_t3.png |
| smollm2-1.7b-q4km | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151832Z_smollm2-1.7b-q4km__pz_t3.png |
| smollm2-1.7b-q4km | pz-int4 | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151910Z_smollm2-1.7b-q4km__pz-int4_t3.png |
| smollm2-1.7b-q4km | llamacpp | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-151953Z_smollm2-1.7b-q4km__llamacpp_t3.png |
| smollm2-1.7b-q4km | ik | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152043Z_smollm2-1.7b-q4km__ik_t3.png |
| bitnet-2b | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152128Z_bitnet-2b__pz_t3.png |
| bitnet-2b | pz-int4 | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152205Z_bitnet-2b__pz-int4_t3.png |
| bitnet-2b | bitnetcpp | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152250Z_bitnet-2b__bitnetcpp_t3.png |
| bitnet-2b | bitnetcpp-fix | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152327Z_bitnet-2b__bitnetcpp-fix_t3.png |
| bitnet-2b-iq2bn | ik | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152403Z_bitnet-2b-iq2bn__ik_t3.png |
| mamba-130m | pz | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152446Z_mamba-130m__pz_t3.png |
| mamba-130m | pz-int4 | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152457Z_mamba-130m__pz-int4_t3.png |
| mamba-130m | llamacpp | 3 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152509Z_mamba-130m__llamacpp_t3.png |
| smollm2-135m | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152519Z_smollm2-135m__pz_t4.png |
| smollm2-135m | pz-int4 | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152527Z_smollm2-135m__pz-int4_t4.png |
| smollm2-135m | llamacpp | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152535Z_smollm2-135m__llamacpp_t4.png |
| smollm2-135m | ik | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152545Z_smollm2-135m__ik_t4.png |
| smollm2-1.7b-q4km | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152554Z_smollm2-1.7b-q4km__pz_t4.png |
| smollm2-1.7b-q4km | pz-int4 | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152628Z_smollm2-1.7b-q4km__pz-int4_t4.png |
| smollm2-1.7b-q4km | llamacpp | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152704Z_smollm2-1.7b-q4km__llamacpp_t4.png |
| smollm2-1.7b-q4km | ik | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152743Z_smollm2-1.7b-q4km__ik_t4.png |
| bitnet-2b | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152813Z_bitnet-2b__pz_t4.png |
| bitnet-2b | pz-int4 | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152844Z_bitnet-2b__pz-int4_t4.png |
| bitnet-2b | bitnetcpp | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152923Z_bitnet-2b__bitnetcpp_t4.png |
| bitnet-2b | bitnetcpp-fix | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-152951Z_bitnet-2b__bitnetcpp-fix_t4.png |
| bitnet-2b-iq2bn | ik | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-153019Z_bitnet-2b-iq2bn__ik_t4.png |
| mamba-130m | pz | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-153057Z_mamba-130m__pz_t4.png |
| mamba-130m | pz-int4 | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-153107Z_mamba-130m__pz-int4_t4.png |
| mamba-130m | llamacpp | 4 | benchmark_results/h2h_20261118-053147Z_published-sweep/screenshots/20261118-153117Z_mamba-130m__llamacpp_t4.png |


## Chart

![throughput vs threads](throughput_vs_threads.png)
