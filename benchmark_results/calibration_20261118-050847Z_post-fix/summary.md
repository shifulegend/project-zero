# Calibration tests (profile, sweeps, tiers)

UTC start: 2026-11-18T05:08:47Z  |  host: Intel(R) Xeon(R) Processor @ 2.10GHz  |  pz commit: 864eeea

## Hardware profile

```
┌─────────────────────────────────────────────────────┐
│          Hardware Profile (auto-detected)           │
├─────────────────────────────────────────────────────┤
│ SIMD backend  : AVX-512 VNNI                        │
│ Cores         : 4 physical / 4 logical              │
│ Threads       : 4                                   │
│ L2 cache      : 2048   KiB per core                 │
│ L3 cache      : 260    MiB shared                   │
│ Free RAM      : 15005  MiB                          │
│ DRAM bandwidth: 53.7   GB/s (measured)              │
├─────────────────────────────────────────────────────┤
│ Classifier    : BF16 (full precision)               │
│ Prefetch rows : 32                                  │
│ Model in L3   : no                                  │
│ Data/token    : 1149   MB (pre-load est.)           │
│ Ceiling       : 57.6   tok/s (at 100% BW)           │
└─────────────────────────────────────────────────────┘
```

## Calibration sweeps

3 forced calibrations (`pz --calibrate`, isolated cache).

| SIMD pick | count |
|---|---|
| auto | 3 |

| thread pick | count |
|---|---|
| 3 | 1 |
| 4 | 2 |

## tiers (reference tier-default)

| model | variant | threads | runs | best tok/s | mean tok/s | stdev | mean / reference |
|---|---|---|---|---|---|---|---|
| bitnet-2b | tier-avx2 | 4 | 3 | 7.82 | 7.73 | 0.13 | 31% |
| bitnet-2b | tier-avx512f | 4 | 3 | 12.94 | 12.92 | 0.02 | 51% |
| bitnet-2b | tier-default | 4 | 3 | 25.50 | 25.27 | 0.37 | 100% |
| bitnet-2b | tier-vnni | 4 | 3 | 25.85 | 25.25 | 0.57 | 100% |
| smollm2-135m | tier-avx2 | 4 | 3 | 109.40 | 107.64 | 3.00 | 100% |
| smollm2-135m | tier-avx512f | 4 | 3 | 115.50 | 113.78 | 2.04 | 106% |
| smollm2-135m | tier-default | 4 | 3 | 110.19 | 107.74 | 2.62 | 100% |
| smollm2-135m | tier-vnni | 4 | 3 | 112.00 | 108.60 | 5.76 | 101% |

