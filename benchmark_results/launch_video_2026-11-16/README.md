# Throughput on a 4-vCPU Xeon (2026-11-16)

Decode speed of `pz` on two models, 4 threads, greedy sampling, best of 3.

| Model | File | tok/s per run | Best |
|---|---|---|---|
| SmolLM2-135M-Instruct F16 | `SmolLM2-135M-Instruct-f16.gguf` | 151.75, 135.94, 134.24 | **151.8** |
| BitNet b1.58 2B-4T I2_S | `bitnet-b1.58-2B-4T-i2_s.gguf` | 25.20, 24.84, 19.94 | **25.2** |

- **Host:** Intel Xeon @ 2.10 GHz, 4 vCPU, AVX-512 VNNI, 15 GB RAM.
- **Build:** `make release CC=gcc` at `378ebc1`.
- **Command:** `pz --model <gguf> --prompt "Explain in two sentences why running an LLM on a CPU is useful." --max-tokens 128 --ignore-eos --temperature 0 --threads 4`. BitNet adds `--chat-template-file docs/templates/bitnet-b1.58.jinja` (the GGUF's own template is broken, see the README).
- **Runs:** each run's full output is in `run_bitnet_N.txt` and `run_smol_N.txt`; the `[gen]` line is the decode speed. Calibration had already run once and was cached.
- **Models:** `microsoft/bitnet-b1.58-2B-4T-gguf` (`ggml-model-i2_s.gguf`, sha256 `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`) and `bartowski/SmolLM2-135M-Instruct-GGUF` (sha256 `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57`).

These numbers use 128 tokens and a different host from the 500-token head-to-head tables in the main README, so they are not comparable with them.
