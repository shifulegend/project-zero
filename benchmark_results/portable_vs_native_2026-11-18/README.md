# Portable binary vs native build, 4-vCPU Xeon (2026-11-18)

Decode speed of `pz` built three ways on the same host, interleaved (the three builds take turns every
round), `--no-calibrate`, 4 threads, greedy, 128 generated tokens, 5 rounds. "Native" is `make release`
(`-march=native`); "portable" is `make dist`, the build the release download and `install.sh` ship.

| Model | Native | Portable, before | Portable, after | After / native |
|---|---|---|---|---|
| BitNet b1.58 2B-4T I2_S (mean tok/s) | 25.6 | 16.3 | 23.5 | 92% (before: 64%) |
| SmolLM2-135M F16 (mean tok/s) | 121.6 | 108.4 | 108.8 | 89% (before: 89%) |

"Before" is the `make dist` binary of `83af62a`; "after" adds the AVX-512 activation quantiser as its own
ISA-flagged file, the pre-quantised VNNI path in every build, and the VNNI kernel compiled a second time
with the VBMI bit-unpack (`ternary_matmul_packed_vnni_vbmi.c`), bound at startup when the CPU's VBMI is
execution-verified. SmolLM2 F16 and the BF16 LM head still use code that the portable build leaves out
(`tools/check_isa_guards.py` lists the files).

Where the time goes for BitNet (`TN_STEP_TIMING=1`, 64 tokens, total ms over the run):

| Step | Native | Portable, after | After / native |
|---|---|---|---|
| Q projection | 175 | 197 | 1.12x |
| Post-attention output | 116 | 124 | 1.07x |
| Dense FFN | 907 | 947 | 1.04x |
| Attention score / softmax | 72 | 71 | 0.99x |
| LM head | 1564 | 1812 | 1.16x |

- **Host:** Intel(R) Xeon(R) Processor @ 2.10GHz (4 vCPU).
- **Command:** `pz --model <gguf> [--chat-template-file docs/templates/bitnet-b1.58.jinja] --prompt "Explain in two sentences why running an LLM on a CPU is useful." --max-tokens 128 --ignore-eos --temperature 0 --threads 4 --no-update-check --no-calibrate`.
- **Raw output:** `bitnet_<build>_r<N>.txt`, `smollm2_<build>_r<N>.txt` (the `[gen]` line is the decode speed), `steps_bitnet_*.txt`, `system_info.txt` (binary checksums).
- Run-to-run noise on this shared 4-vCPU host is about +/-10% for SmolLM2 and +/-3% for BitNet.
