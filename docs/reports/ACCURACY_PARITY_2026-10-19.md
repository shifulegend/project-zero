# Accuracy parity vs llama.cpp / bitnet.cpp (2026-10-19)

**Verdict:** on six GGUF models spanning dense transformers (F16, Q8_0, Q4_K_M), a state-space
model and a ternary BitNet, Project Zero's perplexity matches the reference engines within
**0.7 %** (Mamba 0.03 %, SmolLM2-135M 0.01 %), and on BitNet b1.58 2B-4T it is *lower* than
bitnet.cpp's on the same tokens. Getting here fixed nine pre-existing correctness gaps — see
`docs/ai/mistakes.md` 2026-10-19.

## Method

- Same text for every run: `docs/ai/project-overview.md + docs/WEBUI_GUIDE.md + README.md`
  (60,837 bytes, markdown/code heavy — absolute PPL is high for that reason).
- Project Zero: `adaptive_ai_engine --perplexity FILE --ppl-ctx 256 --ppl-chunks 4 --threads 4`.
- Reference: `llama-perplexity -f FILE -c 256 --chunks 4 -t 4 -b 256` (upstream llama.cpp
  `ff5a233`; bitnet.cpp `0b341e5` for the I2_S model).
- Identical chunking (256 tokens, BOS per the model's `add_bos_token`, second half of each chunk
  scored = 508 tokens). Tokenization is token-exact between the engines (8/8 test strings, four
  vocabularies). PPL ± is the standard error.
- Host: Xeon @ 2.10 GHz (KVM), 4 vCPU, AVX-512F + AVX-VNNI, 16 GB (Addendum AK hardware).

## Results

| Model | Format | Reference PPL | Project Zero PPL | Δ |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | F16 | 63.5728 ± 9.61 (llama.cpp) | 63.5784 ± 9.61 | +0.01 % |
| SmolLM2-1.7B-Instruct | Q4_K_M | 30.4288 ± 4.26 (llama.cpp) | 30.4147 ± 4.25 | −0.05 % |
| Qwen2.5-0.5B-Instruct | Q8_0 | 44.6597 ± 6.73 (llama.cpp) | 44.4414 ± 6.68 | −0.49 % |
| Qwen3-0.6B | Q8_0 | 43.7338 ± 6.83 (llama.cpp) | 43.4294 ± 6.76 | −0.70 % |
| Mamba-130m | F16 | 57.9703 ± 8.98 (llama.cpp) | 57.9542 ± 8.96 | −0.03 % |
| BitNet b1.58 2B-4T | I2_S | 419.78 ± 72.2 (bitnet.cpp) | 253.03 ± 51.3 | −39.7 % |

Small negative deltas are expected where Project Zero dequantises Q8_0 weights to F32 and keeps
an F32 KV cache (llama.cpp quantises activations to Q8_0 for those dot products and stores an
F16 cache by default).

**BitNet.** Both engines see the same 18-token-style sequences (the GGUF has no
`tokenizer.ggml.pre`, so both use llama.cpp's "default" pre-tokenizer; forcing Llama-3 rules
gives bitnet.cpp 290.8). Project Zero's lower PPL comes from its ternary kernels' activation
handling; the gap is consistent per chunk (291/269/251/253 vs 326/368/401/420). Neither number is
a statement about the model's general quality — the text is short-context markdown.

## What had to be fixed first

| Before | Symptom | Fix |
|---|---|---|
| I2_S unsupported (34.2b.2 still PLANNED under a ✅) | official BitNet GGUF refused | `core/gguf_i2s` lossless repack |
| RoPE always NORM | Qwen/BitNet/... attention mis-rotated | `rope_layout_for_arch` (NEOX group) |
| No BPE pre-tokenization | `?\n\n` one token, outputs diverge | `tokenizer/pretokenize` |
| `add_bos_token` ignored | Qwen3-0.6B PPL +5 % | honoured, llama.cpp defaults |
| Qwen2 biases not loaded | Qwen2.5 garbage | `w->bq/bk/bv` |
| head_dim from dim/n_heads | Qwen3-0.6B garbage | QK-norm path with key_length |
| sliding window uninitialised for direct forward calls | first `--perplexity` PPL ≈ vocab size | auto-init |

## Reproduce

```bash
cat docs/ai/project-overview.md docs/WEBUI_GUIDE.md README.md > /tmp/ppl.txt
./adaptive_ai_engine --model models/<m>.gguf --perplexity /tmp/ppl.txt --ppl-ctx 256 --ppl-chunks 4 --threads 4
llama-perplexity -m models/<m>.gguf -f /tmp/ppl.txt -c 256 --chunks 4 -t 4 -b 256
```
