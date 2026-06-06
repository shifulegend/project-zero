# Weight Loading Analysis - Complete Documentation

This directory contains comprehensive analysis of the BitNet-2B weight loading mechanism in Project Zero.

## 📄 Reference Documents

### 1. **QUICK_SUMMARY.txt** ⭐ START HERE
   - Quick reference guide
   - Exact reading order (1-14)
   - Key assertions for validation
   - 10-point checklist
   - File: `/home/<USER>/Documents/project-zero/QUICK_SUMMARY.txt`

### 2. **WEIGHT_LOADING_REFERENCE.md** (Comprehensive)
   - Complete technical specification
   - Function signature & parameters
   - Exact reading order with detailed specs
   - Size calculations with examples (Qwen 2B)
   - Offset calculation algorithm with examples
   - Memory mapping implementation (POSIX & Windows)
   - Bounds checking & validation
   - Ternary unpacking format
   - Integration example
   - File: `/home/<USER>/Documents/project-zero/WEIGHT_LOADING_REFERENCE.md`

### 3. **weight_loading_summary.md**
   - File locations
   - High-level overview
   - Per-layer weight breakdown
   - Memory mapping details
   - Example size calculations
   - File: `/home/<USER>/Documents/project-zero/weight_loading_summary.md`

### 4. **offset_calculation_sequence.txt**
   - Detailed offset calculation walkthrough
   - Alignment macro explanation
   - Read macros (MAP_PTR, ALIGN64, COPY_VAL_ALIGN)
   - Layer-by-layer sequence with actual calculations
   - Bounds checking throughout
   - File: `/home/<USER>/Documents/project-zero/offset_calculation_sequence.txt`

## 🎯 Key Findings

### Function Location
- **File**: `src/core/weights.c`
- **Function**: `weights_map(TransformerWeights *w, const Config *cfg, tn_i8 *data, size_t data_size)`
- **Lines**: 52-140

### Exact Weight Reading Order (Sequential, No Exceptions)
```
BEFORE LAYERS:
  1. Token Embedding Table (float32, vocab_size × dim)

FOR EACH LAYER (l = 0 to n_layers-1):
  2. rms_att_weight[l]
  3. rms_ffn_weight[l]
  4. wq[l] + sq[l]
  5. wk[l] + sk[l]
  6. wv[l] + sv[l]
  7. rms_attn_sub_norm[l]    ← BitNet-specific
  8. wo[l] + so[l]
  9. w1[l] + s1[l]
  10. w3[l] + s3[l]
  11. rms_ffn_sub_norm[l]    ← BitNet-specific
  12. w2[l] + s2[l]

AFTER LAYERS:
  13. rms_final_weight
  14. wcls (tied to embedding table)
```

### Offset Strategy
- **Method**: Direct mmap pointer arithmetic (NOT fseek)
- **Alignment**: 64-byte boundaries in absolute file coordinates
- **Base**: File offset 64 (after 8-byte magic + 40-byte config + 16-byte padding)
- **Formula**: `padding = (64 - ((rel_offset + 64) % 64)) % 64`

### Ternary Packing
- **Format**: 4 weights per byte, 2 bits each
- **Encoding**: `-1 → 0b00`, `0 → 0b01`, `1 → 0b10`
- **Calculation**: `packed_bytes(count) = ceil(count / 4)`

### Bounds Checking
- Every read validates: `(ptr - data + bytes) <= data_size`
- Returns `TN_ERR_INVALID_WEIGHTS` on any failure
- Catches truncation, overflow, alignment issues

## 📊 Size Example (Qwen 2B - 24 layers)
- Embedding table: ~524 MB
- Per layer: ~54 MB
- Total 24 layers: ~1.29 GB
- Final norm: ~16 KB
- **Total file**: ~1.8 GB

## 🔧 Related Source Files

| File | Purpose |
|------|---------|
| `src/core/weights.c` | Main implementation |
| `include/core/weights.h` | TransformerWeights struct |
| `include/core/unpack.h` | `packed_bytes()` & unpacking |
| `include/core/config.h` | Config struct |
| `src/memory/mapped_file.c` | mmap/MapViewOfFile |
| `src/cli/main.c` | Usage example |

## ✅ Validation Checklist

When comparing a converter to this loader:

- [ ] Weight order matches exactly (1-14 sequence)
- [ ] Both BitNet sub-norms present (rms_attn_sub_norm, rms_ffn_sub_norm)
- [ ] Ternary weights packed 4 per byte, LSB first
- [ ] Scale factors are 4-byte floats AFTER each matrix
- [ ] 64-byte alignment applied in absolute file coordinates
- [ ] wcls tied to embedding table (no separate export)
- [ ] Bounds checking on all reads
- [ ] File structure: [8 magic][40 config][16 padding][weights]

## 🚀 Quick Start

1. Read `QUICK_SUMMARY.txt` (10 min)
2. Review exact reading order (step 1)
3. Understand 64-byte alignment (step 3)
4. Check validation checklist (step 10)
5. Compare your converter output against this sequence

## 📝 Notes

- All documentation generated from analysis of actual source code
- Line numbers referenced from `src/core/weights.c`
- Examples use Qwen 2B configuration (dim=4096, hidden_dim=14336, n_layers=24)
- BitNet sub-norms confirmed present in Microsoft BitNet-b1.58-2B-4T

---

**Last Updated**: March 15, 2025
**Analysis Scope**: Project Zero, src/core/weights.c lines 1-141
**Format**: 2-bit ternary, 64-byte aligned, mmap-based loading
