# Weight Loading Analysis - START HERE

> ⚠️ **READ [`GOLDEN_RULES.md`](GOLDEN_RULES.md) FIRST — development rules, test protocol, no-hardcoding mandate.**

## What You Need to Know (TL;DR)

This is the **complete weight loading specification** for BitNet-2B models in Project Zero.

**Main Function**: `weights_map()` in `src/core/weights.c` (lines 52-140)

**Key Points**:
1. Weights are read in a **FIXED SEQUENTIAL ORDER** (no reordering)
2. Uses **mmap** (NOT fseek) for direct memory access
3. Weights are **64-byte aligned** in absolute file coordinates
4. Ternary weights are **4 per byte** (2 bits each)
5. Includes **bounds checking** on every read

---

## The 14-Step Reading Order

This is **CRITICAL** - your converter must output weights in this EXACT order:

### Before Layers
**Step 1**: Token Embedding Table
- Format: `float32`
- Size: `vocab_size × dim × 4` bytes
- Example: 32000 × 4096 × 4 = 524 MB

### Per Layer (repeat 24 times for Qwen 2B)
**Step 2**: Input Attention Norm (`rms_att_weight[l]`)
- Format: `float32`
- Size: `dim × 4` bytes

**Step 3**: Post-Attention Norm (`rms_ffn_weight[l]`)
- Format: `float32`
- Size: `dim × 4` bytes

**Step 4**: Query Projection (`wq[l]` + `sq[l]`)
- Weights: `ceil(dim×dim/4)` bytes (packed ternary)
- Scale: 4 bytes (float32)

**Step 5**: Key Projection (`wk[l]` + `sk[l]`)
- Weights: `ceil(dim×kv_dim/4)` bytes (packed ternary)
- Scale: 4 bytes (float32)

**Step 6**: Value Projection (`wv[l]` + `sv[l]`)
- Weights: `ceil(dim×kv_dim/4)` bytes (packed ternary)
- Scale: 4 bytes (float32)

**Step 7**: Attention Sub-Norm (`rms_attn_sub_norm[l]`)
- Format: `float32`
- Size: `dim × 4` bytes
- ⚠️ **BitNet-specific - MUST be present**

**Step 8**: Output Projection (`wo[l]` + `so[l]`)
- Weights: `ceil(dim×dim/4)` bytes (packed ternary)
- Scale: 4 bytes (float32)

**Step 9**: FFN Gate (`w1[l]` + `s1[l]`)
- Weights: `ceil(dim×hidden_dim/4)` bytes (packed ternary)
- Scale: 4 bytes (float32)

**Step 10**: FFN Up (`w3[l]` + `s3[l]`)
- Weights: `ceil(dim×hidden_dim/4)` bytes (packed ternary)
- Scale: 4 bytes (float32)

**Step 11**: FFN Sub-Norm (`rms_ffn_sub_norm[l]`)
- Format: `float32`
- Size: `hidden_dim × 4` bytes
- ⚠️ **BitNet-specific - MUST be present**

**Step 12**: FFN Down (`w2[l]` + `s2[l]`)
- Weights: `ceil(dim×hidden_dim/4)` bytes (packed ternary)
- Scale: 4 bytes (float32)

### After Layers
**Step 13**: Final Layer Norm (`rms_final_weight`)
- Format: `float32`
- Size: `dim × 4` bytes

**Step 14**: Output Classifier (`wcls`)
- **NO SEPARATE WEIGHTS** - tied to embedding table
- Just set pointer to `token_embedding_table`

---

## Ternary Packing Details

### Format: 4 weights per byte, 2 bits each

```
Byte (8 bits):
[7:6] [5:4] [3:2] [1:0]
 w3   w2    w1    w0

Encoding:
-1 → 0b00
 0 → 0b01
 1 → 0b10
```

### Calculation
```c
packed_bytes(count) = ceil(count / 4) = ((count + 3) >> 2)

Example:
Query matrix: 4096 × 4096 = 16,777,216 weights
Packed: ceil(16777216 / 4) = 4,194,304 bytes
```

---

## 64-Byte Alignment

### Why?
- Cache-line efficiency
- mmap page alignment
- Memory access optimization

### How?
```
1. File structure:
   [0:8]    Magic/Version
   [8:48]   Config
   [48:64]  Padding
   [64:...] Weight data starts here

2. For each read at position 'ptr':
   relative_offset = ptr - data
   absolute_offset = relative_offset + 64
   padding_needed = (64 - (absolute_offset % 64)) % 64

3. Advance ptr to next 64-byte boundary in ABSOLUTE coordinates
```

### Examples
```
rel_offset=0:           padding=0   (already aligned at byte 64)
rel_offset=16384:       padding=0   (aligned at byte 16448)
rel_offset=4227076:     padding=60  (byte 4227140 % 64 = 4, need 60)
rel_offset=4227136:     padding=0   (aligned at byte 4227200)
```

---

## Bounds Checking

### Every Read is Validated

```c
if ((ptr - data + bytes_to_read) > data_size)
    return TN_ERR_INVALID_WEIGHTS;
```

Catches:
- File truncation
- Integer overflow
- Invalid sizes
- Alignment padding extending past EOF

---

## File Format

```
Offset    Size    Content
0         8       Magic/Version
8         40      Config struct
48        16      Padding
64        ...     Weight data

Config contains:
  dim           (4096 for Qwen 2B)
  hidden_dim    (14336 for Qwen 2B)
  n_layers      (24 for Qwen 2B)
  n_heads       (32)
  n_kv_heads    (8)
  vocab_size    (32000)
  ...
```

---

## Memory Mapping (NOT File I/O)

### POSIX (Linux/Unix)
```c
int fd = open(path, O_RDONLY);
flock(fd, LOCK_SH | LOCK_NB);
void *data = mmap(fd, size, PROT_READ, MAP_PRIVATE, ...);
posix_madvise(data, size, POSIX_MADV_SEQUENTIAL);
```

### Windows
```c
HANDLE hFile = CreateFileA(path, GENERIC_READ, ...);
HANDLE hMapping = CreateFileMappingA(hFile, ..., PAGE_READONLY, ...);
void *data = MapViewOfFile(hMapping, FILE_MAP_READ, ...);
```

### Key Point
- **NO fseek()** - all access via direct pointers
- Direct memory access from mapped file region
- Highly efficient for sequential reads

---

## Size Example (Qwen 2B)

```
Config:
  dim = 4096
  hidden_dim = 14336
  n_layers = 24
  n_heads = 32
  n_kv_heads = 8
  vocab_size = 32000

Per-layer calculation:
  norms:              ~64 KB      (4 float arrays)
  attention (Q,K,V,O): ~10 MB    (4 packed matrices)
  FFN (w1,w3,w2):     ~43 MB    (3 packed matrices)
  FFN/attn sub-norms: ~80 KB
  ─────────────────────────────
  Per layer:          ~54 MB

Total:
  24 layers:          ~1.29 GB
  Embedding table:    ~524 MB
  Final norm:         ~16 KB
  ─────────────────────────────
  Total file:         ~1.8 GB
```

---

## Validation Checklist

✓ Are weights read in the EXACT order listed above?
✓ Are 64-byte alignments applied AFTER scale factors?
✓ Are norms (float32) separate from projection matrices?
✓ Do you include BOTH `rms_attn_sub_norm` AND `rms_ffn_sub_norm`?
✓ Are ternary weights packed as 4 per byte, LSB first?
✓ Are scale factors always 4-byte floats, never packed?
✓ Is the file structure exactly: [8 magic][40 config][16 padding][weights]?
✓ Do all weight sections start/end at 64-byte boundaries?
✓ Is `wcls` pointer tied to embedding table (no separate export)?
✓ Is every read bounds-checked?

---

## Related Documents

After reading this, check:

1. **QUICK_SUMMARY.txt** - 10-point quick reference
2. **WEIGHT_LOADING_REFERENCE.md** - Complete technical spec
3. **VISUAL_WEIGHT_LOADING_FLOW.txt** - Flowcharts and diagrams
4. **offset_calculation_sequence.txt** - Offset calculation walkthrough
5. **SOURCE_CODE_WITH_LINE_NUMBERS.txt** - Full annotated source
6. **WEIGHT_LOADING_ANALYSIS_INDEX.md** - Navigation guide

---

## Source Files

| Path | Purpose |
|------|---------|
| `src/core/weights.c` | Main implementation (lines 52-140) |
| `include/core/weights.h` | TransformerWeights struct |
| `include/core/unpack.h` | packed_bytes() and ternary unpacking |
| `include/core/config.h` | Config struct and config_kv_dim() |
| `src/memory/mapped_file.c` | mmap/MapViewOfFile wrapper |
| `src/cli/main.c` | Usage example |

---

## Key Assertions

1. **Sequential Order**: No reordering, no skipping
2. **BitNet Sub-Norms**: BOTH must be present
3. **Ternary Packing**: Always 4 weights/byte, 2 bits each
4. **Scale Factors**: Always after packed matrices, always float32
5. **Alignment**: 64-byte boundaries in absolute file coordinates
6. **Tied Embeddings**: Output classifier = embedding table
7. **Validation**: All reads bounds-checked
8. **No Seeking**: Pure mmap pointer arithmetic

---

**Last Updated**: March 15, 2025
**Analysis Scope**: Project Zero, src/core/weights.c
**Implementation**: BitNet-2B (Microsoft b1.58-2B-4T)
**Format**: 2-bit ternary, 64-byte aligned, mmap-based

