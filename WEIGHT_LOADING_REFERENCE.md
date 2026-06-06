# Complete Weight Loading Reference for BitNet-2B
**File**: `/home/<USER>/Documents/project-zero/src/core/weights.c`

---

## 1. FUNCTION SIGNATURE

```c
TernaryError weights_map(TransformerWeights *w, const Config *cfg,
                         tn_i8 *data, size_t data_size);
```

---

## 2. EXACT READING ORDER (SEQUENTIAL)

### **PRE-LAYER**
1. **Token Embedding Table**
   - Size: `vocab_size * dim * sizeof(float)` bytes
   - Format: Raw float32
   - Scale: Fixed `w->token_embedding_scale = 1.0f`
   - Alignment: 64-byte after read

### **PER-LAYER LOOP** (for each layer l from 0 to n_layers-1)

2. **Input Attention Norm** (`rms_att_weight[l]`)
   - Size: `dim * 4` bytes
   - Format: Float32
   - Alignment: 64-byte

3. **Post-Attention Norm** (`rms_ffn_weight[l]`)
   - Size: `dim * 4` bytes
   - Format: Float32
   - Alignment: 64-byte

4. **Query Projection** (`wq[l]` + `sq[l]`)
   - Weights: `ceil(dim*dim/4)` bytes (2-bit ternary packed)
   - Scale: 4 bytes (float32)
   - Ternary Encoding: `-1 → 0b00, 0 → 0b01, 1 → 0b10`
   - Alignment: 64-byte after scale

5. **Key Projection** (`wk[l]` + `sk[l]`)
   - Weights: `ceil(dim*kv_dim/4)` bytes
   - where: `kv_dim = (dim/n_heads)*n_kv_heads`
   - Scale: 4 bytes (float32)
   - Alignment: 64-byte after scale

6. **Value Projection** (`wv[l]` + `sv[l]`)
   - Weights: `ceil(dim*kv_dim/4)` bytes
   - Scale: 4 bytes (float32)
   - Alignment: 64-byte after scale

7. **Attention Sub-Norm** (`rms_attn_sub_norm[l]`)
   - Size: `dim * 4` bytes
   - Format: Float32
   - **NOTE**: BitNet-specific, CONFIRMED present in b1.58-2B
   - Alignment: 64-byte

8. **Output Projection** (`wo[l]` + `so[l]`)
   - Weights: `ceil(dim*dim/4)` bytes
   - Scale: 4 bytes (float32)
   - Alignment: 64-byte after scale

9. **FFN Gate** (`w1[l]` + `s1[l]`)
   - Weights: `ceil(dim*hidden_dim/4)` bytes
   - Scale: 4 bytes (float32)
   - Alignment: 64-byte after scale

10. **FFN Up** (`w3[l]` + `s3[l]`)
    - Weights: `ceil(dim*hidden_dim/4)` bytes
    - Scale: 4 bytes (float32)
    - Alignment: 64-byte after scale

11. **FFN Sub-Norm** (`rms_ffn_sub_norm[l]`)
    - Size: `hidden_dim * 4` bytes
    - Format: Float32
    - **NOTE**: BitNet-specific, CONFIRMED present in b1.58-2B
    - Alignment: 64-byte

12. **FFN Down** (`w2[l]` + `s2[l]`)
    - Weights: `ceil(dim*hidden_dim/4)` bytes
    - Scale: 4 bytes (float32)
    - Alignment: 64-byte after scale

### **POST-LAYER**

13. **Final Layer Norm** (`rms_final_weight`)
    - Size: `dim * 4` bytes
    - Format: Float32
    - Alignment: 64-byte

14. **Output Classifier** (`wcls`)
    - Value: `w->wcls = (tn_i8*)w->token_embedding_table`
    - Scale: `w->wcls_scale = 1.0f`
    - No separate read (tied embeddings)

---

## 3. SIZE CALCULATIONS

### Formula for Packed Ternary Weights
```c
static inline size_t packed_bytes(int count) {
    return ((size_t)count + 3) >> 2;  // = ceil(count / 4)
}
```

### Example for Qwen 2B (dim=4096, hidden_dim=14336, n_layers=24)

```
Configuration:
  dim = 4096
  hidden_dim = 14336
  n_heads = 32
  n_kv_heads = 8
  kv_dim = (4096/32) * 8 = 1024

Per-Layer Calculations:
  rms_att_weight:      4096 * 4 = 16,384 bytes
  rms_ffn_weight:      4096 * 4 = 16,384 bytes
  wq:                  ceil(4096*4096/4) = 4,194,304 bytes
  sq:                  4 bytes
  wk:                  ceil(4096*1024/4) = 1,048,576 bytes
  sk:                  4 bytes
  wv:                  ceil(4096*1024/4) = 1,048,576 bytes
  sv:                  4 bytes
  rms_attn_sub_norm:   4096 * 4 = 16,384 bytes
  wo:                  ceil(4096*4096/4) = 4,194,304 bytes
  so:                  4 bytes
  w1:                  ceil(4096*14336/4) = 14,680,064 bytes
  s1:                  4 bytes
  w3:                  ceil(4096*14336/4) = 14,680,064 bytes
  s3:                  4 bytes
  rms_ffn_sub_norm:    14336 * 4 = 57,344 bytes
  w2:                  ceil(4096*14336/4) = 14,680,064 bytes
  s2:                  4 bytes

Per-layer raw size: ~54,129,600 bytes ≈ 54.1 MB

Total for 24 layers: ~1.29 GB

Token embedding table: 32000 * 4096 * 4 = 524,288,000 bytes ≈ 524 MB

Final layer norm: 4096 * 4 = 16,384 bytes

Total model file (weights only): ~1.8 GB
```

---

## 4. OFFSET CALCULATION ALGORITHM

### File Structure
```
Byte Offset    Content
[0:8]          Magic/Version
[8:48]         Config struct (40 bytes)
[48:64]        Padding to 64-byte boundary
[64:...]       Weight data (passed to weights_map as 'data')
```

### Alignment Macro
```c
const size_t offset_base = 64;  // File offset where weights start

#define ALIGN64() do {
    size_t abs_offset = (size_t)(ptr - data) + offset_base;
    size_t padding = (64 - (abs_offset % 64)) % 64;
    if (padding > 0) {
        if ((size_t)(ptr - data + padding) > data_size) 
            return TN_ERR_INVALID_WEIGHTS;
        ptr += padding;
    }
} while(0)
```

### How It Works
1. `ptr - data` = relative offset within weight section
2. `(ptr - data) + offset_base` = absolute file offset
3. `abs_offset % 64` = remainder when divided by 64
4. `64 - (abs_offset % 64)` = bytes needed to reach next multiple of 64
5. `(64 - ...) % 64` = 0 if already aligned, else padding needed

### Example Calculation
```
If relative offset = 0:     abs_offset = 64  → 64 % 64 = 0   → padding = 0
If relative offset = 10:    abs_offset = 74  → 74 % 64 = 10  → padding = 54
If relative offset = 4194304: abs_offset = 4194368
                                          → 4194368 % 64 = 0  → padding = 0
If relative offset = 4194308: abs_offset = 4194372
                                          → 4194372 % 64 = 4  → padding = 60
```

---

## 5. MEMORY MAPPING APPROACH (NOT fseek)

### File Mapping (POSIX - Linux/Unix)
```c
int fd = open(path, O_RDONLY);
flock(fd, LOCK_SH | LOCK_NB);              // Prevent concurrent modification
struct stat sb;
fstat(fd, &sb);
void *data = mmap(NULL, sb.st_size, 
                   PROT_READ, MAP_PRIVATE, fd, 0);
posix_madvise(data, sb.st_size, POSIX_MADV_SEQUENTIAL);
```

### File Mapping (Windows)
```c
HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
void *data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
```

### Key Characteristics
- **No fseek()**: All accesses are via pointer arithmetic on already-mapped memory
- **Read-only mapping**: `PROT_READ` / `FILE_MAP_READ` ensures safety
- **Private mapping**: `MAP_PRIVATE` on POSIX means modifications don't affect file
- **Sequential hints**: `POSIX_MADV_SEQUENTIAL` tells OS to prefetch data
- **Shared lock**: POSIX uses `flock(LOCK_SH)` to prevent concurrent modifications

---

## 6. BOUNDS CHECKING & VALIDATION

### Bounds Check Macro
```c
#define MAP_PTR(dest, bytes) do { \
    if ((size_t)(ptr - data + (bytes)) > data_size) \
        return TN_ERR_INVALID_WEIGHTS; \
    (dest) = (void *)ptr; \
    ptr += (bytes); \
} while(0)
```

### Validation Points
1. **Before every read**: Check `(ptr - data + bytes_to_read) <= data_size`
2. **Before alignment**: Check padding doesn't exceed `data_size`
3. **Alignment bounds**: Confirm `(ptr - data + padding) <= data_size`
4. **All macros fail-fast**: Return `TN_ERR_INVALID_WEIGHTS` on any violation

### Error Scenarios
- File truncated mid-layer
- `packed_bytes()` calculation causes integer overflow
- Scale factor read extends beyond buffer
- Alignment padding extends beyond file
- Config values result in invalid sizes

---

## 7. TERNARY UNPACKING (2-bit Format)

### Packed Format
```
4 ternary weights per byte (2 bits each)

Byte layout (LSB first):
  bits [1:0]   = weight[0]  (values: 0b00, 0b01, 0b10)
  bits [3:2]   = weight[1]
  bits [5:4]   = weight[2]
  bits [7:6]   = weight[3]

Encoding:
  -1 → 0b00 (decoded as 0b00 - 1 = -1)
   0 → 0b01 (decoded as 0b01 - 1 = 0)
   1 → 0b10 (decoded as 0b10 - 1 = 1)
```

### Unpacking Function
```c
static inline tn_i8 unpack_ternary(const tn_u8 *packed, int index) {
    tn_u8 byte = packed[index >> 2];       // index / 4
    int shift = (index & 3) << 1;           // (index % 4) * 2
    return (tn_i8)((byte >> shift) & 0x03) - 1;
}
```

### Unpacking Examples
```
Byte value: 0b01_10_01_00 = 0x68 = 104
  Weight[0]: bits[1:0] = 0b00 → -1
  Weight[1]: bits[3:2] = 0b01 → 0
  Weight[2]: bits[5:4] = 0b10 → 1
  Weight[3]: bits[7:6] = 0b01 → 0
```

---

## 8. SOURCE FILES REFERENCE

| File | Purpose |
|------|---------|
| `src/core/weights.c` | Main weight loading implementation (`weights_map()`) |
| `include/core/weights.h` | TransformerWeights struct definition |
| `include/core/unpack.h` | `packed_bytes()` and ternary unpacking functions |
| `include/core/config.h` | Config struct and helper functions |
| `src/memory/mapped_file.c` | `mapped_file_open()` and `mapped_file_close()` |
| `src/cli/main.c` | Example usage: calling `weights_map()` |

---

## 9. CRITICAL NOTES FOR CONVERTER VALIDATION

✓ **Weight Order is SEQUENTIAL**: Do not reorder, do not skip
✓ **Alignment is ABSOLUTE**: Uses file offset 64 as base
✓ **Ternary Packing**: Always 4 weights per byte, LSB first
✓ **BitNet Sub-Norms**: Both `rms_attn_sub_norm` and `rms_ffn_sub_norm` MUST be present
✓ **Scale Factors**: Always 4 bytes (float32), not included in packed weight size
✓ **Tied Embeddings**: Output classifier reuses token embedding table
✓ **No Floats in Ternary Weights**: Only the explicit scale factors are floats

---

## 10. INTEGRATION EXAMPLE

```c
// From main.c
MappedFile mf;
mapped_file_open(&mf, model_path);

tn_i8 *weight_data = (tn_i8*)mf.data + 64;  // Skip config header
size_t data_size = mf.size - 64;

TransformerWeights w;
weights_alloc_pointers(&w, &config);

if (weights_map(&w, &config, weight_data, data_size) != TN_OK) {
    fprintf(stderr, "Weight loading failed\n");
    return 1;
}

// Now w contains pointers to all weights in memory-mapped region
// Direct access: w.wq[layer_idx][element_idx] (but need unpacking for ternary)
```

