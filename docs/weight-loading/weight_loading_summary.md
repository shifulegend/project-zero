# Weight Loading Function Analysis

## File Locations
- **Primary Implementation**: `/home/<USER>/Documents/project-zero/src/core/weights.c`
- **Header Definition**: `/home/<USER>/Documents/project-zero/include/core/weights.h`
- **Unpacking Logic**: `/home/<USER>/Documents/project-zero/include/core/unpack.h`
- **File Mapping**: `/home/<USER>/Documents/project-zero/src/memory/mapped_file.c`

## Key Function: `weights_map()`

### Function Signature
```c
TernaryError weights_map(TransformerWeights *w, const Config *cfg,
                         tn_i8 *data, size_t data_size)
```

### Offset Computation Approach
- **Method**: Direct pointer arithmetic on memory-mapped data (uses mmap on POSIX, MapViewOfFile on Windows)
- **NOT fseek**: Uses direct memory access to already-mapped file
- **Alignment Strategy**: 64-byte alignment boundaries

### File Structure
```
File Layout:
[0:8]       - Magic/Version
[8:48]      - Config (40 bytes)
[48:64]     - Padding to 64-byte boundary
[64:...]    - Weight data (passed to weights_map)
```

### Alignment Macro Details
```c
offset_base = 64  // Absolute alignment base
// For each element:
//   abs_offset = (ptr - data) + offset_base
//   padding = (64 - (abs_offset % 64)) % 64
//   ptr += padding
```

---

## EXACT WEIGHT LOADING ORDER (per layer)

For each layer `l` from 0 to n_layers-1:

### 1. Input Layer Norm
- **Field**: `w->rms_att_weight[l]`
- **Size**: `dim * sizeof(float)` bytes
- **Format**: Raw float32
- **Alignment**: 64-byte aligned after copy

### 2. Post-Attention Layer Norm
- **Field**: `w->rms_ffn_weight[l]`
- **Size**: `dim * sizeof(float)` bytes
- **Format**: Raw float32
- **Alignment**: 64-byte aligned after copy

### 3. Query Projection (wq)
- **Field**: `w->wq[l]`
- **Size**: `packed_bytes(dim * dim)` bytes
  - Calculation: `ceil((dim * dim) / 4)` (4 ternary weights per byte)
- **Format**: 2-bit ternary packed (4 weights/byte)
  - Encoding: -1 → 0b00, 0 → 0b01, 1 → 0b10
- **Scale**: `w->sq[l]` (float32, 4 bytes)
- **Alignment**: 64-byte aligned after scale copy

### 4. Key Projection (wk)
- **Field**: `w->wk[l]`
- **Size**: `packed_bytes(dim * kv_dim)` bytes
  - Where: `kv_dim = (dim / n_heads) * n_kv_heads`
- **Format**: 2-bit ternary packed
- **Scale**: `w->sk[l]` (float32, 4 bytes)
- **Alignment**: 64-byte aligned after scale copy

### 5. Value Projection (wv)
- **Field**: `w->wv[l]`
- **Size**: `packed_bytes(dim * kv_dim)` bytes
- **Format**: 2-bit ternary packed
- **Scale**: `w->sv[l]` (float32, 4 bytes)
- **Alignment**: 64-byte aligned after scale copy

### 6. Attention Sub-Norm (BitNet-specific)
- **Field**: `w->rms_attn_sub_norm[l]`
- **Size**: `dim * sizeof(float)` bytes
- **Format**: Raw float32
- **Alignment**: 64-byte aligned after copy
- **Note**: CONFIRMED PRESENT in BitNet-2B model

### 7. Output Projection (wo)
- **Field**: `w->wo[l]`
- **Size**: `packed_bytes(dim * dim)` bytes
- **Format**: 2-bit ternary packed
- **Scale**: `w->so[l]` (float32, 4 bytes)
- **Alignment**: 64-byte aligned after scale copy

### 8. FFN Gate Projection (w1)
- **Field**: `w->w1[l]`
- **Size**: `packed_bytes(dim * hidden_dim)` bytes
- **Format**: 2-bit ternary packed
- **Scale**: `w->s1[l]` (float32, 4 bytes)
- **Alignment**: 64-byte aligned after scale copy

### 9. FFN Up Projection (w3)
- **Field**: `w->w3[l]`
- **Size**: `packed_bytes(dim * hidden_dim)` bytes
- **Format**: 2-bit ternary packed
- **Scale**: `w->s3[l]` (float32, 4 bytes)
- **Alignment**: 64-byte aligned after scale copy

### 10. FFN Sub-Norm (BitNet-specific)
- **Field**: `w->rms_ffn_sub_norm[l]`
- **Size**: `hidden_dim * sizeof(float)` bytes
- **Format**: Raw float32
- **Alignment**: 64-byte aligned after copy
- **Note**: CONFIRMED PRESENT in BitNet-2B model

### 11. FFN Down Projection (w2)
- **Field**: `w->w2[l]`
- **Size**: `packed_bytes(dim * hidden_dim)` bytes
  - But w2 is: hidden_dim * dim (Down layer)
- **Format**: 2-bit ternary packed
- **Scale**: `w->s2[l]` (float32, 4 bytes)
- **Alignment**: 64-byte aligned after scale copy

---

## After All Layers:

### Final Layer Norm
- **Field**: `w->rms_final_weight`
- **Size**: `dim * sizeof(float)` bytes
- **Format**: Raw float32
- **Alignment**: 64-byte aligned after copy

### Output Classifier (Tied to Embedding)
- **Field**: `w->wcls`
- **Value**: Pointer to `w->token_embedding_table` (tied embeddings)
- **Note**: No separate weights loaded; reuses token embedding table

---

## Pre-Layer Weights (loaded FIRST, before loop):

### Token Embedding Table
- **Field**: `w->token_embedding_table`
- **Size**: `vocab_size * dim * sizeof(float)` bytes
- **Format**: Raw float32
- **Scale**: `w->token_embedding_scale = 1.0f` (fixed)
- **Alignment**: 64-byte aligned after copy

---

## Validation & Error Checking

All reads include bounds checking:
```c
#define MAP_PTR(dest, bytes) do { \
    if ((size_t)(ptr - data + (bytes)) > data_size) \
        return TN_ERR_INVALID_WEIGHTS; \
    (dest) = (void *)ptr; \
    ptr += (bytes); \
} while(0)
```

**Validation Points**:
1. Each MAP_PTR call validates remaining buffer space
2. ALIGN64() padding checks do not exceed data_size
3. COPY_VAL_ALIGN() validates both copy and alignment
4. Final check ensures all required weights fit in file

---

## Memory Mapping Implementation

### POSIX (Linux/Unix)
```c
int fd = open(path, O_RDONLY);
flock(fd, LOCK_SH | LOCK_NB);  // Shared lock prevents modification
void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
posix_madvise(data, file_size, POSIX_MADV_SEQUENTIAL);
```

### Windows
```c
HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, ...);
HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, ...);
void *data = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
```

**Key Points**:
- Read-only access (PROT_READ / FILE_MAP_READ)
- Private mapping (MAP_PRIVATE) so modifications don't affect file
- Sequential access hint for OS prefetching
- Shared lock on POSIX prevents concurrent modifications
- No seeks or file I/O after mmap — all access via pointers

---

## Size Calculations (Example for Qwen 2B model)

```
Config values:
  dim = 4096
  hidden_dim = 14336
  n_layers = 24
  n_heads = 32
  n_kv_heads = 8
  kv_dim = (4096/32) * 8 = 128 * 8 = 1024

Per-layer sizes:
  rms_att_weight:      4096 * 4 = 16 KB
  rms_ffn_weight:      4096 * 4 = 16 KB
  wq:   ceil(4096*4096/4) = 4194304 bytes = 4 MB
  wk:   ceil(4096*1024/4) = 1048576 bytes = 1 MB
  wv:   ceil(4096*1024/4) = 1048576 bytes = 1 MB
  rms_attn_sub_norm:   4096 * 4 = 16 KB
  wo:   ceil(4096*4096/4) = 4194304 bytes = 4 MB
  w1:   ceil(4096*14336/4) = 14680064 bytes ≈ 14 MB
  w3:   ceil(4096*14336/4) = 14680064 bytes ≈ 14 MB
  rms_ffn_sub_norm:    14336 * 4 = 57.3 KB
  w2:   ceil(14336*4096/4) = 14680064 bytes ≈ 14 MB
  
  Total per layer (with padding): ~54 MB
  Total all 24 layers: ~1.3 GB
```

---

## Packed Bytes Calculation

```c
static inline size_t packed_bytes(int count) {
    return ((size_t)count + 3) >> 2;  // ceil(count / 4)
}

// Example: 4096 * 4096 = 16,777,216 ternary values
// packed_bytes = (16777216 + 3) >> 2 = 16777219 >> 2 = 4194305 bytes
```

