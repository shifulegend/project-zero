# Consolidated Terminal Logs: BitNet Debugging

## 1. Weight Verification (Bit-Perfect Match)
**Command:** `python3 -c "diagnostic script"`
**Output:**
```
Original packed shape: (640, 2560), dtype=uint8
Real weight matrix: (2560, 2560)
Ternary unique: [-1  0  1]
Distribution: -1=1649645, 0=3251715, 1=1652240
-1: 25.2%, 0: 49.6%, 1: 25.2%
Our packed: 1638400 bytes (expected 1638400)
Round-trip match: True
Q weight match vs HF original: True
```

## 2. Model Incoherence Sample
**Command:** `./build/project-zero --model models/bitnet-b1.58-2B-4T.bin ... --prompt "The capital of France is"`
**Output:**
```
[INFO] Mapping weights: Layer 30/30...
...
inta-dropÃ¤tuin Lovexpect unfavor offending tissues-ahead apologiseertime/mit lodged avail Academlevard-ahead bes Lafayette Durant male-paneuso sight uphol bola eclExpired welt weltzzo downwards objetÂłhSie Profilests Herald indictment-commentuida

--- 42 tokens in 160.27s (0.3 tok/s) ---
```

## 3. Norm Verification
**Command:** `python3 /tmp/check_norms.py`
**Output:**
```
model.layers.0.input_layernorm.weight:
  shape=(2560,), min=0.002487, max=0.063965, mean=0.017690
Binary att_norm: min=0.002487, max=0.063965, mean=0.017690
  MATCH with original!
```

## 4. RAM and Disk Status
**Command:** `free -h`
**Output:**
```
               total        used        free      shared  buff/cache   available
Mem:           7.5Gi       6.0Gi       1.3Gi       655Mi       1.1Gi       1.5Gi
```
**Note:** Model requires ~1.8GB for full mmap resident caching. Currently swapping/thrashing.
