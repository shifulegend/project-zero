# GitNexus Execution Flow Traces — project-zero

> Step-by-step execution traces for all major code paths.
> Use these to understand exactly what happens during any operation.

---

## Flow 1: TokenGenerationFlow — CLI Invocation to First Output Token

```
./adaptive_ai_engine --model foo.gguf --prompt "Hello" --threads 4

main()
  parse_args()                         ← --model, --prompt, --threads extracted
  g_tn_verbose / g_dump_fp setup       ← --verbose, --dump-tensors
  
  ── SIMD init ──
  if args.simd_override → setenv("TN_FORCE_BACKEND", override)
  tn_simd_init()                       ← CPUID → set all fn pointers
    tn_cpu_features_detect()           ← cpuid → TnCpuFeatures{avx2, avx512f…}
    select: VNNI > AVX-512 > AVX2 > scalar
    set: tn_ternary_matmul_packed = ternary_matmul_packed_avx2  (on this machine)
    set: tn_rmsnorm/softmax/vec_* = avx2 variants
    return "AVX2"
  
  ── Hardware profile ──
  tn_hardware_profile_init()           ← cores, RAM, L2/L3 cache sizes
  
  ── Calibration ──
  tn_calibration_load() or tn_calibrate()
    → best_threads=4, best_simd="avx2", classifier_fmt=TN_CLS_INT8

  ── Model load ──
  mapped_file_open("foo.gguf")         ← mmap(fd, size, PROT_READ, MAP_PRIVATE)
  file_magic == GGUF_MAGIC (0x46554747)
  gguf_read_header()                   ← parse n_meta KV pairs, tensor index
  config_from_gguf()                   ← read dim/n_layers/… from metadata
  weights_alloc_pointers()             ← malloc wq/wk/… pointer arrays
  weights_from_gguf()                  ← set each w->wq[l] → mmap tensor ptr
  weights_build_classifier_quant()     ← build wcls_i8 from w->wcls (BF16→INT8)
  
  ── KV + state ──
  kv_strategy_select()                 ← check free RAM → max_seq_len
  threadpool_create(4)                 ← 3 OS threads + caller = 4 HW slots
  tokenizer_from_gguf()                ← extract BPE vocab, tok->chat_template
  run_state_alloc()                    ← alloc x/xb/hb/q/att/logits/key_cache…
  
  ── Generation ──
  generate("Hello", max_tokens=128, temperature=0.7, top_p=0.9)
    → generate_with_callback(..., stdout_token_callback, NULL)
    
      chat_template_apply(tmpl, "user", "Hello", ...)  ← Jinja2 → formatted
      tokenizer_encode(formatted)      ← BPE → prompt_tokens[0..n_prompt-1]
      
      sw_init(&s->sw, max_seq, n_prompt)
      rng_seed(&rng_state, time(NULL))
      
      ── Step 0: first forward (token=prompt_tokens[0], pos=0) ──
      transformer_forward(token=1, pos=0, ...)   ← see Flow 2
      
      ── Steps 1..n_prompt-2: feed prompt tokens ──
      for step in [1..n_prompt-1]:
        transformer_forward(prompt_tokens[step], step, ...)
        next = prompt_tokens[step+1]   ← no sampling, teacher-forced
      
      ── Step n_prompt-1: first generated token ──
      transformer_forward(prompt_tokens[n_prompt-1], pos)
      apply_temperature(logits)
      sample_top_p(logits) → next_token
      piece = tokenizer_decode(tok, prev, next_token)
      stdout_token_callback(piece, NULL)   ← fprintf(stdout, "%s", piece)
      ← FIRST OUTPUT TOKEN APPEARS HERE
```

---

## Flow 2: DecodingLoop — Per-Token Decode: Embedding → Forward → Sample → Output

```
for step = n_prompt-1 to max_steps:

  token = (step < n_prompt) ? prompt_tokens[step] : next_sampled

  ── FORWARD ──
  transformer_forward(token, abs_pos, cfg, w, s, mc, tp)
    embed_token(s->x, token, w->embd_f32, w->token_embedding_table, dim)
    for l = 0 to n_layers-1:
      attention_forward(s, w, cfg, mc, l, pos, tp)
      ffn_forward(s, w, cfg, mc, l, tp)
    tn_rmsnorm(s->x, s->x, w->rms_final_weight, dim, eps)
    parallel_matmul_* (s->logits, s->x, wcls, dim, vocab_size, tp)
    return s->logits

  ── SAMPLING ──
  if step < n_prompt-1:
    next = prompt_tokens[step+1]     ← no sampling during prompt
  else:
    if temperature <= 0:
      next = sample_argmax(logits, vocab_size)
    else:
      apply_temperature(logits, vocab_size, temperature)
      if top_p < 1.0:
        next = sample_top_p(logits, vocab_size, top_p, &rng_state)
      else:
        tn_softmax(logits, vocab_size)
        r = rng_float(&rng_state)
        CDF-sample → next

  ── EOS CHECK ──
  if next == cfg->eos_token_id || next in tok->eos_list: break

  ── OUTPUT ──
  piece = tokenizer_decode(tok, prev_token, next)
  callback(piece, userdata)   ← stdout / SSE / agent buffer
  tokens_generated++
  prev_token = next

fprintf(stderr, "\n[gen] %.2f tok/s (%d tokens)\n", ...)
```

---

## Flow 3: ForwardPassDense — Dense Model (SmolLM2 F16) Forward

```
transformer_forward(token=T, pos=P, cfg, w, s, mc=dense, tp)

  ── Step 1: Token embedding ──
  embed_token(s->x, T, w->embd_f32=NULL, w->token_embedding_table, dim=576)
    // F16 GGUF path: embd_f32 is NULL for SmolLM2 (Q4_K token_embd)
    // Actually SmolLM2-f16: embd is F16, loaded into embd_f32 via tensor_to_f32
    // → s->x[0..575] = F32 embedding row T

  ── Steps 2a/2b per layer (×30 layers) ──
  attention_forward(s, w, cfg, mc=dense, l=0, pos=P, tp)
    tn_rmsnorm(s->xb, s->x, w->rms_att_weight[l], 576, 1e-5)
    // w->layer_weight_type == WEIGHT_TYPE_F16
    parallel_matmul_f16(s->q,  s->xb, (tn_u16*)w->wq[l], 576, 576, tp)
    parallel_matmul_f16(k_buf, s->xb, (tn_u16*)w->wk[l], 576, 192, tp)  // GQA: 3 KV heads
    parallel_matmul_f16(v_buf, s->xb, (tn_u16*)w->wv[l], 576, 192, tp)
    apply_rope(s->q, k_buf, s->rope_freq, 64, P, 9, 3, ...)  // 9 heads, 3 KV heads
    kv_nt_store(key_cache[l][h][mapped_P], k_head, head_dim=64)  // non-temporal
    kv_nt_store(value_cache[l][h][mapped_P], v_head, head_dim=64)
    for h in [0..8]:
      att[t] = tn_vec_dot(q_head, key_cache[l][h/3][t], 64) / sqrt(64)
      tn_softmax(att, valid_ctx)
      tn_vec_saxpy(xb[h*64], att[t], value_cache[l][h/3][t], 64)  // V-sum
    parallel_matmul_f16(s->xb2, s->xb, (tn_u16*)w->wo[l], 576, 576, tp)
    tn_vec_add(s->x, s->x, s->xb2, 576)   // residual

  ffn_forward(s, w, cfg, mc=dense, l=0, tp)
    tn_rmsnorm(s->xb, s->x, w->rms_ffn_weight[l], 576, 1e-5)
    parallel_matmul_f16(s->hb,  s->xb, w->w1[l], 576, 1536, tp)  // gate
    parallel_matmul_f16(s->hb2, s->xb, w->w3[l], 576, 1536, tp)  // up
    tn_silu(s->hb, 1536)           // SiLU activation on gate
    tn_vec_mul(s->hb, s->hb, s->hb2, 1536)  // gate × up (SwiGLU)
    parallel_matmul_f16(s->xb, s->hb, w->w2[l], 1536, 576, tp)   // down
    tn_vec_add(s->x, s->x, s->xb, 576)  // residual

  ── Step 3: Final RMSNorm ──
  tn_rmsnorm(s->x, s->x, w->rms_final_weight, 576, 1e-5)

  ── Step 4: lm_head ──
  // TN_CLS_INT8 (calibrated for this machine's dotprod)
  parallel_matmul_i8(s->logits, s->x, w->wcls_i8, w->wcls_i8_scales,
                      576, 49152, tp)   // → 49152 logit values

  return s->logits
```

---

## Flow 4: ForwardPassDeepSeek — DeepSeek-V2-Lite: embed → MLA_attn → MoE_FFN → norm

```
transformer_forward(token=T, pos=P, cfg, w, s, mc, tp)
  // cfg: dim=2048, n_layers=27, n_heads=16, hidden_dim=11008
  // mc: is_moe=1, first_k_dense=1, has_mla=1, lora=512, rope=64, nope=128

  embed_token(s->x, T, w->embd_f32, NULL, 2048)  // F32 pre-dequanted embeddings

  for l = 0 to 26:
    attention_forward(s, w, cfg, mc, l, P, tp)
      // mc->has_mla=1 → routes immediately to:
      mla_attention_forward(s, w, cfg, mc, l, P, tp)  // see Flow 6-subset

    ffn_forward(s, w, cfg, mc, l, tp)
      if l == 0 (first_k_dense=1): dense SwiGLU FFN (F32 dequanted)
      else l >= 1: moe_layer_is_moe(mc, l) = true →
        moe_ffn_forward(s, w, cfg, mc, l, tp)  // see Cluster 7 detail

  tn_rmsnorm(s->x, s->x, w->rms_final_weight, 2048, 1e-6)
  parallel_matmul_bf16(s->logits, s->x, w->wcls, 2048, 102400, tp)
  // (wcls is BF16; INT8/INT4 quantised if hardware supports)
  return s->logits
```

---

## Flow 5: WeightLoadFlow — mmap GGUF file → parse metadata → set weight pointers

```
mapped_file_open("model.gguf")
  open(path, O_RDONLY)
  fstat() → file size
  mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0) → data ptr
  close(fd)
  return MappedFile{data, size}

gguf_read_header(&hdr, data, size)
  // Check magic: data[0..3] == 0x46554747 "GGUF"
  // Read: version, n_tensors, n_kv_pairs
  // Parse n_kv_pairs: {key, val_type, value} metadata entries
  //   → hdr.arch = "llama" or "deepseek2"
  //   → rope_theta, n_layers, vocab_size, etc.
  // Parse n_tensors: {name, n_dims, dims[], type, offset}
  //   → hdr.tensors[i] = {name, type, data=base+offset, n_elems}

config_from_gguf(&cfg, &hdr)
  // Read: arch.embedding_length → cfg.dim
  //       arch.block_count → cfg.n_layers
  //       arch.feed_forward_length → cfg.hidden_dim
  //       tokenizer.ggml.bos_token_id → cfg.bos_token_id
  //       attention.layer_norm_rms_epsilon → cfg.rms_norm_eps
  //       rope.scaling.* → YaRN params

weights_from_gguf(&w, &cfg, &hdr, &store)
  // For each layer l:
  gguf_find_tensor(hdr, "blk.%d.attn_q.weight") → GGUFTensor*
  if tensor.type == GGUF_TYPE_F16:
    w.wq[l] = tensor.data   // zero-copy mmap pointer
    w.layer_weight_type = WEIGHT_TYPE_F16
  elif tensor.type == GGUF_TYPE_Q4_K and has_mla:
    w.mla_wq[l] = tensor.data  // zero-copy Q4K bytes
    w.has_mla_quant = true
  else:
    w.wq[l] = tensor_to_f32(tensor, n_elems, store)  // heap dequant

  // Norm weights: always F32 (or dequanted to F32)
  w.rms_att_weight[l] = norm_to_f32(hdr, "blk.%d.attn_norm.weight", dim, store)

  // Embedding:
  embd_tensor = gguf_find_tensor(hdr, "token_embd.weight")
  if embd_tensor.type == GGUF_TYPE_Q4_K:
    w.embd_f32 = tensor_to_f32(embd_tensor, vocab_size*dim, store)  // F32 copy
  elif embd_tensor.type == GGUF_TYPE_F16:
    w.token_embedding_table = embd_tensor.data  // zero-copy

weights_build_classifier_quant(&w, &cfg)
  // Build wcls_i8 (INT8 per-row symmetric, +128 biased for VNNI dpbusds)
  // Build wcls_i4 (INT4 packed, w_signed+8 nibble encoding)
  // Only if hardware supports VNNI/dotprod for INT8, or VNNI for INT4
```

---

## Flow 6: SIMDAutoSelect — cpuid → feature flags → dispatch table init → runtime dispatch

```
main()
  setenv("TN_FORCE_BACKEND", args.simd_override)  // optional user override

tn_simd_init()
  // 1. Read override env
  force = getenv("TN_FORCE_BACKEND")  // "scalar", "avx2", "avx512f", "vnni", "vnni256"

  // 2. Probe CPU at runtime
  tn_cpu_features_detect()
    // x86: __cpuid_count(7, 0) → ebx/ecx bits
    cpu.avx2      = (ebx >> 5)  & 1   // CPUID leaf 7 ebx bit 5
    cpu.avx512f   = (ebx >> 16) & 1   // CPUID leaf 7 ebx bit 16
    cpu.avx512vnni= (ecx >> 11) & 1   // CPUID leaf 7 ecx bit 11
    cpu.avx_vnni  = (eax >> 4)  & 1   // CPUID leaf 7.1 eax bit 4
    // ARM: read from __builtin_cpu_supports or HWCAP
    cpu.arm_dotprod = (HWCAP2 & HWCAP2_ASIMDDP)
    return &cpu

  // 3. Utility ops: AVX-512 > AVX2 > scalar
  #if TN_HAS_AVX512
    if !force_scalar && !force_avx2 && cpu.avx512f:
      tn_rmsnorm = rmsnorm_avx512
      tn_softmax = softmax_avx512
      tn_vec_* = avx512 variants
  #elif TN_HAS_AVX2
    if !force_scalar && cpu.avx2:
      tn_rmsnorm = rmsnorm_avx2
      ...
  else: scalar

  // 4. Packed matmul (main ternary hot path): priority chain
  #if TN_HAS_AVX512VNNI
    if cpu.avx512vnni && !force_avx2 && !force_avx512f && !force_vnni256:
      tn_ternary_matmul_packed = ternary_matmul_packed_vnni
      return "AVX-512 VNNI"
    if cpu.avx512vnni && force_vnni256:
      tn_ternary_matmul_packed = ternary_matmul_packed_vnni256
      return "VNNI-256 (no throttle)"
  #if TN_HAS_AVXVNNI
    if cpu.avx_vnni:
      tn_ternary_matmul_packed = ternary_matmul_packed_avx_vnni
      return "AVX-VNNI"
  #if TN_HAS_AVX512
    if cpu.avx512f:
      tn_ternary_matmul_packed = ternary_matmul_packed_avx512
      return "AVX-512F"
  #if TN_HAS_AVX2
    if cpu.avx2:
      tn_ternary_matmul_packed = ternary_matmul_packed_avx2
      return "AVX2"
  #if TN_HAS_ARM_DOTPROD
    if cpu.arm_dotprod:
      tn_ternary_matmul_packed = ternary_matmul_packed_dotprod
      return "NEON+dotprod"
  tn_ternary_matmul_packed = ternary_matmul_packed   // scalar
  return "Scalar"

// At call site: always indirect through fn pointer, no #ifdef
tn_ternary_matmul_packed(out, x, w, n, d, &scale, 0)
  // dispatches to whichever variant was selected above
```

---

## Flow 7: F16MatmulPath — parallel_matmul_f16 → thread dispatch → AVX2 8-wide FMA loop

```
parallel_matmul_f16(out, x, w_f16, n=576, d=576, tp)
  MatmulF16Args args = {out, x, w, n=576, d=576}
  threadpool_dispatch(tp, matmul_f16_task, &args, d=576)
    // tp->n_threads=4: divides 576 rows into 4 slices [0,144) [144,288) [288,432) [432,576)
    // 3 OS workers claim slices 0-2 via atomic fetch-add
    // caller (main thread) executes slice 3: [432,576)

matmul_f16_task(&args, thread_id=?, start=0, end=144)
  // AVX2 path:
  for i = start(0) to end(144):
    row = w + i * n   // row pointer: i-th row of weight matrix
    acc = _mm256_setzero_ps()
    for j = 0 to n-1 step 8:
      f16_8 = _mm_loadu_si128(&row[j])      // load 8 × F16 (128 bits)
      wv    = _mm256_cvtph_ps(f16_8)        // F16 → F32 (F16C extension)
      xv    = _mm256_loadu_ps(&x[j])        // load 8 × F32 activation
      acc   = _mm256_fmadd_ps(wv, xv, acc)  // fused multiply-add
    // horizontal sum of 8-wide accumulator:
    hi   = _mm256_extractf128_ps(acc, 1)
    lo   = _mm256_castps256_ps128(acc)
    sum4 = _mm_add_ps(lo, hi)
    // + scalar tail for j=n%8..n-1
    out[i] = horizontal_sum + scalar_tail
```

---

## Flow 8: TernaryMatmulPath — tn_ternary_matmul → pack → SIMD variant → scale

```
// Weight format: packed ternary, 2 bits per weight, stored in tn_u8
// Each byte encodes 4 weights: {-1,0,+1} → 2-bit encoding
// scale: single float scale factor (or per-group for group_size > 0)

parallel_ternary_matmul_packed(out, x, packed_w, n, d, scale, tp)
  // On VNNI machine with n <= 16384:
  //   Pre-quantise x once in dispatcher (K-4 R-3 optimisation):
  act_scale = quantize_row_to_i8_avx512(x, q_x_buf, n)
  sum_qx    = sum_i8_avx512(q_x_buf, n)
  // → ParallelMatmulPackedArgs.q_x = q_x_buf, .act_scale, .sum_qx set

  threadpool_dispatch(tp, matmul_packed_task, &args, d)

matmul_packed_task(&args, thread_id, start, end)
  // VNNI path (pre-quantised):
  ternary_matmul_packed_vnni_preq(
    out+start, q_x, act_scale, sum_qx,
    packed_w + start * ceil(n/4),
    n, end-start, &scale, group_size=0)
  // Non-VNNI AVX2 path:
  ternary_matmul_packed_avx2(out+start, x,
    packed_w + start * ceil(n/4),
    n, end-start, &scale, 0)

// Inside ternary_matmul_packed_avx2 (8-wide FP32 FMA loop):
for i = start to end:
  row_bytes = packed_w + i * ceil(n/4)
  acc = _mm256_setzero_ps()
  for j = 0 to n step 32:
    // unpack 32 ternary weights from 8 bytes (2 bits each):
    // w_unpacked[k] ∈ {-1, 0, +1}
    load 8 bytes → 256-bit mask → shuffle → unpack to 32 × float {-1,0,+1}
    xv = _mm256_loadu_ps(&x[j])
    acc += xv * w_float[0..7]
    ... (4 iterations for 32 elements)
  out[i] = horizontal_sum(acc) * scale
```

---

## Flow 9: KVCacheFlow — Strategy select → per-layer KV alloc → compress/sliding window

```
kv_strategy_select(cfg, hw, &max_seq_len)
  free_ram = hw->free_ram_bytes
  full_kv_bytes = n_layers * n_kv_heads * cfg.seq_len * head_dim * 4 * 2  // K+V
  if full_kv_bytes <= free_ram * 0.6:
    strategy = FULL;  max_seq_len = cfg.seq_len
  elif sliding_window_bytes <= free_ram * 0.6:
    strategy = SLIDING_WINDOW;  max_seq_len = WINDOW_SIZE
  else:
    strategy = COMPRESS; max_seq_len = COMPRESSED_SIZE

run_state_alloc(s, cfg, max_seq_len)
  s->key_cache   = aligned_alloc(n_layers * n_kv_heads * max_seq * head_dim * 4)
  s->value_cache = aligned_alloc(same)

// During attention (per token, per layer):
sw_init(&s->sw, max_seq_len, n_prompt)       // at generate() start
  // sw.capacity = max_seq_len
  // sw.write_head = 0 (mod capacity)
  // sw.n_filled = 0

attention_forward(s, ..., l, pos)
  mapped_pos = sw_map_position(&s->sw, pos)  // pos % capacity (circular)
  kv_nt_store(&key_cache[layer,head,mapped_pos], k_head, head_dim)
  sw_advance(&s->sw)                          // write_head = (write_head+1) % capacity

// During attention score computation:
valid_ctx = sw_valid_count(&s->sw, pos)     // min(pos+1, capacity)
for t = 0 to valid_ctx-1:
  hist_logical = (pos >= valid_ctx) ? (pos - valid_ctx + 1 + t) : t
  mapped_t = sw_map_position(&s->sw, hist_logical)
  k_vec = &key_cache[KV_CACHE_IDX(layer, kv_h, mapped_t, 0, ...)]
  score = tn_vec_dot(q_head, k_vec, head_dim)
```

---

## Flow 10: CalibrationFlow — hardware_profile → thread sweep → SIMD select → optimal config

```
tn_calibrate(&calib, hw)
  // Step 1: Check background CPU load
  bg_load = sample_background_load()   // reads /proc/stat for 500ms
  if bg_load > 50: warn, proceed anyway

  // Step 2: For each thread count T in [1..hw.logical_cores]:
  for T = 1 to hw.logical_cores:
    tp = threadpool_create(T)
    // Warmup: run CALIB_MATMULS_PER_TOK (210) ternary matmuls, dim=2560
    // Time CALIB_MAX_REPS (9) runs, take median
    times[T] = median_time(ternary_matmul_sweep(tp, CALIB_DIM=2560, CALIB_MATMULS_PER_TOK=210))
    threadpool_destroy(tp)

  // Step 3: Find best T (lowest total time / highest tok/s)
  calib.best_threads = argmin(times)

  // Step 4: SIMD benchmark (if multiple options compiled in)
  for simd in ["avx512vnni", "avx512f", "avx2", "scalar"]:
    setenv("TN_FORCE_BACKEND", simd)
    tn_simd_init()
    calib.simd_times[simd] = timed_matmul_sweep()
  calib.best_simd = argmin(simd_times)

  // Step 5: Classifier benchmark
  for fmt in [BF16, INT8, INT4]:
    calib.cls_times[fmt] = timed_lm_head(fmt)
  calib.best_classifier = argmin(cls_times)

  // Step 6: Save fingerprint
  get_cpu_model(calib.cpu_model)
  calib.physical_cores = hw.physical_cores

tn_calibration_save(&calib)
  → write to ~/.project-zero/calibration.bin
  // binary format: TnCalibrationResult struct

tn_calibration_load(&calib, hw)
  → read from ~/.project-zero/calibration.bin
  fingerprint_matches(cached, hw):
    compare cpu_model string, physical_cores, logical_cores, L2/L3 cache
  → if mismatch: return false → re-calibrate
```

---

## Flow 11: APIServerFlow — http_server → parse OpenAI request → generate_with_callback → SSE stream

```
api_server_start(port=8080, &ctx)
  bind + listen(sock, port)
  pthread_create(&listener_thread, api_listener_fn, &ctx)
  return TN_OK   ← main thread returns immediately

api_listener_fn(&ctx):
  while ctx.running:
    client_fd = accept(server_fd)
    read_http_request(client_fd) → buf
    // Parse request line: POST /v1/chat/completions
    // Find JSON body
    
    // JSON parse:
    json_parse(buf, "messages") → messages[]
    json_parse(buf, "max_tokens") → max_tokens
    json_parse(buf, "temperature") → temperature
    json_parse(buf, "stream") → stream_flag
    
    // Compile prompt:
    chat_compile(messages, n_messages) → prompt_str
    
    // Write SSE headers if streaming:
    write(client_fd, "Content-Type: text/event-stream\r\n\r\n")
    
    // Generate with SSE callback:
    generate_with_callback(cfg, weights, run_state, moe_cfg, tok, tp,
                            prompt_str, max_tokens, temperature, top_p,
                            sse_token_callback, &{client_fd})
    
    // sse_token_callback(piece, &client_fd):
    //   sse_send_token(client_fd, piece)
    //   → write: "data: {\"choices\":[{\"delta\":{\"content\":\"<piece>\"}}]}\n\n"
    
    // Final SSE event:
    write(client_fd, "data: [DONE]\n\n")
    close(client_fd)
```

---

## Flow 12: AgentToolFlow — agent_loop → intercept tool call → cmd_exec → inject output

```
run_agent_loop(prompt, cfg, w, s, tok, tp, max_tokens, temp, top_p, rag)

  // Optional RAG: auto-retrieve memories
  if rag != NULL:
    relevant = auto_retrieve(rag, prompt)
    prompt = prepend_memories(prompt, relevant)

  system_prompt = load_agent_system_prompt()  // tool list, JSON format spec
  full_prompt = system_prompt + user_prompt

  accumulated = ""
  in_tool_call = false

  generate_with_callback(..., agent_token_callback, &agent_state)

  // agent_token_callback accumulates tokens and scans for tool call markers:
  agent_token_callback(piece, state):
    accumulated += piece
    if detect_tool_call_start("<tool_call>", accumulated):
      in_tool_call = true
    if in_tool_call && detect_tool_call_end("</tool_call>", accumulated):
      tool_json = extract_between("<tool_call>", "</tool_call>", accumulated)
      tool_interceptor_parse(tool_json) → ToolCall{name, args}
      
      // User approval gate:
      if user_approval_prompt(tool_json.command): approved=true
      else: inject "<tool_result>User denied</tool_result>"
      
      if approved:
        // Execute command:
        cmd_exec(tool_json.command, &result_str)  // popen/read/pclose
        
        // Inject result into context:
        inject_text = "<tool_result>\n" + result_str + "\n</tool_result>"
        output_inject(s, tok, inject_text)  // tokenise + feed forward
        
        // Optional RAG auto-save:
        if rag != NULL && is_memorable(result_str):
          auto_save(rag, result_str)

      accumulated = ""  // reset for next turn
      in_tool_call = false
```

---

*Processes file generated by GitNexus. Refresh: `make -j4 && git add .claude/ && git commit -m 'gitnexus: refresh index'`*
