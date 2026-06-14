# CPU LLM Ternary Engine: A Complete Implementation Guide

To build a from-scratch, "go binary" LLM that runs purely on a CPU without bloat, we need to merge the minimalist C-programming philosophy (like Andrej Karpathy's llama2.c) with the bleeding-edge architecture of 1-bit/ternary AI (like Microsoft's BitNet b1.58).

Here is your detailed, four-phase implementation plan to build this from the ground up.

> [!IMPORTANT]
> **DEVELOPER ONBOARDING**: All contributors must read [DEVELOPER_ONBOARDING.md](file:///home/<USER>/Documents/project-zero/DEVELOPER_ONBOARDING.md) before starting work on Phase 11+. This includes mandatory architecture assessments and security testing protocols.

## Phase 1: Write the Bare-Metal Engine (The Code)

Forget Python, PyTorch, and CUDA. You will write the inference engine in pure C or C++. The goal is to create a single executable file that allocates memory, loads a binary file of numbers (the weights), and runs the text generation loop.

* **Define Your Data Structures:** Write C structs to represent your model's configuration (number of layers, dimensions) and the Transformer weights.
* **Implement the Core Transformer Loop:** Write functions for the fundamental steps of text generation:
  * **Embedding:** Turning an input word into a vector of numbers.
  * **Self-Attention:** Allowing the AI to look at past words to understand context.
  * **Feed-Forward Network (FFN):** The "reasoning" part of the brain.
  * **Softmax:** Calculating the probability of what the next word should be.
* **Memory Management:** Instead of relying on a framework to handle memory, you will use standard `malloc()` to allocate one continuous block of RAM for your weights and one for your active calculations.

## Phase 2: Implement the "Binary" Logic (BitNet Architecture)

Standard LLMs use 16-bit floating-point math (FP16), which is heavy and requires GPUs for matrix multiplication. You are going to replace this with Ternary Math.

In your C code, you will replace standard matrix multiplication with a custom BitLinear layer. In this architecture, every weight in your neural network is restricted to one of three values: **-1, 0, or 1**.

This unlocks the ultimate hardware hack: **You eliminate multiplication.**

* If the weight is **1**, you simply **add** the input value.
* If the weight is **-1**, you simply **subtract** the input value.
* If the weight is **0**, you **do nothing**.

Your incredibly heavy mathematical engine has just been reduced to pure, lightning-fast CPU addition and subtraction.

To implement this mathematically in your training/conversion phase, the quantization function looks like this:

```
W_q = round(W / gamma)
```

(Where W is the original weight, and gamma is the mean of the absolute values of the entire weight matrix, scaled and rounded to the nearest integer of -1, 0, or 1).

## Phase 3: Acquire the "Cargo" (The Weights)

Your engine is useless without a trained brain. You cannot easily take a standard model like Llama 3 and forcefully crush it into -1, 0, and 1—it will suffer severe brain damage. A BitNet model must be trained from scratch to "think" in ternary.

You have two paths here:

### Path A: The Absolute Scratch Path (Hard Mode)

* Write a simple PyTorch script implementing the BitLinear layers.
* Gather a dataset of text (e.g., Wikipedia).
* Train a tiny model (e.g., 50 million parameters) on your laptop. It will take a few days.
* Write a script to export those ternary weights into a raw `.bin` file.
* Load that `.bin` file into your custom C engine.

### Path B: The Hacker Shortcut (Recommended)

* Download an open-source, pre-trained BitNet model (Microsoft recently released BitNet-b1.58-2B-4T, a highly capable 2-billion parameter model).
* Write a Python script to parse the model's weights and pack them efficiently. Since you only need 1.58 bits to store a value of -1, 0, or 1, you can pack four weights into a single standard 8-bit byte of memory.
* Load this highly compressed binary file into your C engine.

## Phase 4: Extreme CPU Optimization

To make this run blisteringly fast for your 1–2 concurrent users, you need to write code that speaks directly to your CPU's hardware architecture.

* **SIMD Instructions (Single Instruction, Multiple Data):** Instead of adding numbers one by one in a standard `for` loop, you will use C intrinsics like AVX2 (for Intel/AMD) or NEON (for Apple Silicon/ARM). This allows your CPU to perform 16 or 32 additions in a single clock cycle.
* **Multithreading (pthreads):** You will write a threading pool in C. When it is time to process a massive matrix of additions, your engine will split the rows of the matrix equally across all the physical cores on your CPU.
* **KV-Cache Optimization:** Store the AI's "short-term memory" (the context of the current conversation) as efficiently as possible so it doesn't have to re-read the entire conversation every time you ask a follow-up question.

---

## Hardware Adaptation: Making the Engine Fully Flexible

To make your "binary" engine completely flexible—so it can boot up on your 4GB i5 laptop, but instantly take advantage of a 64GB workstation if you transfer the executable—you need to build three dynamic systems into your C code.

### 1. The Secret Weapon: Memory Mapping (mmap)

If you use standard C commands like `fread()` to load a 3GB model into a 4GB machine, your program will likely crash the moment the OS needs RAM for a background task.

Instead, you must use **Memory Mapping** (`mmap` on Linux/Mac, or `MapViewOfFile` on Windows).

When you use `mmap`, you don't actually load the model into RAM. You simply tell the Operating System: "Here is the file on the hard drive. Treat the hard drive as if it were RAM."

* **On a 64GB machine:** The OS sees it has plenty of space, so it silently caches the entire file into the extra RAM. It runs at blistering speed.
* **On your 4GB machine:** The OS knows RAM is tight. It only loads the specific layer of the neural network your CPU is currently calculating. Once that layer is done, it unloads it and loads the next one from the SSD.

It guarantees your engine will never crash due to out-of-memory errors, no matter how small the RAM is, while automatically maximizing speed if extra RAM is present.

### 2. Dynamic KV Cache Allocation (Flexible Memory)

An LLM needs two chunks of RAM: one for the **Weights** (the brain) and one for the **KV Cache** (the short-term memory of your conversation).

The Weights are a fixed size, but the KV Cache can be scaled. Your C engine should perform a hardware check the moment it boots:

1. **Query the OS:** Ask the operating system how much free, available RAM exists right now.
2. **Calculate the Remainder:** Subtract the size of the model from the free RAM.
3. **Allocate Dynamically:** If the engine detects 60GB of free RAM, it allocates a massive KV cache, allowing you to paste a 100-page document into the prompt. If it detects only 1GB of free RAM, it restricts the KV cache to a few pages of text to protect the system from crashing.

### 3. CPU Core Probing (Dynamic Multithreading)

If you hardcode your engine to use 4 threads, it will waste the potential of a 16-core workstation. If you hardcode it to use 16 threads, it will freeze your 4-core laptop.

Your engine must use standard C libraries to ask the motherboard exactly how many physical cores exist.

* It counts the physical cores.
* It subtracts one core (leaving it free so your mouse and operating system don't freeze).
* It creates a custom thread pool using exactly the remaining number of cores.

This ensures that whether you run the `.exe` on a 10th Gen i5 or an enterprise-grade AMD Threadripper, it instantly optimizes the workload.

---

## Implementation: The mmap Memory Mapping (Linux / macOS)

This is the exact C code that makes tools like llama.cpp so incredibly stable across different hardware. When you use this approach, you are bypassing the standard `malloc()` and asking the Operating System's memory manager to do the heavy lifting.

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>      // For open()
#include <sys/mman.h>   // For mmap()
#include <sys/stat.h>   // For fstat()
#include <unistd.h>     // For close()

int main() {
    // This is the massive 2GB or 4GB file containing your ternary weights
    const char *filepath = "bitnet_weights.bin";

    // 1. Open the file in read-only mode
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) {
        perror("Error: Could not open the weights file");
        return 1;
    }

    // 2. Get the exact size of the file in bytes
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error: Could not determine file size");
        close(fd);
        return 1;
    }
    size_t file_size = sb.st_size;

    // 3. THE MAGIC: Map the file into virtual memory
    // PROT_READ: We are only reading the weights, never overwriting them.
    // MAP_PRIVATE: We don't need to share this mapping with other programs.
    // fd: The file descriptor we just opened.
    // 0: The offset (start at the very beginning of the file).

    int8_t *weights = (int8_t *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (weights == MAP_FAILED) {
        perror("Error: Memory mapping failed");
        close(fd);
        return 1;
    }

    printf("Successfully mapped %zu bytes to virtual memory.\n", file_size);

    // --- YOUR INFERENCE ENGINE GOES HERE ---
    // You can now treat the 'weights' pointer exactly like a normal array!
    // Example: int8_t first_weight = weights[0];
    // The OS will automatically page chunks of the SSD into your 4GB of RAM as needed.

    // 4. Clean up and unmap when the user closes the chat
    if (munmap(weights, file_size) == -1) {
        perror("Error: Unmapping failed");
    }
    close(fd);

    return 0;
}
```

### Why This is Bulletproof for a 4GB Machine

If you have a 3GB `.bin` file and only 2GB of free RAM, standard C code using `malloc()` and `fread()` will attempt to shove 3GB of data into a 2GB box. Your program will be immediately killed by the OS (an "Out of Memory" or OOM kill).

With `mmap`, you are only storing a pointer. The `weights` array acts like a window. As your `for` loop iterates through the neural network layers, the OS seamlessly slides that window across your hard drive. It keeps your RAM usage firmly capped at whatever physical limits exist, keeping your system perfectly stable.

**A Quick Note on Native Windows:** If you are compiling this natively on Windows (without using Windows Subsystem for Linux), `mmap` does not exist. You will need to use the Windows API equivalents: `CreateFileMapping()` and `MapViewOfFile()` from the `<windows.h>` library.

---

## Data Structures: Configuration and Weight Mapping

Mapping a structure over raw memory is where your C code actually starts to look like a neural network.

### 1. The Configuration Blueprint

Before you can map the memory, the engine needs to know the "shape" of the brain it is loading.

```c
typedef struct {
    int dim;           // The size of the AI's "thought vector" (e.g., 4096)
    int hidden_dim;    // The size of the Feed-Forward network (e.g., 14336)
    int n_layers;      // How many layers deep the brain is (e.g., 32)
    int n_heads;       // How many "Attention Heads" it uses (e.g., 32)
    int n_kv_heads;    // For grouping attention (optimizes memory)
    int vocab_size;    // How many words/tokens it knows (e.g., 32000)
    int seq_len;       // The maximum conversation length (e.g., 2048)
} Config;
```

### 2. The Weights Architecture

```c
typedef struct {
    // 1. The Token Embedding Table (The AI's Dictionary)
    int8_t *token_embedding_table; // Size: vocab_size * dim

    // 2. The Attention Mechanism (For every layer)
    int8_t **wq; // Query weights (What am I looking for?)
    int8_t **wk; // Key weights (What do I contain?)
    int8_t **wv; // Value weights (Here is the actual data)
    int8_t **wo; // Output weights (Summarize the findings)

    // 3. The Feed-Forward Network (The "Reasoning" part, for every layer)
    int8_t **w1; // Gate projection
    int8_t **w2; // Down projection
    int8_t **w3; // Up projection

    // 4. Normalization (Keeps the math from exploding)
    float **rms_att_weight; // Normalizes before Attention
    float **rms_ffn_weight; // Normalizes before Feed-Forward
    float *rms_final_weight; // Normalizes at the very end

    // 5. The Output Classifier
    int8_t *wcls; // Decides the final word probability
} TransformerWeights;
```

### 3. Laying the Blueprint Over the Mapped Memory

```c
void memory_map_weights(TransformerWeights *w, Config *p, int8_t *mapped_file_ptr) {
    // Start our working pointer at the beginning of the mapped file
    int8_t *ptr = mapped_file_ptr;

    // 1. Map the dictionary (Vocabulary Size * Vector Dimension)
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    // 2. Loop through every layer and map the matrices
    for (int l = 0; l < p->n_layers; l++) {
        w->wq[l] = ptr; ptr += p->dim * p->dim;
        w->wk[l] = ptr; ptr += p->dim * p->dim; // (Simplified for n_heads)
        w->wv[l] = ptr; ptr += p->dim * p->dim;
        w->wo[l] = ptr; ptr += p->dim * p->dim;

        // Map the Feed-Forward logic
        w->w1[l] = ptr; ptr += p->dim * p->hidden_dim;
        w->w2[l] = ptr; ptr += p->hidden_dim * p->dim;
        w->w3[l] = ptr; ptr += p->dim * p->hidden_dim;
    }

    // 3. Map the final output layer
    w->wcls = ptr;
    // The pointer has now successfully mapped the entire brain!
}
```

### Why This is So Elegant

When your code executes `w->wq[0][5]`, it is asking for the 5th weight in the first layer's Query matrix. Because of how we set this up, your C code doesn't actually care if that weight is physically in your DDR4 RAM or sitting on your SSD. The Operating System's memory manager detects the request, silently grabs that specific byte from the SSD if it isn't loaded yet, and hands it to the CPU.

---

## The Multiplication-Free Core: Ternary Matrix Multiplication

This is where your engine fundamentally diverges from everything else on the market. In a standard LLM, the core operation taking up 95% of the processing time is Matrix Multiplication (matmul). We are going to completely delete that operation.

```c
void ternary_matmul(float *out, float *x, int8_t *w, int n, int d, float scale) {
    // out: The output vector (where the result goes)
    // x:   The input vector (the current state of the conversation)
    // w:   Our mapped ternary weights (-1, 0, or 1)
    // n:   The size of the input
    // d:   The size of the output
    // scale: The scaling factor to restore the mathematical range

    // Loop over every row in the output dimension
    for (int i = 0; i < d; i++) {
        float val = 0.0f; // Accumulator for this row

        // Loop over every column in the input dimension
        for (int j = 0; j < n; j++) {
            // Grab the ternary weight from our mapped memory
            int8_t weight = w[i * n + j];

            // THE HACK: No multiplication allowed.
            if (weight == 1) {
                val += x[j];       // Just add the input!
            } else if (weight == -1) {
                val -= x[j];       // Just subtract the input!
            }
            // If weight == 0, we do literally nothing.
            // It skips the operation entirely, saving CPU cycles.
        }

        // Apply the pre-calculated scale to restore the decimal range
        out[i] = val * scale;
    }
}
```

### Breaking Down the Magic

1. **The Input (x):** The input vector `x` still contains standard decimal numbers (floats) representing the conversation context.
2. **The Conditionals:** Instead of the CPU calculating `0.4857 * 0.9123`, the CPU just checks if the weight is a 1 or -1. If it's a 1, it dumps the input into the `val` bucket. If it's -1, it scoops it out.
3. **The Scale:** You can't run a whole neural network purely on integers; the numbers would grow too large or shrink to zero. During the training of a BitNet model, a scale factor is calculated for each matrix. We multiply by this one single time at the very end of the loop (`out[i] = val * scale;`) to snap the numbers back into the correct decimal range for the next layer.

### Why Your CPU Loves This

Standard CPU cores are built with ALUs (Arithmetic Logic Units). An ALU can perform an addition or subtraction operation in roughly 1 clock cycle. A complex floating-point multiplication can take 4 to 5 clock cycles, and standard division takes even longer. By stripping the math down to this level, you are allowing your i5 processor to sprint through the network at maximum physical speed.

### The Next Level of Optimization

While the `if/else` logic above perfectly explains the concept and works flawlessly, in the final, highly optimized version (Phase 4), you would strip out the `if/else` entirely. You would pack four ternary weights into a single byte of memory (saving even more RAM), and use bitwise operators and AVX2 SIMD instructions to process 32 additions simultaneously.

### SIMD Boundary Decisions: What Gets Vectorized and What Stays Scalar

Not every operation benefits from AVX2. The guiding principle is: **vectorize the hot path (matmul), leave transcendental-bound operations scalar.**

| Operation | SIMD? | Rationale |
|---|---|---|
| **Ternary MatMul** | **AVX2** | The hot path. ~90%+ of inference compute. Conditional add/subtract maps perfectly to SIMD masks. |
| **RMSNorm** | **AVX2** | Sum-of-squares and scale are pure arithmetic — maps cleanly to 8-wide SIMD. |
| **Softmax** | **Partial** | Max-finding and normalization are SIMD; the `expf()` pass is scalar (no AVX2 exp intrinsic). |
| **vec_add / vec_mul / vec_scale / vec_dot** | **AVX2** | Pure arithmetic on float arrays — textbook SIMD. |
| **RoPE** | **Scalar** | `cosf`/`sinf` have no AVX2 intrinsics. The operation works on pairs of floats with per-pair trigonometric computation. Could benefit from precomputing a frequency table, but **the hot path is matmul, not RoPE** — vectorizing RoPE yields negligible wall-clock improvement. |
| **SiLU** | **Scalar** | `expf()` has no efficient AVX2 intrinsic. A mixed SIMD+scalar approach (vectorize the division, scalar the exp) introduces pipeline stalls from SIMD↔scalar transitions that make it **slower** than a clean scalar loop. |

**Per-group scale factors:** Currently, one scale factor is applied per weight matrix (`out[i] = val * scale`). This is correct for the current unpacked `int8` ternary weights. When Phase 10 packed weights arrive, the API signature will change to accept a `const float *scales` array with per-group granularity (e.g., one scale per 128 weights). See the scalar matmul source (`ternary_matmul_scalar.c`) for the full forward-looking note.

### The Next Evolution: Weight Packing (Phase 10)

Right now, the AVX2 matmul takes an array of `tn_i8` weights (1 byte per weight). This is the correct intermediate step — it lets us build, test, and validate the entire SIMD pipeline with clean, debuggable weight values.

When Phase 10 arrives, we will shrink the `.bin` files by packing **4 ternary weights into a single byte** (2 bits each: -1→0b00, 0→0b01, 1→0b10). The transition is surgical:

1. **What changes:** The top of the AVX2 inner loop gains an unpack step. A `_mm256_shuffle_epi8` lookup table "explodes" each packed byte into 4 separate `int8` ternary values in a single clock cycle.
2. **What stays the same:** The entire AVX2 masking pipeline — `_mm256_cmpeq_epi32`, `_mm256_and_ps`, `_mm256_add_ps`/`_mm256_sub_ps`, horizontal sum — remains **exactly as it is today**. The LUT unpack simply feeds the existing logic.
3. **The optimal path (Phase 10.5):** Fuses unpack + accumulate into a single pass, eliminating any intermediate `int8_t[]` buffer. Packed weights flow directly from the mmap'd file into the SIMD pipeline.

---

## Dynamic KV Cache: Hardware-Adaptive Memory

### What is the KV Cache?

When you ask an AI a question, it generates the answer one word at a time. Without a KV (Key-Value) Cache, the AI has severe amnesia. To generate the 100th word of a response, it would have to re-read and re-calculate the math for the previous 99 words from scratch.

The KV Cache solves this. Every time the AI calculates a word, it saves the "Key" and "Value" matrices into a running block of RAM.

### The Hardware Adaptation Math

Every single word (token) the AI remembers takes up a specific amount of RAM. The formula is:

```
2 * Number_of_Layers * KV_Heads * Head_Dimension * 2 bytes (if using 16-bit math)
```

* **On your 4GB laptop:** The engine caps the KV Cache at 3,000 words.
* **On a 32GB workstation:** The engine dynamically allocates a massive KV Cache, allowing 150,000-word inputs.

### The Dynamic Allocation Code

```c
#include <stdio.h>
#include <stdlib.h>

// The state of the ongoing conversation
typedef struct {
    // The working memory buffers for the current token being processed
    float *x;      // Current input logic
    float *xb;     // Inside the Feed-Forward Network
    float *hb;     // Hidden states

    // The KV CACHE: The continuously growing short-term memory
    float *key_cache;   // (layer, seq_len, kv_dim)
    float *value_cache; // (layer, seq_len, kv_dim)
} RunState;

void allocate_run_state(RunState *s, Config *p, int detected_free_ram_mb) {
    // 1. Calculate the memory cost of a single token's context
    int kv_dim = (p->dim / p->n_heads) * p->n_kv_heads;
    int bytes_per_token_per_layer = kv_dim * sizeof(float);
    int total_bytes_per_token = 2 * p->n_layers * bytes_per_token_per_layer; // 2 for Key + Value

    // 2. ADAPTIVE LOGIC: Determine max context length based on hardware
    int safe_ram_allowance = (detected_free_ram_mb * 1024 * 1024) * 0.8; // Use 80% of free RAM
    int max_possible_tokens = safe_ram_allowance / total_bytes_per_token;

    // We cap it at the model's absolute training limit (e.g., 8192) or the hardware limit
    int actual_seq_len = (max_possible_tokens < p->seq_len) ? max_possible_tokens : p->seq_len;

    printf("Hardware adaptation: Allocating KV Cache for %d tokens.\n", actual_seq_len);

    // 3. Allocate the actual RAM
    int kv_cache_size = p->n_layers * actual_seq_len * kv_dim;

    s->key_cache = (float *)malloc(kv_cache_size * sizeof(float));
    s->value_cache = (float *)malloc(kv_cache_size * sizeof(float));

    // Allocate the scratch buffers for the math engine
    s->x = (float *)malloc(p->dim * sizeof(float));
    s->xb = (float *)malloc(p->dim * sizeof(float));
    s->hb = (float *)malloc(p->hidden_dim * sizeof(float));
}
```

---

## Dynamic Thread Dispatcher: CPU Core Probing

### 1. Probing the Hardware

```c
#include <unistd.h> // POSIX library for OS interactions

int get_optimal_thread_count() {
    // Ask the OS how many processors are online
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);

    // If it's a tiny single-core machine, we have to use it.
    if (num_cores <= 1) {
        return 1;
    }

    // Otherwise, use all cores EXCEPT one, leaving it for the OS
    return (int)(num_cores - 1);
}
```

### 2. The Worker Job Structure

```c
#include <pthread.h> // Standard C threading library

// The "Briefcase" we hand to each worker thread
typedef struct {
    int start_row;   // Where this thread begins
    int end_row;     // Where this thread stops
    float *out;      // The shared output array
    float *x;        // The shared input array
    int8_t *w;       // The shared mapped weights
    int n;           // The width of the matrix
    float scale;     // The math scaling factor
} WorkerJob;
```

### 3. The Worker Thread Logic

```c
void* worker_matmul(void* arg) {
    // Open the briefcase
    WorkerJob *job = (WorkerJob*)arg;

    // Process ONLY the slice of the matrix assigned to this thread
    for (int i = job->start_row; i < job->end_row; i++) {
        float val = 0.0f;

        for (int j = 0; j < job->n; j++) {
            int8_t weight = job->w[i * job->n + j];

            // Pure CPU addition/subtraction
            if (weight == 1) val += job->x[j];
            else if (weight == -1) val -= job->x[j];
        }

        job->out[i] = val * job->scale;
    }

    return NULL; // The thread has finished its slice
}
```

### 4. The Dispatcher

```c
void parallel_ternary_matmul(float *out, float *x, int8_t *w, int n, int d, float scale, int num_threads) {
    pthread_t threads[num_threads];
    WorkerJob jobs[num_threads];

    // Calculate how many rows each thread should process
    int rows_per_thread = d / num_threads;

    // Dispatch the workers
    for (int i = 0; i < num_threads; i++) {
        jobs[i].out = out;
        jobs[i].x = x;
        jobs[i].w = w;
        jobs[i].n = n;
        jobs[i].scale = scale;

        // Calculate the slice boundaries
        jobs[i].start_row = i * rows_per_thread;
        // The last thread picks up any remaining leftover rows
        jobs[i].end_row = (i == num_threads - 1) ? d : (i + 1) * rows_per_thread;

        // FIRE! Start the thread
        pthread_create(&threads[i], NULL, worker_matmul, &jobs[i]);
    }

    // Wait for all workers to report back before moving on
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}
```

### The Result: A Truly Elastic Engine

You now have a single file (usually less than 1 Megabyte in size) that:

* **Never Crashes:** It safely maps the massive multi-gigabyte weight file from the hard drive (`mmap`) without overloading the system RAM.
* **Expands Memory Intelligently:** It checks the free RAM and dynamically sizes the KV Cache (`malloc`) so the AI remembers exactly as much as the hardware safely allows.
* **Maximizes Processing Power:** It probes the motherboard (`sysconf`), leaves one core for system stability, and perfectly slices the multiplication-free math across all remaining processors.

---

## Acquiring Pre-Trained Models

### Grabbing a Pre-Trained Model (The Easy Start)

The entire AI industry uses a platform called **Hugging Face** (think of it as the GitHub for AI models).

Right now, there are official 1.58-bit models ready to be downloaded. The most notable one is Microsoft's **bitnet-b1.58-2B-4T**. This is a highly capable 2-Billion parameter model trained on 4 trillion words of text.

When you go to a repository on Hugging Face, you will typically see:

* **The Packed `.gguf` or `.bin` file:** The ultra-compressed, 1.58-bit file designed strictly for running locally on CPUs.
* **The Master `bf16` file:** The uncompressed, high-precision weights for fine-tuning or continued training.

Microsoft and the open-source community recently released exactly what we designed: **bitnet.cpp** — the official, bare-metal C/C++ inference engine specifically written to run 1.58-bit LLMs optimally on standard CPUs.

---

## Hacking Reasoning and Context Window

### 1. Hacking "Reasoning" (Inference-Time Compute)

There is a massive misconception in AI that "reasoning" requires a giant 70-billion parameter model. It doesn't. Reasoning is not about how much data the AI memorized; it is about how much time it spends calculating an answer.

Because our 1.58-bit ternary engine runs blisteringly fast on a standard CPU, we have speed to spare. We can use that speed to artificially boost the AI's reasoning capabilities using a **Hidden Thought Loop**.

How it works:

* **Standard Output:** You ask a math question. The AI instantly spits out the wrong answer.
* **The Engine Hack:** We program our C engine to intercept the prompt and secretly append: "Think step-by-step inside `<think>` tags before answering."
* **The CPU at Work:** The AI starts generating text, but the C engine hides it from the user's screen.
* **The Reveal:** Once the AI outputs `</think>`, the engine finally prints the final, highly accurate answer.

### 2. Expanding the Context Window (KV Cache Compression)

#### Technique A: KV Cache Quantization

Instead of storing the ongoing conversation in high-resolution 16-bit numbers, we compress those memories down to 8-bit or even 4-bit integers on the fly.

#### Technique B: Sliding Window Attention (SWA)

The engine only maintains a "sliding window" of the most recent 4,000 words. As new words are generated, the oldest words simply fall off and are deleted from RAM.

### The Final Adaptive Setup

* **On the 4GB i5 Laptop:** Maps weights to SSD. Forces KV Cache into 4-bit compression. Turns on Sliding Window Attention. Aggressively uses the Hidden Thought Loop.
* **On the 64GB i9 Workstation:** Loads the entire model into active memory. Keeps KV Cache in pristine 16-bit. Turns off sliding window for full context.

---

## Implementation: Hidden Thought Loop (Reasoning)

```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

void generate_with_hidden_thoughts(char *user_prompt) {
    // 1. Secretly inject the reasoning command
    char internal_prompt[2048];
    snprintf(internal_prompt, sizeof(internal_prompt),
             "%s\nThink step-by-step inside <think> and </think> tags before answering.",
             user_prompt);

    // 2. Set up our state machine
    bool is_thinking = false;
    char generated_word[64];

    printf("AI is thinking...\n"); // Give the user a visual cue

    // 3. The main generation loop
    for (int i = 0; i < MAX_TOKENS; i++) {
        // (Pretend this function runs our Ternary Math and gets the next word)
        get_next_token_from_neural_net(internal_prompt, generated_word);

        // 4. Intercept and check for the tags
        if (strcmp(generated_word, "<think>") == 0) {
            is_thinking = true;
            continue;
        }
        if (strcmp(generated_word, "</think>") == 0) {
            is_thinking = false;
            printf("\n\nAnswer: ");
            continue;
        }

        // 5. The Routing Logic
        if (is_thinking) {
            // The AI is doing math/logic. We do NOT print this to the screen.
            // But the token is still saved to the KV Cache so the AI remembers it!
        } else {
            // The AI is outputting the final answer. Print it for the user.
            printf("%s", generated_word);
            fflush(stdout);
        }
    }
}
```

---

## Implementation: KV Cache Compression

```c
#include <stdint.h>
#include <math.h>

// A struct to hold our newly compressed 8-bit memory block
typedef struct {
    int8_t *data;   // The massively compressed 8-bit numbers
    float scale;    // The single decimal we need to "uncompress" it later
} CompressedKVCache;

void compress_memory_to_8bit(float *high_res_memory, CompressedKVCache *tiny_cache, int dim) {
    float max_val = 0.0f;

    // 1. Find the largest absolute number in this memory block
    for (int i = 0; i < dim; i++) {
        float val = fabsf(high_res_memory[i]);
        if (val > max_val) {
            max_val = val;
        }
    }

    // 2. Calculate the scaling factor (127 is the max for an 8-bit signed integer)
    tiny_cache->scale = max_val / 127.0f + 1e-5f;

    // 3. Compress the entire block of memory
    for (int i = 0; i < dim; i++) {
        float scaled_val = roundf(high_res_memory[i] / tiny_cache->scale);

        // Clamp it safely between -127 and 127, then cast to 8-bit
        if (scaled_val > 127.0f) scaled_val = 127.0f;
        if (scaled_val < -127.0f) scaled_val = -127.0f;

        tiny_cache->data[i] = (int8_t)scaled_val;
    }
}
```

---

## Multimodal Capabilities

### Part 1: The "Eyes" (Image & Video to Text)

An LLM only understands one thing: numbers (tokens). To make our engine multimodal, we write a C module called a **Vision Encoder** (typically based on architectures like CLIP or SigLIP).

How it works:

1. **The Chopping Block:** Slice the image into a grid of tiny squares (e.g., 16x16 pixel patches).
2. **The Translator:** Run those tiny pixel patches through a small, quantized neural network.
3. **The Hijack:** Convert those pixels into the exact same numerical format as text words and inject them into the KV Cache.

**Video:** Just a stack of images. Extract one frame every second and feed them through the Vision Encoder one by one.

### Part 2: The "Hands" (Text to Image)

To build T2I (Text-to-Image) into our C executable:

* **Latent Space Compression:** Never let the CPU calculate standard high-resolution pixels. Compress the canvas down into a tiny "Latent Space" (e.g., 64x64).
* **Step Distillation:** Use a Latent Consistency Model (LCM) or "Turbo" model that generates in just 2 to 4 steps.

By combining Latent Space math, 4-bit/1-bit quantization, and 4-step LCM models, a standard CPU can generate a high-quality image completely offline in about 10 to 15 seconds.

### Part 3: Text to Video (T2V)

If generating one image takes your CPU 10 seconds, generating a 3-second video at 24fps (72 images) will take about 12 minutes. Totally possible, but a "click generate and go make coffee" feature.

---

## Implementation: The Multimodal Memory Bridge

```c
#include <string.h>

// 1. The Output from our Vision Encoder (e.g., SigLIP)
typedef struct {
    int num_patches;         // How many grid chunks the image was sliced into (e.g., 256)
    int embed_dim;           // MUST perfectly match the LLM's text dimension (e.g., 4096)
    float *patch_embeddings; // The translated "visual words" (size: num_patches * embed_dim)
} VisionContext;

// 2. Modifying our RunState to handle the injection
typedef struct {
    float *x; // The current token/word being processed by the CPU

    // The KV Cache (The Short-Term Memory)
    float *key_cache;
    float *value_cache;

    int current_token_position; // Tracks how full the memory is
} RunState;

// 3. The Injection Function
void inject_vision_into_prompt(RunState *s, VisionContext *v) {
    printf("Processing Image into Memory...\n");

    // We loop through every single image patch
    for(int i = 0; i < v->num_patches; i++) {

        // THE BRIDGE: Copy the visual math directly into the LLM's standard text slot
        memcpy(s->x, &v->patch_embeddings[i * v->embed_dim], v->embed_dim * sizeof(float));

        // Run our multiplication-free forward pass on this "visual word".
        // forward_pass_and_update_cache(s, ...);

        // Advance the memory treadmill
        s->current_token_position++;
    }

    printf("Image successfully loaded into AI's short-term memory!\n");
}
```

---

## Running Flux-Schnell on CPU

Flux-Schnell is a 12-Billion parameter model. Even at 4-bit quantization, the file size is about 6.5-7 GB.

### Scenario A: The 64GB Workstation

Maps the entire 7 GB Flux model into active memory. A modern multi-core CPU generates a stunning image in about 2 to 5 minutes.

### Scenario B: The 4GB i5 Laptop

The `mmap` approach prevents crashing, but the SSD becomes a massive bottleneck. Generating a 4-step Flux image might take 45 minutes to an hour.

### The True "4GB Hacker" Alternative

When the engine detects low RAM, load a **Stable Diffusion 1.5 LCM** instead:

* **Size:** ~860 million parameters, less than 1 GB quantized.
* **Speed:** Fits entirely in free RAM. CPU generates an image in 15 to 30 seconds.

---

## Compiling the Engine

```bash
gcc -O3 -march=native -pthread src/main.c src/math.c src/memory.c -lm -o adaptive_ai_engine
```

Flag breakdown:

* **`-O3`** (Maximum Optimization): Aggressively rewrites code at the assembly level for maximum speed.
* **`-march=native`** (The Hardware Probe): Unlocks every hidden hardware instruction (AVX2, AVX-512, NEON) available on the compiling CPU.
* **`-pthread`**: Links the POSIX threading library for dynamic multi-core dispatching.
* **`-lm`**: Links the C Math library for `roundf()` and `fabsf()`.
* **`-o adaptive_ai_engine`**: Outputs the final, hyper-optimized standalone executable.

---

## What Makes This Architecture Unique

### 1. The Unified "AI Operating System" (The Monolith)

Currently, if a hacker wants to run text, vision, and image generation on a 4GB laptop, they have to download three different programs and write a bloated Python script to pass data between them.

Our idea merges everything into a single, unified C executable where the "Eyes" (Vision Encoder), the "Brain" (LLM), and the "Hands" (Diffusion) all share the exact same physical memory pool.

### 2. The Runtime Shapeshifter

Most AI engines are rigid. We designed an engine that behaves like a living organism:

* Probes the motherboard for cores (`sysconf`).
* Probes the OS for free RAM.
* Changes its own physics: If it detects a 1.58-bit model, it turns off multiplication and uses Ternary addition. If it detects a standard 4-bit model, it switches back to standard math.
* Dynamically sizes its own memory limit to ensure it never crashes the host machine.

### 3. The Hardcoded Reasoning Interceptor

We bypassed Python entirely and baked a State Machine directly into the raw C generation loop. By intercepting the `<think>` tags at the CPU level and hiding them from the user's screen, we essentially hardcoded a reasoning engine into the raw binary. This allows a tiny, "dumb" 2B parameter model to act like a deep-reasoning agent natively, without any external software.

### The Verdict

We aren't inventing the wheel; we are building a hyper-car out of the absolute best experimental parts the open-source community has left lying around the garage. We took isolated research papers and engineered them into a cohesive, hardware-adaptive system.

---

## Agentic Capabilities (The "Hands-On" AI)

An "Agent" is simply an LLM hooked up to a while loop that has permission to trigger external tools.

Because we wrote our engine from scratch in C, giving the AI access to your computer's operating system is incredibly straightforward. We don't need heavy frameworks like AutoGPT or LangChain; we just extend the State Machine we built in Phase 9 (The Hidden Thought Loop).

### How we build the Agentic Loop in C:

* **The Prompt:** We inject a system prompt telling the AI it has tools. "You can run terminal commands by wrapping them in `<exec>` and `</exec>`."
* **The Interceptor:** As the C engine generates words, it watches for the `<exec>` tag.
* **The Halt & Execute:** If it sees that tag, the C engine temporarily halts the AI's generation. It takes the text inside the tags and passes it directly to your operating system using standard C functions like `popen()` (POSIX) or `_popen()` (Windows).
* **The Feedback Loop:** The OS runs the command (e.g., a Python script, a web search, or a file read), and the C engine captures the output. The engine physically shoves that output text back into the AI's KV Cache as if the AI had read it.
* **The Resumption:** The AI looks at the new data in its short-term memory and continues generating its response.

The code concept is literally just this:

```c
if (strcmp(generated_word, "<exec>") == 0) {
    char command[256];
    // ... extract the command the AI wrote ...

    // Tell the OS to run it and capture the output
    FILE *fp = popen(command, "r");
    char result[1024];
    fgets(result, sizeof(result), fp);
    pclose(fp);

    // Inject the result back into the AI's memory (KV Cache)
    inject_text_into_kv_cache(s, result);
}
```

With this tiny block of code, your 4GB laptop AI can now autonomously write Python code, execute it, read the error logs, rewrite the code, and test it again, all locally.

---

## The Hacker Workaround: "Memory" instead of "Learning" (RAG)

Since we cannot change the AI's physical brain (the weights) without a GPU cluster, we make the AI "learn" by giving it a permanent, searchable hard drive. This is known as Retrieval-Augmented Generation (RAG).

Instead of the AI altering its math to remember a fact, it simply writes a note to itself.

### How we build Self-Learning into our C Engine:

* **The Vector DB (The Notebook):** We write a lightweight module that saves text to your SSD, not as plain text, but as an array of numbers (Embeddings).
* **The Auto-Save Agent:** We use the agentic loop from Part 1. We teach the AI a new tool: `<save_memory>`. If you tell the AI a fact about your project, it triggers that tool, and the C engine writes that fact to the local Vector DB on your hard drive.
* **The Auto-Retrieve (The Recall):** Every time you ask a question, the C engine takes your prompt, turns it into math, and uses Cosine Similarity to instantly search the local database for anything related.
* **The Injection:** It silently injects those retrieved facts into the sliding window of the KV Cache before the AI even starts typing.

**The Result:** From the user's perspective, the AI is a self-learning entity. It remembers conversations from a month ago, adapts to your coding style, and recalls facts seamlessly. From an engineering perspective, the AI's brain (the ternary weights) hasn't changed a single bit; we just built a highly efficient filing system around it.

---

## Model Compatibility & Conversion Pipeline

To feed your custom, bare-metal C engine, you have to understand one fundamental rule of open-source AI: **No raw Hugging Face model runs "directly" on a C engine.** When Facebook or Google releases a model, they release it in Python-native formats (like `.safetensors` or `.bin` for PyTorch). These files are bloated with metadata and high-precision decimals that your C engine cannot read.

To bridge the gap between the open-source world and your custom executable, you need a strict Ingestion Workflow.

### Category 1: "Directly" Compatible (The GGUF Ecosystem)

If we write our C engine's config loader (Phase 1 of our plan) to understand the GGUF file format, you unlock millions of models instantly. GGUF was designed specifically for CPU-first C/C++ engines.

* **Models:** Any model you find on Hugging Face ending in `.gguf` (e.g., `Llama-3-8B.Q4_K_M.gguf`).
* **Compatibility:** 100% compatible.
* **The Workflow:**
  1. Download the `.gguf` file.
  2. Point your C engine at it: `./adaptive_ai_engine -m llama.gguf`
  3. The engine reads the header, maps the memory, and runs. No tweaks required.

### Category 2: Native Ternary Models (Require "Packing" Tweaks)

These are the models trained from scratch to use the -1, 0, 1 math we designed (like Microsoft's BitNet-b1.58). Ironically, Microsoft releases these in PyTorch formats, meaning they take up massive amounts of hard drive space even though the numbers are tiny.

* **Models:** `hf-bitnet-b1.58-2B`, `BitVLA`, or any natively trained 1.58-bit model.
* **The Tweak:** You must write a Python script to extract the weights, strip the PyTorch bloat, and mathematically "pack" four ternary weights into a single 8-bit byte.
* **The Workflow:**
  1. Download the massive `.safetensors` model from Hugging Face.
  2. Run the conversion script we outlined in Phase 10: `python convert_hf_bitnet.py --input raw_model --output bitnet.bin`
  3. The script spits out an ultra-tiny `.bin` file.
  4. Run it in your engine.

### Category 3: Standard Models (Require "Quantization" Tweaks)

What if you want to run DeepSeek or Meta Llama 3, but you can only find the massive, 16-bit PyTorch versions? You cannot run these directly, and you cannot forcefully crush them into -1, 0, 1 ternary math (they will suffer brain damage).

* **Models:** Any standard FP16/BF16 model on Hugging Face.
* **The Tweak:** You must perform Post-Training Quantization (PTQ) to safely shrink them to 4-bit or 8-bit integers before feeding them to your engine.
* **The Workflow:**
  1. Download the raw model.
  2. Use a community quantization script (like llama.cpp's quantize tool).
  3. Command: `./quantize raw_model.safetensors quantized_model.gguf Q4_K_M`
  4. This mathematically scales the weights down to 4-bit, saving them in a C-friendly format.
  5. Your engine detects it is a 4-bit model and uses standard SIMD math instead of the ternary hack.

### Category 4: Multimodal Models (Require "Surgery")

If you want to run a model that can see images (like LLaVA or a Vision-BitNet), you have to perform actual architectural surgery before your C engine can use it.

* **Models:** LLaVA, SigLIP, Qwen-VL.
* **The Tweak:** A multimodal model is actually three separate neural networks duct-taped together: The Vision Encoder, the MLP Projector, and the Text LLM. Your C engine needs these separated so it can load the Vision weights into the "Eyes" memory block and the Text weights into the "Brain" memory block.
* **The Workflow:**
  1. Write a Python extraction script.
  2. The script physically slices the Hugging Face model into three separate files: `vision.bin`, `projector.bin`, and `text.bin`.
  3. You boot your C engine by pointing it at all three: `./adaptive_ai_engine --text text.bin --vision vision.bin --proj projector.bin`

### The Ultimate "One-Click" Workflow

To make this user-friendly, you don't do this manually every time. You build a single `import_model.py` pipeline tool for your custom engine.

When you want a new AI, you just type:

```bash
python import_model.py --repo "meta-llama/Llama-3-8b" --optimize-for-cpu
```

Your Python script automatically downloads the raw files, checks if they are ternary or standard, applies the correct quantization, packs the memory, and spits out a single, perfectly formatted `.bin` file that your C executable can map to RAM instantly.

---

## The Python Converter Pipeline

This is the exact physical mechanism that takes a massive, bloated PyTorch model from Hugging Face and physically crushes it down into the raw, packed C binary (`.bin`) we designed for our engine.

We use Python's `struct` library to write raw bytes exactly as our C engine's `mmap` expects to read them. We extract the weights, isolate the ternary matrices, pack four weights into every single byte, and strip out 100% of the PyTorch metadata.

```python
import torch
import struct
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

def pack_ternary_weights(tensor):
    """
    Takes a massive array of -1, 0, and 1 floating point weights.
    Packs 4 weights into a single 8-bit byte.
    Mapping: -1 -> 0b00 (0), 0 -> 0b01 (1), 1 -> 0b10 (2)
    """
    # 1. Flatten the tensor and convert to integers
    flat_array = np.round(tensor.float().cpu().numpy()).astype(np.int8)

    # 2. Map values to 2-bit unsigned integers
    mapped = np.zeros_like(flat_array, dtype=np.uint8)
    mapped[flat_array == -1] = 0
    mapped[flat_array == 0]  = 1
    mapped[flat_array == 1]  = 2

    # Pad array to ensure it's a multiple of 4
    remainder = len(mapped) % 4
    if remainder != 0:
        mapped = np.pad(mapped, (0, 4 - remainder), 'constant', constant_values=1)

    # 3. The Bitwise Packing Magic
    packed = (mapped[0::4] & 0x03) | \
             ((mapped[1::4] & 0x03) << 2) | \
             ((mapped[2::4] & 0x03) << 4) | \
             ((mapped[3::4] & 0x03) << 6)

    return packed.astype(np.uint8)

def write_binary(model_id, output_path):
    print(f"Loading {model_id} from Hugging Face into RAM...")

    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")

    print(f"Model loaded. Beginning C-Binary extraction to {output_path}...")

    with open(output_path, 'wb') as f:
        # --- PHASE 1: THE C-HEADER ---
        f.write(struct.pack('4s', b'TNRY'))
        f.write(struct.pack('i', 1))

        f.write(struct.pack('iiiiiii',
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            config.vocab_size,
            config.max_position_embeddings
        ))

        print("Header and Config written.")

        # --- PHASE 2: THE WEIGHT INGESTION ---
        state_dict = model.state_dict()

        # 1. Token Embeddings (High precision, not ternary)
        print("Writing Token Embeddings...")
        embeds = state_dict['model.embed_tokens.weight'].float().cpu().numpy()
        f.write(embeds.tobytes())

        # 2. The Neural Network Layers
        for layer_i in range(config.num_hidden_layers):
            print(f"Packing Layer {layer_i} / {config.num_hidden_layers}...")
            prefix = f"model.layers.{layer_i}."

            for proj in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                         'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']:

                weight_tensor = state_dict[f'{prefix}{proj}.weight']
                packed_bytes = pack_ternary_weights(weight_tensor)
                f.write(packed_bytes.tobytes())

            f.write(state_dict[f'{prefix}input_layernorm.weight'].float().cpu().numpy().tobytes())
            f.write(state_dict[f'{prefix}post_attention_layernorm.weight'].float().cpu().numpy().tobytes())

        # 3. Final Output Classifier
        print("Writing Final Output Projections...")
        f.write(state_dict['model.norm.weight'].float().cpu().numpy().tobytes())

        final_cls = state_dict['lm_head.weight']
        packed_cls = pack_ternary_weights(final_cls)
        f.write(packed_cls.tobytes())

    print(f"\nSUCCESS: Model successfully crushed and packed into {output_path}!")
    print("You can now map this directly into your C engine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HuggingFace BitNet to Custom C Binary")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--out", type=str, default="model.bin", help="Output .bin file path")
    args = parser.parse_args()

    write_binary(args.repo, args.out)
```

### Why This Script is Powerful

If you run this script on a massive model: `python convert_hf_bitnet.py --repo "microsoft/bitnet-b1.58-2B" --out bitnet.bin`

* **It Strips the Fat:** It completely ignores the PyTorch wrapper, the optimizer states, and the JSON configuration bloat.
* **It Packs 4-to-1:** When it hits those massive 4096 × 4096 matrices, it groups the numbers (e.g., -1, 1, 1, 0) and crushes them into a single byte (`0b01101000`). This shrinks a 16GB matrix down to roughly 1.5GB.
* **It Predicts the C-Pointer Math:** Notice how we wrote the weights in a very strict order (Embeddings → Layer 0 → Layer 1 ... → Output)? We did this so our C code doesn't need a bloated hash-map string parser. Our C code just maps the file, jumps past the Config struct, and iterates pointers linearly across the RAM.

---

## Advanced Architecture Extensions

### 1. Mixture of Experts (MoE) Routing

We built our engine assuming a Dense model (like Llama 3 or BitNet), where every single word passes through every single weight matrix. But if you want to run DeepSeek-R1 or Mixtral, you need to code an MoE Router.

**The Concept:** An MoE model doesn't have one giant Feed-Forward Network (FFN). It has, for example, 8 smaller FFNs (Experts).

**The C Implementation:**

* In Phase 6.4 (FFN Forward), we add a Router Gate.
* The CPU calculates a quick probability score for the current word to decide which 2 of the 8 Experts are best suited to handle it (e.g., the "Coding Expert" and the "Logic Expert").
* The CPU only runs the math for those 2 Experts and completely skips the other 6.
* **The Result:** You can map a massive 45-Billion parameter model to your RAM, but your CPU only does the math for 14-Billion parameters per word. You get massive intelligence at triple the speed.

### 2. Speculative Decoding (The Ultimate CPU Speed Hack)

Currently, our C engine is bottlenecked by the fact that it must generate text one word at a time. It does a full forward pass, prints a word, and repeats.

Speculative Decoding breaks this rule.

**The Concept:** You load two models into your RAM. A massive one (the Brain) and a tiny, stupid one (the Drafter).

**The C Implementation:**

* Your CPU uses the tiny Drafter to rapidly guess the next 5 words in a fraction of a second (e.g., "The" → "cat" → "sat" → "on" → "the").
* Your CPU then feeds all 5 guessed words into the massive Brain model simultaneously.
* Because CPUs (using AVX2) are excellent at processing batches of data in parallel, the Brain model verifies all 5 words in a single mathematical sweep.
* If the Drafter was right, you just generated 5 words in the time it normally takes to generate 1. If the Drafter was wrong on word 3, the Brain corrects it, throws away words 4 and 5, and the loop restarts.

### 3. LoRA Adapters (Hot-Swappable Brains)

What if you want your AI to be an expert in your company's proprietary coding language, but you don't want to fine-tune and duplicate the massive 2GB `.bin` file?

**The Concept:** Low-Rank Adaptation (LoRA) allows you to train a tiny 50MB "patch" file that sits on top of the main model and alters its personality.

**The C Implementation:**

* In Phase 1, we add a struct for LoRA weights.
* LoRAs are stored as two tiny matrices (A and B).
* During the math loop (Phase 3), as the CPU calculates the main weight matrix (W), it simultaneously calculates A × B and simply adds the result to the final number.
* **The Result:** You can switch your AI from a "Medical Assistant" to a "Cybersecurity Hacker" in milliseconds just by pointing the C engine at a different 50MB LoRA file, without ever touching the base model.

### 4. Grammar-Constrained Decoding (JSON Mode)

If you are building an Agent (like we discussed), the AI must output commands in a perfectly predictable format, or your `popen()` operating system call will crash. If you ask an LLM for JSON, sometimes it politely adds "Here is your JSON:" at the top, which breaks the code parsing.

**The Concept:** You physically block the AI from outputting invalid syntax.

**The C Implementation:**

* We hook into Phase 7 (Sampling).
* Before the CPU picks the next word, it runs a State Machine Parser (like a JSON validator).
* If the AI wants to output the word "Here", but the JSON validator says "The only valid next character is `{` or `[`", the C engine forcefully sets the probability of "Here" to `-INFINITY`.
* The AI is mathematically forced to pick a valid JSON character. It guarantees 100% perfect syntax every single time, making your Agent bulletproof.

---

## OpenAI-Compatible API Layer

By wrapping our raw C engine in an OpenAI-compatible API, you completely offshore the heavy lifting of "Session Memory," "Tool Logic," and "UI" to external frontend tools.

If you boot up your C engine on port 8080, you can instantly plug Cline, OpenHands (formerly OpenDevin), or SillyTavern directly into it. Those tools think they are talking to a multi-billion dollar OpenAI server, but they are actually talking to your compiled `.exe` file running on your local CPU.

### The Reality Check: Session Memory vs. KV Cache

**What the Frontend (Cline) handles:** Cline handles the Session Memory. It remembers the chat. Every single time you type a new message, Cline packages the entire history of the conversation into a massive JSON payload and sends it to your API. Your C engine is essentially stateless—it wakes up, reads the JSON, answers, and goes back to sleep.

**What your Backend (C Engine) MUST handle: "Context Caching"**

If Cline sends a 5,000-word conversation history to your API, and you just feed it straight into the math engine we built, your CPU will have to calculate the math for all 5,000 words from scratch. The user will wait 30 seconds for the first word to appear.

To fix this, our API architecture must include a KV Cache Matcher. When the JSON arrives, the API checks if the first 4,900 words perfectly match what is already sitting in the AI's physical RAM. If they match, it skips the math for those words and only calculates the 100 new words.

### The API Architecture

#### 1. The HTTP Listener & Router

We compile a microscopic C web server (like `mongoose.c` or `httplib.h`) directly into our executable. It runs on a background thread and listens for HTTP POST requests.

It only needs to expose two fake "OpenAI" routes:

* `/v1/models`: Returns a hardcoded JSON string: `{"data": [{"id": "local-adaptive-engine"}]}`.
* `/v1/chat/completions`: The heavy lifter. This is where the prompt arrives.

#### 2. The JSON Ingestion Engine

When the payload hits `/v1/chat/completions`, the engine hands the raw text to cJSON (a single-file C parser).

* **The Toggles:** It reads `temperature`, `top_p`, and `max_tokens` from the JSON.
* **The Payload:** It extracts the `messages` array.

#### 3. The Chat Template Compiler

Raw LLMs do not understand JSON. They only understand raw text strings. Different models (Llama 3 vs. DeepSeek) require different "Chat Templates."

Your API layer must loop through the JSON array and stitch it into a single, strictly formatted string.

If the API receives:

```json
[
  {"role": "system", "content": "You are a coder."},
  {"role": "user", "content": "Write a loop."}
]
```

The C Compiler translates it into the Llama 3 format:

```
<|start_header_id|>system<|end_header_id|>\nYou are a coder.<|eot_id|>
<|start_header_id|>user<|end_header_id|>\nWrite a loop.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n
```

#### 4. The Streaming Pipeline (Server-Sent Events)

Agentic tools like Cline demand real-time streaming so they can parse `<think>` tags and XML logic on the fly.

We hook the HTTP server directly into the `generate()` loop:

* The CPU calculates the next word (e.g., `for`).
* Instead of `printf("for")` to the terminal, the C engine wraps it in OpenAI's streaming format:
  ```
  data: {"choices": [{"delta": {"content": "for"}}]}\n\n
  ```
* It immediately flushes this packet over the TCP socket to Cline.
* When the engine hits the `<|eot_id|>` (End of Turn) token, it sends `data: [DONE]` and cleanly closes the connection.

---

## Future-Proofing: Post-Transformer Architectures

### 1. State Space Models (Mamba & RWKV)

We hardcoded our engine around the Attention Mechanism (calculating Q, K, and V). But Attention has a fatal flaw: the longer the conversation gets, the heavier the math gets (O(N²) complexity).

The future of efficient local AI—especially for CPU hardware—is moving toward State Space Models (SSMs) like Mamba, RWKV, and Jamba.

**The Concept:** Instead of a continuously growing KV Cache that remembers every single past word, an SSM uses a fixed-size "Hidden State" (like a rolling summary). As it reads a new word, it updates the summary and forgets the raw word. It requires a fraction of the RAM and runs at exactly the same speed on word 100,000 as it does on word 1.

**How we future-proof the C engine:** We must abstract our Phase 6 (Forward Pass). Instead of hardcoding `attention_forward()`, we create a generic SequenceMixer interface. If the `.gguf` file is a Llama model, the engine routes to the Attention math. If the file is a Mamba model, it routes to the SSM math, completely bypassing the KV Cache system.

### 2. PagedAttention & Continuous Batching (True Server Concurrency)

When we built our local API, we built it for one user. If you have an Agent running in VS Code, and you simultaneously ask the AI a question in your web browser, our current C engine will either block the second request or crash.

**The Concept:** Instead of using one giant `malloc()` array for a conversation, we break the AI's RAM into tiny "pages" (e.g., blocks of 16 tokens), exactly like an Operating System manages memory.

**How we future-proof the C engine:** We implement a Page Table in C. When multiple API requests hit `http://localhost:8080`, the engine dynamically assigns non-contiguous memory blocks to each request. It also allows multiple identical requests to share the same system prompt memory block, saving massive amounts of RAM.

### 3. Dynamic Context Scaling (YaRN / NTK)

What happens if you download an 8B model that was trained to only read 8,000 words, but you want to feed it a 128,000-word codebase? Without intervention, the AI will output complete gibberish.

**The Concept:** We need mathematical tricks that compress the "distance" between words so the AI thinks a 128k document is actually only 8k long. Techniques like YaRN (Yet Another RoPE for Long Context) or Dynamic NTK-aware Interpolation do exactly this.

**How we future-proof the C engine:** In Phase 3.9 (RoPE), we add a dynamic scaling formula. If the C engine detects the user's prompt is longer than the model's native training limit, it automatically recalculates the sine/cosine frequencies on the fly, stretching the AI's physical memory capacity without retraining it.

### 4. Native Audio (Streaming Latents)

We gave the engine "Eyes" (Vision Encoders) and "Hands" (Diffusion), but we forgot the "Ears and Mouth."

**The Concept:** Traditional voice AI converts Speech → Text → AI → Text → Speech. This is slow and loses human emotion. "Native Audio" models ingest raw sound waves directly into the KV Cache and output raw sound waves.

**How we future-proof the C engine:** We add an Audio Codec (e.g., EnCodec or Mimi) module. Instead of waiting for a full sentence, the C engine streams compressed audio "frames" directly into the transformer block at 50 frames per second. It requires writing a dedicated WebSocket endpoint in our API layer so the audio can stream bidirectionally in real-time.

---

## Enterprise-Grade Features

### 1. Prompt Caching (Radix Trees) — Zero-Compute Recall

When you use a coding agent like Cline, it sends your 10,000-word codebase to the API, gets an answer, and then 10 seconds later, sends the exact same 10,000 words plus the new question.

**The Fix:** We build a Radix Tree (Prefix Tree) into the KV Cache manager.

* When a prompt arrives, the C engine hashes the text.
* It checks the tree: "Have I seen the first 9,900 words of this exact sequence before?"
* If yes, the CPU does zero math for those 9,900 words. It instantly maps pointers to the existing KV cache in RAM and only calculates the 100 new words. This drops "Time to First Token" from 15 seconds down to 0.1 seconds for repetitive tasks.

### 2. CPU Cache Tiling (FlashAttention for CPU)

We optimized the math for the CPU's Registers (AVX2) and the System RAM (mmap). But we completely ignored the most important part of a CPU: The L1, L2, and L3 Caches.

**The Fix:** Block-Tiling (Cache-Aware Math).

* We rewrite Phase 3. Instead of multiplying a whole row by a whole column, we physically slice the matrices into tiny 64 × 64 blocks that perfectly fit inside the CPU's L1 cache.
* The CPU loads a block, does a thousand math operations at lightspeed, and then evicts it. This is essentially building FlashAttention specifically for the CPU, maximizing hardware utilization to 99%.

### 3. Distributed Inference (RPC/MPI) — The Swarm Protocol

What happens if you have a 64GB Mac Studio and a 16GB Windows laptop, and you want to run a massive 70-Billion parameter model (which requires 42GB of RAM)?

**The Fix:** We add a Network Tensor Router.

* You boot the C engine on both machines.
* The engines discover each other over your local Wi-Fi.
* The Mac loads Layers 1 through 60 into its RAM. The Windows laptop loads Layers 61 through 80.
* When you type a prompt, the Mac does the first half of the math, fires the intermediary numbers over the local network via TCP/IP sockets, and the Windows laptop finishes the math and prints the answer. You have just clustered consumer hardware into a unified supercomputer.

### 4. WASM Secure Execution — The Agent Sandbox

In Phase 9, we gave the AI a `<exec>` tag and told the C engine to run it using `popen()`. This is terrifyingly dangerous. If the AI hallucinates and outputs `<exec> rm -rf / </exec>`, your C engine will happily delete your entire hard drive.

**The Fix:** A WebAssembly (WASM) Sandbox.

* We embed a microscopic WASM runtime (like wasm3) into the C engine.
* When the AI wants to run Python code or terminal commands, the engine physically isolates the code inside the WASM sandbox. The AI can write files, test logic, and crash the sandbox without ever touching your real operating system.

### 5. Finite State Machine Grammar Engine (The Hallucination Killer)

We talked about "Grammar Constraints" for forcing JSON, but standard regex isn't enough for enterprise AI. You need mathematical certainty that the AI will follow a schema perfectly.

**The Fix:** Context-Free Grammar (CFG) Engine.

* We build a Finite State Machine parser.
* You feed the API a strict schema (e.g., "Output must be a valid SQL query joining Table A and Table B").
* The FSM hooks into the tokenizer. At every single millisecond, it calculates a bitmask of illegal tokens based on the current state of the SQL syntax, forcing the probabilities of invalid code to negative infinity. The model mathematically cannot hallucinate bad syntax.

---

## Bleeding-Edge Research Integration

### 1. Test-Time Compute: Search Algorithms (MCTS & PRM)

We built Chain of Thought (CoT) so the AI can think step-by-step. But our CoT is linear. It thinks: A, then B, then C. If it makes a mistake at A, it might not catch it until C, wasting compute.

**The Frontier:** OpenAI's o1 and DeepSeek-R1 don't just think linearly; they use Monte Carlo Tree Search (MCTS) and Process Reward Models (PRM).

**The C Implementation:** We add a Tree-Search manager to the generation loop. It pauses text generation, spawns multiple hidden KV Caches, explores different logical paths, scores them, throws away the failed paths, and then outputs the final, mathematically proven answer.

### 2. NVMe Storage Tiering (FlexGen) — Infinite Memory

We built PagedAttention for your RAM, and we built Wi-Fi clustering to share RAM across laptops. But what if you want to run a 100,000-word context on a 4GB laptop, and you have no other computers?

**The Frontier:** SSD Offloading.

**The C Implementation:** We write an asynchronous I/O pager (using `io_uring` on Linux). When the KV Cache hits 95% of your physical RAM limit, the C engine silently starts swapping the oldest conversation pages out of RAM and writing them directly to your SSD. It retrieves them seamlessly when the AI needs to "remember" them. You get practically infinite context memory, bounded only by your hard drive space.

### 3. Test-Time Training (TTT) — The Transformer Killer

We future-proofed the engine for State Space Models (Mamba) and Transformers. But there is a brand new, radical architecture called Test-Time Training (TTT).

**The Concept:** Instead of the AI using a KV Cache or a fixed hidden state to remember the prompt, the model actually contains a microscopic, untrained neural network inside of it. As the AI reads your prompt, it literally runs gradient descent and trains that tiny internal network on your words in real-time.

**The C Implementation:** We build a lightweight backpropagation (training) loop directly into the forward pass. Instead of just adding and multiplying numbers, the engine updates its own internal weights on the fly to compress your context into its brain.

### 4. Cryptographic Watermarking (Toggleable Output Security)

If you are building an engine that can spawn autonomous, audio-capable, reasoning agents, you are building something that can generate massive amounts of undetectable synthetic data.

**The Frontier:** Mathematical Watermarking (like Google's SynthID).

**The C Implementation:** Inside the Sampling Loop (Phase 7), we introduce a cryptographic hash function seeded by a private key. It slightly bumps the probability of specific tokens in a pattern invisible to humans, but detectable by an algorithm. This allows you to cryptographically prove whether a piece of text or code was generated by your specific local engine.

---

## The "God-Engine" Architecture: Master Blueprint

**Philosophy:** Zero bloat, zero Python at runtime, pure bare-metal C. The engine maps the models directly from the hard drive, reshapes its own physics based on available RAM, and routes tasks using AVX2 SIMD and L1 cache tiling.

### PILLAR I: Memory & Hardware Physics (The Foundation)

* **Hardware Probing:** `sysconf` and `CPUID` dynamically detect AVX2/AVX-512 instructions, L1 cache size, and total physical RAM on boot.
* **Aligned Allocation:** Uses `posix_memalign(64)` to guarantee every memory block perfectly aligns with 256-bit AVX cache lines.
* **Disk-to-RAM Bridge (mmap):** Maps the `.bin` or `.gguf` weights directly to virtual memory.
* **NVMe Storage Tiering (FlexGen):** Uses `io_uring` to asynchronously swap old conversation data from RAM to the SSD when physical memory hits 95% capacity.

### PILLAR II: The Math & Sequence Mixing (The Brain)

* **AVX2 Ternary LUT Engine:** For 1.58-bit models, uses `_mm256_shuffle_epi8` to unpack 4-to-1 ternary weights in a single clock cycle.
* **L1 Cache Block Tiling:** Slices massive 4096 × 4096 matrices into 64 × 64 chunks that never spill out of the CPU's L1 cache.
* **Universal Architecture Router:** Routes to Transformers (Attention/KV Caches) or Mamba/SSMs (fixed-state vectors) depending on the model.
* **Dynamic RoPE Stretching (YaRN):** Mathematically interpolates the sine/cosine frequencies on the fly.

### PILLAR III: Advanced Context Management (The API Backend)

* **PagedAttention:** Chops the KV Cache into non-contiguous blocks of 16 tokens for Continuous Batching.
* **Radix Tree Prompt Caching:** Hashing system prompt text for instant retrieval, dropping Time-To-First-Token to 0.1s.

### PILLAR IV: Cognition & Constraint (The Control Unit)

* **MCTS & Process Reward Models:** Spawns multiple hidden thought branches, scores them, outputs the winning logic.
* **FSM Grammar Constraining (JSON Mode):** Forces illegal syntax probability to negative infinity.
* **Cryptographic Watermarking (SynthID):** Private-key seeded hash function for invisible watermarks.

### PILLAR V: Multimodal IO (The Senses)

* **Vision ViT Projectors:** Uses `stb_image.h` to read images, MLP projector for dimension matching.
* **Audio Latent Streaming:** WebSocket connected to EnCodec/Mimi for native 300ms-latency voice.

### PILLAR VI: Autonomy & Networking (The Swarm)

* **WASM Sandboxing:** Embedded WebAssembly runtime for safe AI tool execution.
* **Distributed Wi-Fi Inference (RPC):** Split model layers across multiple machines over TCP/IP.
* **The OpenAI-Compatible API:** Single-file `mongoose.c` HTTP server with SSE streaming.

### Technology Stack

| Component | Library / Technique | Purpose |
|---|---|---|
| Language | Pure C99 / POSIX | Maximum portability, zero bloat |
| Math SIMD | `<immintrin.h>` | Bypassing multiplication; AVX2/AVX-512 |
| Memory | mmap, posix_memalign | Memory-mapped neural network loading |
| Web API | mongoose.c, cJSON.h | OpenAI API compatibility |
| Images | stb_image.h | Dependency-free image ingestion |
| Sandboxing | wasm3 | Secure embedded code execution |

---

## Phase 16: SIMD Upgrade Path — Beyond 22 tok/s

*Added: 2026-03-17 — Based on empirical benchmarks, BitNet.cpp comparison (Addendum J), and comprehensive SIMD landscape analysis (Addendum K, PERFORMANCE_CEILING_REPORT.md)*

---

### 16.0 Current State Assessment

The engine as of Phase 15 achieves **15–16 tok/s** (T=6, AVX-512F float32 FMA, i5-11300H DDR4-2667).
The root cause of the performance gap vs BitNet.cpp (22–23 tok/s) is **not memory bandwidth** —
it is the compute kernel: we use `_mm512_fmadd_ps` (16 float32 MACs/cycle) while BitNet.cpp uses
`_mm512_dpbusds_epi32` (64 int8 MACs/cycle).

The DRAM bandwidth ceiling (~33–37 tok/s) has not been hit by any engine on this hardware.

### 16.1 SIMD Instruction Compatibility Matrix (i5-11300H Tiger Lake)

#### Available on Hardware — Relevant to Inference

| Instruction Set | Availability | MACs/cycle | Current Use | Priority |
|----------------|-------------|------------|-------------|----------|
| AVX-512F float32 FMA | ✅ Active | 16 | Core kernel | Baseline |
| **AVX-512 VNNI int8** | ✅ Available | **64** | **Not used** | 🔴 **#1 Implement** |
| AVX-512 VBMI (byte permutation) | ✅ Available | N/A | Not used | 🟡 #2 (LUT kernel) |
| AVX-512 VBMI2 (compress/shift) | ✅ Available | N/A | Not used | 🟢 #3 (unpack) |
| AVX-512 BW (byte/word ops) | ✅ Active | N/A | Used for pack | Secondary |
| AVX-512 DQ / VL | ✅ Active | N/A | Used implicitly | Secondary |
| AVX-512 BITALG (popcount8) | ✅ Available | N/A | Not used | Future (binary models) |
| AVX-512 VPOPCNTDQ | ✅ Available | N/A | Not used | Future (binary models) |
| AVX-512 IFMA (52-bit int) | ✅ Available | N/A | Not used | Skip (wrong precision) |
| GFNI (Galois field) | ✅ Available | N/A | Not used | Skip (<2% gain) |
| AVX2 fallback | ✅ Active | 8 | Tail loops | Already optimal |

#### NOT Available on i5-11300H Tiger Lake (DO NOT ATTEMPT)

| Instruction Set | Why Unavailable | Notes |
|----------------|----------------|-------|
| AVX-512 BF16 | Tiger Lake lacks it | Available on Sapphire Rapids, Zen4 only |
| AVX-512 FP16 | Tiger Lake lacks it | Available on Sapphire Rapids, Meteor Lake+ |
| AMX | Server-only (Sapphire Rapids Xeon) | 1024 MACs/cycle but not in any laptop/desktop Tiger Lake |
| ARM NEON / SVE2 | x86 architecture | Engine already stubs NEON in simd_dispatch.c for ARM targets |

### 16.2 Phase K-1: AVX-512 VNNI Core Kernel

**Goal:** Match BitNet.cpp at 22–24 tok/s. Expected gain: +40–60% over current.

#### 16.2.1 Architecture

```
Current data flow per token:
  x (float32) ──► matmul_packed_avx512 ──► out (float32)
                   [unpack 2-bit → int32 → float32 → FMA]

Phase K-1 data flow:
  x (float32) ──► quantize_to_i8 ──► xi8 (int8)   ← ONE TIME per layer
                                          │
  w_packed (2-bit) ──► [model load] ──► w_i8 (int8) + bias (int32) per row
                                          │
  xi8 + w_i8 ──► matmul_vnni ──► out (int32) ──► dequant ──► out (float32)
                  [dpbusds: 64 int8 MACs/instruction]
```

#### 16.2.2 New Files Required

| File | Purpose |
|------|---------|
| `src/math/ternary_matmul_vnni.c` | VNNI int8 matmul kernel |
| `src/math/quantize_i8.c` | Float32→int8 activation quantizer |
| `include/math/ternary_matmul_vnni.h` | Function declarations |
| `include/math/quantize_i8.h` | Quantizer declarations |

#### 16.2.3 Files to Modify

| File | Change |
|------|--------|
| `src/core/weights.c` | Expand packed ternary → int8 at model load; allocate `wq_i8`, `wk_i8`, etc. |
| `include/core/weights.h` | Add `int8_t *wq_i8[30]` and `int32_t *wq_bias[30]` etc. to `TransformerWeights` |
| `src/math/simd_dispatch.c` | Add `TN_HAS_AVX512VNNI` tier; new `tn_ternary_matmul_vnni` function pointer |
| `include/math/simd_dispatch.h` | Declare new dispatch entry points |
| `include/core/platform.h` | Add `TN_HAS_AVX512VNNI` compile-time detection |
| `src/transformer/forward.c` | Call `quantize_to_i8` before each layer's matmuls |
| `src/math/parallel_matmul.c` | Add `parallel_ternary_matmul_vnni()` variant |

#### 16.2.4 VNNI Kernel Design

```c
/* Key inner loop (conceptual — see ternary_matmul_vnni.c for full implementation) */
/* dpbusds(acc, a_unsigned, b_signed): acc += sum_k(a[k] * b[k]) for groups of 4 */

for (j = 0; j + 63 < n; j += 64) {
    __m512i a_u8 = load_and_offset(xi8 + j);   /* shift +127 to make unsigned */
    __m512i b_i8 = load(wi8_row + j);           /* signed ternary {-1, 0, +1} */
    acc = _mm512_dpbusds_epi32(acc, a_u8, b_i8);
}
/* Reduce + subtract bias + apply x_scale * w_scale */
out[i] = ((float)reduce_add(acc) - bias[i]) * x_scale * w_scale;
```

**Memory cost:** 521 MB for 30 layers of int8 weights (vs 1.18 GB packed ternary already loaded).
Total inference footprint after Phase K-1: ~1.7 GB (up from ~1.2 GB). With 16 GB RAM, this is fine.

#### 16.2.5 Dispatch Tier Addition

```
Priority: AVX-512 VNNI > AVX-512F > AVX2 > Scalar

New simd_dispatch.c tier:
  #if TN_HAS_AVX512VNNI
    tn_ternary_matmul_vnni = ternary_matmul_vnni;   /* new VNNI path */
    tn_ternary_matmul_packed = ternary_matmul_packed_avx512;  /* keep for non-VNNI hardware */
    ...
  #endif
```

#### 16.2.6 Expected Performance After Phase K-1

| Metric | Current | After K-1 |
|--------|---------|-----------|
| tok/s (T=6) | 15.29 | ~22–25 |
| DRAM utilization | 43% | ~75–80% |
| Compute bottleneck | FMA pipeline | DRAM bandwidth |
| IPC | ~1.0 | ~0.8–0.9 (DRAM-bound) |

---

### 16.3 Phase K-2: Kernel Fusion + Compile Optimization

**Goal:** 26–30 tok/s. Builds on Phase K-1.

#### 16.3.1 Fused RMSNorm + VNNI MatMul

Create `ternary_matmul_vnni_normed()` that:
1. Computes `rms = 1/sqrt(mean(x^2))` inline (single AVX-512 pass over x)
2. Applies `w_norm[j] * rms` as a per-element scaling factor inside the VNNI loop
3. Eliminates write of `xb` and subsequent read in matmul (saves 2 × 10 KB = 20 KB per layer call)

**Files:** New `src/math/ternary_matmul_vnni_normed.c`

#### 16.3.2 2× Loop Unrolling

In `ternary_matmul_vnni.c`, compute 2 output rows per outer loop iteration:
- Load `xi8` once into registers, compute dot products for rows `i` and `i+1` simultaneously
- Reduces effective bandwidth by 15% (sharing the same 64-byte activation chunk)

#### 16.3.3 LTO + PGO Compilation

Add to `Makefile`:
```makefile
CFLAGS_PROFILE = $(CFLAGS_COMMON) -O3 -march=native -fprofile-generate
CFLAGS_OPTIMIZED = $(CFLAGS_COMMON) -O3 -march=native -fprofile-use -flto=thin
```

Usage:
```bash
make CFLAGS="$(CFLAGS_PROFILE)"              # Build with profiling
./adaptive_ai_engine --prompt "..." --max-tokens 50  # Run to generate profile
make CFLAGS="$(CFLAGS_OPTIMIZED)"            # Rebuild with profile data
```

---

### 16.4 Phase K-3: RAM Optimizations

**Goal:** Additional 5–10% on top of K-2. Minimal code changes.

#### 16.4.1 KV Cache Strategy Threshold Correction

In `src/kv_cache/kv_strategy.c`, for 16 GB systems:
- MemAvailable ~7–9 GB should trigger `KV_FULL_F16` not `KV_SLIDING_I4`
- Raises `GB_12` threshold check to properly apply on this system
- Benefit: Full 4096-token context with float16 KV quality

#### 16.4.2 MADV_HUGEPAGE on Weight mmap

In `src/memory/mapped_file.c`, after `mmap()`:
```c
madvise(ptr, size, MADV_HUGEPAGE);  /* request 2 MB huge page backing */
```

#### 16.4.3 NT Stores for KV Cache Writes

In `src/transformer/attention.c`, replace `memcpy` for KV cache writes with NT stores:
```c
/* Use _mm512_stream_ps instead of memcpy for KV cache slots far from current position */
```

#### 16.4.4 Vectorized SiLU exp (libmvec)

In `Makefile`, add to math kernel sources only (not the whole project):
```makefile
src/math/elementwise_avx512.o: CFLAGS += -ffast-math
```
This enables GCC libmvec to vectorize `expf()` in `silu_avx512()`.

---

### 16.5 Phase K-4: Speculative Decoding (Future)

Requires a small draft model (~100M parameters, ternary). Not currently available.
Architecture: run draft model 4× faster → predict N tokens → verify with main model.
Expected gain: +3–6 tok/s for a ×3–4 acceptance rate draft model.

---

### 16.6 Practical Benchmark Data (SIMD Comparison)

Measurements taken 2026-03-17 on CI Xeon @ 2.10 GHz with identical AVX-512 VNNI support:

| Path | Time/iter (N=2560, D=6912) | GOPS | Speedup vs current |
|------|---------------------------|------|-------------------|
| AVX-512F float32 FMA | 1.467 ms | 24.1 | 1.00× (baseline) |
| AVX-512 VNNI int8 | 0.636 ms | 55.6 | **2.31×** |
| AVX-512 BF16 dpbf16 | N/A (Tiger Lake lacks) | N/A | Incompatible |

Memory bandwidth profile (sequential single-thread):

| Buffer | BW |
|--------|-----|
| 1 MB (L3) | 53 GB/s |
| 16 MB (L3/DRAM) | 16 GB/s |
| 256 MB (DRAM) | 14–15 GB/s |
| 4-thread aggregate | 27–46 GB/s |

**Key finding:** VNNI delivers 2.3× compute speedup in isolated benchmarks. In full inference
(DRAM-bound), the practical gain is ~1.5× (matching BitNet.cpp). With Phase K-2 fusion,
~2× over current = **~30 tok/s** is achievable.

---

### 16.7 Maximum Performance Ceiling Revised

| Stage | tok/s (T=6, i5-11300H) | Bottleneck |
|-------|----------------------|------------|
| Current (Phase 15) | 15–16 | Compute (AVX-512F FMA, 43% DRAM) |
| After Phase K-1 (VNNI) | 22–25 | DRAM bandwidth (~75–80%) |
| After Phase K-2 (fusion) | 26–30 | DRAM bandwidth (~85–90%) |
| After Phase K-3 (RAM) | 28–33 | DRAM bandwidth (~90–95%) |
| Hard ceiling | ~33–37 | 100% DRAM saturation |
| Thermal ceiling (sustained) | ~28–30 | TDP exhaustion after ~4 consecutive runs |

**The prior claim of "15–16 tok/s as the ceiling" was incorrect.** That was the ceiling
for the float32 FMA compute kernel, not the hardware. The true hardware ceiling is 33–37 tok/s.


---

## Phase 16-S Implementation Record

*Added: 2026-03-17 — Documents the actual implementation of Phase 16 SIMD kernels,
multi-architecture hardware auto-detection, and microbenchmark results.*

### 16-S.1 What Was Built

Phase 16-S delivers the full SIMD upgrade across all supported hardware tiers. The
implementation is in pure C with compile-time `#if` guards for each ISA so the codebase
compiles to a stub on unsupported hardware without error.

**New files:**

| File | Purpose |
|------|---------|
| `include/math/cpu_features.h` | `TnCpuFeatures` struct + detection API |
| `src/math/cpu_features.c` | CPUID (x86) + auxval/sysctl (ARM) runtime detection |
| `include/math/quantize_i8.h` | float32 → int8 quantization API |
| `src/math/quantize_i8.c` | Scalar + AVX-512 + AVX2 quantize/sum_i8 |
| `src/math/ternary_matmul_packed_vnni.c` | AVX-512 VNNI kernel (64 MACs/cycle) |
| `src/math/ternary_matmul_packed_avx_vnni.c` | AVX-VNNI 256-bit kernel (32 MACs/cycle) |
| `src/math/ternary_matmul_packed_dotprod.c` | ARM SDOT kernel (16 MACs/cycle) |
| `tools/bench_simd.c` | Synthetic microbenchmark measuring all tiers |
| `tests/test_simd_vnni.c` | 50-test correctness suite for all new kernels |

**Modified files:**

| File | Change |
|------|--------|
| `include/core/platform.h` | Added `TN_HAS_AVX512VNNI`, `TN_HAS_AVXVNNI`, `TN_HAS_ARM_DOTPROD`, `TN_HAS_ARM_SVE2`, `TN_ARCH_X86/ARM` macros |
| `include/math/ternary_matmul_packed.h` | Added `_vnni`, `_avx_vnni`, `_dotprod` declarations |
| `src/math/simd_dispatch.c` | Full rewrite: runtime-detected 6-tier dispatch chain |
| `src/memory/mapped_file.c` | Added `MADV_HUGEPAGE` (Linux) / `MADV_ZERO_WIRED_PAGES` (macOS) |
| `src/kv_cache/kv_strategy.c` | Fixed RAM thresholds: GB_12→GB_8, GB_10→GB_5 (K-3 fix) |
| `Makefile` | Per-file CFLAGS for VNNI kernels; dynamic SIMD test flags; `bench` target |

### 16-S.2 Hardware Auto-Detection Architecture

The engine uses **two-layer detection** to maximize portability:

**Layer 1 — Compile-time** (`include/core/platform.h`):
```c
#if defined(__AVX512VNNI__)
    #define TN_HAS_AVX512VNNI 1   /* set by -march=native or -mavx512vnni */
#endif
```
Guards which kernel *code* is compiled. Without the flag, the function compiles to
`ternary_matmul_packed_avx512` fallback (or a no-op on non-x86). Zero SIGILL risk.

**Layer 2 — Runtime** (`src/math/cpu_features.c`):
```c
const TnCpuFeatures *tn_cpu_features_detect(void);  /* cached, thread-safe after init */
```
Uses CPUID leaf 7 sub-leaf 0/1 on x86, `getauxval(AT_HWCAP)` on Linux/ARM, and
`sysctlbyname("hw.optional.arm.FEAT_DotProd", ...)` on macOS. Enables portable binaries
that select the best kernel at startup even on CPUs unknown at compile time.

**Dispatch priority chain** (`src/math/simd_dispatch.c`):

```
AVX-512 VNNI (64 MACs/cycle)  ← avx512vnni=1 (Ice Lake+, Zen 4+)
    ↓
AVX-VNNI 256-bit (32 MACs/cycle) ← avx_vnni=1, no avx512 (Alder Lake, Zen 3)
    ↓
AVX-512F float32 (16 MACs/cycle) ← avx512f=1
    ↓
AVX2 float32 (8 MACs/cycle) ← avx2=1 (Haswell+, any Ryzen)
    ↓
ARM SDOT int8 (16 MACs/cycle) ← arm_dotprod=1 (Apple M1+, Cortex-A75+)
    ↓
ARM NEON float32 (4 MACs/cycle) ← neon=1
    ↓
Scalar (any CPU)
```

`tn_simd_init()` returns the name of the selected backend as a string, which is logged
at startup. The function pointer `tn_ternary_matmul_packed` is set once on first call
and cached for the process lifetime.

### 16-S.3 VNNI Bias Correction (w_enc trick)

AVX-512 VNNI and AVX-VNNI implement `_mm512_dpbusds_epi32(acc, a_u8, b_i8)` where `a`
must be **unsigned** int8. Ternary weights w ∈ {-1, 0, 1} cannot be stored as uint8.

**Solution:** Pack weights as `w_enc = w + 1 ∈ {0, 1, 2}` (always uint8-safe).

```
dpbusds(w_enc, q_x) = Σ (w+1) * q_x = Σ w*q_x + Σ q_x
                    = true_dot_product + sum_qx

→ true_dot = dpbusds_result - sum_qx
```

`sum_qx` is computed once per activation vector using `sum_i8_avx512()`, then subtracted
before dequantization. No extra memory. No extra passes over the weight matrix.

ARM SDOT (`vdotq_s32`) is signed×signed, so weights can be stored as `int8 ∈ {-1, 0, 1}`
directly — no bias correction needed.

### 16-S.4 Microbenchmark Results

Run `make bench` to reproduce. Results on Intel Xeon @ 2.80 GHz (avx512_vnni):

```
=== Phase 16-S SIMD Kernel Microbenchmarks ===
    Matrix dims: 2048 × 2048  (N=input, D=output)
    Warmup: 5 iters, Measured: 50 iters

Hardware: AVX-512 VNNI
  AVX2=1  AVX-512F=1  AVX-512VNNI=1  AVX-VNNI=0  ARM-DOTPROD=0

  Kernel                      ms/call       GMAC/s  ~tok/s
  ----------------------------------------------------------------------
  AVX2 float32              1.419 ms/call    2.95 GMAC/s  ~  4.9 tok/s
  AVX-512F float32          0.387 ms/call   10.83 GMAC/s  ~ 17.9 tok/s
  AVX-512 VNNI int8         0.334 ms/call   12.57 GMAC/s  ~ 20.8 tok/s
  → VNNI speedup        (4.3× vs AVX2 float32)
  → Dispatched            0.342 ms/call   12.25 GMAC/s  ~ 20.3 tok/s
```

**Notes:**
- The synthetic tok/s assumes 144 matmuls per token (24 layers × 6 matmuls/layer). Real
  end-to-end tok/s is 3–5× lower due to attention, sampling, embedding, and DRAM latency.
- VNNI is only 16% faster than AVX-512F on this Xeon because the Xeon's DRAM bandwidth
  (not compute) is already the bottleneck at N=2048. At smaller N (L3-fitting) the
  compute gap is larger.
- AVX-VNNI (Alder Lake) not present on this Xeon; falls back to AVX-512 VNNI correctly.
- The dispatch function pointer adds < 1% overhead (0.008 ms per call).

### 16-S.5 Test Coverage

Run `make build/tests/test_simd_vnni && build/tests/test_simd_vnni`.

The 50-test suite covers:
- `quantize_row_to_i8`: range, zero-input, max→127, scale consistency, AVX-512 matches scalar
- `sum_i8`: scalar, all-127, AVX-512 matches scalar
- AVX-512 VNNI matmul: small (64×16), large (2048×128), all-zero weights, odd-n tail
- AVX-VNNI matmul: medium (512×64), odd-n tail (skipped if not available)
- CPU detection: struct populated, caching, best_backend name
- SIMD dispatch: returns backend name, all function pointers non-NULL, VNNI selected
- KV strategy: 16 GB, 8 GB, 4 GB, and server-class RAM thresholds

Tolerance for int8 VNNI vs float32 reference: `10% rtol + 5% absolute`. This accounts for
quantization error: expected RMS error ≈ `sqrt(n_nonzero)/127 × w_scale` per output element.

### 16-S.6 Build System Notes

Per-file Makefile overrides ensure SIMD flags don't bleed into unrelated objects:
```makefile
build/math/ternary_matmul_packed_vnni.o: $(CC) $(CFLAGS) -mavx512vnni
build/math/ternary_matmul_packed_avx_vnni.o: $(CC) $(CFLAGS) -mavxvnni
build/memory/mapped_file.o: $(CC) $(CFLAGS) -D_GNU_SOURCE   # for MADV_HUGEPAGE
```

The test binary probes `/proc/cpuinfo` to add only the VNNI flags the build machine
actually supports, preventing SIGILL when running debug tests on CPUs that have
`avx512_vnni` but not `avx_vnni` (VEX-encoded 256-bit VNNI).

---

## Phase K-4/K-5 Implementation Record

*Implemented: 2026-03-19. Full benchmark data in Addenda P–AC of PERFORMANCE_CEILING_REPORT.md.*

### K-4/K-5.1 Parallel Dispatch Model

The original thread pool used a "boss-waits" model: main thread dispatches work to N workers
and spins/sleeps until all workers report done. Phase K-5 replaced this with
**caller-participates dispatch**:

```
Old model:
  main → [dispatch] → worker_0, worker_1, worker_2, worker_3
  main → [spins waiting on barrier]         ← wasted core

New model (K-5.1):
  main → [dispatch] → worker_0, worker_1, worker_2
  main → [runs worker_3 itself]             ← free throughput
  main → [then waits on remaining workers]
```

At T=4, this converts 3 worker threads + 1 idle main thread into 4 fully active threads,
yielding +15% throughput with zero extra hardware cost. Entry point:
`threadpool_dispatch_caller_participates()` in `src/threading/thread_pool.c`.

### K-4/K-5.2 T=8 Adaptive Blocking-Wait

The spinlock-based wait strategy (`SPIN_ITERS=40,000`) works well when T ≤ physical cores:
each thread maps to a unique physical core, and the spin holds the core warm between dispatches.
But at T=8 on Tiger Lake (4P × 2 HT), all 8 logical threads are scheduled, and sibling HT
threads sharing one physical core will spin against each other, consuming the shared execution
pipeline and starving actual matmul work.

Fix: `detect_physical_cores()` reads `/proc/cpuinfo` unique `(physical_id, core_id)` pairs.
`use_blocking_wait = (n_threads >= physical_cores * 2)`. When active, workers skip the spin
loop and go straight to `pthread_cond_wait`.

Result on Tiger Lake:

| T | Before fix | After fix |
|---|---|---|
| 4 | 19.48 tok/s | 19.48 tok/s (no change) |
| 8 | 2.6 tok/s | **21.53 tok/s** (+188%) |

### K-4/K-5.3 Layer-Level Pre-Quantisation

In INT8 mode, each matmul call quantises the activation vector `x` from float32 to int8 before
calling the VNNI kernel. Without pre-quantisation, `Q`, `K`, and `V` projections each
independently quantise `s->xb` — producing three identical outputs from three separate AVX-512
passes over the same 2560-float vector.

Phase K-5 introduces `TnPreqActivation`:

```c
typedef struct {
    int8_t  *buf;          // caller-owned aligned buffer
    float    inv_scale;    // 1.0 / quantisation_scale
    int      valid;        // 1 = buf is populated
} TnPreqActivation;
```

Callers in `attention.c` (Q/K/V group) and `ffn.c` (gate/up group) call `tn_preq_prepare()`
once and pass the struct to all projections in that group. The kernel checks `preq->valid`
and skips requantisation if already done.

Savings per token: ~2 µs vs 42 ms total = 0.005%. The workload is DRAM-bound; the
improvement is primarily an architectural correctness gain (no divergent quantisation rounding
between projections that should see identical inputs).

### K-4/K-5.4 Calibration System Architecture

**Design goals:**
1. First-run only (or hardware-change re-trigger) — no per-inference overhead
2. Produce reliable numbers free of thermal burst (PL2 turbo) and background load false positives
3. Report progress to user — calibration takes 10–20 s

**Hardware fingerprint:**
`tn_calibration_load()` computes a fingerprint from CPU model string, physical core count,
DRAM bandwidth probe, and SIMD tier. If the cached `.calib` file matches, results are used
directly. If not (first run, or hardware changed), `tn_calibrate()` runs.

**Two-phase benchmark (`bench_robust`):**

```
Phase 1 — SIMD comparison (all available backends, T=4):
    warmup: run matmul for 4 seconds (first), 1.5 s (subsequent) — covers PL2 window
    measure: time individual reps, take median of 20 reps
    report:  dots during warmup, result per backend

Phase 2 — Thread sweep (best SIMD backend, T=1..N_max):
    N_max = min(physical_cores × 2, TN_CALIB_MAX_THREADS)
    for each T: warmup 1.5 s, measure 20 reps (median)
    report: live result per T, best_threads updated
```

**Background load guard:**
Before each benchmark, `/proc/stat` is sampled for 500 ms with our threads idle.
If background CPU usage > 30%, result is flagged `[!]` in output and written to
`thread_sysload_pct[]` in the cache file. Users can judge whether to re-run.

**Previous benchmark bug (K-5.6 fix):**
The original `bench_matmul(n_threads, reps)` called `threadpool_create(n_threads)` but then
dispatched work via `tn_ternary_matmul_packed` — the single-threaded function pointer.
The pool was created and immediately destroyed, unused. T had zero effect on calibration.
All pre-fix calibration results are invalid. `TN_CALIB_VERSION` bumped 1→2 to auto-invalidate.

### K-4/K-5.5 INT8 Bandwidth Advantage

The LM head is the dominant DRAM consumer per token: 32,768 output classes × 2,560 input
dimensions = 83.9 M ternary weights. Per-token:

| Mode | Weight bytes read | Effective DRAM | Tok/s (Xeon, measured) |
|---|---|---|---|
| BF16 (float16 accumulation) | 131 MB | 16 GB/s ceiling = 122 tok/s | 20–22 |
| INT8 per-row | 65.5 MB | 32 GB/s effective ceiling | 30–36 |

The +36% average speedup from INT8 is purely from halving LM-head DRAM traffic. Weight
quality is preserved by per-row dynamic scale factors (`scale = max_abs / 127`).

VNNI `_mm512_dpbusds_epi32` requires unsigned × signed operands. Since ternary weights
{-1, 0, +1} encode to {0, 1, 2} (`w_enc = w + 1`), the kernel corrects:
```
true_dot = dpbusds(w_enc_u8, q_x_i8) - 128 * sum(q_x_i8)
```
`sum_i8_avx512()` uses `_mm512_cvtepi8_epi32` in four quarters to avoid signed-overflow.



---

## Phase 17: Mixture of Experts (MoE) Routing *(Implementation Record)*

### Architecture Motivation

Dense models (BitNet, Llama) activate every FFN weight for every token. MoE models store
N independent FFN "experts" per layer and use a tiny gate network to route each token to
only the top-k of them. This gives the model a large parameter knowledge base while keeping
per-token compute identical to a much smaller dense model.

**Example -- DeepSeek-V2-Lite:**
- 16B total parameters, 64 experts per MoE layer, top-k=6 active per token
- Effective active parameters per token: ~2.4B (same compute as a 2.4B dense model)
- Throughput: 29-31 tok/s @ T=3-4 on a single Xeon socket (see Addendum AJ)

### MoE Subsystem Architecture

    transformer_forward()
      +-- attention_forward()           <- standard (or MLA if has_mla=1)
      \-- ffn_forward()
           +-- dense path (mc->is_moe=0)  <- BitNet/Llama -- UNCHANGED, zero overhead
           \-- moe_ffn_forward()         <- MoE path (mc->is_moe=1)
                +-- moe_router_forward() <- gate: dim*N matmul -> softmax -> top-k
                \-- for each of top-k experts:
                     SwiGLU FFN (w1/w3/w2 for that expert) -> scale by gate score
                     accumulate into output buffer

**Key data structures:**
- MoEConfig (moe_config.h): num_experts, num_experts_per_tok, is_moe, + MLA fields
- TransformerWeights.moe_w1[layer][expert], .moe_w2, .moe_w3 -- expert weight pointers
- TransformerWeights.moe_gate_w[layer] -- gate projection (dim * num_experts)
- All NULL for dense models -- zero memory/compute overhead

---

## Phase 17.5-17.12: Multi-Head Latent Attention (MLA) Fix

### Problem: Degenerate MoE Output

DeepSeek-V2-Lite uses Multi-head Latent Attention (MLA). The original converter used
expand_mla_attention() which pre-multiplied KV projections (kv_b * kv_a -> standard wk/wv).
This approximation: (1) lost k_rope -- the 64-dim shared positional encoding on K was dropped;
(2) truncated q_rope -- query positional heads incorrectly sized. Result: model cannot track
token positions -> repetitive garbage output ("blueprint blueprint...").

### MLA Architecture

Standard attention: Q = wq @ x, K = wk @ x, V = wv @ x

MLA (DeepSeek) -- 3-matrix low-rank decomposition:
  q_full (3072)  = mla_wq (3072*2048) @ x          -> split: q_nope (2048) + q_rope (1024)
  kvc (576)      = mla_wkv_a (576*2048) @ x         -> split: kv_latent (512) + k_rope (64)
  kv_full (4096) = mla_wkv_b (4096*512) @ kv_latent -> split: k_nope (2048) + v (2048)

  attention score per head h = (q_nope_h . k_nope_h + q_rope_h . k_rope) / sqrt(192)

Key insight: k_rope is shared across all heads (absorbed dim=64, RoPE applied once).
Impossible to reproduce from the pre-expanded wk/wv approximation.

### Extended .bin Header for MLA (Non-Breaking)

  Bytes  0-27:  Original MoE header (num_experts, top_k, expert_hidden, is_moe, ...)
  Bytes 28-47:  MLA extension (has_mla, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
                -- was padding (all zeros) -> old files have has_mla=0 -> MLA path never taken
  Bytes 48-127: Reserved padding
  Byte 128+:    Weight data (UNCHANGED)

Old model files have bytes 28-47 = 0, so has_mla=0 and the MLA path is never taken.
No format break for any existing model.

---

## Phase 34.2b: Quantized GGUF Support (Q8_0, Q4_K, I2_S)

Q8_0: 32-weight blocks; 1*FP16 scale/block. out[i] = int8[i] * f16_scale.
Q4_K: 256-weight super-blocks; 12-byte header with 8 scales+mins packed FP6 per sub-block.
I2_S: 2 bits/weight; groups of 32; FP16 scale/group. {-1,0,+1} decode.

All three add cases to tensor_to_f32() switch in gguf_loader.c -- callers fully transparent.
New isolated file: include/core/gguf_quant.h + src/core/gguf_quant.c.

---

## Phase 34.5: GGUF-Native Tokenizer Extraction

GGUF files embed full vocabulary under tokenizer.ggml.* metadata keys.
New files: tokenizer_gguf.h + tokenizer_gguf.c -- reads tokens/scores/token_type arrays.
Requires ARRAY metadata support added to gguf_reader.c (GGUFMeta union extension).
Auto-detection in main.c: if GGUF model + no --tokenizer given -> load from model file.

---

## Phase 16-E: GGUF-Aware Bandwidth Ceiling

Removes hardcoded MODEL_TERNARY_BYTES/MODEL_VOCAB/MODEL_DIM from hardware_profile.c.
New API: tn_hardware_profile_update_model(const TnModelParams *mp)
Five-strategy cascade: S5 (init: N/A) -> S3 (.bin Config) -> S2 (GGUF metadata) ->
S4 (MoE sparsity-aware) -> S1 (file-size proxy fallback).
Model switch/add/delete -> caller re-invokes update_model() -> always accurate.

---

## Model Compatibility Reference

**Master reference for all model types — what is supported, how to load, and what is not supported.**

### BF16 with sufficient RAM

Yes — the engine fully supports BF16/F16 GGUF files regardless of model size, *provided RAM is available*.
The current GGUF loader dequantises all weight tensors to float32 at load time, which means a BF16 GGUF
requires 2× its file size in RAM (e.g., DeepSeek 16B BF16 = 32 GB file → 64 GB RAM needed).
A future optimisation (Phase 16-E / BF16-native matmul) would allow keeping weights as BF16 in memory,
halving the RAM requirement. For now, the practical limit is:

| Model | BF16 file | RAM needed (current) | RAM needed (future BF16-native) |
|---|---|---|---|
| SmolLM2 1.7B | 3.4 GB | 7 GB | 3.4 GB |
| Llama 3.2 3B | 6 GB | 12 GB | 6 GB |
| Mistral 7B | 14 GB | 28 GB | 14 GB |
| DeepSeek-V2-Lite 16B MoE | 32 GB | 64 GB | 32 GB |
| Llama 3 70B | 140 GB | 280 GB | 140 GB |

For any model where BF16 fits: download the F16 GGUF and pass it directly — the engine will load it at
100% quality with no conversion step.

---

### Conversion & Loading Workflow

```mermaid
flowchart TD
    START([You have a model]) --> Q1{What kind\nof model?}

    %% ── Branch 1: Ternary-trained ──────────────────────────────────────────
    Q1 -->|Ternary-trained\nweights ∈ {-1, 0, +1}\ne.g. BitNet b1.58| TERN_ARCH{Dense or MoE?}

    TERN_ARCH -->|Dense\ne.g. BitNet b1.58-2B\nBitNet-3B-4T| TERN_DENSE_CONV
    TERN_ARCH -->|MoE ternary\nfuture variant| TERN_MOE_CONV

    TERN_DENSE_CONV["python convert_hf.py\n--model ./path\n--output model.bin"] --> BIN_FILE
    TERN_MOE_CONV["python convert_hf.py\n--model ./path --moe\n--output model.bin"] --> BIN_FILE

    BIN_FILE[".bin ternary file\n2 bits/weight\n100% quality preserved\nFastest inference"] --> ENGINE_BIN
    ENGINE_BIN["Engine: load .bin\nternary matmul path\nINT4 / INT8 / BF16 classifier\n~25–48 tok/s"] --> OK1([✅ Coherent output])

    %% ── Branch 2: Standard float model ────────────────────────────────────
    Q1 -->|Standard float model\nFP16 / BF16 weights\ne.g. Llama, Mistral,\nDeepSeek, SmolLM2| RAM_Q

    RAM_Q{"How much RAM\ndo you have?"}

    RAM_Q -->|"RAM ≥ model × 2 bytes\n(BF16 fits)"| GET_F16["Get F16/BF16 GGUF\nHuggingFace direct download\n100% quality — no loss"]
    RAM_Q -->|"RAM ≥ model × 1 byte\n(Q8_0 fits)"| GET_Q8["Get or create Q8_0 GGUF\nllama-quantize model.gguf out.gguf Q8_0\n~99.5% quality"]
    RAM_Q -->|"RAM ≥ model × 0.5 bytes\n(Q4_K fits)"| GET_Q4["Get Q4_K_S GGUF\nHuggingFace or\nllama-quantize model.gguf out.gguf Q4_K_S\n~97–99% quality"]
    RAM_Q -->|"RAM < model × 0.5 bytes\neven Q4 doesn't fit"| TOO_BIG

    TOO_BIG["❌ Model too large\nfor available RAM\nNot supported\n\nOptions:\n• Get a smaller model\n• Increase RAM\n• Wait for chunked-load support"]

    GET_F16 --> GGUF_LOAD
    GET_Q8  --> GGUF_LOAD
    GET_Q4  --> GGUF_LOAD

    GGUF_LOAD["Engine: gguf_loader.c\nDequantises tensors → float32\nBuilds KV cache, run state"] --> ARCH_Q

    ARCH_Q{"Model\narchitecture?"}

    ARCH_Q -->|"LlamaForCausalLM\nMistralForCausalLM\nQwen2ForCausalLM\nSmolLM2\nPhi-3 etc."| DENSE_PATH
    ARCH_Q -->|"DeepSeekV2ForCausalLM\nMLA + MoE"| DS_PATH
    ARCH_Q -->|"MixtralForCausalLM\nStandard attn + MoE"| MIX_PATH
    ARCH_Q -->|"Anything else"| NOT_IMPL_ARCH

    DENSE_PATH["Dense transformer\nStandard RoPE attention\nDense SwiGLU FFN"] --> OK2([✅ Coherent output])
    DS_PATH["MLA attention\n+ MoE routing\n64 experts top-6\nmla_attention.c"] --> OK3([✅ Coherent output\n~12 tok/s Q4_K_S on i5])
    MIX_PATH["Standard attention\n+ MoE routing\nmoe_router + moe_ffn"] --> OK4([✅ Coherent output])
    NOT_IMPL_ARCH["⚠️ Architecture not\nyet implemented\nNeeds new forward pass"] --> CONTRIB([Open issue / contribute])

    %% ── Branch 3: Unsupported model types ──────────────────────────────────
    Q1 -->|Unsupported\narchitecture| UNSUP

    UNSUP{Which type?}

    UNSUP -->|"BERT / RoBERTa\nDeBERTa / ALBERT\nEncoder-only models"| NO1["❌ Encoder-only\nNo causal generation\nNot planned — different paradigm\n(classification, not generation)"]
    UNSUP -->|"T5 / FLAN-T5\nmT5 / UL2\nEncoder-Decoder"| NO2["❌ Encoder-Decoder\nDual-stack required\nNot planned — different paradigm\n(seq2seq, not causal)"]
    UNSUP -->|"Whisper / EnCodec\nAudio-native models"| NO3["⏳ Planned — Phase 25\nNative Audio Streaming\nEnCodec/Mimi codec\nWebSocket endpoint"]
    UNSUP -->|"Stable Diffusion\nFlux / SDXL\nDiffusion models"| NO4["⏳ Phase 36 — Text-to-Image\nFlux-Schnell Q4 / SD-LCM / SD-Turbo\nNew: DiT + VAE + Scheduler\nSee Phase 36 in IMPLEMENTATION_PLAN.md"]
    UNSUP -->|"Mamba / RWKV\nJamba hybrid\nSSM models"| NO5["⏳ Planned — Phase 22\nUniversal Architecture Router\nARCH_MAMBA / ARCH_RWKV\nFixed-state, no KV cache"]
    UNSUP -->|"GPTQ-quantized\nAWQ-quantized\nExLlama2 format"| NO6["⚠️ Workaround available\nConvert to GGUF first:\nllama-quantize → Q4_K_S\nNo native GPTQ/AWQ planned"]
    UNSUP -->|"LoRA adapters\nfine-tune checkpoints"| NO7["⏳ Planned — Phase 19\nHot-swappable LoRA\n.lora.bin format\nSwitch personality in ms"]
    UNSUP -->|"Vision-language\nLLaVA / Qwen-VL\nInternVL"| VISION_Q

    VISION_Q{"Vision encoder\nextracted?"}
    VISION_Q -->|"LLM backbone\nonly — yes"| OK_VIS([✅ Text backbone works\nVision encoder partial\nPhase 16-V stub])
    VISION_Q -->|"Full multimodal\nfusion"| NO8["⚠️ Full fusion\nnot yet implemented"]
```

---

### Quick-Reference Compatibility Table

| Model family | Recommended format | Quality vs BF16 | Approx. tok/s (i5-11300H) | Status |
|---|---|---|---|---|
| BitNet b1.58 (any size) | `.bin` ternary | 100% (native) | 25–48 | ✅ Supported |
| SmolLM2 135M–1.7B | GGUF F16 | 100% | 40–120 | ✅ Supported |
| Llama 3.2 1B–3B | GGUF F16 | 100% | 25–60 | ✅ Supported |
| Llama 3.1 8B | GGUF Q4_K_S | ~98% | ~8–10 | ✅ Supported |
| Mistral 7B v0.3 | GGUF Q4_K_S | ~98% | ~8–10 | ✅ Supported |
| Qwen2.5 7B | GGUF Q4_K_S | ~98% | ~8–10 | ✅ Supported |
| DeepSeek-V2-Lite 16B MoE | GGUF Q4_K_S | ~97% | ~12 (MoE sparse) | ✅ Supported (MLA) |
| Mixtral 8×7B | GGUF Q4_K_S | ~97% | ~8 (MoE sparse) | ✅ Supported |
| Llama 3.1 70B | GGUF Q4_K_S (needs 35 GB) | ~97% | ~2–3 | ✅ If RAM allows |
| Any model, full BF16 | GGUF F16 (if RAM allows) | 100% | Bandwidth-limited | ✅ If RAM allows |
| BERT / RoBERTa | — | — | — | ❌ Encoder-only — not planned (different paradigm) |
| T5 / FLAN | — | — | — | ❌ Encoder-Decoder — not planned (seq2seq) |
| Whisper / Audio | — | — | — | ⏳ Phase 25 — Native Audio Streaming (planned) |
| Stable Diffusion / Flux | — | — | — | ⏳ Phase 36 — Text-to-Image (Flux-Schnell Q4 / SD-LCM) |
| Mamba / RWKV / Jamba | — | — | — | ⏳ Phase 22 — Universal Architecture Router (planned) |
| GPTQ / AWQ | Convert to GGUF Q4_K_S first | — | — | ⚠️ Workaround: llama-quantize → GGUF |
| LoRA adapters | — | — | — | ⏳ Phase 19 — Hot-swappable `.lora.bin` (planned) |
| LLaVA / Qwen-VL (full fusion) | — | — | — | ⏳ Phase 11 stub exists — full fusion planned |

---

### The One Rule

> **Ternary `.bin`** is only for models *trained* as ternary (BitNet family).  
> **GGUF Q4_K or better** is for every other model — never force ternary on float weights.  
> **GGUF F16/BF16** is the gold standard when RAM allows it.

---

## What We Are Missing Without BERT/RoBERTa and T5/FLAN Support

These are permanently out-of-scope for this engine — not because they are low
quality, but because they solve a **fundamentally different problem** from
causal generation.

### BERT / RoBERTa / DeBERTa / ALBERT (Encoder-only)

**What they do:** Deep bidirectional encoding — every token attends to every
other token simultaneously (left AND right context). Optimised for
understanding, not generation.

**Use-cases we cannot serve without them:**

| Task | Examples | Why BERT is used |
|---|---|---|
| Semantic search / RAG retrieval | Embedding documents, query matching | Produces dense vector embeddings, not tokens |
| Named-entity recognition (NER) | Extracting dates, names, places from text | Per-token classification over full context |
| Sentiment / intent classification | Support-ticket routing, review scoring | CLS-token regression head |
| Extractive question-answering | Highlight the answer span in a document | Span-start / span-end head |
| Re-ranking search results | Colbert, cross-encoder | Bidirectional query × doc scoring |

**The gap in plain English:** Our engine can *generate* an answer; it cannot
efficiently *score* or *embed* a query. Retrieval pipelines (RAG) need an
embedding model. We work around this today by using external APIs (OpenAI
embeddings) or by running a separate embedding server. A native BERT encoder
would make the engine fully self-contained for RAG.

### T5 / FLAN-T5 / mT5 / UL2 (Encoder-Decoder)

**What they do:** Encode a full input sequence bidirectionally, then decode
a target sequence. The encoder and decoder are separate transformer stacks
that communicate via cross-attention.

**Use-cases we cannot serve without them:**

| Task | Examples | Why T5 is preferred |
|---|---|---|
| Translation | English → French at high fidelity | Encoder sees full source before any output |
| Summarisation | Long-doc → short summary | Full document context in encoder |
| Data-to-text | Table or JSON → natural language | Structured input → fluent output |
| Rewriting / paraphrase | Style transfer | Full input context first, then rewrite |
| Instruction following (FLAN) | Multi-task zero-shot | FLAN-T5 is extremely parameter-efficient |

**The gap in plain English:** Causal decoders (Llama/BitNet) can do most of
these tasks with prompting, but T5 is 3–5× more parameter-efficient for
conditional generation tasks. For on-device summarisation at 2–4 GB, FLAN-T5
Large (770M) would outperform Llama 3.2 1B in translation accuracy. We
recommend using an external T5 server for production summarisation pipelines.

**Why they are permanently out-of-scope:** Supporting encoder-decoder would
require a second transformer stack, a cross-attention mechanism between encoder
outputs and decoder layers, a separate tokeniser training objective, and
completely different KV cache semantics. This is architecturally incompatible
with the single-pass causal decoder design — it would be a parallel engine, not
an extension.

---

## GPTQ / AWQ: Native Support vs Workaround — Intelligence and Speed

### What GPTQ and AWQ Are

- **GPTQ** (Generative Pre-trained Quantization): PTQ that minimises per-layer
  quantization error by solving a least-squares problem on calibration data.
  Typically 4-bit asymmetric, sometimes 3-bit or 2-bit. Produces `.safetensors`
  or ExLlama2 format files.
- **AWQ** (Activation-aware Weight Quantization): Scales weights based on
  per-channel activation magnitudes before rounding. Slightly better quality
  than GPTQ at the same bit-width, especially for smaller models.

### Intelligence (Quality) Comparison

Both GPTQ and AWQ are **4-bit PTQ methods** targeting the same quality tier as
GGUF Q4_K. Once converted to GGUF Q4_K_S:

| Scenario | Method | PPL (Wikitext-2) | Quality loss |
|---|---|---|---|
| Full BF16 (baseline) | — | ~7.5 | 0% |
| Native GPTQ 4-bit | GPTQ | ~8.0 | ~2–3% |
| Native AWQ 4-bit | AWQ | ~7.9 | ~1–2% |
| GGUF Q4_K_S (GPTQ source) | convert → GGUF | ~8.1 | ~3–4% |
| GGUF Q4_K_S (fresh quant) | llama-quantize | ~8.1 | ~3–4% |

**The conversion loss is ~0.5–1 PPL** — almost nothing. If your model is
already in GPTQ/AWQ format, converting it to GGUF Q4_K_S loses no more
intelligence than the original quantization did.

> **Answer:** Converting GPTQ/AWQ → GGUF Q4_K_S loses negligible
> intelligence compared to native GPTQ/AWQ inference. The dominant quality
> loss came from the original 4-bit quantization, not from the format conversion.

### Speed Comparison

Native GPTQ (via ExLlama2/AutoGPTQ) uses highly optimised GPU CUDA kernels.
On CPU:

| Mode | Backend | Memory | Throughput |
|---|---|---|---|
| Native GPTQ | ExLlama2 (CUDA) | ~4 GB VRAM | 60–100 tok/s (GPU) |
| Native AWQ | AutoAWQ (CUDA) | ~4 GB VRAM | 55–90 tok/s (GPU) |
| **GGUF Q4_K_S** | **Our engine (CPU)** | **~4 GB RAM** | **8–25 tok/s** |
| llama.cpp Q4_K_M | llama.cpp (CPU) | ~4 GB RAM | 8–22 tok/s |

> **Answer:** On CPU, native GPTQ/AWQ offers **no speed advantage** over
> GGUF Q4_K_S — both are bandwidth-limited by DRAM. The CUDA advantage only
> applies to GPU inference. Converting first adds zero overhead at runtime.

### Practical Recommendation

```
User has GPTQ/AWQ model on disk
        │
        ├─ GPU available? → Use ExLlama2 / vLLM natively (fastest)
        │
        └─ CPU only?      → llama-quantize model.gguf out.gguf Q4_K_S
                            → Use our engine (same speed, same quality)
```

Conversion command (requires llama.cpp `llama-quantize` binary):
```bash
llama-quantize input_model.gguf output_model_Q4_K_S.gguf Q4_K_S
```

---

## GGUF: The Format That Needs No Conversion

### The Short Answer

If you already have a `.gguf` file, **you need zero conversion steps**.  
Point the engine directly at it — that is the entire workflow.

```bash
# No conversion. No Python. No setup. Just run.
./adaptive_ai_engine --model deepseek-v2-lite-chat-Q4_K_S.gguf --tokenizer ...
```

### What GGUF Actually Is

GGUF (GPT-Generated Unified Format) was created in 2023 by Georgi Gerganov, the author
of **llama.cpp**, specifically to be the final, runtime-ready format for CPU inference.
Before GGUF there was GGML (predecessor), and before that you had to run Python/PyTorch
just to load a model.

GGUF is a binary container. One file holds:

| What | How stored |
|---|---|
| Hyperparameters (n_layers, dim, heads, vocab…) | Metadata header |
| Full vocabulary + merge rules | Metadata (tokenizer section) |
| All weight tensors | Raw binary, packed in declared quant format |
| Quant type per tensor | Per-tensor metadata |

The CPU reads the header, memory-maps the tensor data, and runs — no decompression,
no format translation, no Python script required.

### Why "Conversion" Ever Comes Up

The word "conversion" in this project refers to three distinct, often confused things:

| Scenario | What you actually do | Is it conversion? |
|---|---|---|
| You have a GGUF file | `./engine --model foo.gguf` | ❌ No conversion — just run |
| You have HuggingFace `.safetensors` (float) | `llama-quantize → foo.gguf Q4_K_S` | ✅ Yes — PTQ to 4-bit GGUF |
| You have BitNet HuggingFace weights | `python convert_hf.py → foo.bin` | ✅ Yes — pack ternary to .bin |
| You have GPTQ / AWQ files | `llama-quantize → foo.gguf Q4_K_S` | ✅ Yes — change container format |

The rule: **GGUF is the destination, not the source that needs converting.**

### What llama.cpp Does — and Why We Exist Alongside It

**llama.cpp** is the gold-standard open-source project for CPU/GPU inference. It:

- Reads every GGUF file directly (`llama-cli -m model.gguf`)
- Supports all architectures natively: LLaMA, Mistral, DeepSeek MLA+MoE, Mixtral,
  Qwen, Phi-3, Falcon, GPT-2, and 70+ more
- Handles all quantization types: Q4_K, Q5_K, Q6_K, Q8_0, F16, BF16, I2_S, …
- Reads chat templates from GGUF metadata automatically
- Implements YaRN RoPE, MLA attention, sliding-window attention
- Has GPU offload (CUDA, Metal, Vulkan, SYCL)
- Is used as the backend by Ollama, LM Studio, and many other apps

**Can llama.cpp run DeepSeek-V2-Lite directly?**  
Yes — with one command:

```bash
llama-cli -m deepseek-v2-lite-chat-Q4_K_S.gguf \
          --chat-template deepseek2 \
          -p "What is the capital of France?"
```

No tokenizer file needed — llama.cpp reads the vocabulary from the GGUF metadata.

**Then why does this engine exist?**

| Feature | llama.cpp | This engine |
|---|---|---|
| BitNet ternary native inference | ❌ Uses I2_S (slower) | ✅ Native packed ternary matmul |
| Ternary INT4 classifier | ❌ Not implemented | ✅ VNNI-optimised |
| Single-binary, zero CMake deps | ❌ Requires cmake build system | ✅ Plain `make` |
| Custom hardware ceiling profiler | ❌ | ✅ |
| MoE expert activation tracing | ❌ | ✅ Built-in |
| Per-model adaptive thread tuning | ❌ | ✅ Phase 16 auto-tuner |
| Target use-case | All GPU/CPU, general purpose | BitNet + edge CPU, minimalist |
| GGUF standard model support | ✅ Full (reference implementation) | ✅ Subset (growing) |

**Use llama.cpp when:** you want the most compatible, battle-tested GGUF runner for
standard models (DeepSeek, LLaMA, Mistral) with GPU offload.

**Use this engine when:** you are running BitNet ternary models at maximum CPU speed
with the smallest possible binary, or you need the hardware profiling / expert tracing
built in.

---

## Model Taxonomy: Base, Instruct, and Chat Models

### Why This Matters for Our Engine

Every model file has two independent dimensions:

1. **Architecture** — how the transformer is built (Dense, MoE, MLA, …)
2. **Alignment** — what training stage it went through (base, SFT, RLHF)

The architecture determines *what code runs*. The alignment determines *how you must
format the prompt*. Getting alignment wrong produces garbage output even from a
perfectly functional model.

### The Three Alignment Tiers

```
Raw pretrained             +SFT                    +RLHF / DPO
─────────────────────────────────────────────────────────────────
Base Model          →    Instruct Model     →    Chat Model
"predict next token"     "follow instruction"     "converse naturally"
```

#### Tier 1 — Base Models

Trained solely on next-token prediction over vast text corpora. They have no concept
of questions, answers, or roles.

**Correct usage:** Provide a continuation prompt. Do NOT use chat markers.

```
Prompt:  "The capital of France is"
Output:  " Paris, located on the banks of the Seine river, and home to…"
```

```
Prompt:  "def fibonacci(n):"
Output:  "\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
```

Examples: `deepseek-v2-lite`, `llama-3-8b`, `mistral-7b-v0.1`, `bitnet-b1.58-2B-4T`

#### Tier 2 — Instruct Models (SFT)

Fine-tuned on curated (instruction, response) pairs. They understand commands but may
use a simple prompt format.

**Correct usage:** Use the specific prompt template documented by the model. Many use
plain `### Human:` / `### Assistant:` markers.

```
[INST] What is the capital of France? [/INST]
```

Examples: `llama-3-8b-instruct`, `mistral-7b-instruct-v0.3`, `phi-3-mini-instruct`

#### Tier 3 — Chat Models (SFT + RLHF/DPO)

Further aligned via Reinforcement Learning from Human Feedback or Direct Preference
Optimisation. These expect a structured **chat template** with explicit role markers.
Sending a plain "continuation" prompt to a chat model causes it to complete the
template syntax, not answer naturally.

**Each model family has its own template** — embedded in the GGUF metadata under
`tokenizer.chat_template`. Using the wrong template (or no template) for a chat model
is the single most common cause of "garbled output" in otherwise working models.

| Model family | Chat template format |
|---|---|
| DeepSeek-V2/V3 | `<\|im_start\|>user\n{text}<\|im_end\|>\n<\|im_start\|>assistant\n` |
| LLaMA 3 Chat | `<\|start_header_id\|>user<\|end_header_id\|>\n\n{text}<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n` |
| ChatML (Qwen, SmolLM2) | `<\|im_start\|>user\n{text}<\|im_end\|>\n<\|im_start\|>assistant\n` |
| Mistral / Mixtral | `[INST] {text} [/INST]` |
| Phi-3 / Phi-4 | `<\|user\|>\n{text}<\|end\|>\n<\|assistant\|>\n` |
| Vicuna / Alpaca | `### Human: {text}\n### Assistant:` |

Examples: `deepseek-v2-lite-chat`, `llama-3-8b-instruct` (Tier 3 variant), `qwen2.5-7b-instruct`

### Full Model Type Taxonomy

```mermaid
flowchart TD
    MT([Model File]) --> A1{Architecture}

    A1 -->|Decoder-only\nstandard attention + dense FFN| DENSE["Dense Decoder\nLLaMA / Mistral / GPT2\nBitNet / SmolLM2 / Phi"]
    A1 -->|Decoder + Sparse FFN\nMoE routing| MOE["MoE Decoder\nMixtral 8x7B\nDeepSeek-V2-Lite\nQwen2-MoE"]
    A1 -->|Decoder + Latent KV\nMLA attention| MLA["MLA Decoder\nDeepSeek-V2/V3/R1"]
    A1 -->|Encoder only\nbidirectional| ENC["Encoder Only\nBERT / RoBERTa\n❌ Not supported"]
    A1 -->|Encoder + Decoder| ENCDEC["Seq2Seq\nT5 / FLAN\n❌ Not supported"]
    A1 -->|Diffusion backbone\nnon-autoregressive| DIFF["Diffusion\nSD / Flux\n⏳ Phase 36"]
    A1 -->|SSM / hybrid| SSM["SSM\nMamba / RWKV\n⏳ Phase 22"]

    DENSE --> AL1{Alignment\ntier}
    MOE   --> AL1
    MLA   --> AL1

    AL1 -->|No SFT| BASE["Base\nNo chat template\nUse continuation prompts"]
    AL1 -->|SFT only| INST["Instruct\nSimple template\n[INST]…[/INST]"]
    AL1 -->|SFT + RLHF/DPO| CHAT["Chat\nFull chat template\n&lt;|im_start|&gt;…"]

    BASE --> W1{Weight\nformat}
    INST --> W1
    CHAT --> W1

    W1 -->|Ternary {-1,0,+1}\nBitNet trained| BINF[".bin ternary\nThis engine native path\nFastest CPU inference"]
    W1 -->|Float GGUF\nQ4_K / Q6_K / F16| GGUF_F["GGUF\nDirect load — no conversion\nllama.cpp or this engine"]
    W1 -->|HuggingFace safetensors\nFP16/BF16| HF["→ llama-quantize\nConvert to GGUF first\nthen load"]
    W1 -->|GPTQ / AWQ| GPTQ_W["→ llama-quantize\nConvert to GGUF Q4_K_S\nthen load"]
```

### How Our Engine Handles Each Tier

| Tier | Our engine support | What you must do |
|---|---|---|
| **Base model (GGUF)** | ✅ Load and run directly | Use continuation prompts: `"The capital of France is"` |
| **Instruct model (GGUF)** | ✅ Load and run directly | Wrap prompt in model's documented template |
| **Chat model (GGUF)** | ✅ Load and run | Manually wrap in chat template (see table above) |
| **Chat model (GGUF) — auto template** | ✅ Phase 34.5 complete | Engine auto-reads template from GGUF; `--tokenizer` optional |
| **Any model, HuggingFace `.safetensors`** | ⚠️ Requires conversion | Run `llama-quantize` → GGUF first |
| **BitNet ternary HuggingFace** | ✅ `convert_hf.py` → `.bin` | One-time conversion, then native path |

### Current DeepSeek Status

DeepSeek-V2-Lite-Chat is a **Tier 3 Chat model** (MLA+MoE architecture, SFT+RLHF).  
The required chat template for all DeepSeek-V2/V3 chat models:

```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
{your question here}<|im_end|>
<|im_start|>assistant

```

Without this template, the model treats the prompt as a continuation task and outputs
garbled multi-language text because it is completing the "assistant" token pattern
rather than generating a grounded answer.

> **✅ STATUS UPDATE (2026-03-22):** All three requirements below are **complete and
> working**. DeepSeek-V2-Lite-Chat produces correct factual output (`"The capital of
> France is Paris."`) both with and without `--tokenizer`. No garbled output.

Previously required (now **DONE**):
- **YaRN RoPE scaling** (factor=40, mscale=0.707) — ✅ implemented (commit `4e6fdfe`)
- **GGUF tokenizer extraction** (100k+ vocab without external .bin file) — ✅ Phase 34.5 complete (commit `26e772a`)
- **Chat template auto-injection** from GGUF metadata — ✅ Phase 34.5 complete; engine reads `tokenizer.chat_template` key from GGUF and applies it automatically

Validated on `2026-03-22` (commit `1435a4b`): `--tokenizer` flag is now optional for GGUF models; the engine auto-loads vocab + chat template from the model file itself.
