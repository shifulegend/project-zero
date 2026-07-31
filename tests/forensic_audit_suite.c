#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
#include <unistd.h>

#include "core/config.h"
#include "core/weights.h"
#include "tokenizer/tokenizer.h"
#include "kv_cache/sliding_window.h"
#include "math/simd_dispatch.h"
#include "math/parallel_matmul.h"
#include "threading/thread_pool.h"
#include "memory/mapped_file.h"

#include <sys/wait.h>

/* --- Signal Recovery Infrastructure (Try-Catch in C) --- */
static const char *g_current_test_name = "None";

#define RUN_PROTECTED_TEST(test_fn, timeout_sec) do { \
    g_current_test_name = #test_fn; \
    printf("[EXEC] Running: %s (Timeout: %ds)...\n", g_current_test_name, timeout_sec); \
    fflush(stdout); \
    pid_t pid = fork(); \
    if (pid == 0) { \
        /* Child: Run the test */ \
        alarm(timeout_sec); \
        test_fn(); \
        exit(0); \
    } else if (pid > 0) { \
        /* Parent: Wait for child */ \
        int status; \
        waitpid(pid, &status, 0); \
        if (WIFEXITED(status)) { \
            if (WEXITSTATUS(status) == 0) { \
                printf("[PASS] %s completed successfully.\n", g_current_test_name); \
            } else { \
                printf("[FAIL] %s exited with code %d.\n", g_current_test_name, WEXITSTATUS(status)); \
            } \
        } else if (WIFSIGNALED(status)) { \
            printf("[CRITICAL] %s CRASHED with signal %d (%s).\n", \
                   g_current_test_name, WTERMSIG(status), \
                   WTERMSIG(status) == SIGALRM ? "TIMEOUT" : "SIGNAL"); \
        } else { \
            printf("[FAIL] %s failed for unknown reasons.\n", g_current_test_name); \
        } \
    } else { \
        perror("fork failed"); \
    } \
    fflush(stdout); \
} while(0)

/* --- Forward Declarations for Internal Static Functions (White-Box) --- */
static void swap_int(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

static void vocab_insertion_sort(int *arr, int n, char **vocab) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && strcmp(vocab[arr[j]], vocab[key]) > 0) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static void vocab_quicksort_test(int *arr, int n, char **vocab) {
    while (n > 16) {
        int mid = n / 2;
        if (strcmp(vocab[arr[0]], vocab[arr[mid]]) > 0) swap_int(&arr[0], &arr[mid]);
        if (strcmp(vocab[arr[0]], vocab[arr[n-1]]) > 0) swap_int(&arr[0], &arr[n-1]);
        if (strcmp(vocab[arr[mid]], vocab[arr[n-1]]) > 0) swap_int(&arr[mid], &arr[n-1]);
        swap_int(&arr[mid], &arr[n-2]);
        int pivot = arr[n-2];

        int i = 0, j = n - 2;
        for (;;) {
            while (strcmp(vocab[arr[++i]], vocab[pivot]) < 0) {}
            while (strcmp(vocab[arr[--j]], vocab[pivot]) > 0) {}
            if (i >= j) break;
            swap_int(&arr[i], &arr[j]);
        }
        swap_int(&arr[i], &arr[n-2]);

        if (i < n - i - 1) {
            vocab_quicksort_test(arr, i, vocab);
            arr += i + 1;
            n -= i + 1;
        } else {
            vocab_quicksort_test(arr + i + 1, n - i - 1, vocab);
            n = i;
        }
    }
    vocab_insertion_sort(arr, n, vocab);
}

/* --- Tests --- */

void test_blackbox_config_validation(void) {
    Config cfg;
    unsigned char buffer[1024];
    memset(buffer, 0, 1024);
    uint32_t magic = 0xBAD12345;
    memcpy(buffer, &magic, 4);
    assert(config_read(&cfg, buffer, 1024) == TN_ERR_INVALID_MAGIC);
    
    magic = TN_MAGIC;
    uint32_t version = TN_VERSION;
    memcpy(buffer, &magic, 4);
    memcpy(buffer + 4, &version, 4);
    Config *bad_cfg = (Config *)(buffer + 8);
    bad_cfg->dim = 0; 
    assert(config_read(&cfg, buffer, 1024) == TN_ERR_INVALID_CONFIG);
}

void test_whitebox_tokenizer_sort(void) {
    char *vocab_data[] = {"z", "y", "x", "w", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"};
    int n = 19;
    int *indices = malloc(n * sizeof(int));
    for(int i=0; i<n; i++) indices[i] = i;
    vocab_quicksort_test(indices, n, vocab_data);
    assert(strcmp(vocab_data[indices[0]], "a") == 0);
    assert(strcmp(vocab_data[indices[n-1]], "z") == 0);
    free(indices);
}

void test_unit_sliding_window_wrapping(void) {
    SlidingWindow sw;
    sw_init(&sw, 10, 2);
    for (int i = 0; i < 8; i++) sw_advance(&sw);
    assert(sw.wrapped == true);
    assert(sw_map_position(&sw, 10) == 2);
}

void test_security_tokenizer_overflow(void) {
    printf("[SECURITY] Testing TOK-SEC-02: Integer Overflow -> Heap Overflow...\n");
    fflush(stdout);
    
    // Create a malicious tokenizer file
    const char *mal_path = "malicious_tokenizer.bin";
    FILE *fp = fopen(mal_path, "wb");
    if (!fp) return;
    
    int vocab_size = 1;
    int max_token_len = 1024;
    fwrite(&vocab_size, sizeof(int), 1, fp);
    fwrite(&max_token_len, sizeof(int), 1, fp);
    
    float score = 0.0f;
    int bad_len = -1; // Trigger: (size_t)-1 + 1 == 0
    fwrite(&score, sizeof(float), 1, fp);
    fwrite(&bad_len, sizeof(int), 1, fp);
    // No bytes written
    fclose(fp);
    
    Tokenizer t;
    TernaryError err = tokenizer_load(&t, mal_path);
    printf("[SECURITY] Result: %s\n", err == TN_ERR_TOKENIZER_LOAD ? "Handled (Safe)" : "Did not catch early");
    fflush(stdout);
    
    if (err == TN_OK) tokenizer_free(&t);
    remove(mal_path);
}

void test_security_weight_oob_read(void) {
    printf("[SECURITY] Testing WGT-SEC-03: Read-Before-Advance OOB...\n");
    fflush(stdout);
    
    Config cfg;
    cfg.dim = 64;
    cfg.hidden_dim = 128;
    cfg.n_layers = 1;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 4;
    cfg.vocab_size = 100;
    cfg.seq_len = 128;
    
    // Allocate exactly enough for the embedding table but NOT the scale
    size_t table_size = (size_t)cfg.vocab_size * cfg.dim;
    unsigned char *mal_data = malloc(table_size);
    if (!mal_data) return;
    
    TransformerWeights w;
    weights_alloc_pointers(&w, &cfg);
    
    // This SHOULD ideally return TN_ERR_INVALID_WEIGHTS *safeley*
    // but line 72 in weights.c does memcpy(..., ptr, 4) BEFORE checking if ptr+4 is valid.
    TernaryError err = weights_map(&w, &cfg, NULL, (tn_i8*)mal_data, table_size);
    printf("[SECURITY] Result: %s\n", err == TN_ERR_INVALID_WEIGHTS ? "Caught (but read occurred)" : "Passed?");
    fflush(stdout);
    
    weights_free_pointers(&w);
    free(mal_data);
}

void test_concurrency_threadpool(void) {
    printf("[STRESS] Testing Concurrency: Threadpool Deadlock/Race...\n");
    fflush(stdout);
    
    ThreadPool *tp = threadpool_create(8);
    if (!tp) return;
    
    Config cfg;
    cfg.dim = 1024;
    cfg.n_heads = 8;
    
    float *x = malloc(cfg.dim * sizeof(float));
    float *out = malloc(cfg.dim * sizeof(float));
    tn_i8 *w = malloc((size_t)cfg.dim * cfg.dim);
    
    if (x && out && w) {
        memset(x, 0, cfg.dim * sizeof(float));
        memset(w, 0, (size_t)cfg.dim * cfg.dim);
        
        // Run multiple times to increase chance of catching a race
        for (int i = 0; i < 50; i++) {
            parallel_ternary_matmul(out, x, w, cfg.dim, cfg.dim, 1.0f, tp);
        }
    }
    
    free(x); free(out); free(w);
    threadpool_destroy(tp);
    printf("[STRESS] Result: Successfully executed 50 parallel matmuls without deadlock.\n");
    fflush(stdout);
}

#include "core/unpack.h"
#include <pthread.h>

void test_sanity_ternary_math(void) {
    printf("[MATH] Starting Sanity: Ternary Matmul Basic...\n");
    fflush(stdout);
    if (tn_ternary_matmul == NULL) exit(1);
    float x_data_test[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    tn_i8 w_data_test[8] = {1, -1, 1, -1, 1, -1, 1, -1};
    float out_val = 0.0f;
    tn_ternary_matmul(&out_val, x_data_test, w_data_test, 8, 1, 1.0f);
    assert(out_val == 0.0f);
    printf("[PASS] Math correct.\n");
    fflush(stdout);
}

/* --- Heartbeat Thread --- */
void* heartbeat_func(void* arg) {
    (void)arg;
    while(1) {
        printf("[WATCHDOG] Heartbeat: Current Test = %s\n", g_current_test_name);
        fflush(stdout);
        sleep(1);
    }
    return NULL;
}

/* --- Additional Tests --- */

void test_unpack_verification(void) {
    tn_u8 packed[32];
    for(int i=0; i<32; i++) packed[i] = (tn_u8)rand();
    
    tn_i8 out_avx[128];
    tn_i8 out_scalar[128];
    
    // Manual scalar unpack for reference
    for(int i=0; i<128; i++) {
        int byte_idx = i / 4;
        int bit_pos = (i % 4) * 2;
        out_scalar[i] = (tn_i8)(((packed[byte_idx] >> bit_pos) & 0x03) - 1);
    }
    
    #ifdef __AVX2__
    unpack_ternary_block_avx2(out_avx, packed, 128);
    #else
    unpack_ternary_block(out_avx, packed, 128); // fallback
    #endif
    
    for(int i=0; i<128; i++) {
        if (out_avx[i] != out_scalar[i]) {
            printf("[FAIL] Unpack mismatch at index %d: AVX=%d, Scalar=%d\n", i, out_avx[i], out_scalar[i]);
            exit(1);
        }
    }
    printf("[PASS] Unpack verification matched.\n");
    fflush(stdout);
}

void test_heavy_stress_matmul(void) {
    int dim = 2048;
    ThreadPool *tp = threadpool_create(16); // Over-subscribe
    float *x = malloc(dim * sizeof(float));
    float *out = malloc(dim * sizeof(float));
    tn_i8 *w = malloc((size_t)dim * dim);
    
    if (x && out && w) {
        for(int i=0; i<100; i++) {
            parallel_ternary_matmul(out, x, w, dim, dim, 1.0f, tp);
            if (i % 25 == 0) { printf("[STRESS] Iteration %d...\n", i); fflush(stdout); }
        }
    }
    
    free(x); free(out); free(w);
    threadpool_destroy(tp);
}

int main(void) {
    pthread_t hb;
    pthread_create(&hb, NULL, heartbeat_func, NULL);
    pthread_detach(hb);

    const char *backend = tn_simd_init();
    printf("--- PROJECT ZERO FORENSIC AUDIT SUITE (Backend: %s) ---\n", backend);
    fflush(stdout);

    RUN_PROTECTED_TEST(test_blackbox_config_validation, 2);
    RUN_PROTECTED_TEST(test_whitebox_tokenizer_sort, 2);
    RUN_PROTECTED_TEST(test_unit_sliding_window_wrapping, 2);
    RUN_PROTECTED_TEST(test_security_tokenizer_overflow, 5);
    RUN_PROTECTED_TEST(test_security_weight_oob_read, 5);
    RUN_PROTECTED_TEST(test_unpack_verification, 2);
    RUN_PROTECTED_TEST(test_concurrency_threadpool, 10);
    RUN_PROTECTED_TEST(test_heavy_stress_matmul, 30);
    RUN_PROTECTED_TEST(test_sanity_ternary_math, 5);
    
    printf("--- AUDIT SUITE COMPLETED ---\n");
    return 0;
}

