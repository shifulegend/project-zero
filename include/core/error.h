#ifndef TN_ERROR_H
#define TN_ERROR_H

typedef enum {
    TN_OK = 0,
    TN_ERR_FILE_OPEN,
    TN_ERR_FILE_STAT,
    TN_ERR_MMAP_FAILED,
    TN_ERR_OOM,
    TN_ERR_INVALID_CONFIG,
    TN_ERR_INVALID_WEIGHTS,
    TN_ERR_INVALID_MAGIC,
    TN_ERR_VERSION_MISMATCH,
    TN_ERR_THREAD_CREATE,
    TN_ERR_TOKENIZER_LOAD,
    TN_ERR_TOKENIZER_ENCODE,
    TN_ERR_IMAGE_LOAD,
    TN_ERR_DIMENSION_MISMATCH,
    TN_ERR_EXEC_BLOCKED,
    TN_ERR_EXEC_TIMEOUT,
    TN_ERR_EXEC_FAILED,
    TN_ERR_DB_OPEN,
    TN_ERR_DB_WRITE,
    TN_ERR_INVALID_ARGS,
    /* Phase 21: API server errors */
    TN_ERR_SOCKET_CREATE,
    TN_ERR_SOCKET_BIND,
    TN_ERR_SOCKET_LISTEN,
    TN_ERR_JSON_PARSE,
    TN_ERR_COUNT  /* sentinel — must be last */
} TernaryError;

#define TN_CHECK(expr, err) do { if (!(expr)) return (err); } while(0)

#define TN_CHECK_ALLOC(ptr) do { if (!(ptr)) return TN_ERR_OOM; } while(0)

const char *tn_error_str(TernaryError err);

#endif /* TN_ERROR_H */
