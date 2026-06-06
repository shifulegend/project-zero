#include "core/error.h"

static const char *error_strings[] = {
    [TN_OK]                   = "Success",
    [TN_ERR_FILE_OPEN]        = "Could not open file",
    [TN_ERR_FILE_STAT]        = "Could not stat file",
    [TN_ERR_MMAP_FAILED]      = "Memory mapping failed",
    [TN_ERR_OOM]              = "Out of memory",
    [TN_ERR_INVALID_CONFIG]   = "Invalid model configuration",
    [TN_ERR_INVALID_WEIGHTS]  = "Invalid or corrupted weights",
    [TN_ERR_INVALID_MAGIC]    = "Invalid file magic number",
    [TN_ERR_VERSION_MISMATCH] = "Unsupported file version",
    [TN_ERR_THREAD_CREATE]    = "Failed to create thread",
    [TN_ERR_TOKENIZER_LOAD]   = "Failed to load tokenizer",
    [TN_ERR_TOKENIZER_ENCODE] = "Tokenizer encoding failed",
    [TN_ERR_IMAGE_LOAD]       = "Failed to load image",
    [TN_ERR_DIMENSION_MISMATCH] = "Dimension mismatch",
    [TN_ERR_EXEC_BLOCKED]     = "Command blocked by security policy",
    [TN_ERR_EXEC_TIMEOUT]     = "Command execution timed out",
    [TN_ERR_EXEC_FAILED]      = "Command execution failed",
    [TN_ERR_DB_OPEN]          = "Failed to open vector database",
    [TN_ERR_DB_WRITE]         = "Failed to write to vector database",
    [TN_ERR_INVALID_ARGS]     = "Invalid arguments",
};

const char *tn_error_str(TernaryError err) {
    if (err < 0 || err >= TN_ERR_COUNT) {
        return "Unknown error";
    }
    return error_strings[err];
}
