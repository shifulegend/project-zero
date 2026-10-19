#ifndef TN_TEXT_FILE_H
#define TN_TEXT_FILE_H

/*
 * core/text_file — read a whole (bounded) text file into a NUL-terminated
 * malloc'd buffer. Used for --perplexity and --chat-template-file.
 */

#include <stddef.h>

/* Returns the buffer (caller frees) and sets *len_out, or NULL when the file
 * cannot be opened/read, is empty, or is larger than max_bytes. */
char *tn_read_text_file(const char *path, size_t max_bytes, size_t *len_out);

#endif /* TN_TEXT_FILE_H */
