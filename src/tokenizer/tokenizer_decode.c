#include "core/platform.h"
#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <string.h>

/**
 * Clean BPE-encoded special characters from decoded tokens.
 * Llama 3 / GPT-style tokenizers use:
 *   Ġ (U+0120, UTF-8: 0xC4 0xA0) for spaces
 *   Ċ (U+010A, UTF-8: 0xC4 0x8A) for newlines
 */
static void clean_bpe_string(char *str) {
    char *p = str;
    while (*p) {
        /* Ġ → space */
        if ((unsigned char)*p == 0xC4 && (unsigned char)*(p+1) == 0xA0) {
            *p = ' ';
            memmove(p+1, p+2, strlen(p+2) + 1);
        }
        /* Ċ → newline */
        else if ((unsigned char)*p == 0xC4 && (unsigned char)*(p+1) == 0x8A) {
            *p = '\n';
            memmove(p+1, p+2, strlen(p+2) + 1);
        } else {
            p++;
        }
    }
}

/* Thread-local buffer for decoded token output.
 * Sized to hold the longest possible token piece with BPE cleanup.
 * Made thread-local to prevent data races (QA-ISS-004). */
#if TN_WIN32
static __declspec(thread) char decode_buf[512];
#else
static __thread char decode_buf[512];
#endif

const char *tokenizer_decode(Tokenizer *t, int prev_token, int token) {
  if (!t || !t->vocab || token < 0 || token >= t->vocab_size)
    return "";

  const char *piece = t->vocab[token];
  if (piece == NULL) return "";

  /* Handle raw byte tokens: <0xHH> pattern */
  if (strlen(piece) >= 6 && piece[0] == '<' &&
      piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
    unsigned int byte_val = 0;
    /* Parse two hex digits */
    for (int i = 3; i < 5; i++) {
      byte_val <<= 4;
      char c = piece[i];
      if (c >= '0' && c <= '9')
        byte_val += (unsigned)(c - '0');
      else if (c >= 'a' && c <= 'f')
        byte_val += (unsigned)(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        byte_val += (unsigned)(c - 'A' + 10);
    }
    decode_buf[0] = (char)byte_val;
    decode_buf[1] = '\0';
    return decode_buf;
  }

  /* Strip leading space when the previous token was BOS.
   * Use the tokenizer's own bos_token_id — never assume a fixed ID. */
  if (t->bos_token_id >= 0 && prev_token == t->bos_token_id && piece[0] == ' ') {
    piece = piece + 1;
  }

  /* Copy into mutable buffer and apply BPE cleanup */
  size_t len = strlen(piece);
  if (len >= sizeof(decode_buf)) len = sizeof(decode_buf) - 1;
  memcpy(decode_buf, piece, len);
  decode_buf[len] = '\0';
  clean_bpe_string(decode_buf);

  return decode_buf;
}
