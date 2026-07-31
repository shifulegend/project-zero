#include "core/platform.h"
#include "tokenizer/tokenizer.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * GPT-2 byte-level BPE reverse mapping: unicode codepoint -> raw byte.
 *
 * Llama 3 / Qwen / GPT-style GGUF vocabs store every token piece through
 * GPT-2's bytes_to_unicode() scheme: each of the 256 possible raw byte
 * values is mapped to a "visual" unicode codepoint before being written into
 * the vocab, so that every byte value (including control chars, spaces, and
 * newlines) can be represented as printable text. Printable bytes (ASCII
 * '!'-'~', Latin-1 0xA1-0xAC, 0xAE-0xFF) map to themselves; the remaining 68
 * "unprintable" byte values (0x00-0x20, 0x7F-0xA0, 0xAD) map to codepoints
 * 0x100-0x143 in ascending-byte-value enumeration order (this is where Ġ
 * U+0120 = space and Ċ U+010A = newline come from).
 *
 * Decoding a piece therefore requires walking its UTF-8 text one codepoint
 * at a time and mapping each codepoint back to a single raw byte via the
 * inverse of this table — NOT just special-casing Ġ/Ċ and passing everything
 * else through unchanged. The old clean_bpe_string() only handled those two
 * markers; every other non-ASCII codepoint (anything in the self-mapped
 * 0xA1-0xFF Latin-1 range, which every multi-byte UTF-8 character's raw
 * bytes decompose into via BPE) was left as its literal 2-byte UTF-8
 * encoding instead of collapsing back to the single raw byte it represents.
 * That is what produced visible mojibake for any non-ASCII model output
 * (e.g. an emoji's raw UTF-8 bytes coming back doubled-and-mangled instead
 * of reassembling into the original character) — found via visual review of
 * a real model's screenshot output, not any existing test (no non-ASCII
 * coverage existed anywhere in the suite before this fix).
 *
 * All 256 mapped codepoints fit in 1 or 2-byte UTF-8 (max codepoint 0x143 <
 * 0x800, the 2-byte UTF-8 ceiling), so the decoder below only needs to
 * handle those two forms.
 */
#define BPE_REV_TABLE_SIZE 324 /* codepoints 0x000..0x143 */
static const int16_t g_bpe_cp_to_byte[BPE_REV_TABLE_SIZE] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, -1, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 173,
};

/* Decodes one UTF-8 codepoint from `s` (NUL-terminated). Returns the
 * codepoint and sets *cp_len to its byte length (1 or 2). Any input outside
 * 1/2-byte UTF-8 (never emitted by the bpe scheme above, but guarded against
 * malformed/foreign input) returns -1 with *cp_len = 1 so the caller can
 * fall back to a raw byte copy without reading out of bounds. */
static int bpe_decode_utf8_cp(const unsigned char *s, int *cp_len) {
    unsigned char c0 = s[0];
    if (c0 < 0x80) {
        *cp_len = 1;
        return c0;
    }
    if ((c0 & 0xE0) == 0xC0 && s[1] != '\0' && (s[1] & 0xC0) == 0x80) {
        *cp_len = 2;
        return ((c0 & 0x1F) << 6) | (s[1] & 0x3F);
    }
    *cp_len = 1;
    return -1;
}

/* Reverse-maps a GPT-2 byte-level BPE piece into its raw bytes, writing up
 * to out_cap-1 bytes plus a NUL terminator into `out`. Codepoints outside
 * the bpe scheme (e.g. literal ASCII inside a special token like
 * "<|im_start|>", or unexpected input) are copied through unchanged rather
 * than dropped, so this is a strict superset of the old passthrough
 * behavior for anything the old code already handled correctly. */
static void bpe_bytes_decode(const char *piece, char *out, size_t out_cap) {
    const unsigned char *s = (const unsigned char *)piece;
    size_t oi = 0;
    while (*s && oi + 1 < out_cap) {
        int cp_len;
        int cp = bpe_decode_utf8_cp(s, &cp_len);
        if (cp >= 0 && cp < BPE_REV_TABLE_SIZE && g_bpe_cp_to_byte[cp] >= 0) {
            out[oi++] = (char)g_bpe_cp_to_byte[cp];
        } else {
            for (int i = 0; i < cp_len && oi + 1 < out_cap; i++)
                out[oi++] = (char)s[i];
        }
        s += cp_len;
    }
    out[oi] = '\0';
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

  /* Strip leading space marker when the previous token was BOS. Two vocab
   * formats reach this function: this project's native binary tokenizer
   * (tokenizer_load, tested by test_tokenizer.c) stores pieces as literal
   * text with a real ' ' byte; GGUF byte-level-BPE vocabs (tokenizer_gguf.c)
   * store the same leading space as the 2-byte Ġ marker (UTF-8 0xC4 0xA0)
   * instead — checking only literal ' ' never matched real byte-level-BPE
   * pieces, so that marker was never stripped for GGUF-loaded models. */
  if (t->bos_token_id >= 0 && prev_token == t->bos_token_id) {
    if ((unsigned char)piece[0] == 0xC4 && (unsigned char)piece[1] == 0xA0)
      piece = piece + 2;
    else if (piece[0] == ' ')
      piece = piece + 1;
  }

  bpe_bytes_decode(piece, decode_buf, sizeof(decode_buf));
  return decode_buf;
}
