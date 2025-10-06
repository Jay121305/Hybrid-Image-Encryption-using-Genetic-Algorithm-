# encryption.py
# Final defensive implementation: safe_uint8(), robust image pipeline
from PIL import Image
import numpy as np
import os
import math
import traceback

# ----------------- utilities -----------------
def safe_uint8(a):
    """
    Ensure array-like a becomes a numpy array with dtype uint8 and values in 0..255.
    Uses modulo 256 to avoid wrap/overflow errors.
    """
    arr = np.asarray(a)
    # convert to signed/int64 to avoid overflow during mod
    if not np.issubdtype(arr.dtype, np.integer):
        # try to cast to int64 first
        try:
            arr = arr.astype(np.int64)
        except Exception:
            arr = np.vectorize(int)(arr)
            arr = np.asarray(arr, dtype=np.int64)
    else:
        arr = arr.astype(np.int64)
    arr_mod = np.mod(arr, 256)
    return arr_mod.astype(np.uint8)

def debug_write_info(name, arr):
    """Write basic diagnostics to results/debug_image_info.txt"""
    try:
        os.makedirs("results", exist_ok=True)
        path = "results/debug_image_info.txt"
        with open(path, "a") as f:
            f.write(f"--- {name} ---\n")
            f.write(f"shape: {getattr(arr,'shape', None)}\n")
            try:
                amin = int(np.min(arr))
                amax = int(np.max(arr))
                f.write(f"min: {amin}, max: {amax}\n")
                f.write(f"dtype: {arr.dtype}\n")
                # find up to 10 out-of-range indices
                bad = np.where((arr < 0) | (arr > 255))
                if bad and bad[0].size > 0:
                    f.write(f"out_of_range_count: {bad[0].size}\n")
                    # record a few sample values
                    sample_indices = list(zip(*(bad[i][:10] for i in range(len(bad)))))
                    f.write(f"sample bad positions/values (up to 10):\n")
                    for idx in sample_indices:
                        v = arr[idx]
                        f.write(f"{idx} -> {v}\n")
                else:
                    f.write("no out-of-range values\n")
            except Exception as e:
                f.write(f"diagnostic failure: {e}\n")
            f.write("\n")
    except Exception:
        pass

# ---------------- TEXT functions (unchanged) ----------------
def vigenere_encrypt(plaintext, key):
    plaintext = plaintext.upper().replace(" ", "")
    key = key.upper()
    ciphertext = []
    klen = len(key)
    for i, ch in enumerate(plaintext):
        shift = (ord(ch) - 65 + (ord(key[i % klen]) - 65)) % 26
        ciphertext.append(chr(shift + 65))
    return ''.join(ciphertext)

def vigenere_decrypt(ciphertext, key):
    ciphertext = ciphertext.upper()
    key = key.upper()
    plaintext = []
    klen = len(key)
    for i, ch in enumerate(ciphertext):
        shift = (ord(ch) - 65 - (ord(key[i % klen]) - 65)) % 26
        plaintext.append(chr(shift + 65))
    return ''.join(plaintext)

def columnar_encrypt(plaintext, key):
    n = len(key)
    if n == 0:
        return plaintext
    num_rows = (len(plaintext) + n - 1) // n
    grid = [['X'] * n for _ in range(num_rows)]
    idx = 0
    for r in range(num_rows):
        for c in range(n):
            if idx < len(plaintext):
                grid[r][c] = plaintext[idx]
                idx += 1
    sorted_key_indices = sorted(range(n), key=lambda x: key[x])
    out = []
    for i in sorted_key_indices:
        for r in range(num_rows):
            out.append(grid[r][i])
    return ''.join(out)

def columnar_decrypt(ciphertext, key):
    n = len(key)
    if n == 0:
        return ciphertext
    num_rows = (len(ciphertext) + n - 1) // n
    sorted_key_indices = sorted(range(n), key=lambda x: key[x])
    col_lengths = [num_rows] * n
    extra = (n * num_rows) - len(ciphertext)
    for i in range(extra):
        col_lengths[sorted_key_indices[-(i + 1)]] -= 1
    cols = {}
    index = 0
    for i in sorted_key_indices:
        length = col_lengths[i]
        cols[i] = list(ciphertext[index:index + length])
        index += length
    out = []
    for _ in range(num_rows):
        for i in range(n):
            if cols[i]:
                out.append(cols[i].pop(0))
    return ''.join(out).rstrip('X')

def hybrid_encrypt(plaintext, key1, key2):
    v = vigenere_encrypt(plaintext, key1)
    return columnar_encrypt(v, key2)

def hybrid_decrypt(ciphertext, key1, key2):
    t = columnar_decrypt(ciphertext, key2)
    return vigenere_decrypt(t, key1)

# ---------------- BYTE helpers ----------------
def vigenere_encrypt_bytes(data_bytes, key_bytes):
    if len(key_bytes) == 0:
        return data_bytes
    data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    key_arr = np.frombuffer(bytes(key_bytes), dtype=np.uint8)
    # build key_seq safely as uint8
    key_seq = key_arr[np.arange(data_arr.size) % key_arr.size]
    out = (data_arr.astype(np.int16) + key_seq.astype(np.int16)) % 256
    return out.astype(np.uint8).tobytes()

def vigenere_decrypt_bytes(data_bytes, key_bytes):
    if len(key_bytes) == 0:
        return data_bytes
    data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    key_arr = np.frombuffer(bytes(key_bytes), dtype=np.uint8)
    key_seq = key_arr[np.arange(data_arr.size) % key_arr.size]
    out = (data_arr.astype(np.int16) - key_seq.astype(np.int16)) % 256
    return out.astype(np.uint8).tobytes()

def columnar_encrypt_bytes(data_bytes, key):
    n = len(key)
    if n == 0:
        return data_bytes
    data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    L = data_arr.size
    rows = math.ceil(L / n)
    pad_len = rows * n - L
    if pad_len > 0:
        padded = np.concatenate([data_arr, np.zeros(pad_len, dtype=np.uint8)])
    else:
        padded = data_arr
    grid = padded.reshape(rows, n)
    sorted_idx = np.argsort(key)
    out = grid[:, sorted_idx].reshape(-1)
    # return as raw bytes (length >= L)
    return out.tobytes()

def columnar_decrypt_bytes(cipher_bytes, key, original_len):
    n = len(key)
    if n == 0:
        return cipher_bytes[:original_len]
    data_arr = np.frombuffer(cipher_bytes, dtype=np.uint8)
    rows = math.ceil(original_len / n)
    expected = rows * n
    if data_arr.size < expected:
        data_arr = np.concatenate([data_arr, np.zeros(expected - data_arr.size, dtype=np.uint8)])
    grid_sorted = data_arr.reshape(rows, n)
    sorted_idx = np.argsort(key)
    # reconstruct original order
    grid_orig = np.empty_like(grid_sorted)
    for j, orig_col in enumerate(sorted_idx):
        grid_orig[:, orig_col] = grid_sorted[:, j]
    out = grid_orig.reshape(-1)[:original_len]
    return out.tobytes()

def add_visual_noise(arr, key1):
    rng = np.random.default_rng(sum(ord(c) for c in str(key1)))
    noise = rng.integers(0, 256, arr.shape, dtype=np.uint8)
    # ensure arr is uint8
    arr8 = np.asarray(arr, dtype=np.uint8)
    res = np.bitwise_xor(arr8, noise)
    return res.astype(np.uint8)

# ---------------- IMAGE functions (defensive) ----------------
def image_encrypt(image_path, key1, key2, save_path="output/encrypted.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    try:
        img = Image.open(image_path)
        arr = np.array(img, dtype=np.uint8)
        mode = img.mode
        shape = arr.shape
        flat = arr.reshape(-1)
        L = flat.size

        key_bytes = [ord(c) % 256 for c in str(key1)]

        vig_enc = vigenere_encrypt_bytes(flat.tobytes(), key_bytes)
        col_enc = columnar_encrypt_bytes(vig_enc, key2)

        # frombuffer should be uint8 and length at least L
        cipher_arr = np.frombuffer(col_enc, dtype=np.uint8)[:L].reshape(shape)

        cipher_arr = add_visual_noise(cipher_arr, key1)

        # final safe conversion using safe_uint8
        final = safe_uint8(cipher_arr)

        # extra diagnostics if something odd
        if final.size == 0:
            raise ValueError("Empty array after processing")

        try:
            Image.fromarray(final, mode=mode).save(save_path)
        except Exception as e_inner:
            # write debug info and attempt forced modulo cast then save
            debug_write_info("encrypt_before_save", cipher_arr)
            final2 = np.mod(np.asarray(cipher_arr, dtype=np.int64), 256).astype(np.uint8)
            Image.fromarray(final2, mode=mode).save(save_path)
        print(f"✅ Encrypted image saved to {save_path}")
    except Exception as e:
        print("⚠️ Error in image_encrypt:", e)
        traceback.print_exc()
        debug_write_info("encrypt_exception", locals().get('cipher_arr', np.array([])))

def image_decrypt(image_path, key1, key2, save_path="output/decrypted.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    try:
        img = Image.open(image_path)
        arr = np.array(img, dtype=np.uint8)
        mode = img.mode
        shape = arr.shape
        L = np.prod(shape)

        arr_n = add_visual_noise(arr, key1)

        key_bytes = [ord(c) % 256 for c in str(key1)]
        col_dec = columnar_decrypt_bytes(arr_n.tobytes(), key2, L)
        vig_dec = vigenere_decrypt_bytes(col_dec, key_bytes)

        dec_arr = np.frombuffer(vig_dec, dtype=np.uint8)[:L].reshape(shape)

        final = safe_uint8(dec_arr)

        try:
            Image.fromarray(final, mode=mode).save(save_path)
        except Exception as e_inner:
            debug_write_info("decrypt_before_save", dec_arr)
            final2 = np.mod(np.asarray(dec_arr, dtype=np.int64), 256).astype(np.uint8)
            Image.fromarray(final2, mode=mode).save(save_path)
        print(f"✅ Decrypted image saved to {save_path}")
    except Exception as e:
        print("⚠️ Error in image_decrypt:", e)
        traceback.print_exc()
        debug_write_info("decrypt_exception", locals().get('dec_arr', np.array([])))
