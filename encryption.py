import numpy as np
from PIL import Image
import random

# ============================================================
# 🧠 TEXT ENCRYPTION / DECRYPTION (Vigenère Cipher)
# ============================================================

def vigenere_encrypt(plaintext: str, key: str) -> str:
    ciphertext = []
    key = key.upper()
    plaintext = plaintext.upper()
    klen = len(key)
    for i, ch in enumerate(plaintext):
        if ch.isalpha():
            shift = ord(key[i % klen]) - ord('A')
            ciphertext.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            ciphertext.append(ch)
    return ''.join(ciphertext)


def vigenere_decrypt(ciphertext: str, key: str) -> str:
    plaintext = []
    key = key.upper()
    ciphertext = ciphertext.upper()
    klen = len(key)
    for i, ch in enumerate(ciphertext):
        if ch.isalpha():
            shift = ord(key[i % klen]) - ord('A')
            plaintext.append(chr((ord(ch) - 65 - shift) % 26 + 65))
        else:
            plaintext.append(ch)
    return ''.join(plaintext)

# ============================================================
# 🖼️ IMAGE ENCRYPTION / DECRYPTION (Shuffle + XOR)
# ============================================================

def _expand_key(key, size):
    """Scale and repeat GA key safely within 0–255."""
    key_arr = np.array(key, dtype=np.uint16)  # temporary higher type
    key_arr = (key_arr * 8) % 256             # scale but clamp to 0–255
    key_arr = key_arr.astype(np.uint8)
    return np.resize(key_arr, size)


def _shuffle_indices(seed, length):
    """Return deterministic shuffle order based on key sum."""
    rng = random.Random(seed)
    idx = list(range(length))
    rng.shuffle(idx)
    return np.array(idx)


def image_encrypt(input_path: str, output_path: str, key: list):
    """
    Encrypt an image with XOR + pixel shuffling.
    Safe for uint8 images.
    """
    try:
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        h, w, c = arr.shape

        # Flatten the image for manipulation
        flat = arr.reshape(-1, c)

        # XOR step
        key_expanded = _expand_key(key, flat.size)
        key_matrix = key_expanded.reshape(flat.shape)
        xor_flat = np.bitwise_xor(flat, key_matrix)

        # Shuffle pixels deterministically
        shuffle_idx = _shuffle_indices(sum(key), len(xor_flat))
        shuffled = xor_flat[shuffle_idx]

        enc_arr = shuffled.reshape(h, w, c)
        Image.fromarray(enc_arr).save(output_path)
        return output_path

    except Exception as e:
        print(f"⚠️ Error in image_encrypt: {e}")
        raise


def image_decrypt(input_path: str, output_path: str, key: list):
    """
    Reverse pixel shuffling + XOR to restore image.
    """
    try:
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        h, w, c = arr.shape

        flat = arr.reshape(-1, c)

        # Deterministic shuffle (to undo)
        shuffle_idx = _shuffle_indices(sum(key), len(flat))
        unshuffled = np.empty_like(flat)
        unshuffled[shuffle_idx] = flat

        # XOR reverse
        key_expanded = _expand_key(key, flat.size)
        key_matrix = key_expanded.reshape(flat.shape)
        dec_flat = np.bitwise_xor(unshuffled, key_matrix)

        dec_arr = dec_flat.reshape(h, w, c)
        Image.fromarray(dec_arr).save(output_path)
        return output_path

    except Exception as e:
        print(f"⚠️ Error in image_decrypt: {e}")
        raise


# ============================================================
# 🔬 LOCAL TEST (optional)
# ============================================================

if __name__ == "__main__":
    msg = "HELLO WORLD"
    key_text = "OMRY"
    enc = vigenere_encrypt(msg, key_text)
    dec = vigenere_decrypt(enc, key_text)
    print("Text original:", msg)
    print("Encrypted:", enc)
    print("Decrypted:", dec)

    img_key = [31, 23, 29, 15, 25, 23, 24, 16, 28, 7, 13, 0, 22, 10, 5, 9,
               31, 19, 1, 14, 11, 21, 11, 27, 8, 6, 26, 4, 12, 20, 30, 18]
    try:
        image_encrypt("test.png", "encrypted_test.png", img_key)
        image_decrypt("encrypted_test.png", "decrypted_test.png", img_key)
        print("✅ Image encryption/decryption complete.")
    except Exception as err:
        print("⚠️", err)
