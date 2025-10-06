# main.py
from genetic_algorithm import run_ga
from encryption import hybrid_encrypt, hybrid_decrypt, image_encrypt, image_decrypt
from PIL import Image
import numpy as np
import os
import traceback

if __name__ == "__main__":
    plaintext = "THISISASAMPLEPLAINTEXTTOEVALUATETHEHYBRIDCIPHERSIMULATION"
    print("Running GA — this may take a bit...")
    result = run_ga(plaintext, vig_len=4, perm_len=32,
                    population_size=30, generations=40,
                    crossover_rate=0.85, mutation_rate_vig=0.12,
                    mutation_rate_perm=0.25, elitism_k=3, verbose=True)
    best_vig, best_perm = result['best_chromosome']
    print("Best keys:", best_vig, best_perm)

    print("Testing text encryption...")
    c = hybrid_encrypt(plaintext, best_vig, best_perm)
    p = hybrid_decrypt(c, best_vig, best_perm)
    print("Decrypted == original:", p == plaintext.replace(" ", ""))

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input_image = "input/test.png"
    enc_img = "output/encrypted.png"
    dec_img = "output/decrypted.png"

    print("=== Image Encryption & Decryption ===")
    try:
        image_encrypt(input_image, best_vig, best_perm, enc_img)
        image_decrypt(enc_img, best_vig, best_perm, dec_img)
        orig = np.array(Image.open(input_image))
        rest = np.array(Image.open(dec_img))
        print("Exact Image Match:", np.array_equal(orig, rest))
    except FileNotFoundError:
        print("Place input/test.png")
    except Exception as e:
        print("Error during image pipeline:")
        traceback.print_exc()
        # read debug file if exists
        debug_file = "results/debug_image_info.txt"
        if os.path.exists(debug_file):
            print("--- debug info head ---")
            with open(debug_file, "r") as f:
                print(f.read()[-2000:])
