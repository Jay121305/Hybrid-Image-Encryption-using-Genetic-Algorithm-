from flask import Flask, render_template, request, send_file, url_for
import os
from werkzeug.utils import secure_filename
from encryption import image_encrypt, image_decrypt, vigenere_encrypt, vigenere_decrypt
from genetic_algorithm import run_genetic_algorithm

# Flask setup
app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Run GA once at startup (generate best keys)
text_key, image_key = run_genetic_algorithm()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_image", methods=["POST"])
def process_image():
    """Handles image encryption/decryption"""
    action = request.form.get("action")
    file = request.files.get("image")

    if not file:
        return render_template("index.html", message="⚠️ Please upload an image.")

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    encrypted_path = os.path.join(app.config["RESULT_FOLDER"], f"encrypted_{filename}")
    decrypted_path = os.path.join(app.config["RESULT_FOLDER"], f"decrypted_{filename}")

    try:
        if action == "encrypt":
            image_encrypt(upload_path, encrypted_path, image_key)
            if not os.path.exists(encrypted_path):
                raise FileNotFoundError("Encrypted image was not saved properly.")
            return render_template(
                "index.html",
                message="✅ Image Encrypted Successfully!",
                image_result=url_for("static", filename=f"results/encrypted_{filename}"),
                result_label="Encrypted Image",
                download_link=url_for("download", filename=f"results/encrypted_{filename}"),
            )

        elif action == "decrypt":
            image_decrypt(upload_path, decrypted_path, image_key)
            if not os.path.exists(decrypted_path):
                raise FileNotFoundError("Decrypted image was not saved properly.")
            return render_template(
                "index.html",
                message="✅ Image Decrypted Successfully!",
                image_result=url_for("static", filename=f"results/decrypted_{filename}"),
                result_label="Decrypted Image",
                download_link=url_for("download", filename=f"results/decrypted_{filename}"),
            )

        else:
            return render_template("index.html", message="⚠️ Invalid option selected.")

    except Exception as e:
        return render_template("index.html", message=f"⚠️ Error: {e}")


@app.route("/process_text", methods=["POST"])
def process_text():
    """Handles text encryption/decryption"""
    text_action = request.form.get("text_action")
    user_text = request.form.get("user_text", "").strip()

    if not user_text:
        return render_template("index.html", text_message="⚠️ Please enter text.")

    try:
        if text_action == "encrypt":
            encrypted = vigenere_encrypt(user_text, text_key)
            return render_template(
                "index.html",
                text_message="✅ Text Encrypted Successfully!",
                text_output=encrypted,
                text_label="Encrypted Text",
            )

        elif text_action == "decrypt":
            decrypted = vigenere_decrypt(user_text, text_key)
            return render_template(
                "index.html",
                text_message="✅ Text Decrypted Successfully!",
                text_output=decrypted,
                text_label="Decrypted Text",
            )

        else:
            return render_template("index.html", text_message="⚠️ Invalid option selected.")
    except Exception as e:
        return render_template("index.html", text_message=f"⚠️ Error: {e}")


@app.route("/download/<path:filename>")
def download(filename):
    """Downloads encrypted/decrypted files"""
    full_path = os.path.join(BASE_DIR, "static", filename)
    if not os.path.exists(full_path):
        return f"⚠️ File not found: {full_path}", 404
    return send_file(full_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
