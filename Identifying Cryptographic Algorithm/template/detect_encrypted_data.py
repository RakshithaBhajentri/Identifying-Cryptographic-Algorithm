from flask import Flask, request, jsonify
from flask_cors import CORS
from Crypto.Cipher import AES, DES, PKCS1_OAEP
from Crypto.PublicKey import RSA
import base64

app = Flask(__name__)
CORS(app)

# Helper function for padding/unpadding
def pad_to_block_size(data, block_size):
    padding_length = block_size - len(data) % block_size
    return data + chr(padding_length) * padding_length

def unpad_from_block_size(data):
    padding_length = ord(data[-1])
    return data[:-padding_length]

# Decryption logic
def decrypt_aes_cbc(ciphertext, key, iv):
    cipher = AES.new(bytes.fromhex(key), AES.MODE_CBC, bytes.fromhex(iv))
    decrypted = cipher.decrypt(bytes.fromhex(ciphertext)).decode()
    return unpad_from_block_size(decrypted)

def decrypt_des_cbc(ciphertext, key, iv):
    cipher = DES.new(bytes.fromhex(key), DES.MODE_CBC, bytes.fromhex(iv))
    decrypted = cipher.decrypt(bytes.fromhex(ciphertext)).decode()
    return unpad_from_block_size(decrypted)

def decrypt_rsa(ciphertext, private_key):
    key = RSA.import_key(private_key)
    cipher = PKCS1_OAEP.new(key)
    decrypted = cipher.decrypt(bytes.fromhex(ciphertext)).decode()
    return decrypted

@app.route('/api/decrypt', methods=['POST'])
def decrypt_data():
    data = request.json
    ciphertext = data.get("ciphertext", "")
    key = data.get("key", "")
    iv = data.get("iv", "")
    algorithm = data.get("algorithm", "").upper()

    try:
        if not ciphertext or not key or not algorithm:
            return jsonify({"error": "Ciphertext, key, and algorithm are required."}), 400

        if algorithm == "AES":
            if not iv:
                return jsonify({"error": "IV is required for AES in CBC mode."}), 400
            decrypted = decrypt_aes_cbc(ciphertext, key, iv)
        elif algorithm == "DES":
            if not iv:
                return jsonify({"error": "IV is required for DES in CBC mode."}), 400
            decrypted = decrypt_des_cbc(ciphertext, key, iv)
        elif algorithm == "RSA":
            decrypted = decrypt_rsa(ciphertext, key)
        else:
            return jsonify({"error": "Unsupported decryption algorithm."}), 400

        return jsonify({
            "algorithm": algorithm,
            "decrypted_text": decrypted
        })

    except ValueError as ve:
        return jsonify({"error": "Decryption failed: " + str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001, debug=True)
