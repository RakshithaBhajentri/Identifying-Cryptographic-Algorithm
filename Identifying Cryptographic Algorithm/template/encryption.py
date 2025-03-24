from flask import Flask, request, jsonify
from flask_cors import CORS
from Crypto.Cipher import AES, DES
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
import os

app2 = Flask(__name__)
CORS(app2)

# Helper function
def pad_to_block_size(data, block_size):
    padding_length = block_size - len(data) % block_size
    return data + chr(padding_length) * padding_length

# Encryption logic
def encrypt_aes(plaintext):
    key = os.urandom(16)
    cipher = AES.new(key, AES.MODE_ECB)
    padded_plaintext = pad_to_block_size(plaintext, 16)
    encrypted = cipher.encrypt(padded_plaintext.encode())
    return encrypted.hex(), key.hex()

def encrypt_des(plaintext):
    key = os.urandom(8)
    cipher = DES.new(key, DES.MODE_ECB)
    padded_plaintext = pad_to_block_size(plaintext, 8)
    encrypted = cipher.encrypt(padded_plaintext.encode())
    return encrypted.hex(), key.hex()

def encrypt_rsa(plaintext):
    key = RSA.generate(1024)
    cipher = PKCS1_OAEP.new(key)
    encrypted = cipher.encrypt(plaintext.encode())
    return encrypted.hex(), key.export_key().decode()

# API endpoint

@app2.route('/api/encrypt', methods=['POST'])
def encrypt_data():
    data = request.json
    plaintext = data.get("plaintext", "")
    algorithm = data.get("algorithm", "").upper()

    if algorithm == "AES":
        encrypted, key = encrypt_aes(plaintext)
    elif algorithm == "DES":
        encrypted, key = encrypt_des(plaintext)
    elif algorithm == "RSA":
        encrypted, key = encrypt_rsa(plaintext)
    else:
        return jsonify({"error": "Unsupported encryption algorithm"}), 400

    return jsonify({
        "algorithm": algorithm,
        "encrypted": encrypted,
        "key": key
    })

if __name__ == '__main__':
    app2.run(host="127.0.0.1", port=5001, debug=True)

    