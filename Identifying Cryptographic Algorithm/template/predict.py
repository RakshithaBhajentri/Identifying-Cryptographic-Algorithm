from flask import Flask, request, jsonify
from flask_cors import CORS
from Crypto.Cipher import AES, DES
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
from matplotlib import pyplot as plt

def calculate_entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def extract_features(ciphertext):
    byte_values = list(ciphertext)
    entropy = calculate_entropy(byte_values)
    length = len(byte_values)
    byte_frequency = np.histogram(byte_values, bins=256, range=(0, 256))[0]
    byte_std_dev = np.std(byte_frequency)
    byte_skewness = np.mean((byte_frequency - np.mean(byte_frequency)) ** 3) / (np.std(byte_frequency) ** 3)
    return [entropy, length, byte_std_dev, byte_skewness]

app = Flask(__name__)
CORS(app)

def load_dataset(file_path):
    try:
        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        X = df[['entropy', 'length', 'byte_std_dev', 'byte_skewness']].values
        y = df['label'].values
        print(f"Dataset loaded: {len(df)} samples.")
        return X, y
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def build_neural_network(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')  # Assuming 3 classes: AES, DES, RSA
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_neural_network(input_dim=X.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=2)
    
    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nModel Evaluation:\n")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AES", "DES", "RSA"])
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.show()
    
    return model

# Predict Algorithm
@app.route('/predict', methods=['POST'])
def predict_algorithm(model, ciphertext):
    features = extract_features(ciphertext)  # Now expects a list of byte values
    prediction_proba = model.predict([features])[0]
    confidence = max(prediction_proba)

    if confidence < 0.5:
        return "Uncertain", features, confidence

    prediction = np.argmax(prediction_proba)
    print(f"Prediction: Features={features}, Predicted Algorithm={prediction}, Confidence={confidence}")
    return prediction, features, confidence

app = Flask(__name__)
CORS(app)

DATASET_PATH = r"D:\Current Data\Desktop\empty_renamed_dataset.csv"
X, y = load_dataset(DATASET_PATH)
model = train_model(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cipher_hex = data.get('ciphertext')

    if not cipher_hex:
        return jsonify({"error": "Ciphertext is required"}), 400

    # Ensure the input is a valid hexadecimal string
    cipher_hex = cipher_hex.strip().replace(" ", "")  # Remove any extra spaces

    # Check if the string is a valid hex string
    if not all(c in '0123456789ABCDEFabcdef' for c in cipher_hex):
        return jsonify({"error": "Invalid ciphertext format. Make sure it's in hexadecimal."}), 400

    try:
        cipher_bytes = bytes.fromhex(cipher_hex)
        prediction, features, confidence = predict_algorithm(model, cipher_bytes)
        algo_mapping = {0: "AES", 1: "DES", 2: "RSA"}
        algo_name = algo_mapping.get(prediction, "Uncertain")
        return jsonify({
            "predicted_algorithm": algo_name,
            "features_used": features,
            "confidence": confidence
        })
    except ValueError as ve:
        return jsonify({"error": "Invalid ciphertext format. Make sure it's in hex."}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
