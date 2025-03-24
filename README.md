# An ML based solution for identifying the cryptographic algorithm from the dataset
Project Overview
This project uses machine learning to identify cryptographic algorithms (AES, DES, RSA) from given ciphertext. By extracting key features such as entropy, byte_skewness, length, it trains a Random Forest Classifier to predict the encryption method used. 

Features
✅ Encrypts user-input/plaintext using AES, DES, and RSA
✅ Extracts entropy and length as classification features
✅ Trains an ML model to predict the encryption algorithm
✅ Accepts user-input ciphertext for prediction
✅ Supports plaintext encryption and decryption

Technologies Used
Python
Cryptography Libraries (PyCryptodome for AES, DES, and RSA)
Machine Learning (scikit-learn – RandomForestClassifier)
