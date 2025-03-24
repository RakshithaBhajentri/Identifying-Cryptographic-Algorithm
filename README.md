<!DOCTYPE html>
<head></head>
<body>
<h1>An ML based solution for identifying the cryptographic algorithm from the dataset</h1>
<h4>Project Overview</h4>
<p>
This project uses machine learning to identify cryptographic algorithms (AES, DES, RSA) from given ciphertext. By extracting key features such as entropy, byte_skewness, length, it trains a Random Forest Classifier to predict the encryption method used. 
</p>
Features
<ul>
<li>Encrypts user-input/plaintext using AES, DES, and RSA</li>
<li>Extracts entropy, byte_skewness, length as classification features</li>
<li>Trains an ML model to predict the encryption algorithm</li>
<li>Accepts user-input ciphertext for prediction</li>
<li>Supports plaintext encryption and decryption</li>
</ul>

<h6>Technologies Used</h6>
<p>Web Technologies: HTML, CSS </p>
<p>Programming language: Python</p>
<p>Cryptography Libraries (PyCryptodome for AES, DES, and RSA)</p>
<p>Machine Learning (scikit-learn – RandomForestClassifier)</p>
</body>
</html>
