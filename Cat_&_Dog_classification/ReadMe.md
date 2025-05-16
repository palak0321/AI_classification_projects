# 🐾 Cat vs Dog Image Classifier using CNN

This project is a deep learning-based image classifier that predicts whether an image is of a **Cat** or a **Dog**. It uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 📌 Overview

- Trained on image data using custom CNN layers
- Preprocessed using `ImageDataGenerator` with rescaling
- Evaluated and visualized training accuracy and loss
- Uses real-world data with binary classification

---

## 🧠 Model Architecture

```python
Conv2D(32, kernel_size=(3,3), activation='relu')
MaxPooling2D(2,2)

Conv2D(64, (3,3), activation='relu')
MaxPooling2D(2,2)

Conv2D(64, (3,3), activation='relu')
MaxPooling2D(2,2)

Flatten()
Dense(64, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')
Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy

🧪 Dataset
Source: Kaggle Dogs vs. Cats Dataset
Contains 25,000 labeled images of dogs and cats
Only a subset used for training and validation to reduce load

📈 Model Training Performance
Epochs: 10
Batch Size: 64
Achieved high training and validation accuracy = 96%


🧰 Requirements
Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
scikit-learn

🧑‍💻 Author
Palak Yaduvanshi
Aspiring Data Scientist and AI Enthusiast
