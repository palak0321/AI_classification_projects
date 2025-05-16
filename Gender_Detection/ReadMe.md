# 👤 Gender Detection from Images using Deep Learning

A deep learning project that predicts the **gender (Male/Female)** of a person from a face image using a Convolutional Neural Network (CNN). This model is trained using labeled face images and achieves high accuracy on unseen data.

---

## 📌 Overview

This project uses deep learning techniques to classify face images as either **Male** or **Female**. It is built using TensorFlow and Keras and trained on a labeled dataset of human faces.

---

## 🧠 Model Architecture

> Custom CNN or Transfer Learning (e.g., MobileNetV2 / VGG16)

```python
Conv2D(32, kernel_size=(3,3), activation='relu')
MaxPooling2D(2,2)

Conv2D(64, (3,3), activation='relu')
MaxPooling2D(2,2)

Conv2D(128, (3,3), activation='relu')
MaxPooling2D(2,2)

Flatten()
Dense(128, activation='relu')
Dropout(0.5)
Dense(1, activation='sigmoid')
Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy

🧪 Dataset
Source: Kaggle dataset

Contains over 20,000 face images with labels for age, gender, and ethnicity
Preprocessed to crop and resize images to a fixed size (e.g., 100x100)

Gender labels:
0 = Male
1 = Female

Dataset is not uploaded to GitHub due to size limitations.

📈 Model Performance
Metric	Value
Training Accuracy	~95%
Validation Accuracy	~94%
Loss	Low & stable
Visualized using Matplotlib
Training monitored with accuracy/loss graphs


🧰 Requirements
Python 3.x
TensorFlow / Keras
NumPy
Pandas
Matplotlib
OpenCV (for image processing)

🧑‍💻 Author
Palak Yaduvanshi
📫 LinkedIn
