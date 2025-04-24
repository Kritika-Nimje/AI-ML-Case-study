# 🧠 Aiml_DeepLearnig
 
**Subject:** Introduction to Deep Learning  
**Name:** Ishwari Kakade  
**PRN:** 22070521183  
**Section:** B  

---

## 📘 Overview

This repository contains solutions to **CA-4** for the *Introduction to Deep Learning* course. The assignment explores two real-world applications of neural networks:

1. **Handwritten Digit Recognition** using the MNIST dataset  
2. **Heart Disease Prediction** using patient clinical data

Both problems follow the complete deep learning pipeline — from preprocessing to model evaluation — using **Keras** and **TensorFlow**.

---

## 🔍 Scenario 1: Handwritten Digit Recognition (MNIST)

### 📝 Problem Statement
A startup is building an automated system to recognize handwritten digits on receipts for expense logging. The goal is to develop a model that can accurately classify digits from **0 to 9** using the MNIST dataset (70,000 grayscale images, 28x28 pixels each).

### 📊 Workflow
- Import essential libraries  
- Load and preprocess the MNIST dataset  
- Normalize pixel values (range: 0–1)  
- Flatten 2D images into 1D vectors  
- One-hot encode class labels  
- Build and train a *Feedforward Neural Network (FNN)* using Keras  
- Evaluate model performance and suggest improvements

### 🧱 Model Architecture
- **Input Layer:** 784 neurons (flattened 28x28 image)  
- **Hidden Layers:** Dense layers with ReLU activation  
- **Output Layer:** 10 neurons with softmax activation (multiclass classification)

### ✅ Result
Achieved ~**98% accuracy** on test data using a simple feedforward model.

---

## ❤️ Scenario 2: Heart Disease Prediction

### 📝 Problem Statement
A healthcare provider aims to develop a predictive model to identify patients at risk of **heart disease**, using clinical features like age, cholesterol level, blood pressure, etc.

### 📊 Workflow
- Load and preprocess patient data from CSV  
- Normalize features and split data into training/testing sets  
- Handle class imbalance using `class_weight`  
- Build and train an *Artificial Neural Network (ANN)* using Keras  
- Evaluate model accuracy and suggest optimizations

### 🧱 Model Architecture
- **Hidden Layers:** Dense layers with ReLU activation  
- **Output Layer:** 1 neuron with sigmoid activation (binary classification)

### ✅ Result
Successfully built a binary classification model that handles class imbalance and performs well on test data.

---

## 🧰 Libraries & Tools
- **Python**  
- **TensorFlow / Keras**  
- **NumPy**, **Pandas**  
- **Scikit-learn**  
- **Matplotlib**, **Seaborn** (for visualizations)

---

## 🚀 Potential Enhancements

### 🔢 MNIST Digit Recognition
- Introduce **Convolutional Neural Networks (CNNs)** for better image feature extraction  
- Apply **Dropout** and **Batch Normalization** to reduce overfitting  
- Use **Data Augmentation** to enhance model generalization

### ❤️ Heart Disease Prediction
- Use **KerasTuner** or **GridSearchCV** for hyperparameter tuning  
- Apply **feature engineering** for better model insights  
- Evaluate using **ROC-AUC** and **Precision-Recall** curves

---

## 📁 Repository Structure

```
Deep Learning CA-4/
│
├── Deep Learning CA-4.ipynb     # Jupyter Notebook with both solutions
├── README.md                    # Project documentation
└── dataset/                     # (Optional) Contains CSV files
```

---

