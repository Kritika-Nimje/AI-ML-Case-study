# 🧠 Aiml_DeepLearning

**Subject:** Introduction to Deep Learning  
**Name:** Kritika Nimje 
**PRN:** 22070521193  
**Section:** B  

---

## 📘 Overview

This repository presents solutions for **CA-4** from the *Introduction to Deep Learning* course. It addresses two practical applications of neural networks:

1. **Handwritten Digit Recognition** using the MNIST dataset  
2. **Heart Disease Prediction** using clinical patient data

Both scenarios walk through the full deep learning workflow — from data preprocessing to model evaluation — utilizing **Keras** and **TensorFlow**.

---

## 🔍 Scenario 1: Handwritten Digit Recognition (MNIST)

### 📝 Problem Statement

A startup is developing an automated system to recognize handwritten digits on receipts to streamline expense logging. The task is to build a model that classifies digits from **0 to 9** using the MNIST dataset (70,000 grayscale images, 28x28 pixels).

### 📊 Workflow

- Import necessary libraries  
- Load and preprocess the MNIST dataset  
- Normalize pixel intensities (range 0–1)  
- Flatten 2D images into 1D arrays  
- One-hot encode the labels  
- Build and train a *Feedforward Neural Network (FNN)* using Keras  
- Evaluate model performance and suggest improvements

### 🧱 Model Architecture

- **Input Layer:** 784 neurons (flattened 28x28 input)  
- **Hidden Layers:** Dense layers with ReLU activations  
- **Output Layer:** 10 neurons with softmax activation for multiclass classification

### ✅ Result

Achieved approximately **98% accuracy** on the test set using a basic feedforward network.

---

## ❤️ Scenario 2: Heart Disease Prediction

### 📝 Problem Statement

A healthcare organization aims to create a predictive model that can assess the risk of **heart disease** in patients based on clinical features like age, cholesterol, blood pressure, and more.

### 📊 Workflow

- Load and clean the patient dataset from a CSV file  
- Normalize feature values and split the dataset into training and testing sets  
- Address class imbalance using `class_weight`  
- Construct and train an *Artificial Neural Network (ANN)* with Keras  
- Evaluate model performance and propose further improvements

### 🧱 Model Architecture

- **Hidden Layers:** Dense layers with ReLU activations  
- **Output Layer:** 1 neuron with sigmoid activation (for binary classification)

### ✅ Result

Built a robust binary classification model that effectively handles class imbalance and performs well on unseen data.

---

## 🧰 Libraries & Tools

- **Python**  
- **TensorFlow / Keras**  
- **NumPy**, **Pandas**  
- **Scikit-learn**  
- **Matplotlib**, **Seaborn** (for visualizations)

---

## 🚀 Future Improvements

### 🔢 MNIST Digit Recognition

- Integrate **Convolutional Neural Networks (CNNs)** for superior image feature learning  
- Apply **Dropout** and **Batch Normalization** to reduce overfitting risks  
- Use **Data Augmentation** techniques to improve generalization

### ❤️ Heart Disease Prediction

- Employ **KerasTuner** or **GridSearchCV** for hyperparameter optimization  
- Implement **feature engineering** for enhanced model performance  
- Evaluate models using **ROC-AUC** scores and **Precision-Recall** curves

---

## 📁 Repository Structure

