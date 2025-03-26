# Handwriting Recognition System using kNN and CNN

## 📌 Overview
This project implements a **handwriting recognition system** using two different machine learning models:
- **k-Nearest Neighbors (kNN)**
- **Convolutional Neural Networks (CNN)**

The models are trained on the **MNIST dataset**, which consists of **70,000 grayscale images** of handwritten digits (0-9).

---

## 🗂 Dataset Details
- **Dataset:** MNIST (28x28 grayscale images)
- **Training set:** 60,000 images
- **Testing set:** 10,000 images
- **Classes:** 10 (Digits 0-9)

---

## ⚡ k-Nearest Neighbors (kNN)
### **Implementation Details:**
- **Feature Scaling:** Standardized pixel values using `StandardScaler()`
- **Hyperparameter tuning:** Best `k` value found using **cross-validation**
- **Dimensionality Reduction:** PCA applied (50 components) to improve speed

### **Results:**
| Model | Accuracy |
|--------|----------|
| kNN (without PCA) | 94.65% |
| kNN (with PCA) | 95.86% |

### **Strengths & Weaknesses:**
✅ Simple to implement, no training required  
✅ Works well for small datasets  
⚠️ Slow for large datasets due to distance computation  
⚠️ Requires high memory for storing all data  

---

## 🧠 Convolutional Neural Network (CNN)
### **Implementation Details:**
- **Architecture:**
  - 3 Convolutional Layers with ReLU activation
  - MaxPooling for feature reduction
  - Fully connected Dense layers
  - Dropout to prevent overfitting
- **Optimizer:** Adam
- **Loss function:** Categorical Crossentropy

### **Results:**
| Model | Accuracy |
|--------|----------|
| CNN | 99.14% |

### **Strengths & Weaknesses:**
✅ High accuracy, learns spatial patterns  
✅ Efficient for large datasets  
⚠️ Requires more computation and longer training time  
⚠️ Needs a larger dataset for best performance  

---

## 🔍 Comparison: kNN vs. CNN
| Feature | kNN | CNN |
|------------|----------------|----------------|
| **Accuracy** | ~95% | ~99% |
| **Training Time** | No training needed | Longer due to deep learning |
| **Prediction Speed** | Slow for large datasets | Fast due to learned weights |
| **Best For** | Small datasets | Large, complex image datasets |

---

## 🚀 Future Improvements
- **Enhancing kNN**:
  - Use **KD-Trees** or **Ball Trees** for faster distance calculations
  - Try different distance metrics (Euclidean, Manhattan, Minkowski)
- **Enhancing CNN**:
  - Train on a larger dataset for even better accuracy
  - Experiment with **data augmentation** to improve generalization
  - Implement **transfer learning** using pretrained models like VGG16

---

## 📂 How to Run the Project
### **Requirements:**
- Python 3.x
- TensorFlow/Keras
- scikit-learn
- NumPy, Matplotlib, Seaborn

### **Run kNN Model:**
```bash
python knn_mnist.py
```

### **Run CNN Model:**
```bash
python cnn_mnist.py
```

---

## 📜 Conclusion
- kNN is **simple and effective** for smaller datasets but suffers from slow performance on large datasets.
- CNN **outperforms kNN** in terms of accuracy and speed, making it the preferred choice for handwriting recognition.
- Future optimizations can further enhance both models!

---

✉ **Author:** Arjun  
📅 **Date:** March 2025  
