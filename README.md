# 🌐 Language Detection System

An end-to-end language classification system built using a combination of traditional machine learning and deep learning models. The system identifies the language of given text inputs with high accuracy, utilizing advanced preprocessing, feature extraction, and multiple evaluation metrics.

---

## 📌 Project Overview

This project demonstrates how machine learning and deep learning techniques can be applied to the problem of automatic **language identification**. It includes everything from data preprocessing to model training and evaluation using various text processing methods and model architectures.

Key highlights:
- **Multiple models**: SVM, Random Forest, KNN, Naive Bayes, Ridge Classifier, MLP, GRU, Char-level CNN
- Feature engineering with **TF-IDF**, **CountVectorizer**, and **GloVe embeddings**
- Accuracy range: **97% – 99%** with **no overfitting**
- Includes model persistence for reuse and deployment

---

## 🧠 Technologies & Tools Used

- **Languages & Libraries**: Python, NumPy, Pandas
- **Machine Learning**: Scikit-learn
- **Deep Learning**: TensorFlow / Keras
- **Text Processing**: TF-IDF, CountVectorizer, Tokenizer, Padding
- **Embeddings**: GloVe (pre-trained word vectors)
- **Model Persistence**: `joblib`, `pickle`
- **Visualization**: Matplotlib, Seaborn

---

## 🧪 Project Workflow

### 1. 🧹 Data Preprocessing & Balancing
- Cleaned and tokenized text data.
- Addressed class imbalance in two datasets using:
  - **Upsampling** of minority classes
  - **Downsampling** of majority classes

### 2. 🧾 Feature Extraction
- Traditional ML models:
  - **TF-IDF**
  - **CountVectorizer**
- Deep learning models:
  - **Tokenization and Padding**
  - **GloVe word embeddings** for semantic understanding

### 3. 🧠 Model Training
- **Traditional Models:**  
  SVM, Random Forest, KNN, Naive Bayes, Ridge Classifier, MLP  
- **Deep Learning Models:**  
  GRU, Char-level CNN  
- Each model was trained and evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**

### 4. ✅ Results
- All models achieved **97%–99% accuracy** on test data
- Deep learning models showed strong performance with no overfitting
- Evaluation metrics confirmed consistent and reliable language detection

### 5. 💾 Model Persistence
- Saved trained models and evaluation metrics using `joblib` and `pickle` for reuse or deployment

---
### 📈 Model Performance

| Model                | Dataset 1 Accuracy | Dataset 2 Accuracy | Type               |
|----------------------|--------------------|--------------------|--------------------|
| SVM                  | 98.76%             | 99.64%             | Machine Learning   |
| Random Forest        | 98.59%             | 97.5%              | Machine Learning   |
| K-Nearest Neighbors  | 99.64%             | 98.0%              | Machine Learning   |
| Naive Bayes          | 97.2%              | 97.9%              | Machine Learning   |
| Ridge Classifier     | 97.5%              | 98.1%              | Machine Learning   |
| MLP (Neural Net)     | 98.0%              | 98.6%              | Machine Learning   |
| GRU                  | 97.68%             | 99.68%             | Deep Learning      |
| Char-level CNN       | 98.5%              | 98.8%              | Deep Learning      |
| 1D-CNN               | 99.995%            | 97.85%             | Deep Learning      |

> ✅ All models achieved high accuracy (97–99%) across both datasets with no overfitting, correctly classifying all test cases.

### Clone the repository:
```bash
git clone https://github.com/yourusername/language-detection-system.git
cd language-detection-system

