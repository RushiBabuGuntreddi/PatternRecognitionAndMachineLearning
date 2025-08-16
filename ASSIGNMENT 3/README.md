# 📧 Spam Email Classifier (PRML Assignment 3)

This repository contains the implementation and report for **Assignment 3: Spam or Ham** as part of the *Pattern Recognition and Machine Learning (PRML)* course.

---

## 📂 Repository Contents

- **Spamclassifier.ipynb**  
  Jupyter Notebook with the complete implementation of the email classifier.

- **Report.pdf**  
  Detailed report explaining dataset, preprocessing, feature extraction, classifiers used, hyperparameter tuning, and final results.

- **PRML_Assignment3.pdf**  
  Original assignment problem statement.

---

## 📝 Problem Statement

The objective is to build a **spam email classifier from scratch** that can read emails from a `test/` folder and classify each as:
- `+1` → Spam  
- `0`  → Not Spam (Ham)

Algorithms (except SVM) are implemented **from scratch**, as required by the assignment.

---

## 📊 Dataset

- **Source:** [Spam Email Dataset (emails.csv)](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset)  
- **Description:** 5728 emails with labels (`1 = spam`, `0 = not spam`)  
- **Columns:**  
  - `text` → email content (subject + body)  
  - `spam_or_not` → binary label  

---

## ⚙️ Preprocessing

Steps applied on raw email text:
- Remove `"Subject"` keyword.  
- Remove non-alphabetic characters using regex.  
- Convert all text to lowercase.  

This ensures standardized inputs for feature extraction and training.

---

## 🧮 Feature Extraction

Three feature extraction strategies were used:
1. **CountVectorizer (binary features)** – for Naive Bayes.  
2. **TfidfVectorizer** – for Logistic Regression.  
3. **CountVectorizer (word frequency)** – for SVM.  

---

## 🤖 Models Implemented

1. **Naive Bayes**  
   - Uses conditional probability of words.  
   - Log-likelihood formulation to handle small probabilities.  

2. **Logistic Regression**  
   - Trained with gradient descent.  
   - Hyperparameter tuning on step size (`η`).  
   - TF-IDF features provided better performance.  

3. **Support Vector Machine (SVM)**  
   - Frequency-based features.  
   - Hyperparameter `C = 1.0` chosen for optimal performance.  

---

## 🏆 Final Classifier

- All three models are trained on the full dataset.  
- For each email in `test/`, predictions are made using all three classifiers.  
- The **final prediction = majority vote** among the three models.  

---

## 📈 Results (Highlights)

- **Logistic Regression (TF-IDF, η = 1e-1):** Achieved ~99.47% accuracy.  
- **SVM (C = 1.0):** High accuracy with frequency-based features.  

TF-IDF features performed better and faster than CountVectorizer for logistic regression.

---

## ▶️ Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/spam-email-classifier.git
   cd spam-email-classifier
