# CodSoft Machine Learning Internship

A collection of machine learning projects completed as part of the CodSoft ML Internship program.

---

## Task 1 — Movie Genre Classification

Predicts the genre of a movie from its plot summary using NLP techniques.

**Approach:** TF-IDF vectorization combined with a genre keyword lexicon, classified using Logistic Regression / SVM.

**Key Libraries:** `scikit-learn`, `pandas`, `numpy`, `re`

**Run:**
```bash
python Model2_optimized__1_.py
```

---

## Task 2 — Credit Card Fraud Detection

Detects fraudulent transactions from credit card data, handling severe class imbalance.

**Approach:** Feature engineering (log amount, geo-distance, hour of day) + SMOTE oversampling. Models compared: Logistic Regression, Decision Tree, Random Forest, XGBoost.

**Key Libraries:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`

**Run:**
```bash
python Card_fraud_detector.py
```

---

## Task 3 — Customer Churn Prediction

Predicts whether a customer will churn from a subscription-based service.

**Approach:** Trained on historical customer data (usage behavior, demographics) using Logistic Regression, Random Forest, and Gradient Boosting, with cross-validation and feature importance analysis.

**Key Libraries:** `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

**Run:**
```bash
python churn_predictor.py
```

---

## Task 4 — Spam SMS Detection

Classifies SMS messages as spam or legitimate (ham).

**Approach:** Text preprocessing (lowercasing, URL/number normalization) + TF-IDF, served via a Flask web app.

**Key Libraries:** `scikit-learn`, `flask`, `joblib`, `numpy`

**Run:**
```bash
python app.py
```
Then open `http://localhost:10000` in your browser.

---

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn flask joblib matplotlib seaborn
```

---

## Repository Structure

```
CODSOFT/
├── Model2_optimized__1_.py   # Task 1 – Movie Genre Classification
├── Card_fraud_detector.py    # Task 2 – Credit Card Fraud Detection
├── churn_predictor.py        # Task 3 – Customer Churn Prediction
├── app.py                    # Task 4 – Spam SMS Detection (Flask app)
└── README.md
```

---

*Internship by [CodSoft](https://www.codsoft.in)*
