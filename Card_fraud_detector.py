"""
CREDIT CARD FRAUD DETECTION PIPELINE
Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
Focus: Imbalanced classification + proper evaluation metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


print("CREDIT CARD FRAUD DETECTION PIPELINE")


# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"D:\Internships\Codsoft ML Internship\Credit Card Fraud Detection\Card fraud\fraudTrain.csv")   # change path if needed

print("\nDataset shape:", df.shape)
print("Fraud rate:", round(df["is_fraud"].mean() * 100, 3), "%")


# ---------------- FEATURE ENGINEERING ----------------
df["log_amt"] = np.log1p(df["amt"])
df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
df["is_night"] = df["hour"].apply(lambda x: 1 if x <= 5 or x >= 22 else 0)

df["geo_dist"] = np.sqrt(
    (df["lat"] - df["merch_lat"])**2 +
    (df["long"] - df["merch_long"])**2
)


# ---------------- FEATURES ----------------
num_features = ["amt", "log_amt", "hour", "is_night", "geo_dist", "city_pop"]
cat_features = ["category", "gender"]

X = df[num_features + cat_features]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ---------------- PREPROCESSING ----------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])


# ---------------- MODELS ----------------
models = {
    "Logistic Regression": ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", DecisionTreeClassifier(max_depth=8))
    ]),

    "Random Forest": ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10))
    ]),

    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss"
        ))
    ])
}


# ---------------- TRAIN + EVALUATE ----------------
results = {}

print("\nMODEL PERFORMANCE\n")

for name, model in models.items():
    start = time.time()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    else:
        prob = model.decision_function(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob)
    }

    print("Model:", name)
    print("Accuracy :", round(results[name]["accuracy"], 4))
    print("Precision:", round(results[name]["precision"], 4))
    print("Recall   :", round(results[name]["recall"], 4))
    print("F1 Score :", round(results[name]["f1"], 4))
    print("ROC-AUC  :", round(results[name]["roc_auc"], 4))
    print("PR-AUC   :", round(results[name]["pr_auc"], 4))
    print("-----------------------------")


# ---------------- BEST MODEL ----------------
best_model = max(results, key=lambda x: results[x]["roc_auc"])
print("\nBest Model:", best_model)


# ---------------- SAMPLE PREDICTION ----------------
sample = X_test.iloc[:5]
best_pipe = models[best_model]

sample_prob = best_pipe.predict_proba(sample)[:, 1]
sample_pred = (sample_prob > 0.5).astype(int)

print("\nSAMPLE PREDICTIONS")
print("-----------------------------")

for i in range(len(sample)):
    print(f"Transaction {i+1}:")
    print("Probability:", round(sample_prob[i], 4))
    print("Prediction :", "FRAUD" if sample_pred[i] == 1 else "LEGIT")
    print("-----------------------------")