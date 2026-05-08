import warnings
warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.naive_bayes import (
    MultinomialNB,
    ComplementNB
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

import re
import time

# Load dataset
df = pd.read_csv(
    r"D:\Internships\Codsoft ML Internship\Spam SMS Detection\Spam sms dataset\spam.csv",
    encoding="latin-1"
)[["v1", "v2"]]

df.columns = ["label", "message"]

df.dropna(inplace=True)

df["label_enc"] = (
    df["label"] == "spam"
).astype(int)

print(f"Dataset loaded: {len(df)} messages")

# Preprocessing
def preprocess(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " url ",
        text
    )

    text = re.sub(
        r"\b\d+\b",
        " num ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

df["clean"] = df["message"].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"],
    df["label_enc"],
    test_size=0.20,
    random_state=42,
    stratify=df["label_enc"]
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Vectorizers
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    min_df=2
)

bow = CountVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2
)

# Models
pipelines = {

    "BoW + MultinomialNB": Pipeline([
        ("vec", bow),
        ("clf", MultinomialNB(alpha=0.1))
    ]),

    "TF-IDF + MultinomialNB": Pipeline([
        ("vec", tfidf),
        ("clf", MultinomialNB(alpha=0.1))
    ]),

    "TF-IDF + ComplementNB": Pipeline([
        ("vec", tfidf),
        ("clf", ComplementNB(alpha=0.1))
    ]),

    "TF-IDF + LogReg": Pipeline([
        ("vec", tfidf),
        ("clf", LogisticRegression(
            C=5,
            max_iter=1000,
            random_state=42
        ))
    ]),

    "TF-IDF + LinearSVC": Pipeline([
        ("vec", tfidf),
        ("clf", LinearSVC(
            C=1.0,
            max_iter=2000,
            random_state=42
        ))
    ]),
}

# Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

print("\nModel Results:\n")

for name, pipe in pipelines.items():

    t0 = time.time()

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    elapsed = time.time() - t0

    results[name] = {
        "mean": scores.mean(),
        "std": scores.std(),
        "time": elapsed
    }

    print(
        f"{name} | "
        f"F1: {scores.mean():.4f} | "
        f"Time: {elapsed:.1f}s"
    )

best_name = max(
    results,
    key=lambda k: results[k]["mean"]
)

print(f"\nBest model: {best_name}")

# Train best model
best_pipe = pipelines[best_name]

best_pipe.fit(
    X_train,
    y_train
)

# Save trained model
joblib.dump(
    best_pipe,
    "spam_classifier_model.pkl"
)

print("Saved: spam_classifier_model.pkl")

# Predictions
y_pred = best_pipe.predict(X_test)

clf = best_pipe.named_steps["clf"]

if hasattr(clf, "predict_proba"):

    y_score = best_pipe.predict_proba(X_test)[:, 1]

else:

    y_score = best_pipe.decision_function(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_score)

print("\nMetrics:\n")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"]
    )
)

# Plots
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 11)
)

# Model comparison
ax = axes[0, 0]

names = list(results.keys())

means = [results[n]["mean"] for n in names]

stds = [results[n]["std"] for n in names]

colors = [
    "#e74c3c" if n == best_name else "#3498db"
    for n in names
]

bars = ax.barh(
    names,
    means,
    xerr=stds,
    color=colors,
    edgecolor="white",
    height=0.6
)

ax.set_xlabel("F1 Score")
ax.set_title("Model Comparison")

for bar, v in zip(bars, means):

    ax.text(
        v + 0.001,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.4f}",
        va="center",
        fontsize=9
    )

ax.invert_yaxis()

# Confusion matrix
ax = axes[0, 1]

cm = confusion_matrix(
    y_test,
    y_pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["ham", "spam"],
    yticklabels=["ham", "spam"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

# ROC curve
ax = axes[1, 0]

fpr, tpr, _ = roc_curve(
    y_test,
    y_score
)

ax.plot(
    fpr,
    tpr,
    lw=2,
    label=f"AUC = {auc:.4f}"
)

ax.plot(
    [0, 1],
    [0, 1],
    "k--",
    lw=1
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")

ax.legend()

# Feature importance
ax = axes[1, 1]

vec = best_pipe.named_steps["vec"]

clf_ = best_pipe.named_steps["clf"]

fn = vec.get_feature_names_out()

if hasattr(clf_, "coef_"):

    coef = clf_.coef_.ravel()

else:

    coef = (
        clf_.feature_log_prob_[1]
        - clf_.feature_log_prob_[0]
    )

top_n = 20

top_idx = np.argsort(coef)[-top_n:]

bot_idx = np.argsort(coef)[:top_n]

combined_idx = np.concatenate([
    bot_idx,
    top_idx
])

combined_vals = coef[combined_idx]

combined_labels = fn[combined_idx]

colors_feat = [
    "#27ae60" if v < 0 else "#e74c3c"
    for v in combined_vals
]

ax.barh(
    combined_labels,
    combined_vals,
    color=colors_feat,
    edgecolor="white"
)

ax.axvline(
    0,
    color="black",
    lw=0.8
)

ax.set_title("Feature Importance")

plt.tight_layout()

plt.savefig(
    "spam_classifier_results.png",
    dpi=150,
    bbox_inches="tight"
)

print("\nSaved: spam_classifier_results.png")

# Class distribution chart
fig2, ax2 = plt.subplots(
    figsize=(5, 5)
)

counts = df["label"].value_counts()

ax2.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=["#27ae60", "#e74c3c"],
    startangle=90
)

ax2.set_title("Class Distribution")

plt.tight_layout()

plt.savefig(
    "class_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

print("Saved: class_distribution.png")

# Sample predictions
samples = [
    "WINNER!! You have been selected for a FREE prize.",
    "Hey, are we still on for lunch tomorrow?",
    "Congratulations! You've won a gift card.",
    "Can you send the notes?",
]

clean_samples = [
    preprocess(s)
    for s in samples
]

preds = best_pipe.predict(clean_samples)

print("\nSample Predictions:\n")

for msg, pred in zip(samples, preds):

    label = "Spam" if pred == 1 else "Ham"

    print(f"{label}: {msg}")

# Interactive prediction
while True:

    user_msg = input(
        "\nEnter message (type 'exit' to quit): "
    )

    if user_msg.lower() == "exit":
        break

    clean_msg = preprocess(user_msg)

    prediction = best_pipe.predict([clean_msg])[0]

    if hasattr(clf, "predict_proba"):

        confidence = best_pipe.predict_proba(
            [clean_msg]
        )[0].max()

    else:

        score = best_pipe.decision_function(
            [clean_msg]
        )[0]

        confidence = 1 / (
            1 + np.exp(-abs(score))
        )

    label = "Spam" if prediction == 1 else "Ham"

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

print("\nFinished")