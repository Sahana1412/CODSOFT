import sys
sys.stdout.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    precision_recall_curve
)

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import time

# =============================================================================
# COLORS
# =============================================================================
C_BLUE   = "#3498db"
C_RED    = "#e74c3c"
C_GREEN  = "#2ecc71"
C_ORANGE = "#e67e22"
C_PURPLE = "#9b59b6"

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv(
    r"D:\Internships\Codsoft ML Internship\Customer Churn Prediction\Customer churn dataset\Churn_Modelling.csv"
)

df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
df["BalanceSalaryRatio"] = (
    df["Balance"] / (df["EstimatedSalary"] + 1)
)

df["TenureByAge"] = (
    df["Tenure"] / (df["Age"] + 1)
)

df["CreditScorePerAge"] = (
    df["CreditScore"] / (df["Age"] + 1)
)

df["IsZeroBalance"] = (
    (df["Balance"] == 0).astype(int)
)

df["ProductsPerTenure"] = (
    df["NumOfProducts"] / (df["Tenure"] + 1)
)

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["<30", "30-40", "40-50", "50-60", "60+"]
)

# =============================================================================
# PREPROCESSING
# =============================================================================
cat_cols = ["Geography", "Gender", "AgeGroup"]

num_cols = [
    c for c in df.columns
    if c not in cat_cols + ["Exited"]
    and df[c].dtype != object
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),

        (
            "cat",
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore"
            ),
            cat_cols
        ),
    ]
)

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# =============================================================================
# MODELS
# =============================================================================
smote = SMOTE(random_state=42, k_neighbors=5)

models = {

    "Logistic Regression": ImbPipeline([
        ("pre", preprocessor),
        ("smote", smote),

        (
            "clf",
            LogisticRegression(
                C=0.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        ),
    ]),

    "Random Forest": ImbPipeline([
        ("pre", preprocessor),
        ("smote", smote),

        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
        ),
    ]),

    "Gradient Boosting": ImbPipeline([
        ("pre", preprocessor),
        ("smote", smote),

        (
            "clf",
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
        ),
    ]),

    "XGBoost": ImbPipeline([
        ("pre", preprocessor),
        ("smote", smote),

        (
            "clf",
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        ),
    ]),
}

# =============================================================================
# CROSS VALIDATION
# =============================================================================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = {}

for name, pipe in models.items():

    auc = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    f1 = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    cv_results[name] = {
        "auc_mean": auc.mean(),
        "auc_std": auc.std(),
        "f1_mean": f1.mean(),
        "f1_std": f1.std()
    }

best_name = max(
    cv_results,
    key=lambda k: cv_results[k]["auc_mean"]
)

# =============================================================================
# TRAIN BEST MODEL
# =============================================================================
best_pipe = models[best_name]

best_pipe.fit(X_train, y_train)

y_pred = best_pipe.predict(X_test)

y_prob = best_pipe.predict_proba(X_test)[:, 1]

# =============================================================================
# METRICS
# =============================================================================
acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

ap = average_precision_score(y_test, y_prob)

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
pre_step = best_pipe.named_steps["pre"]

ohe = pre_step.named_transformers_["cat"]

ohe_names = list(
    ohe.get_feature_names_out(cat_cols)
)

feat_names = num_cols + ohe_names

clf = best_pipe.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
else:
    importances = np.abs(clf.coef_[0])

imp_df = (
    pd.DataFrame({
        "Feature": feat_names,
        "Importance": importances
    })
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(20)
)

# =============================================================================
# ALL MODEL RESULTS
# =============================================================================
all_results = {}

for name, pipe in models.items():

    pipe.fit(X_train, y_train)

    yp = pipe.predict(X_test)

    ypr = pipe.predict_proba(X_test)[:, 1]

    all_results[name] = {

        "acc": accuracy_score(y_test, yp),

        "prec": precision_score(y_test, yp),

        "rec": recall_score(y_test, yp),

        "f1": f1_score(y_test, yp),

        "auc": roc_auc_score(y_test, ypr),

        "prob": ypr,

        "pred": yp,
    }

# =============================================================================
# DASHBOARD
# =============================================================================
fig = plt.figure(figsize=(20, 22))

fig.patch.set_facecolor("#0f1117")

gs = gridspec.GridSpec(
    4,
    3,
    figure=fig,
    hspace=0.45,
    wspace=0.35
)

TITLE_KW = dict(
    color="white",
    fontsize=12,
    fontweight="bold",
    pad=10
)

LABEL_KW = dict(
    color="#cccccc",
    fontsize=10
)

SPINE_CLR = "#333333"

def style_ax(ax):

    ax.set_facecolor("#1a1d27")

    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_CLR)

    ax.tick_params(
        colors="white",
        labelsize=9
    )

# =============================================================================
# CHURN DISTRIBUTION
# =============================================================================
ax = fig.add_subplot(gs[0, 0])

style_ax(ax)

counts = y.value_counts()

bars = ax.bar(
    ["Stay", "Churn"],
    counts.values,
    color=[C_GREEN, C_RED]
)

ax.set_title("Class Distribution", **TITLE_KW)

ax.set_ylabel("Count", **LABEL_KW)

# =============================================================================
# CV AUC COMPARISON
# =============================================================================
ax = fig.add_subplot(gs[0, 1])

style_ax(ax)

names_list = list(cv_results.keys())

auc_means = [
    cv_results[n]["auc_mean"]
    for n in names_list
]

bar_colors = [
    C_RED if n == best_name else C_BLUE
    for n in names_list
]

ax.barh(
    names_list,
    auc_means,
    color=bar_colors
)

ax.set_title("ROC-AUC Comparison", **TITLE_KW)

ax.set_xlabel("ROC-AUC", **LABEL_KW)

# =============================================================================
# HEATMAP
# =============================================================================
ax = fig.add_subplot(gs[0, 2])

style_ax(ax)

metric_df = pd.DataFrame({
    n: {
        k: v
        for k, v in all_results[n].items()
        if k in ["acc", "prec", "rec", "f1", "auc"]
    }
    for n in all_results
}).T

metric_df.columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC"
]

sns.heatmap(
    metric_df,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    ax=ax,
    cbar=False
)

ax.set_title("Metrics Heatmap", **TITLE_KW)

# =============================================================================
# ROC CURVES
# =============================================================================
ax = fig.add_subplot(gs[1, 0:2])

style_ax(ax)

line_colors = [
    C_RED,
    C_BLUE,
    C_GREEN,
    C_ORANGE
]

for (name, res), lc in zip(
    all_results.items(),
    line_colors
):

    fpr, tpr, _ = roc_curve(
        y_test,
        res["prob"]
    )

    ax.plot(
        fpr,
        tpr,
        label=f"{name} ({res['auc']:.4f})",
        color=lc
    )

ax.plot([0, 1], [0, 1], "w--")

ax.legend()

ax.set_title("ROC Curves", **TITLE_KW)

# =============================================================================
# CONFUSION MATRIX
# =============================================================================
ax = fig.add_subplot(gs[1, 2])

style_ax(ax)

cm = confusion_matrix(
    y_test,
    all_results[best_name]["pred"]
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    ax=ax,
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"]
)

ax.set_title("Confusion Matrix", **TITLE_KW)

# =============================================================================
# PRECISION RECALL CURVE
# =============================================================================
ax = fig.add_subplot(gs[2, 0])

style_ax(ax)

for (name, res), lc in zip(
    all_results.items(),
    line_colors
):

    prec_curve, rec_curve, _ = precision_recall_curve(
        y_test,
        res["prob"]
    )

    ax.plot(
        rec_curve,
        prec_curve,
        label=name,
        color=lc
    )

ax.legend()

ax.set_title("Precision Recall Curve", **TITLE_KW)

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
ax = fig.add_subplot(gs[2, 1:])

style_ax(ax)

top15 = imp_df.head(15)

ax.barh(
    top15["Feature"],
    top15["Importance"],
    color=C_BLUE
)

ax.invert_yaxis()

ax.set_title(
    f"Feature Importance ({best_name})",
    **TITLE_KW
)

# =============================================================================
# CHURN BY GEOGRAPHY
# =============================================================================
ax = fig.add_subplot(gs[3, 0])

style_ax(ax)

geo_churn = (
    df.groupby("Geography")["Exited"]
    .mean()
    .sort_values(ascending=False)
)

ax.bar(
    geo_churn.index,
    geo_churn.values * 100,
    color=[C_RED, C_ORANGE, C_BLUE]
)

ax.set_ylabel("Churn Rate (%)", **LABEL_KW)

ax.set_title("Churn by Geography", **TITLE_KW)

# =============================================================================
# CHURN BY AGE GROUP
# =============================================================================
ax = fig.add_subplot(gs[3, 1])

style_ax(ax)

age_churn = (
    df.groupby("AgeGroup", observed=True)["Exited"]
    .mean()
)

ax.bar(
    age_churn.index.astype(str),
    age_churn.values * 100,
    color=[C_GREEN, C_BLUE, C_ORANGE, C_RED, C_PURPLE]
)

ax.set_ylabel("Churn Rate (%)", **LABEL_KW)

ax.set_title("Churn by Age Group", **TITLE_KW)

# =============================================================================
# CHURN PROBABILITY DISTRIBUTION
# =============================================================================
ax = fig.add_subplot(gs[3, 2])

style_ax(ax)

prob_churn = all_results[best_name]["prob"]

ax.hist(
    prob_churn[y_test == 0],
    bins=40,
    alpha=0.7,
    color=C_GREEN,
    label="Stay"
)

ax.hist(
    prob_churn[y_test == 1],
    bins=40,
    alpha=0.7,
    color=C_RED,
    label="Churn"
)

ax.legend()

ax.set_title("Prediction Probability Distribution", **TITLE_KW)

# =============================================================================
# SAVE DASHBOARD
# =============================================================================
fig.text(
    0.5,
    0.995,
    f"Customer Churn Prediction Dashboard",
    ha="center",
    va="top",
    fontsize=16,
    fontweight="bold",
    color="white"
)

plt.savefig(
    "churn_dashboard.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#0f1117"
)

# =============================================================================
# FINAL OUTPUT
# =============================================================================
print("\nModel Performance\n")

metrics = [
    ("Accuracy", acc),
    ("Precision", prec),
    ("Recall", rec),
    ("F1 Score", f1),
    ("ROC AUC", auc)
]

for label, value in metrics:
    print(f"{label}: {value:.4f}")

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Stay", "Churn"]
))

# =============================================================================
# SAMPLE PREDICTIONS
# =============================================================================
print("\nSample Predictions\n")

sample_idx = X_test.sample(
    8,
    random_state=7
).index

sample_X = X_test.loc[sample_idx]

sample_y = y_test.loc[sample_idx]

sample_prob = best_pipe.predict_proba(sample_X)[:, 1]

sample_pred = best_pipe.predict(sample_X)

for i, (idx, row) in enumerate(sample_X.iterrows()):

    actual = "Churn" if sample_y.loc[idx] == 1 else "Stay"

    predicted = "Churn" if sample_pred[i] == 1 else "Stay"

    print(
        f"Customer {i+1} | "
        f"Age: {int(row.Age)} | "
        f"Country: {row.Geography} | "
        f"Balance: {row.Balance:,.0f} | "
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"Probability: {sample_prob[i]:.3f}"
    )

print("\nDashboard saved as churn_dashboard.png")

# =============================================================================
# CUSTOM CUSTOMER PREDICTION
# =============================================================================
print("\nCustomer Churn Prediction")

credit_score = int(input("Credit Score: "))
geography = input("Geography (France/Germany/Spain): ")
gender = input("Gender (Male/Female): ")
age = int(input("Age: "))
tenure = int(input("Tenure: "))
balance = float(input("Balance: "))
num_products = int(input("Number of Products: "))
has_card = int(input("Has Credit Card (1/0): "))
is_active = int(input("Is Active Member (1/0): "))
salary = float(input("Estimated Salary: "))

# Feature engineering
balance_salary_ratio = balance / (salary + 1)

tenure_by_age = tenure / (age + 1)

credit_score_per_age = credit_score / (age + 1)

is_zero_balance = int(balance == 0)

products_per_tenure = num_products / (tenure + 1)

if age < 30:
    age_group = "<30"
elif age < 40:
    age_group = "30-40"
elif age < 50:
    age_group = "40-50"
elif age < 60:
    age_group = "50-60"
else:
    age_group = "60+"

# Create dataframe
custom_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary,

    "BalanceSalaryRatio": balance_salary_ratio,
    "TenureByAge": tenure_by_age,
    "CreditScorePerAge": credit_score_per_age,
    "IsZeroBalance": is_zero_balance,
    "ProductsPerTenure": products_per_tenure,
    "AgeGroup": age_group
}])

prediction = best_pipe.predict(custom_data)[0]

probability = best_pipe.predict_proba(custom_data)[0][1]

result = "Customer is likely to CHURN" if prediction == 1 else "Customer is likely to STAY"

print("\nPrediction Result")
print(result)

print(f"Churn Probability: {probability:.2%}")