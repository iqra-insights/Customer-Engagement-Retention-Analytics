"""
Predictive Churn Model — Logistic Regression
------------------------------------------------------------------
Builds a genuine, trained logistic regression model to predict churn,
using the engagement/product/balance features engineered earlier in this
project. Reports real train/test performance (accuracy, precision, recall,
F1, ROC-AUC) and odds ratios for each feature's independent effect on churn,
controlling for all other features simultaneously.

This is deliberately kept as a simple, interpretable baseline model (not a
tuned/ensembled production model) since the project's goal is explanatory
insight for a retention strategy, not maximum predictive accuracy.
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

CHART_DIR = "/home/claude/engagement-retention-analytics/outputs/charts"

df = pd.read_csv("/home/claude/engagement-retention-analytics/outputs/processed_customer_engagement_data.csv")

# ---------------------------------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------------------------------
model_df = df.copy()
model_df["Germany"] = (model_df["Geography"] == "Germany").astype(int)
model_df["Spain"] = (model_df["Geography"] == "Spain").astype(int)
model_df["Gender_Male"] = (model_df["Gender"] == "Male").astype(int)

feature_cols = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Germany", "Spain", "Gender_Male",
]
X = model_df[feature_cols]
y = model_df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale continuous features (logistic regression coefficients are more
# interpretable and stable when features are standardized)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# REAL PERFORMANCE METRICS (test set — held out, not seen during training)
# ---------------------------------------------------------------------------
metrics = {
    "train_size": len(X_train),
    "test_size": len(X_test),
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, y_pred_proba)), 4),
    "baseline_accuracy_predict_majority_class": round(float(1 - y_test.mean()), 4),
}

cm = confusion_matrix(y_test, y_pred)
metrics["confusion_matrix"] = {
    "true_negative": int(cm[0][0]), "false_positive": int(cm[0][1]),
    "false_negative": int(cm[1][0]), "true_positive": int(cm[1][1]),
}

# ---------------------------------------------------------------------------
# ODDS RATIOS (exp(coefficient) — independent effect of each feature,
# controlling for all others, since features were standardized)
# ---------------------------------------------------------------------------
odds_ratios = []
for feat, coef in zip(feature_cols, model.coef_[0]):
    odds_ratios.append({
        "feature": feat,
        "coefficient": round(float(coef), 4),
        "odds_ratio": round(float(np.exp(coef)), 3),
    })
odds_ratios.sort(key=lambda d: abs(d["coefficient"]), reverse=True)

output = {
    "model": "Logistic Regression (scikit-learn, standardized features, 75/25 train/test split, random_state=42)",
    "performance_on_held_out_test_set": metrics,
    "odds_ratios_per_1_std_dev_increase": odds_ratios,
    "note": "Odds ratios >1 increase churn odds; <1 decrease churn odds. All effects are per 1 standard deviation increase in the (standardized) feature, holding all other features constant.",
}

with open("/home/claude/engagement-retention-analytics/outputs/predictive_model_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# CHART: Odds ratio tornado chart
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
labels = [o["feature"] for o in odds_ratios]
values = [o["odds_ratio"] for o in odds_ratios]
colors = ["#E63946" if v > 1 else "#2E86AB" for v in values]

plt.figure(figsize=(8, 6))
bars = plt.barh(labels, values, color=colors)
plt.axvline(x=1, color="black", linewidth=1, linestyle="--")
plt.xlabel("Odds Ratio (per 1 SD increase, holding others constant)")
plt.title("Independent Effect of Each Feature on Churn Odds\n(Logistic Regression)", fontsize=12, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_logistic_regression_odds_ratios.png", dpi=150)
plt.close()

print("\nChart saved.")
