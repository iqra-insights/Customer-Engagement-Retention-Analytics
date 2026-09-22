"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
----------------------------------------------------------------------------
Data ingestion, validation, engagement classification, product utilization
analysis, financial-commitment vs engagement analysis, retention strength
scoring, and KPI computation.

Author: Iqra
Dataset: European Bank customer records (10,000 rows)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import chi2_contingency, pointbiserialr

sns.set_theme(style="whitegrid")
CHART_DIR = "/home/claude/engagement-retention-analytics/outputs/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATA INGESTION & VALIDATION
# ---------------------------------------------------------------------------
df = pd.read_csv("/home/claude/engagement-retention-analytics/data/customer_engagement_data.csv")

validation_report = {}
validation_report["total_rows"] = len(df)
validation_report["duplicate_customer_ids"] = int(df["CustomerId"].duplicated().sum())
validation_report["missing_values"] = df.isnull().sum().to_dict()
validation_report["hascrcard_unique"] = sorted(df["HasCrCard"].unique().tolist())
validation_report["isactivemember_unique"] = sorted(df["IsActiveMember"].unique().tolist())
validation_report["exited_unique"] = sorted(df["Exited"].unique().tolist())
validation_report["geography_values"] = sorted(df["Geography"].unique().tolist())
validation_report["numofproducts_range"] = [int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max())]

# ---------------------------------------------------------------------------
# 2. ENGAGEMENT CLASSIFICATION
# ---------------------------------------------------------------------------
median_balance = df["Balance"].median()
high_balance_threshold = df["Balance"].quantile(0.75)

def classify_engagement(row):
    active = row["IsActiveMember"] == 1
    multi_product = row["NumOfProducts"] >= 2
    high_balance = row["Balance"] >= high_balance_threshold

    if active and multi_product:
        return "Active Engaged"
    elif not active and high_balance:
        return "Inactive High-Balance"
    elif active and not multi_product:
        return "Active Low-Product"
    elif not active and not high_balance:
        return "Inactive Disengaged"
    else:
        return "Other"

df["EngagementProfile"] = df.apply(classify_engagement, axis=1)

engagement_summary = df.groupby("EngagementProfile").agg(
    Customers=("CustomerId", "count"),
    ChurnRate=("Exited", "mean"),
    AvgBalance=("Balance", "mean"),
    AvgProducts=("NumOfProducts", "mean"),
).round(3).reset_index()
engagement_summary["ChurnRatePct"] = (engagement_summary["ChurnRate"] * 100).round(2)

# ---------------------------------------------------------------------------
# 3. PRODUCT UTILIZATION ANALYSIS
# ---------------------------------------------------------------------------
product_churn = df.groupby("NumOfProducts").agg(
    Customers=("CustomerId", "count"),
    ChurnRate=("Exited", "mean"),
).round(3).reset_index()
product_churn["ChurnRatePct"] = (product_churn["ChurnRate"] * 100).round(2)

df["ProductTier"] = np.where(df["NumOfProducts"] == 1, "Single-Product", "Multi-Product")
single_vs_multi = df.groupby("ProductTier").agg(
    Customers=("CustomerId", "count"),
    ChurnRate=("Exited", "mean"),
).round(3).reset_index()
single_vs_multi["ChurnRatePct"] = (single_vs_multi["ChurnRate"] * 100).round(2)

# ---------------------------------------------------------------------------
# 4. FINANCIAL COMMITMENT vs ENGAGEMENT ANALYSIS
# ---------------------------------------------------------------------------
df["BalanceTier"] = pd.qcut(df["Balance"].rank(method="first"), 4,
                             labels=["Low", "Medium", "High", "Premium"])

balance_activity = df.groupby(["BalanceTier", "IsActiveMember"]).agg(
    Customers=("CustomerId", "count"),
    ChurnRate=("Exited", "mean"),
).round(3).reset_index()
balance_activity["ChurnRatePct"] = (balance_activity["ChurnRate"] * 100).round(2)

# Salary-balance mismatch: high salary but low balance (potential under-utilization)
salary_median = df["EstimatedSalary"].median()
df["SalarySalaryMismatch"] = np.where(
    (df["EstimatedSalary"] >= salary_median) & (df["Balance"] < median_balance),
    "High Salary / Low Balance", "Aligned"
)
mismatch_summary = df.groupby("SalarySalaryMismatch").agg(
    Customers=("CustomerId", "count"),
    ChurnRate=("Exited", "mean"),
).round(3).reset_index()
mismatch_summary["ChurnRatePct"] = (mismatch_summary["ChurnRate"] * 100).round(2)

# At-risk premium customers: Premium balance tier + inactive
at_risk_premium = df[(df["BalanceTier"] == "Premium") & (df["IsActiveMember"] == 0)]
at_risk_premium_count = len(at_risk_premium)
at_risk_premium_churn = round(at_risk_premium["Exited"].mean() * 100, 2)

# ---------------------------------------------------------------------------
# 5. RETENTION STRENGTH ASSESSMENT ("sticky customer" scoring)
# ---------------------------------------------------------------------------
def relationship_strength_score(row):
    score = 0
    score += 2 if row["IsActiveMember"] == 1 else 0
    score += min(row["NumOfProducts"], 4) * 1.5
    score += 1 if row["HasCrCard"] == 1 else 0
    score += 1 if row["Tenure"] >= 5 else 0
    return round(score, 2)

df["RelationshipStrengthIndex"] = df.apply(relationship_strength_score, axis=1)

df["StickyCustomer"] = np.where(df["RelationshipStrengthIndex"] >= 6, "Sticky", "At-Risk")
sticky_summary = df.groupby("StickyCustomer").agg(
    Customers=("CustomerId", "count"),
    ChurnRate=("Exited", "mean"),
).round(3).reset_index()
sticky_summary["ChurnRatePct"] = (sticky_summary["ChurnRate"] * 100).round(2)

# Engagement threshold analysis: churn rate by relationship-strength bucket
df["RSIBucket"] = pd.cut(df["RelationshipStrengthIndex"],
                          bins=[-1, 2, 4, 6, 8, 100],
                          labels=["0-2", "2-4", "4-6", "6-8", "8+"])
rsi_bucket_churn = df.groupby("RSIBucket").agg(
    Customers=("CustomerId", "count"),
    ChurnRate=("Exited", "mean"),
).round(3).reset_index()
rsi_bucket_churn["ChurnRatePct"] = (rsi_bucket_churn["ChurnRate"] * 100).round(2)

# ---------------------------------------------------------------------------
# 6. KEY PERFORMANCE INDICATORS
# ---------------------------------------------------------------------------
active_churn = df[df["IsActiveMember"] == 1]["Exited"].mean()
inactive_churn = df[df["IsActiveMember"] == 0]["Exited"].mean()
engagement_retention_ratio = round(inactive_churn / active_churn, 2) if active_churn > 0 else None

# Product Depth Index: correlation-style measure -> churn drop from 1 to max products
churn_1_product = df[df["NumOfProducts"] == 1]["Exited"].mean()
churn_multi_product = df[df["NumOfProducts"] >= 2]["Exited"].mean()
product_depth_index = round((churn_1_product - churn_multi_product) / churn_1_product * 100, 2)

high_balance_disengagement_rate = round(
    df[(df["BalanceTier"] == "Premium") & (df["IsActiveMember"] == 0)].shape[0]
    / df[df["BalanceTier"] == "Premium"].shape[0] * 100, 2
)

cc_churn = df[df["HasCrCard"] == 1]["Exited"].mean()
no_cc_churn = df[df["HasCrCard"] == 0]["Exited"].mean()
credit_card_stickiness_score = round((no_cc_churn - cc_churn) / no_cc_churn * 100, 2)

relationship_strength_index_avg = round(df["RelationshipStrengthIndex"].mean(), 2)

kpi_summary = {
    "Engagement Retention Ratio (inactive churn / active churn)": engagement_retention_ratio,
    "Product Depth Index (% churn reduction, multi vs single product)": product_depth_index,
    "High-Balance Disengagement Rate (% of premium-balance customers inactive)": high_balance_disengagement_rate,
    "Credit Card Stickiness Score (% churn reduction with card ownership)": credit_card_stickiness_score,
    "Average Relationship Strength Index (0-10 scale)": relationship_strength_index_avg,
    "Active Member Churn Rate (%)": round(active_churn * 100, 2),
    "Inactive Member Churn Rate (%)": round(inactive_churn * 100, 2),
    "At-Risk Premium Customers (count)": at_risk_premium_count,
    "At-Risk Premium Customers Churn Rate (%)": at_risk_premium_churn,
    "Overall Churn Rate (%)": round(df["Exited"].mean() * 100, 2),
}

# ---------------------------------------------------------------------------
# 7. STATISTICAL SIGNIFICANCE VALIDATION
# ---------------------------------------------------------------------------
def chi_square_test(col):
    ct = pd.crosstab(df[col], df["Exited"])
    chi2, p, dof, _ = chi2_contingency(ct)
    return {"chi2": round(float(chi2), 2), "p_value": float(p), "dof": int(dof)}

def wilson_ci(p, n, z=1.96):
    se = np.sqrt(p * (1 - p) / n)
    return round((p - z * se) * 100, 2), round((p + z * se) * 100, 2)

r_value, r_p = pointbiserialr(df["Exited"], df["RelationshipStrengthIndex"])

active_n, inactive_n = (df["IsActiveMember"] == 1).sum(), (df["IsActiveMember"] == 0).sum()
active_p, inactive_p = active_churn, inactive_churn

statistical_validation = {
    "EngagementProfile_vs_Exited_chi_square": chi_square_test("EngagementProfile"),
    "ProductTier_vs_Exited_chi_square": chi_square_test("ProductTier"),
    "IsActiveMember_vs_Exited_chi_square": chi_square_test("IsActiveMember"),
    "StickyCustomer_vs_Exited_chi_square": chi_square_test("StickyCustomer"),
    "RSI_vs_Exited_point_biserial_correlation": {"r": round(float(r_value), 4), "p_value": float(r_p)},
    "Active_member_churn_95pct_CI": list(wilson_ci(active_p, active_n)),
    "Inactive_member_churn_95pct_CI": list(wilson_ci(inactive_p, inactive_n)),
}

with open("/home/claude/engagement-retention-analytics/outputs/statistical_validation.json", "w") as f:
    json.dump(statistical_validation, f, indent=2)

print("\n=== STATISTICAL VALIDATION (Chi-Square / Point-Biserial) ===")
print(json.dumps(statistical_validation, indent=2))

# ---------------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------------
df.to_csv("/home/claude/engagement-retention-analytics/outputs/processed_customer_engagement_data.csv", index=False)

with open("/home/claude/engagement-retention-analytics/outputs/kpi_summary.json", "w") as f:
    json.dump(kpi_summary, f, indent=2)

with open("/home/claude/engagement-retention-analytics/outputs/validation_report.json", "w") as f:
    json.dump(validation_report, f, indent=2, default=str)

engagement_summary.to_csv("/home/claude/engagement-retention-analytics/outputs/engagement_profile_summary.csv", index=False)
product_churn.to_csv("/home/claude/engagement-retention-analytics/outputs/product_count_churn.csv", index=False)
single_vs_multi.to_csv("/home/claude/engagement-retention-analytics/outputs/single_vs_multi_product.csv", index=False)
balance_activity.to_csv("/home/claude/engagement-retention-analytics/outputs/balance_tier_activity_churn.csv", index=False)
mismatch_summary.to_csv("/home/claude/engagement-retention-analytics/outputs/salary_balance_mismatch.csv", index=False)
sticky_summary.to_csv("/home/claude/engagement-retention-analytics/outputs/sticky_vs_at_risk.csv", index=False)
rsi_bucket_churn.to_csv("/home/claude/engagement-retention-analytics/outputs/rsi_bucket_churn.csv", index=False)

print("=== VALIDATION REPORT ===")
print(json.dumps(validation_report, indent=2, default=str))
print("\n=== ENGAGEMENT PROFILE SUMMARY ===")
print(engagement_summary.to_string(index=False))
print("\n=== PRODUCT COUNT vs CHURN ===")
print(product_churn.to_string(index=False))
print("\n=== SINGLE vs MULTI PRODUCT ===")
print(single_vs_multi.to_string(index=False))
print("\n=== BALANCE TIER x ACTIVITY CHURN ===")
print(balance_activity.to_string(index=False))
print("\n=== SALARY-BALANCE MISMATCH ===")
print(mismatch_summary.to_string(index=False))
print("\n=== STICKY vs AT-RISK ===")
print(sticky_summary.to_string(index=False))
print("\n=== RSI BUCKET CHURN (Engagement Threshold) ===")
print(rsi_bucket_churn.to_string(index=False))
print("\n=== KPI SUMMARY ===")
print(json.dumps(kpi_summary, indent=2))

# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
palette = {"0": "#2E86AB", "1": "#E63946"}

# Chart 1: Engagement profile vs churn rate
plt.figure(figsize=(8, 5))
order = engagement_summary.sort_values("ChurnRatePct", ascending=False)
sns.barplot(data=order, x="EngagementProfile", y="ChurnRatePct", color="#E63946")
plt.title("Churn Rate by Engagement Profile", fontsize=13, fontweight="bold")
plt.ylabel("Churn Rate (%)")
plt.xlabel("")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_engagement_profile_churn.png", dpi=150)
plt.close()

# Chart 2: Product count vs churn
plt.figure(figsize=(7, 5))
sns.barplot(data=product_churn, x="NumOfProducts", y="ChurnRatePct", color="#2E86AB")
plt.title("Churn Rate by Number of Products", fontsize=13, fontweight="bold")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Number of Products")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_product_count_churn.png", dpi=150)
plt.close()

# Chart 3: Balance tier x activity churn heatmap
pivot = balance_activity.pivot(index="BalanceTier", columns="IsActiveMember", values="ChurnRatePct")
pivot.columns = ["Inactive", "Active"]
plt.figure(figsize=(6, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "Churn Rate (%)"})
plt.title("Churn Rate: Balance Tier x Activity Status", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_balance_activity_heatmap.png", dpi=150)
plt.close()

# Chart 4: RSI bucket vs churn (engagement threshold curve)
plt.figure(figsize=(7, 5))
sns.lineplot(data=rsi_bucket_churn, x="RSIBucket", y="ChurnRatePct", marker="o",
             color="#E63946", linewidth=2.5)
plt.title("Retention Strength Index vs Churn Rate", fontsize=13, fontweight="bold")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Relationship Strength Index Bucket")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_rsi_churn_curve.png", dpi=150)
plt.close()

# Chart 5: Sticky vs At-Risk churn comparison
plt.figure(figsize=(6, 5))
sns.barplot(data=sticky_summary, x="StickyCustomer", y="ChurnRatePct",
            hue="StickyCustomer", palette=["#2E86AB", "#E63946"], legend=False)
plt.title("Churn Rate: Sticky vs At-Risk Customers", fontsize=13, fontweight="bold")
plt.ylabel("Churn Rate (%)")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_sticky_vs_at_risk.png", dpi=150)
plt.close()

# Chart 6: Geography x engagement profile distribution
geo_engagement = df.groupby(["Geography", "EngagementProfile"]).size().reset_index(name="Count")
geo_pivot = geo_engagement.pivot(index="Geography", columns="EngagementProfile", values="Count").fillna(0)
geo_pivot.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="viridis")
plt.title("Engagement Profile Distribution by Geography", fontsize=13, fontweight="bold")
plt.ylabel("Number of Customers")
plt.xlabel("")
plt.legend(title="Engagement Profile", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_geography_engagement_distribution.png", dpi=150)
plt.close()

print("\nAll charts saved to:", CHART_DIR)
