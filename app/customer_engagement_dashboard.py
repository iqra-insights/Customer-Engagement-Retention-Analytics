"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Streamlit Dashboard
----------------------------------------------------------------------------
Run with: streamlit run app/customer_engagement_dashboard.py

Brand palette (matches the companion Research Paper / Presentation deck):
  Navy      #1E2761   -- primary brand color
  Navy Mid  #2A3578   -- secondary panels
  Ice Blue  #CADCFC   -- light accents / highlights
  Accent    #E63946   -- churn / risk / attention
  Slate     #6E7B9B   -- muted text
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# BRAND PALETTE
# ---------------------------------------------------------------------------
NAVY = "#1E2761"
NAVY_MID = "#2A3578"
ICE = "#CADCFC"
ICE_LIGHT = "#F2F4FA"
ACCENT = "#E63946"
SLATE = "#6E7B9B"
WHITE = "#FFFFFF"

SEQUENTIAL_SCALE = [ICE_LIGHT, ICE, "#7B93D6", NAVY_MID, NAVY]
DIVERGING_SCALE = [NAVY, "#5B6BAE", ICE_LIGHT, "#F4A6AC", ACCENT]
CATEGORY_SCALE = [NAVY, ACCENT, "#4F7CAC", "#F1A208"]

st.set_page_config(
    page_title="Customer Engagement & Retention Analytics",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# BRAND CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{ background-color: {WHITE}; }}

    h1, h2, h3 {{ color: {NAVY}; font-family: 'Georgia', serif; }}

    [data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
    [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stMultiSelect label {{
        color: {ICE} !important; font-weight: 600;
    }}

    div[data-testid="stMetric"] {{
        background-color: {ICE_LIGHT};
        border-left: 5px solid {ACCENT};
        border-radius: 8px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {SLATE} !important; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY} !important; font-weight: 700; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {ICE_LIGHT};
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: {NAVY};
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {NAVY} !important;
        color: {WHITE} !important;
    }}

    .brand-header {{
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_MID} 100%);
        padding: 28px 32px;
        border-radius: 12px;
        margin-bottom: 18px;
    }}
    .brand-header h1 {{ color: {WHITE} !important; margin: 0; font-size: 2rem; }}
    .brand-header p {{ color: {ICE} !important; margin: 4px 0 0 0; font-size: 1rem; }}

    .insight-box {{
        background-color: #FDECEC;
        border-left: 5px solid {ACCENT};
        border-radius: 8px;
        padding: 14px 18px;
        color: #333333;
        font-size: 0.95rem;
    }}
    .insight-box-navy {{
        background-color: {ICE_LIGHT};
        border-left: 5px solid {NAVY};
        border-radius: 8px;
        padding: 14px 18px;
        color: #333333;
        font-size: 0.95rem;
    }}
    .footer-note {{
        text-align: center;
        color: {SLATE};
        font-size: 0.85rem;
        padding-top: 12px;
    }}
    div[data-testid="stExpander"] {{
        border: 1px solid {ICE};
        border-radius: 8px;
    }}
    .stDownloadButton button {{
        background-color: {NAVY} !important;
        color: {WHITE} !important;
        border-radius: 6px !important;
        border: none !important;
    }}
    .stDownloadButton button:hover {{
        background-color: {NAVY_MID} !important;
    }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    font=dict(family="Georgia, serif", color="#333333"),
    title_font=dict(family="Georgia, serif", color=NAVY, size=18),
    plot_bgcolor=WHITE,
    paper_bgcolor=WHITE,
    legend=dict(bgcolor=WHITE),
)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "customer_engagement_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
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
        return "Other"

    df["EngagementProfile"] = df.apply(classify_engagement, axis=1)
    df["ProductTier"] = np.where(df["NumOfProducts"] == 1, "Single-Product", "Multi-Product")
    df["BalanceTier"] = pd.qcut(df["Balance"].rank(method="first"), 4,
                                 labels=["Low", "Medium", "High", "Premium"])

    def rsi(row):
        score = 0
        score += 2 if row["IsActiveMember"] == 1 else 0
        score += min(row["NumOfProducts"], 4) * 1.5
        score += 1 if row["HasCrCard"] == 1 else 0
        score += 1 if row["Tenure"] >= 5 else 0
        return round(score, 2)

    df["RelationshipStrengthIndex"] = df.apply(rsi, axis=1)
    df["StickyCustomer"] = np.where(df["RelationshipStrengthIndex"] >= 6, "Sticky", "At-Risk")
    return df


df = load_data()

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🏦 Filters")
st.sidebar.markdown("---")

geo_filter = st.sidebar.multiselect(
    "Geography", options=sorted(df["Geography"].unique()), default=sorted(df["Geography"].unique())
)
engagement_filter = st.sidebar.multiselect(
    "Engagement Profile", options=sorted(df["EngagementProfile"].unique()),
    default=sorted(df["EngagementProfile"].unique())
)
product_range = st.sidebar.slider(
    "Number of Products", int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max()),
    (int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max()))
)
balance_range = st.sidebar.slider(
    "Balance Threshold", float(df["Balance"].min()), float(df["Balance"].max()),
    (float(df["Balance"].min()), float(df["Balance"].max()))
)
salary_range = st.sidebar.slider(
    "Estimated Salary Threshold", float(df["EstimatedSalary"].min()), float(df["EstimatedSalary"].max()),
    (float(df["EstimatedSalary"].min()), float(df["EstimatedSalary"].max()))
)

st.sidebar.markdown("---")
st.sidebar.caption("Customer Engagement & Product Utilization Analytics for Retention Strategy — Iqra")

with st.sidebar.expander("ℹ️ About this dashboard"):
    st.markdown(
        "Built on 10,000 European bank customer records. "
        "Full methodology, statistical validation (chi-square tests, "
        "confidence intervals), and a supplementary predictive model are "
        "documented in the companion **Research Paper** (`docs/`) and "
        "**Jupyter Notebook** (`notebooks/`)."
    )

filtered = df[
    (df["Geography"].isin(geo_filter)) &
    (df["EngagementProfile"].isin(engagement_filter)) &
    (df["NumOfProducts"].between(*product_range)) &
    (df["Balance"].between(*balance_range)) &
    (df["EstimatedSalary"].between(*salary_range))
]

# ---------------------------------------------------------------------------
# BRAND HEADER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="brand-header">
    <h1>🏦 Customer Engagement &amp; Product Utilization Analytics</h1>
    <p>Retention Strategy Dashboard — European Bank Customer Data</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight-box-navy">
📌 <b>Headline finding:</b> Inactive customers churn at <b>1.88x</b> the rate of active
customers — regardless of balance. Engagement, not wealth, drives retention.
Use the filters on the left to explore every segment behind that number.
</div>
""", unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------------------
# TOP KPIs
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
overall_churn = filtered["Exited"].mean() * 100 if len(filtered) else 0
active_churn = filtered[filtered["IsActiveMember"] == 1]["Exited"].mean() * 100 if len(filtered) else 0
inactive_churn = filtered[filtered["IsActiveMember"] == 0]["Exited"].mean() * 100 if len(filtered) else 0
avg_rsi = filtered["RelationshipStrengthIndex"].mean() if len(filtered) else 0
sticky_pct = (filtered["StickyCustomer"] == "Sticky").mean() * 100 if len(filtered) else 0

k1.metric("Customers", f"{len(filtered):,}")
k2.metric("Overall Churn Rate", f"{overall_churn:.1f}%")
k3.metric("Active Member Churn", f"{active_churn:.1f}%")
k4.metric("Inactive Member Churn", f"{inactive_churn:.1f}%")
k5.metric("Sticky Customers", f"{sticky_pct:.1f}%")

st.write("")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Engagement vs Churn Overview",
    "🧩  Product Utilization Impact",
    "🚨  High-Value Disengaged Detector",
    "🏆  Retention Strength Scoring",
])

# ---------------------------------------------------------------------------
# TAB 1: ENGAGEMENT VS CHURN OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        eng_summary = filtered.groupby("EngagementProfile").agg(
            Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean")
        ).reset_index()
        eng_summary["ChurnRatePct"] = eng_summary["ChurnRate"] * 100
        fig = px.bar(eng_summary.sort_values("ChurnRatePct", ascending=False),
                     x="EngagementProfile", y="ChurnRatePct", color="ChurnRatePct",
                     color_continuous_scale=DIVERGING_SCALE, title="Churn Rate by Engagement Profile")
        fig.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="", **PLOTLY_LAYOUT)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        geo_eng = filtered.groupby(["Geography", "EngagementProfile"]).size().reset_index(name="Count")
        fig2 = px.bar(geo_eng, x="Geography", y="Count", color="EngagementProfile",
                      title="Engagement Profile Distribution by Geography", barmode="stack",
                      color_discrete_sequence=CATEGORY_SCALE)
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Engagement Profile Summary")
    st.dataframe(eng_summary.style.format({"ChurnRatePct": "{:.2f}%", "ChurnRate": "{:.3f}"})
                 .background_gradient(subset=["ChurnRatePct"], cmap="Reds"),
                 use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Insight:</b> Active Engaged customers churn at under 10%, while Inactive
    High-Balance customers churn at over 30% — a 3x gap despite similar or higher
    balances. This is the paper's central finding: engagement predicts loyalty far
    better than wealth (χ² = 325.46, p &lt; 0.001).
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Engagement Summary (CSV)",
        eng_summary.to_csv(index=False).encode("utf-8"),
        "engagement_profile_summary.csv",
        "text/csv",
    )

# ---------------------------------------------------------------------------
# TAB 2: PRODUCT UTILIZATION IMPACT
# ---------------------------------------------------------------------------
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        prod_churn = filtered.groupby("NumOfProducts").agg(
            Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean")
        ).reset_index()
        prod_churn["ChurnRatePct"] = prod_churn["ChurnRate"] * 100
        fig3 = px.bar(prod_churn, x="NumOfProducts", y="ChurnRatePct",
                      title="Churn Rate by Number of Products", color="ChurnRatePct",
                      color_continuous_scale=SEQUENTIAL_SCALE)
        fig3.update_layout(**PLOTLY_LAYOUT)
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        tier_churn = filtered.groupby("ProductTier").agg(
            Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean")
        ).reset_index()
        tier_churn["ChurnRatePct"] = tier_churn["ChurnRate"] * 100
        fig4 = px.bar(tier_churn, x="ProductTier", y="ChurnRatePct",
                      title="Single-Product vs Multi-Product Churn", color="ProductTier",
                      color_discrete_sequence=[NAVY, ACCENT])
        fig4.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Insight:</b> Customers holding exactly 2 products show the lowest churn.
    Customers with 3-4 products show a sharp churn spike — likely a segment sold
    extra products without genuine engagement (cross-sell without retention design).
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Product Utilization Data (CSV)",
        prod_churn.to_csv(index=False).encode("utf-8"),
        "product_count_churn.csv",
        "text/csv",
    )

# ---------------------------------------------------------------------------
# TAB 3: HIGH-VALUE DISENGAGED DETECTOR
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("##### At-Risk Premium Customer Detector")
    premium_threshold = st.slider("Premium balance threshold (percentile)", 50, 95, 75, key="premium_pct")
    threshold_value = filtered["Balance"].quantile(premium_threshold / 100) if len(filtered) else 0

    at_risk = filtered[(filtered["Balance"] >= threshold_value) & (filtered["IsActiveMember"] == 0)]
    c1, c2, c3 = st.columns(3)
    c1.metric("At-Risk High-Value Customers", f"{len(at_risk):,}")
    c2.metric("Their Churn Rate", f"{at_risk['Exited'].mean()*100:.1f}%" if len(at_risk) else "N/A")
    c3.metric("Total Balance at Risk", f"€{at_risk['Balance'].sum():,.0f}" if len(at_risk) else "€0")

    fig5 = px.scatter(filtered, x="Balance", y="EstimatedSalary", color="IsActiveMember",
                      symbol="Exited", title="Balance vs Salary (colored by activity, shaped by churn)",
                      labels={"IsActiveMember": "Active Member"},
                      color_continuous_scale=[ACCENT, NAVY])
    fig5.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    st.dataframe(
        at_risk[["CustomerId", "Surname", "Geography", "Age", "Balance", "EstimatedSalary",
                 "NumOfProducts", "Tenure"]].sort_values("Balance", ascending=False).head(50),
        use_container_width=True
    )

    st.markdown("""
    <div class="insight-box">
    🚨 <b>Insight:</b> These are customers with real financial weight who have gone
    quiet. 1,247 such customers exist bank-wide (30.5% churn) — a concentrated,
    high-leverage segment for a proactive retention program, prioritized over
    broad-based outreach.
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download At-Risk Customer List (CSV)",
        at_risk.to_csv(index=False).encode("utf-8"),
        "at_risk_premium_customers.csv",
        "text/csv",
        disabled=len(at_risk) == 0,
    )

# ---------------------------------------------------------------------------
# TAB 4: RETENTION STRENGTH SCORING
# ---------------------------------------------------------------------------
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        rsi_bins = pd.cut(filtered["RelationshipStrengthIndex"], bins=[-1, 2, 4, 6, 8, 100],
                          labels=["0-2", "2-4", "4-6", "6-8", "8+"])
        rsi_df = filtered.groupby(rsi_bins, observed=False).agg(
            Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean")
        ).reset_index()
        rsi_df["ChurnRatePct"] = rsi_df["ChurnRate"] * 100
        rsi_df.columns = ["RSIBucket", "Customers", "ChurnRate", "ChurnRatePct"]
        fig6 = px.line(rsi_df, x="RSIBucket", y="ChurnRatePct", markers=True,
                       title="Relationship Strength Index vs Churn Rate",
                       color_discrete_sequence=[ACCENT])
        fig6.update_traces(line=dict(width=3), marker=dict(size=9, color=NAVY))
        fig6.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig6, use_container_width=True)
    with col2:
        sticky_df = filtered.groupby("StickyCustomer").agg(
            Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean")
        ).reset_index()
        sticky_df["ChurnRatePct"] = sticky_df["ChurnRate"] * 100
        fig7 = px.pie(sticky_df, names="StickyCustomer", values="Customers",
                      title="Sticky vs At-Risk Customer Distribution", hole=0.45,
                      color="StickyCustomer",
                      color_discrete_map={"Sticky": NAVY, "At-Risk": ACCENT})
        fig7.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <b>Relationship Strength Index (RSI)</b> = 2×(Active Member) + 1.5×(Products, capped at 4)
    + 1×(Has Credit Card) + 1×(Tenure ≥ 5 years). Customers scoring ≥6 are classified as
    <b>Sticky</b>.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""
<p class="footer-note">
🏦 Customer Engagement &amp; Product Utilization Analytics for Retention Strategy · Built by Iqra · Unified Mentor Internship<br>
Full statistical validation, predictive model, and business recommendations in the companion Research Paper &amp; Jupyter Notebook.
</p>
""", unsafe_allow_html=True)
