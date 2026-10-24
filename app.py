import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. Page Configuration
st.set_page_config(
    page_title="EconoCausal | AI Dynamic Pricing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Universal Adaptive Theme Styling
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.85;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #0284c7;
    }

    /* Badge */
    .pill-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        background-color: rgba(2, 132, 199, 0.15);
        color: #0284c7;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Scored Dataset
data_file = "data/retail_scored_ite.csv"
if not os.path.exists(data_file):
    st.error("Data file `data/retail_scored_ite.csv` not found. Please run `python src/causal_engine_v2.py` in your terminal first.")
    st.stop()

df = pd.read_csv(data_file)

# Normalize labels to simple, understandable terms
label_map = {
    'High-Value Persuadables': 'Best Target Customers',
    'Moderate Persuadables': 'Average Target Customers',
    'Sure Things (No Discount Needed)': 'Always-Buy Customers',
    'Lost Causes': 'Non-Buyers',
    'Sleeping Dogs / Waste': 'Do-Not-Disturb Customers'
}
df['causal_segment'] = df['causal_segment'].replace(label_map)

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("## ⚙️ Campaign Controls")
    budget = st.slider("Campaign Budget ($)", min_value=1000, max_value=30000, value=8000, step=1000)
    coupon_val = st.number_input("Discount Per Customer ($)", min_value=5, max_value=50, value=10, step=5)
    num_targets = budget // coupon_val
    
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    current_view = st.radio(
        "Select Dashboard Section:",
        ["📊 Customer Groups & Share", "📈 Qini & Profit Uplift", "🎯 Best Discount Plan", "🛡️ Model Audit & Refutations"]
    )
    
    st.markdown("---")
    st.markdown("### 🔍 Model Architecture")
    st.markdown("""
    - **Engine:** Double ML (EconML)[cite: 21]
    - **Nuisance Learner:** LightGBM[cite: 21]
    - **Identification:** Backdoor DAG[cite: 21]
    - **Refuter Tests:** Placebo & Common Cause[cite: 21]
    """)

# 5. Header Bar
st.markdown('<div class="pill-badge">ENTERPRISE CAUSAL AI</div>', unsafe_allow_html=True)
st.title("EconoCausal: Dynamic Pricing Engine")
st.caption("Double Machine Learning for Strategic Discount & Promotional Allocation")

# 6. Global Metric Cards
best_targets_count = (df['causal_segment'] == 'Best Target Customers').sum()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Audience Size", f"{len(df):,} users")
col2.metric("Extra Sales / Discount", f"${df['predicted_ite'].mean():.2f}")
col3.metric("Coupons Available", f"{num_targets:,} units")
col4.metric("Best Target Customers", f"{best_targets_count:,}")

st.write("")
st.markdown("---")

# 7. Dynamic Section Content Based on Navigation
# ---------------- VIEW 1: CUSTOMER GROUPS & SHARE ----------------
if current_view == "📊 Customer Groups & Share":
    st.subheader("Customer Responsiveness Breakdown")
    
    col_chart1, col_chart2 = st.columns([1.6, 1])
    
    with col_chart1:
        fig_scatter = px.scatter(
            df.sample(min(800, len(df)), random_state=42),
            x="browsing_mins",
            y="predicted_ite",
            color="causal_segment",
            size="history",
            hover_data=['recency'],
            color_discrete_map={
                "Best Target Customers": "#0284c7",
                "Average Target Customers": "#6366f1",
                "Non-Buyers": "#ec4899",
                "Do-Not-Disturb Customers": "#ef4444"
            },
            title="Extra Sales Potential vs. Web Browsing Time"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Browsing Activity (Minutes)", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(title="Incremental Extra Sales ($)", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
            legend_title_text="Customer Group"
        )
        st.plotly_chart(fig_scatter, width='stretch')
        
    with col_chart2:
        seg_counts = df['causal_segment'].value_counts().reset_index()
        fig_pie = px.pie(
            seg_counts,
            names='causal_segment',
            values='count',
            hole=0.45,
            color='causal_segment',
            color_discrete_map={
                "Best Target Customers": "#0284c7",
                "Average Target Customers": "#6366f1",
                "Non-Buyers": "#ec4899",
                "Do-Not-Disturb Customers": "#ef4444"
            },
            title="Customer Base Share"
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_pie, width='stretch')

    st.markdown("#### 📋 Customer Group Playbook")
    st.markdown("""
    | Customer Group | Behavior | Strategic Action |
    | :--- | :--- | :--- |
    | **Best Target Customers** | Highly responsive; spend significantly more when given discounts. | **Priority Target:** Send promotional coupons first. |
    | **Average Target Customers** | Mildly responsive; spend slightly more when given discounts. | **Secondary Target:** Allocate only if budget remains. |
    | **Always-Buy Customers** | High spending customers who buy regardless of promotions. | **Do Not Discount:** Protect baseline product margin. |
    | **Non-Buyers** | Unresponsive visitors who will not purchase even with discounts. | **Do Not Target:** Avoid wasting marketing capital. |
    | **Do-Not-Disturb Customers** | Annoyed by promotional outreach; high unsubscribe risk. | **Exclude:** Omit completely from marketing lists. |
    """)

# ---------------- VIEW 2: QINI & PROFIT UPLIFT ----------------
elif current_view == "📈 Qini & Profit Uplift":
    st.subheader("Causal AI vs. Random Discounting (Cumulative Revenue)")
    
    df_sorted = df.sort_values(by="predicted_ite", ascending=False).reset_index(drop=True)
    df_sorted['cumulative_treated'] = np.arange(1, len(df_sorted) + 1)
    df_sorted['cumulative_uplift'] = df_sorted['predicted_ite'].cumsum()
    random_uplift = np.linspace(0, df_sorted['cumulative_uplift'].iloc[-1], len(df_sorted))
    
    fig_qini = go.Figure()