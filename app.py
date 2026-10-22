import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EconoCausal | Dynamic Pricing & Causal AI", layout="wide")

st.title("EconoCausal: Dynamic Pricing Engine")
st.markdown("Smart Discount Planning & Profit Optimization")

data_file = "data/retail_scored_ite.csv"
if not os.path.exists(data_file):
    st.warning("Running the scoring engine first...")
    from src.causal_model import generate_advanced_retail_data, train_double_ml
    df_raw = generate_advanced_retail_data()
    _, df = train_double_ml(df_raw)
    df.to_csv(data_file, index=False)
else:
    df = pd.read_csv(data_file)

# Sidebar
st.sidebar.header("Campaign Optimization Settings")
budget = st.sidebar.slider("Promotional Budget ($10 per coupon)", min_value=1000, max_value=25000, value=8000, step=1000)
num_targets = budget // 10

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Extra Sales Per Discount", f"${df['predicted_ite'].mean():.2f}")
col3.metric("Discounts to Give", f"{num_targets:,} coupons")
col4.metric("Average Customer Value", f"${df['revenue'].mean():.2f}")
targets = (df['causal_segment'].str.contains('Target Customers')).sum()
col4.metric("Identified Target Customers", f"{targets:,}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Customer Groups", "Performance Comparison", "Budget & Profit Calculator"])

with tab1:
    st.subheader("Audience Segmentation by Causal Uplift (ITE)")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_scatter = px.scatter(
            df.sample(800, random_state=42),
            x="browsing_mins",
            y="predicted_ite",
            color="causal_segment",
            size="history",
            hover_data=['recency'],
            title="Individual Treatment Effect vs. Browsing Activity (Grouped by Segment)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col_b:
        segment_counts = df['causal_segment'].value_counts().reset_index()
        fig_pie = px.pie(segment_counts, names='causal_segment', values='count', title="Strategic Allocation Share")
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Causal Uplift vs. Random Targeting (Qini Curve)")
    df_sorted = df.sort_values(by="predicted_ite", ascending=False).reset_index(drop=True)
    df_sorted['cumulative_treated'] = np.arange(1, len(df_sorted) + 1)
    df_sorted['cumulative_uplift'] = df_sorted['predicted_ite'].cumsum()
    
    # Baseline random rollout
    random_uplift = np.linspace(0, df_sorted['cumulative_uplift'].iloc[-1], len(df_sorted))
    
    fig_qini = go.Figure()
    fig_qini.add_trace(go.Scatter(x=df_sorted['cumulative_treated'], y=df_sorted['cumulative_uplift'], mode='lines', name='Causal Double ML Policy', line=dict(color='#00CC96', width=3)))
    fig_qini.add_trace(go.Scatter(x=df_sorted['cumulative_treated'], y=random_uplift, mode='lines', name='Random Promotion Policy', line=dict(color='#EF553B', dash='dash')))
    fig_qini.update_layout(title="Cumulative Incremental Revenue Curve", xaxis_title="Customers Targeted", yaxis_title="Cumulative Incremental Revenue ($)")
    st.plotly_chart(fig_qini, use_container_width=True)

with tab3:
    st.subheader("Best Discount Distribution Plan")
    df_ranked = df.sort_values(by="predicted_ite", ascending=False).copy()
    df_ranked['Action'] = 'Holdout (No Promo)'
    df_ranked.iloc[:num_targets, df_ranked.columns.get_loc('Action')] = 'Send $10 Discount'
    
    est_incremental_rev = df_ranked.iloc[:num_targets]['predicted_ite'].sum()
    net_roi = ((est_incremental_rev - budget) / budget) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Extra Sales", f"${est_incremental_rev:,.2f}")
    m2.metric("Total Campaign Cost", f"${budget:,.2f}")
    m3.metric("Estimated Profit Return (ROI)", f"{net_roi:.1f}%")
    
    st.dataframe(df_ranked[['recency', 'history', 'browsing_mins', 'predicted_ite', 'causal_segment', 'Action']].head(20))