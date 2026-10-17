import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="EconoCausal Dashboard", layout="wide")

st.title("EconoCausal: Dynamic Pricing Engine")
st.subheader("Week 1: Observational Data & Causal Problem Setup")

# Sidebar: File Upload & Budget Configuration
st.sidebar.header("Campaign Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Promotion CSV", type=["csv"])
budget_constraint = st.sidebar.number_input("Target Promotional Budget ($)", min_value=1000, value=5000, step=500)

data_path = "data/retail_promotions.csv"
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    st.error("No dataset found. Run `python src/causal_model.py` first.")
    st.stop()

# Key Business Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Audience Size", f"{len(df):,} users")
col2.metric("Treated (Discounted)", f"{int(df['discount'].sum()):,} users")
col3.metric("Baseline Avg Revenue", f"${df[df['discount'] == 0]['revenue'].mean():.2f}")
col4.metric("Treated Avg Revenue", f"${df[df['discount'] == 1]['revenue'].mean():.2f}")

st.markdown("---")

# Visualizations with Plotly
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Confounder vs. Treatment: Browsing Time")
    fig1 = px.histogram(df, x="browsing_time", color="discount", barmode="overlay",
                        title="Distribution of Browsing Time by Discount Status")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("#### Naive Observational Outcome: Revenue by Promo")
    fig2 = px.box(df, x="discount", y="revenue", color="discount",
                title="Raw Revenue Comparison (Biased by Confounders)")
    st.plotly_chart(fig2, use_container_width=True)

st.info("Week 1 Deliverable: Causal Graph formulated. Confounders identified: `user_age`, `browsing_time`, `prior_spend`.")
