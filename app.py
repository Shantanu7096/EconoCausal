import streamlit as st
import pandas as pd
import os
from src.ui_components import (
    inject_styles,
    render_navbar,
    render_kpis,
    render_qini_view,
    render_audience_view,
    render_whatif_sandbox,
    render_policy_solver,
    render_causal_diagnostics,
    render_database_governance,
    render_prescriptive_allocation_matrix,
    render_drift_monitoring_view
)
from src.database import load_scored_customers

# 1. Page Configuration
st.set_page_config(
    page_title="EconoCausal Enterprise | Pricing Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject CSS
inject_styles()

# 3. Data Ingestion (Database First, CSV Fallback)
try:
    df = load_scored_customers()
    if df.empty:
        raise ValueError()
except Exception:
    data_file = "data/retail_scored_ite.csv"
    if not os.path.exists(data_file):
        st.error("Data not found. Run `python src/causal_model.py` first.")
        st.stop()
    df = pd.read_csv(data_file)

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("### 🎛️ Optimization Controls")
    budget = st.slider("Campaign Budget ($)", 1000, 30000, 8000, step=1000)
    coupon_val = st.number_input("Unit Discount Value ($)", min_value=5, max_value=50, value=10, step=5)
    num_targets = int(budget // coupon_val)
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Telemetry")
    st.markdown("""
    <div style='background:rgba(255,255,255,0.03); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.05); font-size:0.8rem;'>
        <b>Causal Estimator:</b> LinearDML (EconML)<br>
        <b>Base Learner:</b> LightGBM Regressor<br>
        <b>Validation:</b> DoWhy Refuters<br>
        <b>Storage:</b> SQLite Relational DB
    </div>
    """, unsafe_allow_html=True)

# 5. Top Bar & Global Metrics
render_navbar()

best_targets = (df['causal_segment'] == 'Best Target Customers').sum()
render_kpis(
    audience_size=len(df),
    avg_ite=df['predicted_ite'].mean(),
    num_coupons=num_targets,
    best_targets=best_targets
)

# 6. Tab Navigation (Each View Rendered Exactly Once Inside Its Own Tab)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Qini & Profit Uplift",
    "🗺️ Audience Discovery & Cohorts",
    "🧪 Live What-If Sandbox",
    "🎯 Prescriptive Allocation Matrix",
    "🛡️ Causal Integrity & DAG",
    "📡 Data Drift & Telemetry",
    "🏛️ Database & Audit Logs"
])

with tab1:
    render_qini_view(df)

with tab2:
    render_audience_view(df)

with tab3:
    render_whatif_sandbox()

with tab4:
    render_prescriptive_allocation_matrix(df, budget)

with tab5:
    render_causal_diagnostics()

with tab6:
    render_drift_monitoring_view(df)

