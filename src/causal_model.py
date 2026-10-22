import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from econml.dml import LinearDML, CausalForestDML
import dowhy
from dowhy import CausalModel
import plotly.express as px
import plotly.graph_objects as go

os.makedirs("data", exist_ok=True)
csv_path = "data/retail_promotions.csv"

# ---------------------------------------------------------
# 1. Advanced Synthetic Retail Data with True Heterogeneity
# ---------------------------------------------------------
def generate_advanced_retail_data(n_samples=4000):
    np.random.seed(42)
    
    # Confounders (W)
    recency = np.random.randint(1, 12, n_samples)          # Months since last purchase
    history = np.random.exponential(150, n_samples)       # Historical total spend ($)
    browsing_mins = np.random.gamma(shape=3, scale=8, size=n_samples) # Active web engagement
    is_mobile = np.random.binomial(1, 0.6, n_samples)     # Device type
    
    # Treatment Assignment Mechanism (T) - Propensity to receive $10 discount
    # Biased: active web browsers with higher past spend are targeted more frequently
    logit_p = -1.5 + 0.04 * browsing_mins + 0.003 * history - 0.15 * recency
    propensity = 1 / (1 + np.exp(-logit_p))
    treatment = np.random.binomial(1, propensity, n_samples)
    
    # Heterogeneous Treatment Effect (True Uplift / CATE)
    # Discounts are far more effective for users with high browsing time and recent visits
    true_ite = 5.0 + 0.8 * browsing_mins - 1.2 * recency + (history * 0.02)
    true_ite = np.maximum(true_ite, -5.0)  # Some customers react neutrally or negatively
    
    # Baseline revenue (without treatment)
    baseline_revenue = 20.0 + 0.10 * history + 0.5 * browsing_mins - 0.8 * recency
    
    # Observed Outcome (Y)
    noise = np.random.normal(0, 4.0, n_samples)
    revenue = baseline_revenue + (true_ite * treatment) + noise
    revenue = np.maximum(0, np.round(revenue, 2))
    
    df = pd.DataFrame({
        'recency': recency,
        'history': np.round(history, 2),
        'browsing_mins': np.round(browsing_mins, 1),
        'is_mobile': is_mobile,
        'treatment': treatment,
        'revenue': revenue,
        'true_ite': np.round(true_ite, 2)
    })
    return df

# ---------------------------------------------------------
# 2. Double Machine Learning (DML) Pipeline (EconML)
# ---------------------------------------------------------
def train_double_ml(df):
    features = ['recency', 'history', 'browsing_mins', 'is_mobile']
    X = df[features]
    T = df['treatment']
    Y = df['revenue']
    
    print("\n--- Training Double Machine Learning (LinearDML) ---")
    # Base estimators for nuisance parameters: E[Y|X] and E[T|X]
    model_y = LGBMRegressor(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
    model_t = LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
    
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        random_state=42
    )
    
    dml.fit(Y, T, X=X)
    
    # Predict Individual Treatment Effect (ITE / CATE)
    predicted_ite = dml.effect(X)
    df['predicted_ite'] = np.round(predicted_ite, 2)
    
    # Segment Audience into Strategic Causal Quadrants
    df['causal_segment'] = 'Non-Buyers'
    df.loc[(df['predicted_ite'] > 12.0), 'causal_segment'] = 'Best Target Customers'
    df.loc[(df['predicted_ite'] > 4.0) & (df['predicted_ite'] <= 12.0), 'causal_segment'] = 'Average Target Customers'
    df.loc[(df['predicted_ite'] <= 4.0) & (df['revenue'] > 50), 'causal_segment'] = 'Always-Buy Customers'
    df.loc[(df['predicted_ite'] <= 0), 'causal_segment'] = 'Do-Not-Disturb Customers'
    
    return dml, df

# ---------------------------------------------------------
# 3. Robustness Refutation Tests (DoWhy)
# ---------------------------------------------------------
def run_refutation_tests(df):
    print("\n--- Running DoWhy Causal Refutation Tests ---")
    causal_graph = """
    digraph {
        recency -> treatment;
        recency -> revenue;
        history -> treatment;
        history -> revenue;
        browsing_mins -> treatment;
        browsing_mins -> revenue;
        is_mobile -> treatment;
        is_mobile -> revenue;
        treatment -> revenue;
    }
    """
    model = CausalModel(
        data=df,
        treatment='treatment',
        outcome='revenue',
        graph=causal_graph.replace("\n", " ")
    )
    
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )
    print(f"Causal Estimate (Average Treatment Effect): {estimate.value:.4f}")
    
    # 1. Placebo Treatment Refuter: Replaces treatment with random noise (Effect should go to 0)
    refute_placebo = model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter")
    print(f"Placebo Treatment Refuter: New Effect = {refute_placebo.new_effect:.4f} (Valid if near 0)")
    
    # 2. Random Common Cause: Adds an unobserved confounder (Effect should not change wildly)
    refute_random = model.refute_estimate(estimand, estimate, method_name="random_common_cause")
    print(f"Random Common Cause Refuter: New Effect = {refute_random.new_effect:.4f}")
    
    return estimate, refute_placebo, refute_random

if __name__ == "__main__":
    df = generate_advanced_retail_data(4000)
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated at {csv_path}")
    
    dml_model, df_scored = train_double_ml(df)
    df_scored.to_csv("data/retail_scored_ite.csv", index=False)
    print("Scored dataset with ITE written to data/retail_scored_ite.csv")
    
    run_refutation_tests(df)