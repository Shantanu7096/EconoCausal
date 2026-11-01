import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from econml.dml import LinearDML
import dowhy
from dowhy import CausalModel
from src.database import save_scored_customers

os.makedirs("data", exist_ok=True)
csv_raw_path = "data/retail_promotions_raw.csv"
csv_scored_path = "data/retail_scored_ite.csv"

# =====================================================================
# WEEK 1: Observational Dataset with Additional Behavioral Parameters
# =====================================================================
def generate_enhanced_retail_dataset(n_samples=5000):
    np.random.seed(42)
    
    # 1. Customer Confounders (W)
    recency = np.random.randint(1, 12, n_samples)                  # Months since last visit
    history_spend = np.random.exponential(200, n_samples)         # Past lifetime spend ($)
    browsing_mins = np.random.gamma(shape=3, scale=8, size=n_samples)   # Time on store (mins)
    cart_abandon_val = np.random.exponential(40, n_samples)       # Abandoned cart value ($)
    email_engagement = np.random.beta(a=2, b=5, size=n_samples)   # Email open/click rate (0-1)
    
    # 2. Treatment Selection Mechanism (T): 0 = No Promo, 1 = $10 Promo, 2 = $20 Promo
    # Confounded: High spenders & engaged users receive discounts more frequently
    latent_score = (
        0.002 * history_spend 
        + 0.03 * browsing_mins 
        + 0.02 * cart_abandon_val 
        - 0.15 * recency 
        + 1.5 * email_engagement 
        - 1.2
    )
    p_control = 1 / (1 + np.exp(latent_score))
    p_10 = (1 - p_control) * 0.6
    p_20 = (1 - p_control) * 0.4
    
    probs = np.vstack([p_control, p_10, p_20]).T
    probs /= probs.sum(axis=1, keepdims=True)
    
    treatment_tier = np.array([np.random.choice([0, 1, 2], p=p) for p in probs])
    
    # Binary treatment flag for standard DML baseline ($10 or $20 vs Control)
    treatment_binary = (treatment_tier > 0).astype(int)
    
    # 3. Individual Treatment Effect (CATE / ITE Uplift)
    # Discounts are most effective for engaged browsers with high abandoned carts
    true_ite_10 = 8.0 + 0.6 * browsing_mins + 0.15 * cart_abandon_val - 1.2 * recency
    true_ite_20 = 14.0 + 0.9 * browsing_mins + 0.25 * cart_abandon_val - 1.5 * recency
    
    # 4. Customer Revenue Outcome (Y)
    baseline_revenue = 15.0 + 0.08 * history_spend + 0.4 * browsing_mins + 0.1 * cart_abandon_val
    noise = np.random.normal(0, 5, n_samples)
    
    incremental_effect = np.where(treatment_tier == 1, true_ite_10, np.where(treatment_tier == 2, true_ite_20, 0.0))
    revenue = baseline_revenue + incremental_effect + noise
    revenue = np.maximum(0, np.round(revenue, 2))
    
    df = pd.DataFrame({
        'recency': recency,
        'history': np.round(history_spend, 2),
        'browsing_mins': np.round(browsing_mins, 1),
        'cart_abandon_val': np.round(cart_abandon_val, 2),
        'email_engagement': np.round(email_engagement, 3),
        'treatment_tier': treatment_tier,
        'treatment': treatment_binary,
        'revenue': revenue
    })
    return df

# =====================================================================
# WEEK 1: Microsoft DoWhy Causal DAG & Estimand Identification
# =====================================================================
def run_dowhy_identification(df):
    print("\n--- [WEEK 1] Formulating Causal DAG & Estimand Identification ---")
    causal_graph = """
    digraph {
        recency -> treatment;
        recency -> revenue;
        history -> treatment;
        history -> revenue;
        browsing_mins -> treatment;
        browsing_mins -> revenue;
        cart_abandon_val -> treatment;
        cart_abandon_val -> revenue;
        email_engagement -> treatment;
        email_engagement -> revenue;
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
    print("Identified Backdoor Estimand successfully.")
    return model, estimand

# =====================================================================
# WEEK 2: Double Machine Learning (LinearDML via EconML & LightGBM)
# =====================================================================
def train_double_ml(df):
    print("\n--- [WEEK 2] Training EconML Double Machine Learning Pipeline ---")
    features = ['recency', 'history', 'browsing_mins', 'cart_abandon_val', 'email_engagement']
    X = df[features]
    T = df['treatment']
    Y = df['revenue']
    
    # Nuisance models
    model_y = LGBMRegressor(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
    model_t = LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)
    
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        random_state=42
    )
    dml.fit(Y, T, X=X)
    
    # Calculate ITE scores
    df['predicted_ite'] = np.round(dml.effect(X), 2)
    
    # Customer Segmentation
    df['causal_segment'] = 'Non-Buyers'
    df.loc[(df['predicted_ite'] > 14.0), 'causal_segment'] = 'Best Target Customers'
    df.loc[(df['predicted_ite'] > 5.0) & (df['predicted_ite'] <= 14.0), 'causal_segment'] = 'Average Target Customers'
    df.loc[(df['predicted_ite'] <= 5.0) & (df['revenue'] > 60), 'causal_segment'] = 'Always-Buy Customers'
    df.loc[(df['predicted_ite'] <= 0), 'causal_segment'] = 'Do-Not-Disturb Customers'
    
    return dml, df

# =====================================================================
# MID-PROJECT REVIEW: Causal Refutation Tests
# =====================================================================
def run_refutation_audit(model, estimand):
    print("\n--- [MID REVIEW] Running DoWhy Causal Refutation Tests ---")
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    print(f"Causal Baseline Estimate: ${estimate.value:.2f}")
    
    # 1. Placebo Refuter: Should converge to ~0
    placebo_refute = model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter")
    print(f"Placebo Test: New Effect = ${placebo_refute.new_effect:.3f} (Passed if near 0)")
    
    # 2. Random Common Cause: Should not change drastically
    random_refute = model.refute_estimate(estimand, estimate, method_name="random_common_cause")
    print(f"Random Common Cause Test: New Effect = ${random_refute.new_effect:.3f} (Passed if stable)")

if __name__ == "__main__":
    df_raw = generate_enhanced_retail_dataset(5000)
    df_raw.to_csv(csv_raw_path, index=False)
    print(f"Saved raw observational dataset to: {csv_raw_path}")
    
    model, estimand = run_dowhy_identification(df_raw)
    _, df_scored = train_double_ml(df_raw)
    
    df_scored.to_csv(csv_scored_path, index=False)
    print(f"Saved ITE-scored dataset to: {csv_scored_path}")
    
    # Save to SQLite Database
    save_scored_customers(df_scored)
    print("✓ Synced scored records with SQLite Database: data/econocausal.db")
    
    run_refutation_audit(model, estimand)