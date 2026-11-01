import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from econml.dml import LinearDML
from src.database import save_scored_customers

os.makedirs("data", exist_ok=True)
csv_path = "data/retail_scored_ite.csv"

def generate_and_train():
    print("1. Generating retail observational dataset...")
    np.random.seed(42)
    n_samples = 4000
    
    recency = np.random.randint(1, 12, n_samples)
    history_spend = np.random.exponential(200, n_samples)
    browsing_mins = np.random.gamma(shape=3, scale=8, n_samples)
    
    # Treatment selection (biased): loyal customers get more promotions
    latent_prob = 1 / (1 + np.exp(-(0.003 * history_spend + 0.04 * browsing_mins - 0.2 * recency - 1.0)))
    treatment = (np.random.rand(n_samples) < latent_prob).astype(int)
    
    # True Causal Uplift (Discounts cause the highest sales for users browsing a lot)
    true_ite = 5.0 + 0.7 * browsing_mins - 1.2 * recency
    true_ite = np.round(true_ite, 2)
    
    # Customer Revenue Outcome
    baseline_revenue = 20.0 + 0.08 * history_spend + 0.3 * browsing_mins
    revenue = baseline_revenue + (treatment * true_ite) + np.random.normal(0, 4, n_samples)
    revenue = np.maximum(0, np.round(revenue, 2))
    
    df = pd.DataFrame({
        'customer_id': range(1001, 1001 + n_samples),
        'recency': recency,
        'history': np.round(history_spend, 2),
        'browsing_mins': np.round(browsing_mins, 1),
        'treatment': treatment,
        'revenue': revenue
    })
    
    print("2. Training Double Machine Learning (LinearDML + LightGBM)...")
    X = df[['recency', 'history', 'browsing_mins']]
    Y = df['revenue']
    T = df['treatment']
    
    dml = LinearDML(
        model_y=LGBMRegressor(n_estimators=80, max_depth=3, random_state=42, verbose=-1),
        model_t=LGBMClassifier(n_estimators=80, max_depth=3, random_state=42, verbose=-1),
        discrete_treatment=True,
        random_state=42
    )
    dml.fit(Y, T, X=X)
    
    df['predicted_ite'] = np.round(dml.effect(X), 2)
    
    # Assign simple business segments
    df['causal_segment'] = 'Non-Buyers'
    df.loc[(df['predicted_ite'] > 12.0), 'causal_segment'] = 'Best Target Customers'
    df.loc[(df['predicted_ite'] > 4.0) & (df['predicted_ite'] <= 12.0), 'causal_segment'] = 'Average Target Customers'
    df.loc[(df['predicted_ite'] <= 4.0) & (df['revenue'] > 60), 'causal_segment'] = 'Always-Buy Customers'
    df.loc[(df['predicted_ite'] <= 0), 'causal_segment'] = 'Do-Not-Disturb Customers'
    
    # Save CSV and SQLite
    df.to_csv(csv_path, index=False)
    save_scored_customers(df)
    print(f"3. Done! Scored dataset saved to CSV and SQLite database.")

if __name__ == "__main__":
    generate_and_train()