import os
import numpy as np
import pandas as pd
from dowhy import CausalModel

# 1. Ensure directory exists
os.makedirs("data", exist_ok=True)
csv_path = "data/retail_promotions.csv"

# 2. Programmatically generate realistic retail promotional data
np.random.seed(42)
n_samples = 4000

# Confounders (W): Customer history and activity
recency = np.random.randint(1, 12, n_samples)          # Months since last visit
history = np.random.exponential(150, n_samples)       # Historical total spend ($)
newbie = np.random.binomial(1, 0.4, n_samples)        # 1 if joined in last 6 months

# Treatment (T): Discount Promo offered (0 = Control, 1 = $10 Discount)
# Confounding: Higher past spenders & recent buyers have higher likelihood of receiving promo
prob = 1 / (1 + np.exp(-(0.003 * history - 0.1 * recency + 0.2 * newbie - 0.5)))
discount_treatment = np.random.binomial(1, prob, n_samples)

# Outcome (Y): Purchase Spend ($)
# True Causal Uplift from discount = +$22.50
spend = (
    15.0
    + 0.08 * history
    - 0.5 * recency
    + (22.5 * discount_treatment)
    + np.random.normal(0, 5, n_samples)
)
spend = np.maximum(0, np.round(spend, 2))

# 3. Save as local dataset
df = pd.DataFrame({
    'recency': recency,
    'history': np.round(history, 2),
    'newbie': newbie,
    'discount_treatment': discount_treatment,
    'spend': spend
})

df.to_csv(csv_path, index=False)
print(f"✓ Dataset generated and saved locally at: {csv_path} ({len(df):,} records)")

# 4. Formulate the Causal DAG using DoWhy
causal_graph = """
digraph {
    recency -> discount_treatment;
    recency -> spend;
    history -> discount_treatment;
    history -> spend;
    newbie -> discount_treatment;
    newbie -> spend;
    discount_treatment -> spend;
}
"""

print("\n--- Initializing DoWhy Causal Model ---")
model = CausalModel(
    data=df,
    treatment='discount_treatment',
    outcome='spend',
    graph=causal_graph.replace("\n", " ")
)

# 5. Identify the Causal Estimand
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\n--- Identified Causal Estimand ---")
print(identified_estimand)