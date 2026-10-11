import numpy as np
import pandas as pd
from dowhy import CausalModel
import matplotlib.pyplot as plt

# 1. Generate synthetic observational retail data
np.random.seed(42)
n_samples = 2000

# Confounders
user_income = np.random.normal(50000, 15000, n_samples)
browsing_time = np.random.exponential(12, n_samples)

# Treatment: Discount offered (0 = no discount, 1 = $10 discount)
# Higher browsing time & lower income increase probability of receiving discount
prob_discount = 1 / (1 + np.exp(-(0.05 * browsing_time - 0.00003 * user_income)))
discount = np.random.binomial(1, prob_discount, n_samples)

# Outcome: Purchase amount ($)
# Direct causal uplift from discount is $25, but income & browsing time confound it
revenue = (
    50
    + 0.001 * user_income
    + 1.8 * browsing_time
    + 25.0 * discount
    + np.random.normal(0, 5, n_samples)
)

df = pd.DataFrame({
    'user_income': user_income,
    'browsing_time': browsing_time,
    'discount': discount,
    'revenue': revenue
})

df.to_csv("data/retail_promotions.csv", index=False)
print("Dataset created: data/retail_promotions.csv")

# 2. Define the Causal Directed Acyclic Graph (DAG) using DoWhy
# W: Confounders, T: Treatment, Y: Outcome
causal_graph = """
digraph {
    user_income -> discount;
    user_income -> revenue;
    browsing_time -> discount;
    browsing_time -> revenue;
    discount -> revenue;
}
"""

model = CausalModel(
    data=df,
    treatment='discount',
    outcome='revenue',
    graph=causal_graph.replace("\n", " ")
)

# 3. Identify the Causal Effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\n--- Identified Causal Estimand ---")
print(identified_estimand)

# 4. View and save the graph
model.view_model(file_name="causal_dag")
print("\nDAG saved as causal_dag.png")