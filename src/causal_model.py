import os
import numpy as np
import pandas as pd
from dowhy import CausalModel

def generate_mock_data(n_samples=2500):
    """Generates synthetic retail data with confounders, treatment, and outcome."""
    np.random.seed(42)
    
    # Confounders (W)
    user_age = np.random.randint(18, 65, n_samples)
    browsing_time = np.random.exponential(15, n_samples)  # in minutes
    prior_spend = np.random.gamma(shape=2, scale=120, size=n_samples)  # past 6-month spend ($)
    
    # Treatment (T): Discount offered (0 = None, 1 = $10 Coupon)
    # Confounding: More active browsers and past high spenders receive discounts more often
    logit_p = -1.8 + 0.04 * browsing_time + 0.002 * prior_spend - 0.01 * user_age
    prob_discount = 1 / (1 + np.exp(-logit_p))
    discount = np.random.binomial(1, prob_discount, n_samples)
    
    # Outcome (Y): Purchase revenue ($)
    # True causal effect of discount is +$30.00
    true_causal_effect = 30.0
    revenue = (
        35.0 
        + 0.45 * user_age 
        + 1.10 * browsing_time 
        + 0.12 * prior_spend 
        + (true_causal_effect * discount) 
        + np.random.normal(0, 6.0, n_samples)
    )
    
    df = pd.DataFrame({
        'user_age': user_age,
        'browsing_time': np.round(browsing_time, 1),
        'prior_spend': np.round(prior_spend, 2),
        'discount': discount,
        'revenue': np.round(revenue, 2)
    })
    return df

def build_causal_graph(df):
    """Defines the DAG and identifies causal effects using DoWhy."""
    causal_graph = """
    digraph {
        user_age -> discount;
        user_age -> revenue;
        browsing_time -> discount;
        browsing_time -> revenue;
        prior_spend -> discount;
        prior_spend -> revenue;
        discount -> revenue;
    }
    """
    model = CausalModel(
        data=df,
        treatment='discount',
        outcome='revenue',
        graph=causal_graph.replace("\n", " ")
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return model, estimand

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_mock_data()
    df.to_csv("data/retail_promotions.csv", index=False)
    print("✓ Dataset created at data/retail_promotions.csv")
    
    model, estimand = build_causal_graph(df)
    print("\n✓ Causal DAG successfully built and identified:")
    print(estimand)