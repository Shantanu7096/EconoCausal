import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
import os

def solve_optimal_budget_allocation(df, budget=8000, cost_tier_1=10.0, cost_tier_2=20.0):
    """
    Formulates a binary integer programming problem:
    Maximize: Sum(ITE_1 * x_i1 + ITE_2 * x_i2)
    Subject to:
      1. Budget: Sum(cost_tier_1 * x_i1 + cost_tier_2 * x_i2) <= budget
      2. Mutual Exclusivity: x_i1 + x_i2 <= 1 (each customer receives at most one discount)
      3. Binary decision variables: x_i1, x_i2 in {0, 1}
    """
    n = len(df)
    
    # Calculate synthetic multi-tier uplift if not already present
    ite_10 = df['predicted_ite'].values
    ite_20 = np.maximum(ite_10 * 1.35, ite_10 + 3.0)  # Diminishing marginal returns for $20
    
    # Decision vector: [x_0_tier1, x_0_tier2, x_1_tier1, x_1_tier2, ..., x_n_tier1, x_n_tier2]
    # Length: 2 * n
    c = np.empty(2 * n)
    c[0::2] = -ite_10  # SciPy minimizes, so negate coefficients to maximize revenue
    c[1::2] = -ite_20
    
    # 1. Budget Constraint
    # cost_tier_1 * x_i1 + cost_tier_2 * x_i2 <= budget
    budget_row = np.empty(2 * n)
    budget_row[0::2] = cost_tier_1
    budget_row[1::2] = cost_tier_2
    
    # 2. Mutually exclusive constraints: x_i1 + x_i2 <= 1 for each customer
    rows = [budget_row]
    lhs = [0.0]
    rhs = [float(budget)]
    
    for i in range(n):
        exclusivity_row = np.zeros(2 * n)
        exclusivity_row[2 * i] = 1.0
        exclusivity_row[2 * i + 1] = 1.0
        rows.append(exclusivity_row)
        lhs.append(0.0)
        rhs.append(1.0)
        
    A = np.vstack(rows)
    constraints = LinearConstraint(A, lhs, rhs)
    
    # Variable bounds: {0, 1}
    bounds = Bounds(0, 1)
    integrality = np.ones(2 * n)  # 1 indicates binary integer variable
    
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    
    optimized_df = df.copy()
    if res.success:
        decisions = np.round(res.x).astype(int)
        d_10 = decisions[0::2]
        d_20 = decisions[1::2]
        
        prescriptions = []
        discount_amounts = []
        expected_lifts = []
        
        for i in range(n):
            if d_20[i] == 1:
                prescriptions.append("Award $20 Incentive")
                discount_amounts.append(cost_tier_2)
                expected_lifts.append(ite_20[i])
            elif d_10[i] == 1:
                prescriptions.append("Award $10 Discount")
                discount_amounts.append(cost_tier_1)
                expected_lifts.append(ite_10[i])
            else:
                prescriptions.append("Holdout (No Promo)")
                discount_amounts.append(0.0)
                expected_lifts.append(0.0)
                
        optimized_df['prescribed_action'] = prescriptions
        optimized_df['prescribed_discount'] = discount_amounts
        optimized_df['prescribed_lift'] = expected_lifts
    else:
        # Fallback greedy ranking
        optimized_df['prescribed_action'] = 'Holdout (No Promo)'
        optimized_df['prescribed_discount'] = 0.0
        optimized_df['prescribed_lift'] = 0.0
        top_k = int(budget // cost_tier_1)
        top_idx = optimized_df.nlargest(top_k, 'predicted_ite').index
        optimized_df.loc[top_idx, 'prescribed_action'] = f'Award ${cost_tier_1} Discount'
        optimized_df.loc[top_idx, 'prescribed_discount'] = cost_tier_1
        optimized_df.loc[top_idx, 'prescribed_lift'] = optimized_df.loc[top_idx, 'predicted_ite']
        
    return optimized_df