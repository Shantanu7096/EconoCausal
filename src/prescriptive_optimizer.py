import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os

def optimize_discount_allocation(df_path="data/retail_scored_ite.csv", total_budget=10000.0, cost_per_discount=10.0):
    """
    Solves the optimal discount assignment policy using Linear Programming (SciPy).
    
    Objective: Maximize sum(ITE_i * x_i) 
    Subject to:
        1. sum(cost_i * x_i) <= total_budget  (Budget Constraint)
        2. 0 <= x_i <= 1                      (Relaxed Binary Decision Variable)
    """
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Missing {df_path}. Run the causal engine first.")
        
    df = pd.read_csv(df_path)
    n_users = len(df)
    
    # Objective coefficients: linprog minimizes c^T * x, so we negate ITE to maximize
    c = -df['predicted_ite'].values
    
    # Inequality constraint: Cost per intervention * x_i <= total_budget
    A_ub = np.full((1, n_users), cost_per_discount)
    b_ub = np.array([total_budget])
    
    # Variable bounds: Each individual can receive between 0 and 1 treatment allocation
    bounds = [(0, 1) for _ in range(n_users)]
    
    print(f"--- Solving Prescriptive Optimization for {n_users:,} users with Budget ${total_budget:,.2f} ---")
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        # Round fractional allocations to binary decisions
        df['optimal_allocation'] = np.round(res.x).astype(int)
        df['allocated_cost'] = df['optimal_allocation'] * cost_per_discount
        df['expected_incremental_revenue'] = df['optimal_allocation'] * df['predicted_ite']
        
        total_spent = df['allocated_cost'].sum()
        total_uplift = df['expected_incremental_revenue'].sum()
        net_profit = total_uplift - total_spent
        roi = (net_profit / total_spent) * 100 if total_spent > 0 else 0
        
        print("\nOptimization Successful:")
        print(f"  • Total Budget Allocated: ${total_spent:,.2f} / ${total_budget:,.2f}")
        print(f"  • Users Targeted: {df['optimal_allocation'].sum():,} / {n_users:,}")
        print(f"  • Expected Incremental Revenue: ${total_uplift:,.2f}")
        print(f"  • Net Return on Investment (ROI): {roi:.2f}%")
        
        output_path = "data/optimized_policy.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Prescriptive policy saved to: {output_path}")
        return df
    else:
        raise RuntimeError(f"Optimization failed: {res.message}")

if __name__ == "__main__":
    optimize_discount_allocation()