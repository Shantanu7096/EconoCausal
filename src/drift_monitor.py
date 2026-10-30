import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detect_covariate_drift(baseline_df, incoming_df, alpha=0.05):
    """
    Automated statistical drift detection using the two-sample Kolmogorov-Smirnov test.
    Returns status and p-values for key customer behavior distributions.
    """
    monitored_features = ['recency', 'history', 'browsing_mins']
    if 'cart_abandon_val' in baseline_df.columns and 'cart_abandon_val' in incoming_df.columns:
        monitored_features.append('cart_abandon_val')
        
    drift_report = {}
    is_drift_detected = False
    
    for feat in monitored_features:
        stat, p_val = ks_2samp(baseline_df[feat], incoming_df[feat])
        drift_found = bool(p_val < alpha)
        if drift_found:
            is_drift_detected = True
        drift_report[feat] = {
            'ks_stat': round(float(stat), 4),
            'p_value': round(float(p_val), 4),
            'drift_detected': drift_found
        }
        
    return is_drift_detected, drift_report

def generate_natural_language_executive_summary(optimized_df, budget):
    """
    Generates dynamic natural language business intelligence summaries.
    """
    treated = optimized_df[optimized_df['prescribed_discount'] > 0]
    total_spend = treated['prescribed_discount'].sum()
    total_lift = treated['prescribed_lift'].sum()
    net_profit = total_lift - total_spend
    
    # Comparison against standard blanket discounting
    blanket_sample = optimized_df.sample(len(treated), random_state=42)
    blanket_lift = blanket_sample['predicted_ite'].sum()
    blanket_profit = blanket_lift - total_spend
    savings_vs_blanket = max(0.0, net_profit - blanket_profit)
    
    summary = (
        f"This prescriptive dynamic pricing strategy allocates ${total_spend:,.2f} across {len(treated):,} high-yield users, "
        f"generating ${total_lift:,.2f} in incremental revenue. Compared to a standard untargeted blanket campaign, "
        f"EconoCausal captures ${savings_vs_blanket:,.2f} in additional net profit, achieving an estimated {((net_profit)/total_spend)*100:.1f}% Net ROI."
    )
    return summary