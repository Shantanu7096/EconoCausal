from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(
    title="EconoCausal Dynamic Pricing REST API",
    description="Enterprise API for Causal ITE scoring and Prescriptive Budget Allocation",
    version="1.0.0"
)

class CustomerProfile(BaseModel):
    recency: int
    history: float
    browsing_mins: float
    cart_abandon_val: float = 0.0
    email_engagement: float = 0.5

class BatchOptimizeRequest(BaseModel):
    budget: float = 5000.0
    cost_tier_1: float = 10.0
    cost_tier_2: float = 20.0

@app.get("/")
def health_check():
    return {"status": "active", "service": "EconoCausal REST API", "version": "1.0.0"}

@app.post("/predict_ite")
def predict_single_ite(profile: CustomerProfile):
    # Dynamic CATE evaluation
    ite = 5.0 + 0.6 * profile.browsing_mins + 0.15 * profile.cart_abandon_val - 1.2 * profile.recency + (profile.history * 0.01)
    ite = round(max(-5.0, ite), 2)
    
    if ite >= 14.0:
        segment = "Best Target Customer"
        recommendation = "Award $20 Incentive"
    elif ite >= 6.0:
        segment = "Average Target Customer"
        recommendation = "Award $10 Discount"
    elif profile.history > 400:
        segment = "Always-Buy Customer"
        recommendation = "Holdout (Preserve Margin)"
    else:
        segment = "Non-Buyer"
        recommendation = "Holdout"
        
    return {
        "predicted_ite": ite,
        "causal_segment": segment,
        "prescribed_action": recommendation
    }