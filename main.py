from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn based on telecom data",
    version="1.0.0"
)

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

class CustomerInput(BaseModel):
    gender: int                # 0=Female, 1=Male
    tenure: int                # Tenure in Months
    monthly_charge: float      # Monthly Charge
    total_charges: float       # Total Charges
    contract: int              # 0=Month-to-Month, 1=One Year, 2=Two Year
    internet_service: int      # 0=No, 1=DSL, 2=Fiber Optic
    online_security: int       # 0=No, 1=Yes
    tech_support: int          # Premium Tech Support (0=No, 1=Yes)
    satisfaction_score: int    # Satisfaction Score (1-5)

@app.get("/")
def root():
    return {"message": "Churn Prediction API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    # Convert to DataFrame with correct column order
    input_data = pd.DataFrame([[
        customer.gender,
        customer.tenure,
        customer.monthly_charge,
        customer.total_charges,
        customer.contract,
        customer.internet_service,
        customer.online_security,
        customer.tech_support,
        customer.satisfaction_score
    ]], columns=[
        'Gender',
        'Tenure in Months',
        'Monthly Charge',
        'Total Charges',
        'Contract',
        'Internet Service',
        'Online Security',
        'Premium Tech Support',
        'Satisfaction Score'
    ])
    
    # Scale
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": round(probability, 2),
        "risk_level": "High" if probability > 0.5 else "Low"
    }