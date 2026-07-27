# Churn Prediction API

A FastAPI-based service that predicts customer churn using a Random Forest model. Built as part of my transition from Data Analysis to AI Engineering.

## 🚀 Features
- Accepts customer data via JSON
- Returns churn prediction, churn probability and risk level
- Interactive Swagger docs at `/docs`
- Ready for Docker containerization

## 🛠️ Tech Stack
- Python 3.9+
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Uvicorn

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Ajtech27/churn-prediction-api.git
cd churn-prediction-api

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Locally
uvicorn app.main:app --reload 
```

## 📊 API Endpoint
### POST /predict
```bash
{
  "gender": 1,
  "tenure": 12,
  "monthly_charge": 65.0,
  "total_charges": 780.0,
  "contract": 0,
  "internet_service": 1,
  "online_security": 0,
  "tech_support": 1,
  "satisfaction_score": 3
}
```
### Response:
```bash
{
  "churn_prediction": 0,
  "churn_probability": 0.05,
  "risk_level": "Low"
}
```
## 👨‍💻 Author
Joseph Afolayan
[Gmail](josephafolay012@gmail.com) | [LinkedIn](linkedin.com/in/afolayan-joseph)