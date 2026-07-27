import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("telco_dataset.csv")

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
print(f"\nChurn distribution:\n{df['Churn Label'].value_counts()}")

# Replace string versions of missing values with actual NaN
df['Internet Service'] = df['Internet Service'].replace(['NaN', 'nan', 'None', '', ' '], np.nan)

# Fill NaN with 'No' (customers without internet)
df['Internet Service'] = df['Internet Service'].fillna('No')

# Select features 
features = [
    'Gender',
    'Tenure in Months',
    'Monthly Charge',
    'Total Charges',
    'Contract',
    'Internet Service',
    'Online Security',
    'Premium Tech Support',
    'Satisfaction Score'
]

# Prepare data
# Gender: Female=0, Male=1
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Contract: Month-to-Month=0, One Year=1, Two Year=2
contract_map = {
    'Month-to-Month': 0,
    'One Year': 1,
    'Two Year': 2
}
df['Contract'] = df['Contract'].map(contract_map)

# Internet Service: No=0, DSL=1, Fiber Optic=2
internet_map = {
    'No': 0,
    'Yes': 1,      # Generic "Yes" = has internet, type unknown
    'DSL': 2,
    'Fiber Optic': 3}
df['Internet Service'] = df['Internet Service'].map(internet_map)

# Encode Yes/No to 1/0
yes_no_cols = ['Online Security', 'Premium Tech Support']
for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Target: Churn Label (Yes=1, No=0)
df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

# Prepare X and y
X = df[features]
y = df['Churn Label']

# Handle any missing values
print(f"\nMissing values in X:\n{X.isnull().sum()}")

# Drop rows with missing values (if any)
initial_shape = X.shape
X = X.dropna()
y = y[X.index]  # Align y with X after dropping
print(f"Dropped {initial_shape[0] - X.shape[0]} rows with missing values")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"\n Model trained successfully!")
print(f"Accuracy: {accuracy:.2f}")
print(f"\nFeature importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")