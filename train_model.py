import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("customer_churn.csv")

# Features
X = data[[
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract"
]]

# Target
y = data["Churn"]

# Train model
model = RandomForestClassifier(random_state=42)

model.fit(X, y)

# Save model
joblib.dump(model, "churn_model.joblib")

print("Model Trained Successfully")
print("Model Saved Successfully")

