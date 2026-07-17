from fastapi import FastAPI
import joblib

app = FastAPI()

# Load Model
model = joblib.load("churn_model.joblib")


@app.get("/")
def home():
    return {
        "message":"Customer Churn Prediction API Running"
    }


@app.get("/predict")
def predict(
    Tenure:int,
    MonthlyCharges:float,
    TotalCharges:float,
    Contract:int
):

    prediction = model.predict([[
        Tenure,
        MonthlyCharges,
        TotalCharges,
        Contract
    ]])

    if prediction[0]==1:
        result="Customer Will Leave"
    else:
        result="Customer Will Stay"

    return {"Prediction":result}
