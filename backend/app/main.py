from fastapi import FastAPI, HTTPException
from .schemas import TransactionInput, PredictionOutput
from .services import fraud_service

app = FastAPI(
    title="FinGuard AI API",
    description="API for Real-time Financial Fraud Detection using ResNet-PolyLoss Model",
)

@app.get("/")
def read_root():
    """
    Health check endpoint to verify if the API is running.
    """
    return {"status": "active", "message": "FinGuard AI Fraud Detection System is Online "}


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    """
    Analyzes a financial transaction and predicts the probability of fraud.
    
    Args:
        transaction (TransactionInput): A list of 30 numerical features [Time, V1...V28, Amount].
        
    Returns:
        PredictionOutput: Contains fraud status (bool), probability (float), and risk level (str).
    """
    try:
        result = fraud_service.predict(transaction.features)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))