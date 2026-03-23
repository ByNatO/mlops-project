from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from datetime import datetime
import os

app = FastAPI()

# Charger le modèle (à adapter selon le déploiement)
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
model = joblib.load(MODEL_PATH)

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float

@app.post("/predict")
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.dict()])
    proba = model.predict_proba(df)[0, 1]
    # Logging
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": os.getenv("MODEL_VERSION", "unknown"),
        "score": proba,
        "features": transaction.dict()
    }
    with open("logs/predictions.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    return {"fraud_probability": proba}

@app.get("/health")
def health():
    return {"status": "ok"}