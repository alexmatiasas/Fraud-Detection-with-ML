import os
import sys

# ───────────────────────────────
# Setup project path
# ───────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# ───────────────────────────────
# Imports
# ───────────────────────────────
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import create_model, Field
from predictor import predict_fraud, load_model
from typing import Dict, Any

# ───────────────────────────────
# Generate TransactionInput model
# ───────────────────────────────
csv_path = os.path.join(project_root, "data", "processed", "train_final_ready.csv")
df_sample = pd.read_csv(csv_path, nrows=1).drop(columns=["isFraud"])

# Create example from first row
example_input = df_sample.iloc[0].to_dict()

# Map pandas dtypes to Pydantic fields with examples
dtype_map = {
    "int64": int,
    "float64": float,
    "object": str,
    "bool": bool,
}
fields = {
    col: (dtype_map.get(str(dtype), str), Field(..., example=example_input[col]))
    for col, dtype in df_sample.dtypes.items()
}
TransactionInput = create_model("TransactionInput", **fields)

# ───────────────────────────────
# FastAPI app
# ───────────────────────────────
app = FastAPI(
    title="Fraud Detection API", 
    description="Predict fraud probability based on transaction input."
)

# Load model once
model = load_model()

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: TransactionInput):
    try:
        df = pd.DataFrame([transaction.dict()])
        result = predict_fraud(df, model=model)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))