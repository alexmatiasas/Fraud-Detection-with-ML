import pandas as pd
from sklearn.impute import SimpleImputer
from data_preprocessing import impute_missing_values, encode_categorical, scale_features
import joblib
import os

MODEL_PATH = os.path.join("..", "models", "final_model_stacking.pkl")

def safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Improved imputation: skips columns with all missing values."""
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    cat_cols = [col for col in cat_cols if df[col].notna().sum() > 0]

    # Solo imputar si hay columnas válidas
    if len(num_cols) > 0:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    if len(cat_cols) > 0:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    return df

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline used during training."""
    df = safe_impute(df)
    df = encode_categorical(df)
    df = scale_features(df)
    return df

def load_model(path: str = MODEL_PATH):
    """Load the trained model from disk."""
    return joblib.load(path)

def predict_fraud(df: pd.DataFrame, model=None):
    """Preprocess input and return prediction + fraud probability."""
    if model is None:
        model = load_model()

    X = preprocess_input(df)

    # Validate and align columns
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    proba = model.predict_proba(X)[:, 1][0]
    pred = model.predict(X)[0]
    return {"prediction": int(pred), "fraud_probability": round(proba, 4)}