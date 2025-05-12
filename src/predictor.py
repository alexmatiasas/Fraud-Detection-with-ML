import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from data_preprocessing import impute_missing_values, encode_categorical, scale_features

# ───────────────────────────────────────────────
# Load trained model
# ───────────────────────────────────────────────
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "final_model_stacking.pkl")
)

def load_model(path: str = MODEL_PATH):
    """Load the trained model from disk."""
    return joblib.load(path)

# ───────────────────────────────────────────────
# Safe preprocessing functions (used only if needed)
# ───────────────────────────────────────────────
def safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numerical
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    valid_num_cols = [col for col in num_cols if df[col].notna().sum() > 0]
    if valid_num_cols:
        imputed = SimpleImputer(strategy="median").fit_transform(df[valid_num_cols])
        df[valid_num_cols] = pd.DataFrame(imputed, columns=valid_num_cols, index=df.index)

    # Categorical
    cat_cols = df.select_dtypes(include=["object"]).columns
    valid_cat_cols = [col for col in cat_cols if df[col].notna().sum() > 0]
    if valid_cat_cols:
        imputed = SimpleImputer(strategy="most_frequent").fit_transform(df[valid_cat_cols])
        df[valid_cat_cols] = pd.DataFrame(imputed, columns=valid_cat_cols, index=df.index)

    return df

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline used during inference (always apply scaling)."""
    df = df.copy()

    # Si aún hay columnas tipo object → imputamos y codificamos
    if not df.select_dtypes(include=["object"]).empty:
        df = safe_impute(df)
        df = encode_categorical(df)

    # Escalado siempre debe aplicarse
    df = scale_features(df, fit=False)

    return df

# ───────────────────────────────────────────────
# Prediction logic
# ───────────────────────────────────────────────
def predict_fraud(df: pd.DataFrame, model=None):
    """Preprocess input and return prediction + fraud probability."""
    if model is None:
        model = load_model()

    X = preprocess_input(df)

    # Verifica si los datos no son todos ceros
    print("📊 Preprocessed sample (first 5 cols):")
    print(X.iloc[0, :5])  # solo algunas columnas

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    return {"prediction": int(pred), "fraud_probability": round(float(proba), 4)}