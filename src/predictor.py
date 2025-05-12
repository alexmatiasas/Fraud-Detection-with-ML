import pandas as pd
from sklearn.impute import SimpleImputer
from data_preprocessing import impute_missing_values, encode_categorical, scale_features
import joblib
import os

# Obtain the absolute path to the model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "final_model_stacking.pkl"
)
MODEL_PATH = os.path.abspath(MODEL_PATH)

def safe_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Robust imputation: skip blocks where there are no valid columns."""
    df = df.copy()

    # Numerical imputation
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    num_cols_valid = [col for col in num_cols if df[col].notna().sum() > 0]
    if num_cols_valid:
        imputed_nums = SimpleImputer(strategy="median").fit_transform(df[num_cols_valid])
        df[num_cols_valid] = pd.DataFrame(imputed_nums, columns=num_cols_valid, index=df.index)

    # Cathegorical imputation
    cat_cols = df.select_dtypes(include=["object"]).columns
    cat_cols_valid = [col for col in cat_cols if df[col].notna().sum() > 0]
    if cat_cols_valid:
        imputed_cats = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols_valid])
        df[cat_cols_valid] = pd.DataFrame(imputed_cats, columns=cat_cols_valid, index=df.index)

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