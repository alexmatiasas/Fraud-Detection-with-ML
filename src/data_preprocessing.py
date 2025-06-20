import os

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Define paths
SCALER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
)
ENCODERS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "label_encoders.pkl")
)


def save_processed_data(train_data: pd.DataFrame, filename: str = "train_clean.csv"):
    """Saves cleaned/preprocessed data to the root data/processed folder."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, filename)
    train_data.to_csv(processed_path, index=False)
    print(f"✅ Processed data saved to {processed_path}")


def impute_missing_values(df):
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if not cat_cols.empty:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
            df[cat_cols]
        )
    return df


def encode_categorical(df, fit=False):
    cat_cols = df.select_dtypes(include=["object"]).columns
    df_copy = df.copy()
    encoders = {}

    if fit:
        for col in cat_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, ENCODERS_PATH)
    else:
        encoders = joblib.load(ENCODERS_PATH)
        for col in cat_cols:
            le = encoders.get(col)
            if le:
                df_copy[col] = le.transform(df_copy[col].astype(str))

    return df_copy


def scale_features(df, fit=False):
    df_copy = df.copy()
    num_cols = df_copy.select_dtypes(include=["float64", "int64"]).columns

    scaler = joblib.load("models/scaler.pkl")

    if hasattr(scaler, "feature_names_in_"):
        cols_to_use = scaler.feature_names_in_
    else:
        cols_to_use = num_cols  # fallback si no está definida

    df_copy[cols_to_use] = scaler.transform(df_copy[cols_to_use])

    return df_copy


def full_preprocessing_pipeline(df, fit=False):
    df = impute_missing_values(df)
    df = encode_categorical(df, fit=fit)
    df = scale_features(df, fit=fit)
    return df
