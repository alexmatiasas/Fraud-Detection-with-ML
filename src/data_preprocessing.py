import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

def save_processed_data(train_data: pd.DataFrame, filename: str = "train_clean.csv"):
    """Saves cleaned/preprocessed data to the root data/processed folder."""

    # Get the project root (assumes this script is in src/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Build absolute path to data/processed
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Final path for saving
    processed_path = os.path.join(processed_dir, filename)
    train_data.to_csv(processed_path, index=False)
    print(f"✅ Processed data saved to {processed_path}")

def impute_missing_values(df):
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    return df

def encode_categorical(df):
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def scale_features(df, exclude=None):
    exclude = exclude or []
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(exclude)
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    return df

def full_preprocessing_pipeline(df):
    df = impute_missing_values(df)
    df = encode_categorical(df)
    return df