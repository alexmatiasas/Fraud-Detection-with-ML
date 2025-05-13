import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pandas as pd
from data_preprocessing import full_preprocessing_pipeline

def test_pipeline_does_not_crash():
    df = pd.read_csv("data/processed/train_final_ready.csv")
    df = df.sample(n=3, random_state=42)
    
    result = full_preprocessing_pipeline(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert not result.isnull().values.any()