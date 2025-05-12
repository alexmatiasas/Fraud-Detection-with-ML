import pandas as pd
import os
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_preprocessing import impute_missing_values, encode_categorical, scale_features

# Load data
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "processed", "train_final_ready.csv")
df = pd.read_csv(data_path)

# Separate features and target variable
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Preprocessing pipeline
X = impute_missing_values(X)
X = encode_categorical(X)
X = scale_features(X, fit=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the stacking model
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
]
model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

model.fit(X_train, y_train)

# Basic evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
model_path = os.path.join(project_root, "models", "final_model_stacking.pkl")
joblib.dump(model, model_path)
print(f"✅ Modelo guardado en {model_path}")