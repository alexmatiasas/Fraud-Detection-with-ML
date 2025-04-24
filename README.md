# Fraud Detection with Machine Learning

This project applies machine learning techniques to detect fraudulent transactions using the IEEE-CIS Fraud Detection dataset.

## Project Structure

```bash
.
├── LICENSE
├── README.md
├── data
│   └── ieee-fraud-detection  # Raw dataset from Kaggle
├── models                    # Trained models and artifacts
├── notebooks                 # Exploratory analysis, modeling, and deployment notebooks
├── reports                   # Generated reports (EDA, model evaluation)
├── requirements.txt
└── src                       # Source code for preprocessing, training, and deployment
```

## Dataset

- **Source**: [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- **Size**: ~1.2 GB
- Contains anonymized transaction and identity data.

## Goals

- Perform Exploratory Data Analysis (EDA) to understand patterns in fraudulent vs. non-fraudulent transactions.
- Build machine learning models to classify transactions.
- Deploy the model via an API for real-time inference.

## Tech Stack

- **R** for EDA (`tidyverse`, `ggplot2`, `data.table`)
- **Python** for modeling and deployment (`scikit-learn`, `XGBoost`, `Flask/FastAPI`)
- **Docker** (optional) for reproducible environments

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/alexmatiasas/Fraud-Detection-with-ML.git
    cd Fraud-Detection-with-ML
    ```

2. Set up Python environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```