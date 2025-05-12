# Fraud Detection with Machine Learning

This project applies machine learning techniques to detect fraudulent transactions using the IEEE-CIS Fraud Detection dataset. It integrates robust preprocessing, model training, evaluation, and deployment via an interactive API.

## 📁 Project Structure

```bash
.
├── dags/                      # Airflow DAGs for pipeline orchestration
├── data/                      # Raw and processed data
│   ├── ieee-fraud-detection/  # Original CSVs from Kaggle
│   └── processed/             # Cleaned datasets
├── docker/airflow/            # Docker setup for Airflow
├── models/                    # Saved ML models
├── notebooks/                 # EDA, modeling, deployment workflows
├── reports/                   # Metrics, ROC curves, confusion matrices
├── src/                       # Source code (preprocessing, API, training)
├── requirements.txt           # Python dependencies (if not using Poetry)
├── pyproject.toml             # Poetry config
├── README.md
└── LICENSE
```

## 📊 Dataset

- **Source**: [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- **Size**: ~1.2 GB
- Includes anonymized features on transactions and user identity.

## 🎯 Goals

- ✅ Exploratory Data Analysis (EDA) in R and Python
- ✅ Build robust ML models (Logistic Regression, Random Forest, XGBoost, LGBM, CatBoost, Stacking)
- ✅ Evaluate performance with metrics & plots
- ✅ Deploy best model with FastAPI
- ✅ Interact with the model through Swagger UI

## 🛠️ Tech Stack

- **EDA**: `tidyverse`, `ggplot2`, `data.table` (R)
- **Modeling**: `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost` (Python)
- **API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Visualization**: `matplotlib`, `seaborn`
- **Deployment tools**: `joblib`, `Docker` (optional), `Poetry`

---

## 📈 R Setup (EDA)

1. Open `notebooks/01_EDA_in_R.Rmd` in RStudio.
2. Required R packages:
   - `tidyverse`, `ggplot2`, `data.table`, `skimr`, `ranger`, `ggcorrplot`, `visdat`

Render to HTML with:
```r
rmarkdown::render("notebooks/01_EDA_in_R.Rmd")
```

## 🧪 Python Setup

1. Clone this repo:
```bash
git clone https://github.com/alexmatiasas/Fraud-Detection-with-ML.git
cd Fraud-Detection-with-ML
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Activate the environment:
```bash
poetry shell
```

4. Set up Jupyter kernel (optional):
```bash
poetry run python -m ipykernel install --user --name=fraud-eda --display-name "Python (fraud-eda)"
```

## 🚀 Run the API

1. Train and save your model to `models/final_model_stacking.pkl`
2. Launch the API locally:
```bash
uvicorn src.main:app --reload
```
3. Visit `http://127.0.0.1:8000/docs` for Swagger UI 🧪

## 🔄 Example API Request (JSON Body)
```json
{
  "TransactionID": 123456,
  "TransactionAmt": 103.0,
  "ProductCD": "W",
  "card1": 1500,
  "card2": 200.0,
  ... (more features)
}
```
Returns:
```json
{
  "prediction": 0,
  "fraud_probability": 0.0241
}
```

## 📊 Reports

Visuals and metrics stored in `reports/figures/` and `reports/model_metrics.csv` include:
- Confusion matrices
- ROC curves
- Ensemble comparison heatmaps

## 📌 Next Steps

- [ ] Add Airflow DAG to automate pipeline from preprocessing to inference
- [ ] Containerize with Docker
- [ ] Deploy to the cloud (Render, Heroku, or AWS)
- [ ] Document pipeline with MLFlow or similar

## 💡 Author
**Manuel Alejandro Matías Astorga**  
Data Scientist | Physicist | Open Source Enthusiast  
📧 Contact: [LinkedIn](https://www.linkedin.com/in/alexmatiasas) | [GitHub](https://github.com/alexmatiasas)