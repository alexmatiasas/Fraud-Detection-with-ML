# Fraud Detection with Machine Learning

[![Docker Pulls](https://img.shields.io/docker/pulls/alexmatiasastorga/fraud-api.svg)](https://hub.docker.com/r/alexmatiasastorga/fraud-api)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/alexmatiasas/Fraud-Detection-with-ML.svg)](https://github.com/alexmatiasas/Fraud-Detection-with-ML/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/alexmatiasas/Fraud-Detection-with-ML.svg)](https://github.com/alexmatiasas/Fraud-Detection-with-ML)

> A complete machine learning pipeline for detecting fraudulent transactions using the IEEE-CIS dataset.
> Includes EDA, ensemble modeling, a FastAPI deployment, and automated tests — ready for production or research.

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
├── pyproject.toml             # Poetry config
├── README.md
└── LICENSE
```

## 📊 Dataset

- **Source**: [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/c/ieee-fraud-detection)
- **Size**: ~1.2 GB
- Includes anonymized features on transactions and user identity.

## 🎯 Goals

- ✅ Exploratory Data Analysis (EDA) in R ([View it on RPubs](https://rpubs.com/Alex_matias_as/fraud-detection-eda)) and Python
- ✅ Build robust ML models (Logistic Regression, Random Forest, XGBoost, LGBM, CatBoost, Stacking)
- ✅ Evaluate performance with metrics & plots
- ✅ Deploy best model with FastAPI
- ✅ Interact with the model through Swagger UI

## 🌟 Highlights

- 📊 Performed deep EDA in R and Python with statistical plots and correlation matrices.
- 🤖 Built and compared 6 ML models (including stacking ensemble) with tuned hyperparameters.
- 🧪 Implemented unit tests with `pytest` for model and preprocessing pipeline.
- 🔌 Deployed model via FastAPI and documented with Swagger UI.

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

## ✅ Testing

Basic unit tests included for:

- Preprocessing pipeline
- Ensemble model prediction

To run tests:

  ```bash
  pytest tests/
  ```

## 📌 Next Steps

- [x] Containerize with Docker
- [x] Unit testing with pytest
- [ ] Add Airflow DAG to automate pipeline
- [ ] Deploy to the cloud (Render, Heroku, or AWS)
- [ ] Document pipeline with MLFlow or similar

## 📬 API Usage (Detailed)

Once the FastAPI app is running locally (via `uvicorn src.main:app --reload`), you can:

### 🔎 1. Access the Swagger UI

Navigate to:

```bash
http://127.0.0.1:8000/docs
```

There, you'll see an interactive interface to:

- Test the `/predict` endpoint
- See the expected schema (auto-generated from the training dataset)
- Send POST requests with transaction data

### 🧾 2. Sample Request Format

The request body must match the model used in training (feature names and types).

✅ Valid example:

  ```json
  {
    "TransactionID": 3457624,
    "TransactionAmt": 724.0,
    "ProductCD": "W",
    "card1": 7826,
    "card2": 481,
    "card3": 150,
    "card4": "visa",
    "card5": 224,
    "card6": "credit",
    "addr1": 387,
    "addr2": 87,
    "P_emaildomain": "gmail.com",
    "C1": 3.0,
    "D1": 45.0,
    "V1": 1.0,
    "id_01": -5.0
    // ... truncated for brevity
  }
  ```

⛔ Avoid passing encoded or preprocessed data unless your API explicitly expects it. The backend handles:

- Missing value imputation
- Categorical encoding
- Feature scaling

✅ 3. Output

If successful, the response will look like:

```json
{
  "prediction": 1,
  "fraud_probability": 0.8237
}
```

Where:

- "prediction": Final label (1 = fraud, 0 = not fraud)

## 🐳 Run with Docker

To pull and run the model without installing anything:

```bash
docker pull alexmatiasastorga/fraud-api:latest
docker run -d -p 8000:8000 alexmatiasastorga/fraud-api
```

Access the API at: [http://localhost:8000/docs](http://localhost:8000/docs) and follow the same instructions as [1. Access the Swagger UI](./README.md###🔎-1.-Access-the-Swagger-UI) in [📬 API Usage (Detailed)](./README.md#📬-API-Usage-(Detailed))

## 🔒 Security Notice

This project includes Apache Airflow as part of an experimental pipeline orchestration setup. However, Airflow is not currently deployed nor exposed to the public in any environment.

GitHub may flag known vulnerabilities in the current Airflow version (`requirements.txt` in `docker/airflow/`). These are **acknowledged** and will be addressed before any production deployment.

**Current status**: Airflow DAGs are paused and not used in active inference or automation.

## 💡 Author

**Manuel Alejandro Matías Astorga**  
Data Scientist | Physicist | Open Source Enthusiast  
📧 Contact: [LinkedIn](https://www.linkedin.com/in/alexmatiasas) | [GitHub](https://github.com/alexmatiasas)
