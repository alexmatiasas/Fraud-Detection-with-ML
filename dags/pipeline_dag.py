import sys
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# ─── Import your training function ────────────────────────
from model_training import train_model

# ─── Add src directory to sys.path ────────────────────────
sys.path.append("/app/src")


# ─── Default DAG arguments ────────────────────────────────
default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

# ─── Define DAG ───────────────────────────────────────────
with DAG(
    dag_id="fraud_detection_pipeline",
    description="Pipeline to train fraud detection model",
    schedule_interval=None,
    default_args=default_args,
    catchup=False,
    tags=["fraud", "ml"],
) as dag:

    train_model_task = PythonOperator(
        task_id="train_model", python_callable=train_model
    )

    train_model_task
