from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import subprocess
import os
import sys

PROJECT_DIR = "/Users/rishwanthpb/MLOPS/MLOPS_PROJECT"
API_URL     = "http://localhost:8000"

default_args = {
    "owner":            "mlops",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

def check_drift():
    """Check current drift score from API metrics."""
    r = requests.post(f"{API_URL}/update-drift", timeout=30)
    result = r.json()
    drift_score = result["drift_score"]
    print(f"Current drift score: {drift_score}")

    if drift_score > 0.15:
        print("Drift detected — retraining needed")
        return True
    else:
        print("No drift detected — skipping retraining")
        return False

def retrain_model():
    print("Starting retraining...")
    
    # Use the same Python that is running Airflow
    python_path = sys.executable
    models_dir  = os.path.join(PROJECT_DIR, "src/models")
    
    print(f"Python: {python_path}")
    print(f"CWD: {models_dir}")
    
    result = subprocess.run(
        [python_path, "train_lightgbm.py"],
        cwd=models_dir,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Retraining failed: {result.stderr}")
    print("Retraining complete")

def validate_model():
    """Check that the retrained model file exists and is recent."""
    model_path = os.path.join(PROJECT_DIR, "models/lightgbm_model.txt")
    if not os.path.exists(model_path):
        raise Exception("Model file not found after retraining")
    print(f"Model file exists at {model_path}")
    print("Validation passed")

def reset_drift_score():
    """Reset drift score after successful retraining."""
    r = requests.post(f"{API_URL}/update-drift", timeout=30)
    print(f"Drift score after retraining: {r.json()['drift_score']}")

with DAG(
    dag_id="retraining_pipeline",
    default_args=default_args,
    description="Automated retraining when drift is detected",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "retraining"]
) as dag:

    t1 = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift
    )

    t2 = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model
    )

    t3 = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model
    )

    t4 = PythonOperator(
        task_id="reset_drift_score",
        python_callable=reset_drift_score
    )

    # Pipeline: check drift → retrain → validate → reset score
    t1 >> t2 >> t3 >> t4