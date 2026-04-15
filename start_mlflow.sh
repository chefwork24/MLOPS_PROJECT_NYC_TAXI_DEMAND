#!/bin/bash
cd /Users/rishwanthpb/MLOPS/MLOPS_PROJECT
source venv/bin/activate
mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000
