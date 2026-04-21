#!/bin/bash

echo "Stopping all services..."

pkill -f "airflow"
pkill -f "uvicorn"
pkill -f "streamlit"
pkill -f "mlflow"

cd /Users/rishwanthpb/MLOPS/MLOPS_PROJECT
docker compose down

echo "All services stopped"
