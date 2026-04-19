#!/bin/bash

PROJECT_DIR="/Users/rishwanthpb/MLOPS/MLOPS_PROJECT"
VENV="$PROJECT_DIR/venv/bin/activate"

echo "=== Stopping everything first ==="
pkill -f "airflow" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
pkill -f "streamlit" 2>/dev/null
pkill -f "mlflow" 2>/dev/null
sleep 3

lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:8502 | xargs kill -9 2>/dev/null

# Stop conflicting Docker containers
docker stop da5402-wordpress-1 2>/dev/null
docker stop assignment-6-chefwork24-airflow-webserver-1 2>/dev/null
docker stop assignment-6-chefwork24-airflow-scheduler-1 2>/dev/null
docker stop prometheus grafana 2>/dev/null

sleep 3

echo "=== Starting MLflow ==="
cd $PROJECT_DIR
source $VENV
mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000 &
sleep 5
echo "MLflow started at http://localhost:5000"

echo "=== Starting FastAPI Backend ==="
cd $PROJECT_DIR/src/api
source $VENV
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 5
echo "Backend started at http://localhost:8000"

echo "=== Starting Streamlit Frontend ==="
cd $PROJECT_DIR/frontend
source $VENV
streamlit run app.py --server.port 8502 &
sleep 5
echo "Frontend started at http://localhost:8502"

echo "=== Starting Airflow ==="
export AIRFLOW_HOME=$PROJECT_DIR/airflow
cd $PROJECT_DIR
source $VENV
airflow webserver --port 8080 &
sleep 5
airflow scheduler &
sleep 5
echo "Airflow started at http://localhost:8080"

echo "=== Starting Docker (Prometheus + Grafana) ==="
cd $PROJECT_DIR
docker compose up -d
sleep 5

echo ""
echo "=== Verifying all services ==="
echo -n "MLflow:     " && curl -s http://localhost:5000 > /dev/null && echo "OK" || echo "FAILED"
echo -n "Backend:    " && curl -s http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])" 2>/dev/null || echo "FAILED"
echo -n "Frontend:   " && curl -s http://localhost:8502 > /dev/null && echo "OK" || echo "FAILED"
echo -n "Prometheus: " && curl -s http://localhost:9090/-/healthy > /dev/null && echo "OK" || echo "FAILED"
echo -n "Grafana:    " && curl -s http://localhost:3001 > /dev/null && echo "OK" || echo "FAILED"

echo ""
echo "=== All services started ==="
echo "Dashboard:  http://localhost:8502"
echo "Backend:    http://localhost:8000"
echo "MLflow:     http://localhost:5000"
echo "Airflow:    http://localhost:8080"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3001 (admin/admin)"

