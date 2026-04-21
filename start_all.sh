#!/bin/bash

PROJECT_DIR="/Users/rishwanthpb/MLOPS/MLOPS_PROJECT"
VENV="$PROJECT_DIR/venv/bin/activate"

echo ""
echo "======================================"
echo " STEP 1: Stop conflicting containers"
echo "======================================"
docker stop assignment-6-chefwork24-airflow-webserver-1 2>/dev/null && echo "Stopped assignment airflow webserver" || echo "Not running"
docker stop assignment-6-chefwork24-airflow-scheduler-1 2>/dev/null && echo "Stopped assignment airflow scheduler" || echo "Not running"
docker stop da5402-wordpress-1 2>/dev/null && echo "Stopped wordpress" || echo "Not running"
docker stop prometheus 2>/dev/null && echo "Stopped old prometheus" || echo "Not running"
docker stop grafana 2>/dev/null && echo "Stopped old grafana" || echo "Not running"

echo ""
echo "======================================"
echo " STEP 2: Kill all local services"
echo "======================================"
pkill -f "airflow" 2>/dev/null && echo "Killed airflow" || echo "Airflow not running"
pkill -f "uvicorn" 2>/dev/null && echo "Killed uvicorn" || echo "Uvicorn not running"
pkill -f "streamlit" 2>/dev/null && echo "Killed streamlit" || echo "Streamlit not running"
pkill -f "mlflow" 2>/dev/null && echo "Killed mlflow" || echo "MLflow not running"
sleep 3

echo ""
echo "======================================"
echo " STEP 3: Free all ports"
echo "======================================"
lsof -ti:5000 | xargs kill -9 2>/dev/null && echo "Freed port 5000" || echo "Port 5000 free"
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "Freed port 8000" || echo "Port 8000 free"
lsof -ti:8080 | xargs kill -9 2>/dev/null && echo "Freed port 8080" || echo "Port 8080 free"
lsof -ti:8502 | xargs kill -9 2>/dev/null && echo "Freed port 8502" || echo "Port 8502 free"
sleep 2

echo ""
echo "======================================"
echo " STEP 4: Start Docker containers"
echo "======================================"
cd $PROJECT_DIR
docker compose up -d
sleep 5
docker compose ps

echo ""
echo "======================================"
echo " STEP 5: Start MLflow"
echo "======================================"
cd $PROJECT_DIR
source $VENV
mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000 > /tmp/mlflow.log 2>&1 &
sleep 5
curl -s http://localhost:5000 > /dev/null && echo "MLflow OK" || echo "MLflow FAILED — check /tmp/mlflow.log"

echo ""
echo "======================================"
echo " STEP 6: Start FastAPI backend"
echo "======================================"
cd $PROJECT_DIR/src/api
source $VENV
uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
sleep 5
curl -s http://localhost:8000/health && echo "" && echo "Backend OK" || echo "Backend FAILED — check /tmp/api.log"

echo ""
echo "======================================"
echo " STEP 7: Start Streamlit frontend"
echo "======================================"
cd $PROJECT_DIR/frontend
source $VENV
streamlit run app.py --server.port 8502 > /tmp/streamlit.log 2>&1 &
sleep 5
curl -s http://localhost:8502 > /dev/null && echo "Streamlit OK" || echo "Streamlit FAILED — check /tmp/streamlit.log"

echo ""
echo "======================================"
echo " STEP 8: Start Airflow"
echo "======================================"
export AIRFLOW_HOME=$PROJECT_DIR/airflow
cd $PROJECT_DIR
source $VENV
airflow webserver --port 8080 > /tmp/airflow_web.log 2>&1 &
sleep 8
airflow scheduler > /tmp/airflow_scheduler.log 2>&1 &
sleep 5
curl -s http://localhost:8080 > /dev/null && echo "Airflow OK" || echo "Airflow FAILED — check /tmp/airflow_web.log"

echo ""
echo "======================================"
echo " STEP 9: Final verification"
echo "======================================"
echo -n "MLflow     (5000): " && curl -s http://localhost:5000 > /dev/null && echo "OK" || echo "FAILED"
echo -n "Backend    (8000): " && curl -s http://localhost:8000/health | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "FAILED"
echo -n "Streamlit  (8502): " && curl -s http://localhost:8502 > /dev/null && echo "OK" || echo "FAILED"
echo -n "Prometheus (9090): " && curl -s http://localhost:9090/-/healthy > /dev/null && echo "OK" || echo "FAILED"
echo -n "Grafana    (3001): " && curl -s http://localhost:3001 > /dev/null && echo "OK" || echo "FAILED"
echo -n "Airflow    (8080): " && curl -s http://localhost:8080 > /dev/null && echo "OK" || echo "FAILED"
echo -n "Drift endpoint:    " && curl -s -X POST http://localhost:8000/update-drift | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"OK — score={d['drift_score']:.4f}\")" 2>/dev/null || echo "FAILED"

echo ""
echo "======================================"
echo " All services started"
echo " Open these tabs in browser:"
echo " Dashboard:  http://localhost:8502"
echo " MLflow:     http://localhost:5000"
echo " Airflow:    http://localhost:8080"
echo " Prometheus: http://localhost:9090"
echo " Grafana:    http://localhost:3001"
echo "======================================"
