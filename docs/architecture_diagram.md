# Architecture Diagram — NYC Taxi Demand Forecasting System

## System Overview

The system consists of 5 independent layers connected via REST APIs and
orchestrated by Airflow. The frontend and backend are strictly decoupled.

## Layers

### Layer 1 — Data Ingestion & Engineering
- Raw monthly Parquet files downloaded from NYC TLC (nyc.gov/tlc)
- Apache Spark processes and engineers features
- Apache Airflow orchestrates the monthly ingestion DAG
- DVC versions all data artifacts

### Layer 2 — Experiment Tracking & Model Development
- MLflow Tracking Server logs all experiments
- 4 models trained: XGBoost, LightGBM, LSTM, TFT
- Best model (TFT — MAE 5.07) registered in MLflow Model Registry
- LightGBM used for real-time REST inference (MAE 10.10, R² 0.95)

### Layer 3 — Model Serving (Backend)
- FastAPI application serves predictions via REST API
- LightGBM model loaded directly from disk
- Endpoints: /predict, /predict/batch, /recommend, /update-drift,
  /health, /ready, /metrics
- Containerized in Docker

### Layer 4 — Frontend Dashboard
- Streamlit web application
- 4 pages: Dashboard, Single Zone Prediction, Zone Heatmap,
  Driver Recommendations
- Calls FastAPI backend exclusively via REST API
- Containerized in Docker

### Layer 5 — Monitoring & Retraining
- Prometheus scrapes /metrics endpoint every 15 seconds
- Grafana visualizes metrics in real-time dashboards
- Drift detector computes KL divergence vs 2019 baseline
- Airflow retraining DAG triggers automatically when drift > 0.15

## Data Flow

NYC TLC Data → Spark → DVC → MLflow → FastAPI → Streamlit
                                   ↑
                              Airflow DAG
                                   ↑
                          Prometheus Alert
                                   ↑
                        demand_drift_score > 0.15