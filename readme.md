# NYC Taxi Demand Forecasting & Driver Dispatch Optimization

A production-grade MLOps system that predicts hourly taxi pickup demand per NYC zone 6 hours ahead, enabling intelligent pre-positioning of drivers before demand materializes.

---

## Problem

Urban taxi fleets are deployed reactively. Dispatchers have no way to know where demand will be in the next few hours, causing driver shortages in high-demand zones and idle oversupply in low-demand zones.

---

## Dataset

- **Source:** NYC TLC Yellow Taxi Trip Records — [nyc.gov/tlc](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Size:** ~40–50 million records across 263 NYC taxi zones
- **Years:** 2019 (baseline), 2020 April (COVID drift), 2023 Jan–Apr (new normal)
- **Format:** Monthly Parquet files, freely downloadable

---

## Models

| Model | Val MAE | Val R² | Use |
|---|---|---|---|
| XGBoost | 10.27 | 0.9467 | Comparison |
| LightGBM | 10.10 | 0.9481 | Real-time REST inference |
| LSTM | 11.05 | 0.9360 | Comparison |
| TFT | 5.07 | — | Batch forecasting |

---

## Tech Stack

| Area | Tool |
|---|---|
| Data Engineering | Apache Spark, Apache Airflow |
| Versioning | DVC, Git, Git LFS |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit, Plotly, Folium |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
MLOPS_PROJECT/
├── data/
│   ├── raw/              # Raw NYC TLC Parquet files (DVC tracked)
│   ├── processed/        # Cleaned data (DVC tracked)
│   └── features/         # Engineered features (DVC tracked)
├── src/
│   ├── ingestion/        # Spark cleaning pipeline
│   ├── features/         # Spark feature engineering
│   ├── models/           # Model training scripts
│   ├── api/              # FastAPI backend
│   └── monitoring/       # Drift detection
├── frontend/             # Streamlit dashboard
├── airflow/dags/         # Airflow retraining DAG
├── docker/               # Dockerfiles and Prometheus config
├── models/               # Saved model files
├── docs/                 # Architecture, HLD, LLD, test plan, user manual
├── tests/                # Unit tests
├── docker-compose.yml
├── start_all.sh          # Start all services
└── stop_all.sh           # Stop all services
```

---

## How to Run

### Prerequisites
- Python 3.12
- Java 17 (for Spark)
- Docker Desktop

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/nyc-taxi-demand-forecast.git
cd nyc-taxi-demand-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start All Services

```bash
./start_all.sh
```

### Stop All Services

```bash
./stop_all.sh
```

### Open in Browser

| Service | URL | Login |
|---|---|---|
| Dashboard | http://localhost:8502 | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin/admin |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin/admin |
| API Docs | http://localhost:8000/docs | — |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API liveness check |
| GET | `/ready` | Model loaded check |
| POST | `/predict` | Single zone prediction |
| POST | `/predict/batch` | Multi-zone prediction |
| POST | `/recommend` | Driver recommendation |
| POST | `/update-drift` | Recompute drift score |
| GET | `/metrics` | Prometheus metrics |

---

## Run Tests

```bash
python -m pytest tests/test_api.py -v
# 6/6 tests passing
```

---

## Run Spark Pipeline

```bash
# Single file
python src/ingestion/spark_clean.py 2019-01
python src/features/spark_features.py 2019-01

# All files
for ym in 2019-01 2019-02 ... 2023-04; do
    python src/ingestion/spark_clean.py $ym
    python src/features/spark_features.py $ym
done
```

---

## Train Models

```bash
cd src/models

python train_xgboost.py
python train_lightgbm.py
python train_lstm.py
python train_tft.py
```

---

## Drift Detection

```bash
python src/monitoring/drift_detector.py

# Expected output:
# 2020-04 drift score (expect HIGH): 0.7695
# 2019-01 drift score (expect LOW):  0.0077
```
