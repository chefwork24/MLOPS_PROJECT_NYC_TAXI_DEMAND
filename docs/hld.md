# High Level Design — NYC Taxi Demand Forecasting System

## 1. Problem Statement
Urban taxi fleets are deployed reactively causing driver shortages in
high-demand zones and oversupply in low-demand zones. This system
predicts hourly taxi pickup demand per NYC zone 6 hours ahead.

## 2. Design Choices & Rationale

### 2.1 Dataset Choice — NYC TLC Yellow Taxi
- 100M+ records across 263 NYC zones
- Freely available at nyc.gov/tlc — no license required
- 2019–2023 date range provides natural drift scenario
- 2020 COVID collapse gives real, measurable drift trigger

### 2.2 Model Choice — LightGBM for Inference, TFT for Batch
- LightGBM: R² 0.95, fast single-row inference via REST API
- TFT: MAE 5.07, best batch forecasting with attention mechanisms
- Two models serve two different use cases — standard industry practice

### 2.3 Feature Engineering
- Lag features (1h, 24h, 168h) capture temporal autocorrelation
- Rolling averages (24h, 7d) capture zone-level trends
- Temporal features (hour, day, month) capture seasonality
- Zone flags (airport, congestion) capture spatial context

### 2.4 Drift Detection — KL Divergence
- 2019 baseline statistics computed during EDA and stored as artifacts
- KL divergence measures distributional shift from baseline
- Threshold 0.15 chosen based on empirical testing
- 2020 April scored 0.77 — confirms threshold is meaningful

### 2.5 Architecture — Loose Coupling
- Frontend and backend are independent Docker containers
- Only connection is configurable REST API URL in environment variable
- Both can be developed, tested, and deployed independently

### 2.6 Technology Stack Rationale
| Tool | Rationale |
|---|---|
| Apache Spark | 40M+ records require distributed processing |
| Apache Airflow | Pipeline orchestration with retry logic |
| MLflow | Experiment reproducibility via run ID + commit hash |
| FastAPI | High performance async Python REST framework |
| Streamlit | Rapid Python dashboard without frontend complexity |
| Prometheus | Industry standard metrics collection |
| Grafana | Real-time visualization of operational metrics |
| Docker Compose | Environment parity across dev and production |
| DVC | Data and model versioning alongside Git |

## 3. MLOps Design Decisions

### 3.1 Reproducibility
Every experiment is reproducible via:
- Git commit hash (code state)
- MLflow Run ID (model state)
- DVC data hash (data state)

### 3.2 Retraining Strategy
- Triggered by drift score, not by schedule
- Drift-based retraining is more efficient than time-based
- Prevents unnecessary retraining when data is stable

### 3.3 No Cloud Constraint
All components run locally or on-premise:
- Spark runs in local[*] mode
- MLflow uses SQLite backend
- Docker containers run on local daemon
- No external API dependencies except NYC TLC data download