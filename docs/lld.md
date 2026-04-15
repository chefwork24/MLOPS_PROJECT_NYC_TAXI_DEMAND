# Low Level Design — API Endpoint Specifications

## Base URL
http://localhost:8000

## Endpoints

---

### GET /health
Returns API health status.

**Request:** None

**Response:**
{
  "status": "ok",
  "model": "lightgbm"
}

**Status Codes:**
- 200: API is running

---

### GET /ready
Returns whether model is loaded and ready.

**Request:** None

**Response:**
{
  "status": "ready",
  "model": "lightgbm"
}

**Status Codes:**
- 200: Model loaded and ready
- 503: Model not loaded

---

### POST /predict
Predict demand for a single zone.

**Request Body:**
{
  "zone_id": 161,
  "hour_of_day": 18,
  "day_of_week": 2,
  "month": 1,
  "is_weekend": 0,
  "is_rush_hour": 1,
  "demand_lag_1h": 450.0,
  "demand_lag_24h": 420.0,
  "demand_lag_168h": 430.0,
  "rolling_mean_24h": 380.0,
  "rolling_mean_7d": 390.0,
  "is_airport_zone": 0
}

**Response:**
{
  "zone_id": 161,
  "predicted_demand": 483.19,
  "demand_status": "Normal"
}

**demand_status values:**
- "Normal": predicted / rolling_mean between 0.3 and 1.5
- "Surge": predicted / rolling_mean > 1.5
- "Dead Zone": predicted / rolling_mean < 0.3

**Status Codes:**
- 200: Success
- 422: Validation error
- 500: Prediction error

---

### POST /predict/batch
Predict demand for multiple zones in one request.

**Request Body:**
{
  "zones": [ { ...PredictRequest... }, { ...PredictRequest... } ]
}

**Response:**
{
  "predictions": [
    { "zone_id": 161, "predicted_demand": 483.19, "demand_status": "Normal" },
    { "zone_id": 132, "predicted_demand": 95.40,  "demand_status": "Normal" }
  ]
}

**Status Codes:**
- 200: Success
- 500: Prediction error

---

### POST /recommend
Get driver pre-positioning recommendation for a zone.

**Request Body:**
{
  "zone_id": 161,
  "predicted_demand": 520.0,
  "current_drivers": 25
}

**Response:**
{
  "zone_id": 161,
  "recommended_drivers": 34,
  "action": "Send 9 more drivers to this zone"
}

**Recommendation Logic:**
- recommended_drivers = max(1, int(predicted_demand / 15))
- diff = recommended - current
- diff > 2:  "Send N more drivers to this zone"
- diff < -2: "Relocate N drivers from this zone"
- else:      "Driver count is optimal"

**Status Codes:**
- 200: Success
- 500: Recommendation error

---

### POST /update-drift
Recompute and update the drift score metric.

**Request:** None

**Response:**
{
  "drift_score": 0.0326,
  "file": "features_2023-04.parquet"
}

**Status Codes:**
- 200: Drift score updated
- 500: Error computing drift

---

### GET /metrics
Prometheus metrics endpoint.

**Response:** Plain text Prometheus metrics format

**Metrics exposed:**
- prediction_requests_total (counter)
- prediction_latency_seconds (histogram)
- model_confidence_score (gauge)
- api_error_total (counter)
- demand_drift_score (gauge)

---

## Airflow DAG — retraining_pipeline

**Schedule:** @daily

**Tasks:**
1. check_drift — calls /update-drift, checks if score > 0.15
2. retrain_model — runs train_lightgbm.py with venv Python
3. validate_model — checks model file exists after retraining
4. reset_drift_score — calls /update-drift to refresh score

**Flow:** check_drift → retrain_model → validate_model → reset_drift_score