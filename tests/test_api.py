import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/api"))

import pytest
from fastapi.testclient import TestClient
import model as model_module
from main import app

# Load model before tests
@pytest.fixture(autouse=True, scope="session")
def load_model():
    model_module_instance = model_module.load_model()
    import main
    main.model = model_module_instance

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_predict():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_demand" in response.json()
    assert "demand_status" in response.json()
    assert response.json()["predicted_demand"] > 0

def test_predict_surge():
    payload = {
        "zone_id": 161,
        "hour_of_day": 18,
        "day_of_week": 2,
        "month": 1,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "demand_lag_1h": 1200.0,
        "demand_lag_24h": 1100.0,
        "demand_lag_168h": 1000.0,
        "rolling_mean_24h": 400.0,
        "rolling_mean_7d": 390.0,
        "is_airport_zone": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["demand_status"] in ["Normal", "Surge", "Dead Zone"]

def test_recommend():
    payload = {
        "zone_id": 161,
        "predicted_demand": 520.0,
        "current_drivers": 25
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    assert "recommended_drivers" in response.json()
    assert "action" in response.json()
    assert response.json()["recommended_drivers"] > 0

def test_predict_batch():
    payload = {
        "zones": [
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
            },
            {
                "zone_id": 132,
                "hour_of_day": 10,
                "day_of_week": 3,
                "month": 6,
                "is_weekend": 0,
                "is_rush_hour": 0,
                "demand_lag_1h": 80.0,
                "demand_lag_24h": 75.0,
                "demand_lag_168h": 70.0,
                "rolling_mean_24h": 60.0,
                "rolling_mean_7d": 65.0,
                "is_airport_zone": 1
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2