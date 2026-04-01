import os
import lightgbm as lgb
import pandas as pd

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/Users/rishwanthpb/MLOPS/MLOPS_PROJECT/models/lightgbm_model.txt"
)

FEATURE_COLS = [
    'PULocationID', 'hour_of_day', 'day_of_week', 'month',
    'is_weekend', 'is_rush_hour',
    'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
    'rolling_mean_24h', 'rolling_mean_7d', 'is_airport_zone'
]

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = lgb.Booster(model_file=MODEL_PATH)
    print("Model loaded successfully")
    return model

def predict_single(model, data: dict) -> float:
    row = pd.DataFrame([{
        'PULocationID':     data['zone_id'],
        'hour_of_day':      data['hour_of_day'],
        'day_of_week':      data['day_of_week'],
        'month':            data['month'],
        'is_weekend':       data['is_weekend'],
        'is_rush_hour':     data['is_rush_hour'],
        'demand_lag_1h':    data['demand_lag_1h'],
        'demand_lag_24h':   data['demand_lag_24h'],
        'demand_lag_168h':  data['demand_lag_168h'],
        'rolling_mean_24h': data['rolling_mean_24h'],
        'rolling_mean_7d':  data['rolling_mean_7d'],
        'is_airport_zone':  data['is_airport_zone'],
    }])
    pred = model.predict(row[FEATURE_COLS])[0]
    return max(0.0, float(pred))

def classify_demand(predicted: float, rolling_mean: float) -> str:
    if rolling_mean == 0:
        return "Normal"
    ratio = predicted / rolling_mean
    if ratio > 1.5:
        return "Surge"
    elif ratio < 0.3:
        return "Dead Zone"
    return "Normal"

def recommend_drivers(predicted_demand: float, current_drivers: int) -> dict:
    recommended = max(1, int(predicted_demand / 15))
    diff = recommended - current_drivers
    if diff > 2:
        action = f"Send {diff} more drivers to this zone"
    elif diff < -2:
        action = f"Relocate {abs(diff)} drivers from this zone"
    else:
        action = "Driver count is optimal"
    return {"recommended_drivers": recommended, "action": action}