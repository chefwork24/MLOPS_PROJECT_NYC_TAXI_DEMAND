from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    zone_id:          int
    hour_of_day:      int
    day_of_week:      int
    month:            int
    is_weekend:       int
    is_rush_hour:     int
    demand_lag_1h:    float
    demand_lag_24h:   float
    demand_lag_168h:  float
    rolling_mean_24h: float
    rolling_mean_7d:  float
    is_airport_zone:  int

class PredictResponse(BaseModel):
    zone_id:          int
    predicted_demand: float
    demand_status:    str

class BatchPredictRequest(BaseModel):
    zones: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

class HealthResponse(BaseModel):
    status: str
    model:  str

class RecommendRequest(BaseModel):
    zone_id:          int
    predicted_demand: float
    current_drivers:  int

class RecommendResponse(BaseModel):
    zone_id:             int
    recommended_drivers: int
    action:              str