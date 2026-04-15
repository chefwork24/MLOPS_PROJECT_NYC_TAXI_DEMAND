import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_client import make_asgi_app
import sys
import os
sys.path.insert(0, "/Users/rishwanthpb/MLOPS/MLOPS_PROJECT/src/monitoring")
from drift_detector import compute_drift_score
import pandas as pd

from schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse,
    RecommendRequest, RecommendResponse
)
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY,
    MODEL_CONFIDENCE, ERROR_COUNT, DRIFT_SCORE
)
from model import load_model, predict_single, classify_demand, recommend_drivers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Starting up — loading model...")
    model = load_model()
    MODEL_CONFIDENCE.set(0.95)
    DRIFT_SCORE.set(0.0)
    logger.info("Startup complete")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="NYC Taxi Demand Forecast API", version="1.0.0", lifespan=lifespan)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model": "lightgbm"}

@app.get("/ready", response_model=HealthResponse)
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model": "lightgbm"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    try:
        pred   = predict_single(model, request.model_dump())
        status = classify_demand(pred, request.rolling_mean_24h)
        REQUEST_COUNT.labels(endpoint="predict", zone_id=str(request.zone_id)).inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.time() - start)
        return {
            "zone_id":          request.zone_id,
            "predicted_demand": round(pred, 2),
            "demand_status":    status
        }
    except Exception as e:
        ERROR_COUNT.labels(endpoint="predict").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    start = time.time()
    results = []
    try:
        for zone in request.zones:
            pred   = predict_single(model, zone.model_dump())
            status = classify_demand(pred, zone.rolling_mean_24h)
            results.append({
                "zone_id":          zone.zone_id,
                "predicted_demand": round(pred, 2),
                "demand_status":    status
            })
        REQUEST_LATENCY.labels(endpoint="predict_batch").observe(time.time() - start)
        return {"predictions": results}
    except Exception as e:
        ERROR_COUNT.labels(endpoint="predict_batch").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        result = recommend_drivers(request.predicted_demand, request.current_drivers)
        return {"zone_id": request.zone_id, **result}
    except Exception as e:
        ERROR_COUNT.labels(endpoint="recommend").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-drift")
def update_drift():
    """
    Called periodically to recompute drift score.
    Reads latest feature file and computes KL divergence.
    """
    try:
        features_dir = "/Users/rishwanthpb/MLOPS/MLOPS_PROJECT/data/features"

        # Load most recent month's features
        files = sorted([
            f for f in os.listdir(features_dir)
            if f.endswith(".parquet")
        ])
        latest_file = os.path.join(features_dir, files[-1])
        df = pd.read_parquet(latest_file)

        score = compute_drift_score(df)
        DRIFT_SCORE.set(score)

        return {"drift_score": score, "file": files[-1]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))