from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "zone_id"]
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"]
)

MODEL_CONFIDENCE = Gauge(
    "model_confidence_score",
    "Model confidence score"
)

ERROR_COUNT = Counter(
    "api_error_total",
    "Total API errors",
    ["endpoint"]
)

DRIFT_SCORE = Gauge(
    "demand_drift_score",
    "KL divergence drift score from training baseline"
)