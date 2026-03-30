import mlflow
import mlflow.xgboost
import xgboost as xgb
from prepare_data import load_all_features, split_data, get_features_and_target
from metrics import evaluate
from config_loader import load_config

cfg = load_config()

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment(cfg['mlflow']['experiment_name'])

data = load_all_features(cfg)
train, val, test = split_data(data, cfg)
X_train, y_train, _ = get_features_and_target(train, cfg)
X_val,   y_val,   _ = get_features_and_target(val,   cfg)
X_test,  y_test,  _ = get_features_and_target(test,  cfg)

params = cfg['xgboost']

with mlflow.start_run(run_name="xgboost"):
    mlflow.log_params(params)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    val_metrics  = evaluate(y_val,  model.predict(X_val),  "XGBoost-Val")
    test_metrics = evaluate(y_test, model.predict(X_test), "XGBoost-Test")

    for k, v in val_metrics.items():
        mlflow.log_metric(f"val_{k}", v)
    for k, v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    mlflow.xgboost.log_model(model, artifact_path="model")
    print("\nXGBoost run logged to MLflow")