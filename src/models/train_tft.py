# train_tft.py
import pandas as pd
import mlflow
import mlflow.pytorch
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl
from prepare_data import load_all_features, split_data
from config_loader import load_config

cfg = load_config()

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment(cfg['mlflow']['experiment_name'])

p = cfg['tft']

# ── Check device ──────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    accelerator = "mps"
    print("Training on Apple MPS (GPU)")
elif torch.cuda.is_available():
    accelerator = "gpu"
    print("Training on CUDA GPU")
else:
    accelerator = "cpu"
    print("Training on CPU")

# ── Load & prepare data ───────────────────────────────────────────────────────
data = load_all_features(cfg)
train_raw, val_raw, _ = split_data(data, cfg)

combined = pd.concat([train_raw, val_raw], ignore_index=True)
combined = combined.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)
combined = combined.dropna()

combined["time_idx"]     = combined.groupby("PULocationID").cumcount()
combined["PULocationID"] = combined["PULocationID"].astype(str)

cutoff     = combined["time_idx"].max() - p['max_prediction_length']
train_data = combined[combined["time_idx"] <= cutoff]

# ── Build TFT dataset ─────────────────────────────────────────────────────────
training = TimeSeriesDataSet(
    train_data,
    time_idx="time_idx",
    target="demand",
    group_ids=["PULocationID"],
    max_encoder_length=p['max_encoder_length'],
    max_prediction_length=p['max_prediction_length'],
    time_varying_known_reals=[
        "time_idx", "hour_of_day", "day_of_week",
        "month", "is_weekend", "is_rush_hour"
    ],
    time_varying_unknown_reals=["demand"],
    target_normalizer=GroupNormalizer(groups=["PULocationID"]),
    add_relative_time_idx=True,
    add_target_scales=True,
)

val_dataset  = TimeSeriesDataSet.from_dataset(training, combined, predict=True, stop_randomization=True)
train_loader = training.to_dataloader(train=True,  batch_size=p['batch_size'], num_workers=0)
val_loader   = val_dataset.to_dataloader(train=False, batch_size=p['batch_size'], num_workers=0)

# ── Train ─────────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="tft"):
    mlflow.log_params({**p, "accelerator": accelerator})

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=p['learning_rate'],
        hidden_size=p['hidden_size'],
        attention_head_size=p['attention_head_size'],
        dropout=p['dropout'],
        loss=MAE(),
        log_interval=10,
    )

    # Train on MPS/GPU
    trainer = pl.Trainer(
        max_epochs=p['max_epochs'],
        accelerator=accelerator,
        enable_progress_bar=True,
        gradient_clip_val=0.1,
    )
    trainer.fit(model, train_loader, val_loader)

    # ── Predict entirely on CPU to avoid device mismatch ─────────────────────
    print("Running prediction on CPU...")
    model = model.cpu()
    model.eval()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch, _ in val_loader:
            # Move batch to CPU
            batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            output = model(batch)
            pred   = output.prediction.cpu()
            target = batch["decoder_target"].cpu()

            all_outputs.append(pred)
            all_targets.append(target)

    outputs = torch.cat(all_outputs).squeeze()
    targets = torch.cat(all_targets).squeeze()

    mae_val = float((outputs - targets).abs().mean())
    print(f"\nTFT val MAE: {mae_val:.2f}")
    mlflow.log_metric("val_mae", mae_val)

    mlflow.pytorch.log_model(model, artifact_path="model")
    print("TFT run logged to MLflow")