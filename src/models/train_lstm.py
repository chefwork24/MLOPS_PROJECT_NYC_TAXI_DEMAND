import numpy as np
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from prepare_data import load_all_features, split_data, get_features_and_target
from metrics import evaluate
from config_loader import load_config
import os

cfg = load_config()

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment(cfg['mlflow']['experiment_name'])

class DemandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

data = load_all_features(cfg)
train, val, test = split_data(data, cfg)
X_train, y_train, _ = get_features_and_target(train, cfg)
X_val,   y_val,   _ = get_features_and_target(val,   cfg)
X_test,  y_test,  _ = get_features_and_target(test,  cfg)

p = cfg['lstm']

train_loader = DataLoader(DemandDataset(X_train, y_train), batch_size=p['batch_size'], shuffle=True)
val_loader   = DataLoader(DemandDataset(X_val,   y_val),   batch_size=p['batch_size'])

device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model     = LSTMModel(X_train.shape[1], p['hidden_size'], p['num_layers']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
criterion = nn.MSELoss()

with mlflow.start_run(run_name="lstm"):
    mlflow.log_params(p)

    for epoch in range(p['epochs']):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch+1}/{p['epochs']} — train: {avg_train:.2f}  val: {avg_val:.2f}")
        mlflow.log_metric("train_loss", avg_train, step=epoch)
        mlflow.log_metric("val_loss",   avg_val,   step=epoch)

    model.eval()
    with torch.no_grad():
        val_preds  = model(torch.tensor(X_val.values,  dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy()
        test_preds = model(torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy()

    val_metrics  = evaluate(y_val,  val_preds,  "LSTM-Val")
    test_metrics = evaluate(y_test, test_preds, "LSTM-Test")

    for k, v in val_metrics.items():
        mlflow.log_metric(f"val_{k}", v)
    for k, v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    # Save model to disk
    os.makedirs("../../models", exist_ok=True)
    torch.save(model.state_dict(), "../../models/lstm_model.pth")
    print("Model saved to models/lstm_model.pth")

    mlflow.pytorch.log_model(model, artifact_path="model")
    print("\nLSTM run logged to MLflow")