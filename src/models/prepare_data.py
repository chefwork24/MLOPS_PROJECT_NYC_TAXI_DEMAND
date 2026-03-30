import os
import pandas as pd
from config_loader import load_config

def load_all_features(cfg):
    features_dir = cfg['data']['features_dir']
    files = sorted([f for f in os.listdir(features_dir) if f.endswith(".parquet")])
    dfs = []
    for f in files:
        df = pd.read_parquet(os.path.join(features_dir, f))
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)
    print(f"Total rows loaded: {len(data)}")
    return data

def split_data(data, cfg):
    train_years = cfg['data']['train_years']
    val_months  = cfg['data']['val_months']
    test_months = cfg['data']['test_months']

    train = data[data['ym'].str[:4].isin(train_years)]
    val   = data[data['ym'].isin(val_months)]
    test  = data[data['ym'].isin(test_months)]

    print(f"Train rows: {len(train)}")
    print(f"Val rows:   {len(val)}")
    print(f"Test rows:  {len(test)}")
    return train, val, test

def get_features_and_target(df, cfg):
    feature_cols = cfg['features']['cols']
    target       = cfg['features']['target']
    drop_na_cols = cfg['features']['drop_na_cols']

    df = df.dropna(subset=drop_na_cols)
    X  = df[feature_cols]
    y  = df[target]
    return X, y, df

if __name__ == "__main__":
    cfg = load_config()
    data = load_all_features(cfg)
    train, val, test = split_data(data, cfg)
    X_train, y_train, _ = get_features_and_target(train, cfg)
    X_val,   y_val,   _ = get_features_and_target(val,   cfg)
    X_test,  y_test,  _ = get_features_and_target(test,  cfg)
    print("\nX_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)