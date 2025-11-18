import numpy as np
import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :2].values.astype(float)
    Y = df.iloc[:, 2].values.reshape(-1,1).astype(float)
    return X, Y

def zscore_normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std
