import pandas as pd
import os

def load_data(filepath='data/raw/creditcard.csv'):
    """Charge les données brutes et retourne un DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def get_features_target(df, target_col='Class'):
    """Sépare les features et la cible."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y