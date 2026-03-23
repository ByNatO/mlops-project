import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import mlflow

def simulate_drift(df, drift_type='amount'):
    df_drift = df.copy()
    if drift_type == 'amount':
        idx = np.random.choice(df.index, size=int(0.2*len(df)), replace=False)
        df_drift.loc[idx, 'Amount'] *= 1.3
    return df_drift

def compare_distributions(ref, new, features):
    drift = False
    for col in features:
        stat, p = ks_2samp(ref[col], new[col])
        if p < 0.05:
            print(f"Drift sur {col}, p={p}")
            drift = True
    return drift

if __name__ == "__main__":
    ref = pd.read_csv('data/raw/creditcard.csv')
    new = simulate_drift(ref, 'amount')
    features = [f'V{i}' for i in range(1,29)] + ['Amount','Time']
    drift = compare_distributions(ref, new, features)
    if drift:
        mlflow.log_metric("drift_detected", 1)
        print("Alerte : dérive détectée")
    else:
        mlflow.log_metric("drift_detected", 0)