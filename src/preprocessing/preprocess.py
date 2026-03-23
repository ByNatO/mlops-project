import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os
import joblib

def preprocess(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Chargement
    df = pd.read_csv(config['data']['raw_path'])
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Séparation train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['data']['val_size'] + config['data']['test_size'],
        random_state=config['data']['random_state'], stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=config['data']['test_size']/(config['data']['val_size']+config['data']['test_size']),
        random_state=config['data']['random_state'], stratify=y_temp
    )

    # Standardisation (optionnelle)
    scaler = StandardScaler()
    if config['preprocessing'].get('scale_amount', True):
        X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
        X_val[['Amount', 'Time']] = scaler.transform(X_val[['Amount', 'Time']])
        X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])
        # Sauvegarde du scaler
        joblib.dump(scaler, os.path.join(config['data']['processed_path'], 'scaler.pkl'))

    # Sauvegarde des datasets
    os.makedirs(config['data']['processed_path'], exist_ok=True)
    X_train.to_csv(os.path.join(config['data']['processed_path'], 'X_train.csv'), index=False)
    X_val.to_csv(os.path.join(config['data']['processed_path'], 'X_val.csv'), index=False)
    X_test.to_csv(os.path.join(config['data']['processed_path'], 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(config['data']['processed_path'], 'y_train.csv'), index=False)
    y_val.to_csv(os.path.join(config['data']['processed_path'], 'y_val.csv'), index=False)
    y_test.to_csv(os.path.join(config['data']['processed_path'], 'y_test.csv'), index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    preprocess()