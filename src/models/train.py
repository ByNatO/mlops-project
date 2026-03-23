import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_data(processed_path):
    X_train = pd.read_csv(os.path.join(processed_path, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(processed_path, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(processed_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_path, 'y_train.csv')).squeeze()
    y_val = pd.read_csv(os.path.join(processed_path, 'y_val.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(processed_path, 'y_test.csv')).squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_precision_recall_curve(y_true, y_proba, output_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (AUC = {pr_auc:.4f})')
    plt.savefig(output_path)
    plt.close()
    return pr_auc

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Définir l'URI MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config['data']['processed_path'])

    # Préparer les paramètres du modèle
    model_type = config['model']['type']
    model_params = config['model']['params']

    # Créer le modèle
    if model_type == 'xgboost':
        model = XGBClassifier(**model_params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Entraînement
    with mlflow.start_run() as run:
        # Log des paramètres
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions sur validation
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= config['model'].get('threshold', 0.5)).astype(int)

        # Métriques
        roc_auc = roc_auc_score(y_val, y_val_proba)
        f1 = f1_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)

        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_val_proba)
        pr_auc = auc(recall_curve, precision_curve)

        mlflow.log_metrics({
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })

        # Log des artefacts
        # Courbe PR
        pr_curve_path = "pr_curve.png"
        plot_precision_recall_curve(y_val, y_val_proba, pr_curve_path)
        mlflow.log_artifact(pr_curve_path)
        # Matrice de confusion
        cm_path = "confusion_matrix.png"
        plot_confusion_matrix(y_val, y_val_pred, cm_path)
        mlflow.log_artifact(cm_path)

        # Log du modèle
        mlflow.sklearn.log_model(model, "model")

        # Sauvegarde locale (optionnel)
        joblib.dump(model, "model.joblib")

        # Ajouter des tags
        mlflow.set_tag("author", "votre_nom")
        mlflow.set_tag("dataset", "creditcard")
        mlflow.set_tag("version", "1.0")

        print(f"Run ID: {run.info.run_id}")
        print(f"PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    train(args.config)