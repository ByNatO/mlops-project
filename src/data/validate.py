import pandas as pd
import sys

def validate_missing(df, threshold=0.05):
    """Vérifie que le pourcentage de valeurs manquantes est inférieur au seuil.
    Lève une exception si le seuil est dépassé."""
    missing_ratio = df.isnull().sum().sum() / df.size
    if missing_ratio > threshold:
        raise ValueError(f"Missing ratio {missing_ratio:.2%} exceeds threshold {threshold:.0%}")
    return True

def validate_data_quality(df, threshold=0.05):
    """Exécute toutes les validations."""
    validate_missing(df, threshold)
    # Ajoutez d'autres contrôles : valeurs négatives pour Amount, etc.
    if (df['Amount'] <= 0).any():
        raise ValueError("Amount has non-positive values")
    return True

if __name__ == "__main__":
    # Exemple d'utilisation en ligne de commande
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--threshold', type=float, default=0.05, help='Missing threshold')
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    try:
        validate_data_quality(df, args.threshold)
        print("Data quality checks passed")
        sys.exit(0)
    except Exception as e:
        print(f"Data quality failed: {e}")
        sys.exit(1)