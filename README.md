# MLOps Project

A fraud detection project for bank transactions using a simple MLOps pipeline: data ingestion, preprocessing, model training, MLflow tracking, and a prediction API service.

## Project structure

- `config.yaml`: central pipeline configuration (data, preprocessing, model, MLflow).
- `requirements.txt`: Python dependencies for the project.
- `data/raw/creditcard.csv`: raw dataset (bank fraud).
- `data/processed/`: processed datasets saved as (`X_train.csv`, `X_val.csv`, `X_test.csv`, `y_train.csv`, `y_val.csv`, `y_test.csv`).
- `mlruns/`: local MLflow experiment tracking.
- `mlartifacts/`: model artifact and version archives.
- `docker/`: Docker files for running MLflow and the pipeline in containers.
- `src/`: project source code.
  - `src/data/load_data.py`: data loading utilities.
  - `src/preprocessing/preprocess.py`: preprocessing, train/val/test split, and standardization.
  - `src/data/validate.py`: data quality checks.
  - `src/models/train.py`: model training with MLflow, metrics calculation, and saving `model.joblib`.
  - `src/api/app.py`: FastAPI app exposing fraud prediction.
- `test/test_validation.py`: unit tests for data validation.

## Features

- Data preprocessing with train/val/test split and standardization of `Amount` and `Time`.
- XGBoost model training (configurable via `config.yaml`) with MLflow tracking.
- Metrics calculation: ROC-AUC, PR-AUC, F1, precision, and recall.
- Model export to `model.joblib` and MLflow artifact generation.
- FastAPI prediction service and health check endpoint.
- Docker support via `Dockerfile` and `docker-compose.yml` for running MLflow and the pipeline.

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place the raw dataset file at `data/raw/creditcard.csv`.

## Preprocessing

Run preprocessing with:

```powershell
python src/preprocessing/preprocess.py
```

This will:
- load `data/raw/creditcard.csv`
- split data into train/validation/test sets
- standardize `Amount` and `Time`
- save processed files to `data/processed/`
- save the scaler to `data/processed/scaler.pkl`

## Training

Run model training with:

```powershell
python src/models/train.py --config config.yaml
```

The script will:
- read `config.yaml`
- load the processed data files
- train an XGBoost model or another configured classifier
- compute and log validation metrics
- generate MLflow artifacts (`pr_curve.png`, `confusion_matrix.png`)
- save the model locally as `model.joblib`

## API deployment

Start the FastAPI app with Uvicorn:

```powershell
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:
- `POST /predict`: fraud probability prediction
- `GET /health`: service health check

Example JSON payload:

```json
{
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "V3": 2.53634673796914,
  "V4": 1.37815522427443,
  "V5": -0.338320769942518,
  "V6": 0.462387777762292,
  "V7": 0.239598554061257,
  "V8": 0.0986979012610507,
  "V9": 0.363786969611213,
  "V10": 0.0907941719789316,
  "V11": -0.551599533260813,
  "V12": -0.617800855762348,
  "V13": -0.991389847235408,
  "V14": -0.311169353699879,
  "V15": 1.46817697209427,
  "V16": -0.470400525259478,
  "V17": 0.207971241929242,
  "V18": 0.0257905801985591,
  "V19": 0.403992960255733,
  "V20": 0.251412098239705,
  "V21": -0.018306777944153,
  "V22": 0.277837575558899,
  "V23": -0.110473910188767,
  "V24": 0.0669280759146731,
  "V25": 0.128539358273528,
  "V26": -0.189114843888824,
  "V27": 0.133558376740387,
  "V28": -0.0210530534538215,
  "Amount": 149.62,
  "Time": 0.0
}
```

## MLflow

This project uses MLflow for experiment tracking.

To run MLflow locally outside Docker:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlruns
```

If using Docker Compose from the `docker/` folder:

```powershell
docker-compose up --build
```

MLflow will be available at `http://localhost:5000`.

## Tests

Run unit tests:

```powershell
pytest
```

## Customization

Edit `config.yaml` to adjust:
- data paths
- validation/test split sizes
- model type and hyperparameters
- decision threshold
- MLflow URI

## Notes

- `src/models/predict.py` is empty in this repository; prediction is currently implemented in `src/api/app.py`.
- `mlartifacts/` contains model versions and artifacts useful for release management.
- The pipeline is designed to be extensible with monitoring, data drift detection, automated validation, and CI/CD.

***

