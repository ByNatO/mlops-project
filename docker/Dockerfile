FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système (si nécessaire)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port pour MLflow UI (si on lance le serveur)
EXPOSE 5000

# Commande par défaut (peut être surchargée)
CMD ["python", "src/models/train.py", "--config", "config.yaml"]