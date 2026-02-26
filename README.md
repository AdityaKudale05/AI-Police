# Smart Expense Categorization & Anomaly Detector 💳🤖

A production-ready starter backend for automatic transaction categorization and unusual spending detection.

## What it does
- **Transaction categorization** using **TF-IDF + Logistic Regression**.
- **Anomaly detection** for unusual amounts using **Isolation Forest**.
- API to train, predict, detect anomalies, and persist transactions.
- Containerized with Docker + Kubernetes manifest.

## Tech Stack
- **ML:** scikit-learn
- **API:** FastAPI
- **Database:** SQLite (easy local dev; can be swapped for PostgreSQL)
- **Deployment:** Docker + K8s

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: `http://localhost:8000/docs`

## Main Endpoints
- `GET /health`
- `POST /train`
- `POST /predict`
- `POST /anomalies`
- `POST /transactions`
- `GET /transactions`

## Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"description":"ZOMATO 3245"}'
```

## Tests
```bash
pytest -q
```

## Docker
```bash
docker build -t expense-ai:latest .
docker run -p 8000:8000 expense-ai:latest
```

## Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

## Create ZIP package
```bash
./scripts/create_zip.sh
```
This generates `release/AI-Police.zip` from tracked files at `HEAD`.

## Future upgrades
- Replace Logistic Regression with fine-tuned BERT for richer text understanding.
- Add merchant embeddings + temporal features for stronger anomaly detection.
- Move DB URL to environment variables and use managed PostgreSQL in production.
