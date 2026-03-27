# Smart Expense Categorization & Anomaly Detector 💳🤖

A production-ready starter backend for automatic transaction categorization and unusual spending detection.

## What it does
- **Hybrid categorization** using contextual TF-IDF + Logistic Regression with optional LLM semantic refinement.
- **Context-aware anomaly detection** using Isolation Forest over amount + time + text entropy features.
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
- `POST /predict` (accepts optional context: merchant, channel, location, notes)
- `POST /anomalies`
- `POST /transactions` (runs contextual anomaly scoring using recent history)
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

## Semantic + context behavior
- If `OPENAI_API_KEY` is set, `/predict` can use an LLM to refine the baseline category while constraining outputs to known labels.
- Without an API key, the service falls back to fast local classification only.
- `/transactions` scores anomalies against the latest ~100 historical records to reduce context misses from single-point checks.

## Future upgrades
- Replace fallback linear model with fine-tuned BERT for richer text understanding.
- Add account-level graph features and seasonality decomposition for stronger anomaly detection.
- Move DB URL to environment variables and use managed PostgreSQL in production.
