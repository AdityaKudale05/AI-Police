from datetime import datetime
from fastapi import FastAPI, HTTPException

from app.db import Transaction, get_session, init_db
from app.ml import ExpenseMLEngine
from app.schemas import (
    AnomalyPoint,
    AnomalyRequest,
    AnomalyResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TransactionCreate,
    TransactionListResponse,
    TransactionResponse,
)

app = FastAPI(title="Smart Expense Categorization & Anomaly Detector")
ml_engine = ExpenseMLEngine()


@app.on_event("startup")
def startup() -> None:
    init_db()
    ml_engine.predict_category("ZOMATO")
    ml_engine.detect_anomalies([120.0, 450.0, 900.0, 15000.0, 300.0])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def train(request: TrainRequest) -> dict[str, int]:
    descriptions = [
        ml_engine.build_contextual_text(
            s.description,
            {
                "merchant": s.merchant,
                "counterparty": s.counterparty,
                "channel": s.channel,
                "location": s.location,
                "notes": s.notes,
            },
        )
        for s in request.samples
    ]
    categories = [s.category for s in request.samples]
    ml_engine.train_classifier(descriptions, categories)
    return {"trained_samples": len(request.samples)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    category = ml_engine.predict_category(
        request.description,
        {
            "merchant": request.merchant,
            "counterparty": request.counterparty,
            "channel": request.channel,
            "location": request.location,
            "notes": request.notes,
        },
    )
    return PredictResponse(description=request.description, category=category)


@app.post("/anomalies", response_model=AnomalyResponse)
def anomalies(request: AnomalyRequest) -> AnomalyResponse:
    if not request.amounts:
        raise HTTPException(status_code=400, detail="amounts cannot be empty")
    flags, scores = ml_engine.detect_anomalies(request.amounts)
    points = [
        AnomalyPoint(amount=a, is_anomaly=flag, anomaly_score=score)
        for a, flag, score in zip(request.amounts, flags, scores)
    ]
    return AnomalyResponse(points=points)


@app.post("/transactions", response_model=TransactionResponse)
def create_transaction(payload: TransactionCreate) -> TransactionResponse:
    context = {
        "merchant": payload.merchant,
        "counterparty": payload.counterparty,
        "channel": payload.channel,
        "location": payload.location,
        "notes": payload.notes,
    }

    category = ml_engine.predict_category(payload.description, context)

    event_time = payload.timestamp or datetime.utcnow()

    with get_session() as session:
        history = session.query(Transaction).order_by(Transaction.timestamp.desc()).limit(100).all()

        context_rows = [
            {
                "amount": item.amount,
                "timestamp": item.timestamp,
                "description": item.description,
            }
            for item in history
        ]
        context_rows.append(
            {
                "amount": payload.amount,
                "timestamp": event_time,
                "description": payload.description,
            }
        )

        flags, scores = ml_engine.detect_contextual_anomalies(context_rows)

        record = Transaction(
            description=payload.description,
            amount=payload.amount,
            category=category,
            is_anomaly=flags[-1],
            anomaly_score=scores[-1],
            timestamp=event_time,
        )

        session.add(record)
        session.commit()
        session.refresh(record)

    return TransactionResponse(
        id=record.id,
        description=record.description,
        amount=record.amount,
        category=record.category,
        is_anomaly=record.is_anomaly,
        anomaly_score=record.anomaly_score,
        timestamp=record.timestamp,
    )


@app.get("/transactions", response_model=TransactionListResponse)
def list_transactions() -> TransactionListResponse:
    with get_session() as session:
        rows = session.query(Transaction).order_by(Transaction.timestamp.desc()).all()

    return TransactionListResponse(
        items=[
            TransactionResponse(
                id=row.id,
                description=row.description,
                amount=row.amount,
                category=row.category,
                is_anomaly=row.is_anomaly,
                anomaly_score=row.anomaly_score,
                timestamp=row.timestamp,
            )
            for row in rows
        ]
    )
