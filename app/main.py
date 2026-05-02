from datetime import datetime
from fastapi import FastAPI, HTTPException, Query

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
    TransactionSearchResponse,
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
    descriptions = [s.description for s in request.samples]
    categories = [s.category for s in request.samples]
    ml_engine.train_classifier(descriptions, categories)
    return {"trained_samples": len(request.samples)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    category = ml_engine.predict_category(request.description)
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
    category = ml_engine.predict_category(payload.description)
    flags, scores = ml_engine.detect_anomalies([payload.amount])

    record = Transaction(
        description=payload.description,
        amount=payload.amount,
        category=category,
        is_anomaly=flags[0],
        anomaly_score=scores[0],
        timestamp=payload.timestamp or datetime.utcnow(),
    )

    with get_session() as session:
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


@app.get("/transactions/search", response_model=TransactionSearchResponse)
def search_transactions(
    category: str | None = Query(default=None),
    min_amount: float | None = Query(default=None, ge=0),
    max_amount: float | None = Query(default=None, ge=0),
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> TransactionSearchResponse:
    if min_amount is not None and max_amount is not None and min_amount > max_amount:
        raise HTTPException(status_code=400, detail="min_amount cannot be greater than max_amount")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise HTTPException(status_code=400, detail="start_time cannot be after end_time")

    with get_session() as session:
        query = session.query(Transaction)

        if category:
            query = query.filter(Transaction.category == category)
        if min_amount is not None:
            query = query.filter(Transaction.amount >= min_amount)
        if max_amount is not None:
            query = query.filter(Transaction.amount <= max_amount)
        if start_time is not None:
            query = query.filter(Transaction.timestamp >= start_time)
        if end_time is not None:
            query = query.filter(Transaction.timestamp <= end_time)

        total = query.count()
        rows = (
            query.order_by(Transaction.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    return TransactionSearchResponse(
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
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
