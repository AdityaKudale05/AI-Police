from datetime import datetime
from pydantic import BaseModel, Field


class LabeledTransaction(BaseModel):
    description: str = Field(..., examples=["ZOMATO 3245"])
    category: str = Field(..., examples=["Food"])
    merchant: str | None = None
    counterparty: str | None = None
    channel: str | None = None
    location: str | None = None
    notes: str | None = None


class TrainRequest(BaseModel):
    samples: list[LabeledTransaction]


class PredictRequest(BaseModel):
    description: str
    merchant: str | None = None
    counterparty: str | None = None
    channel: str | None = None
    location: str | None = None
    notes: str | None = None


class PredictResponse(BaseModel):
    description: str
    category: str


class AnomalyRequest(BaseModel):
    amounts: list[float]


class AnomalyPoint(BaseModel):
    amount: float
    is_anomaly: bool
    anomaly_score: float


class AnomalyResponse(BaseModel):
    points: list[AnomalyPoint]


class TransactionCreate(BaseModel):
    description: str
    amount: float
    timestamp: datetime | None = None
    merchant: str | None = None
    counterparty: str | None = None
    channel: str | None = None
    location: str | None = None
    notes: str | None = None


class TransactionResponse(BaseModel):
    id: int
    description: str
    amount: float
    category: str
    is_anomaly: bool
    anomaly_score: float
    timestamp: datetime


class TransactionListResponse(BaseModel):
    items: list[TransactionResponse]
