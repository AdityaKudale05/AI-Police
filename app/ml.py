from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_DIR = Path("models")
CLASSIFIER_PATH = MODEL_DIR / "category_classifier.joblib"
ANOMALY_PATH = MODEL_DIR / "amount_anomaly.joblib"


class ExpenseMLEngine:
    def __init__(self) -> None:
        self.classifier: Pipeline | None = None
        self.anomaly_model: IsolationForest | None = None
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.llm_api_key = os.getenv("OPENAI_API_KEY")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def train_classifier(self, descriptions: list[str], categories: list[str]) -> None:
        if len(descriptions) < 2:
            raise ValueError("Need at least 2 training samples")
        self.classifier = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, strip_accents="unicode")),
                ("clf", LogisticRegression(max_iter=600, class_weight="balanced")),
            ]
        )
        self.classifier.fit(descriptions, categories)
        joblib.dump(self.classifier, CLASSIFIER_PATH)

    def predict_category(self, description: str, context: dict[str, Any] | None = None) -> str:
        if self.classifier is None:
            self.classifier = self._load_or_default_classifier()

        combined_text = self._build_contextual_text(description, context)
        baseline = str(self.classifier.predict([combined_text])[0])

        llm_guess = self._llm_semantic_guess(description=description, context=context, baseline=baseline)
        return llm_guess or baseline

    def train_anomaly_model(self, amounts: list[float]) -> None:
        if len(amounts) < 5:
            raise ValueError("Need at least 5 amounts to train anomaly model")

        pseudo_context = [
            {"amount": amt, "timestamp": datetime.utcnow(), "description": "historical"}
            for amt in amounts
        ]
        vectors = self._build_anomaly_vectors(pseudo_context)

        self.anomaly_model = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_estimators=250,
        )
        self.anomaly_model.fit(vectors)
        joblib.dump(self.anomaly_model, ANOMALY_PATH)

    def detect_anomalies(self, amounts: list[float]) -> tuple[list[bool], list[float]]:
        context_rows = [{"amount": amount, "timestamp": datetime.utcnow(), "description": "standalone"} for amount in amounts]
        return self.detect_contextual_anomalies(context_rows)

    def detect_contextual_anomalies(
        self,
        transactions: list[dict[str, Any]],
    ) -> tuple[list[bool], list[float]]:
        if self.anomaly_model is None:
            self.anomaly_model = self._load_or_default_anomaly_model()

        vectors = self._build_anomaly_vectors(transactions)
        predictions = self.anomaly_model.predict(vectors)
        scores = self.anomaly_model.decision_function(vectors)
        is_anomaly = [p == -1 for p in predictions]
        return is_anomaly, [float(s) for s in scores]

    def build_contextual_text(self, description: str, context: dict[str, Any] | None = None) -> str:
        return self._build_contextual_text(description, context)

    def _build_contextual_text(self, description: str, context: dict[str, Any] | None) -> str:
        context = context or {}
        merchant = context.get("merchant", "")
        counterparty = context.get("counterparty", "")
        channel = context.get("channel", "")
        location = context.get("location", "")
        notes = context.get("notes", "")

        return (
            f"description={description} "
            f"merchant={merchant} counterparty={counterparty} "
            f"channel={channel} location={location} notes={notes}"
        ).strip()

    def _build_anomaly_vectors(self, transactions: list[dict[str, Any]]) -> np.ndarray:
        if not transactions:
            return np.zeros((0, 6), dtype=float)

        vectors: list[list[float]] = []
        for tx in transactions:
            amount = float(tx.get("amount", 0.0))
            ts = tx.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts is None:
                ts = datetime.utcnow()
            if not isinstance(ts, datetime):
                ts = datetime.utcnow()

            description = str(tx.get("description", "")).upper()
            hour = ts.hour
            day = ts.weekday()
            is_night = 1.0 if hour < 6 or hour > 22 else 0.0
            token_entropy = len(set(description.split())) / max(len(description.split()), 1)

            vectors.append(
                [
                    amount,
                    np.log1p(max(amount, 0.0)),
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * day / 7),
                    is_night,
                    token_entropy,
                ]
            )

        return np.array(vectors, dtype=float)

    def _llm_semantic_guess(
        self,
        description: str,
        context: dict[str, Any] | None,
        baseline: str,
    ) -> str | None:
        if not self.llm_api_key or self.classifier is None:
            return None

        try:
            categories = list(map(str, self.classifier.classes_))
            payload = {
                "model": self.llm_model,
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Classify bank transactions into one label. Return strict JSON: {\"category\":\"...\"}",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Description: {description}\n"
                                    f"Context: {json.dumps(context or {}, default=str)}\n"
                                    f"Allowed categories: {categories}\n"
                                    f"Baseline category: {baseline}"
                                ),
                            }
                        ],
                    },
                ],
                "text": {"format": {"type": "json_object"}},
                "temperature": 0,
            }
            response = httpx.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self.llm_api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=8.0,
            )
            response.raise_for_status()
            data = response.json()
            text = self._extract_text_output(data)
            if not text:
                return None
            parsed = json.loads(text)
            category = str(parsed.get("category", "")).strip()
            if category in categories:
                return category
        except Exception:
            return None
        return None

    def _extract_text_output(self, response_json: dict[str, Any]) -> str | None:
        for block in response_json.get("output", []):
            for content in block.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text")
        return None

    def _load_or_default_classifier(self) -> Pipeline:
        if CLASSIFIER_PATH.exists():
            return joblib.load(CLASSIFIER_PATH)

        starter = [
            ("description=ZOMATO ORDER 3345 merchant=ZOMATO channel=CARD location=BANGALORE", "Food"),
            ("description=UBER TRIP 882 merchant=UBER channel=UPI", "Transport"),
            ("description=NETFLIX SUBSCRIPTION merchant=NETFLIX notes=monthly plan", "Entertainment"),
            ("description=BIG BAZAAR merchant=BIG BAZAAR", "Groceries"),
            ("description=APOLLO PHARMACY merchant=APOLLO", "Healthcare"),
            ("description=AMAZON PAYMENT merchant=AMAZON", "Shopping"),
            ("description=SALARY CREDIT counterparty=EMPLOYER", "Income"),
        ]
        descriptions, categories = zip(*starter)
        self.train_classifier(list(descriptions), list(categories))
        return self.classifier  # type: ignore[return-value]

    def _load_or_default_anomaly_model(self) -> IsolationForest:
        if ANOMALY_PATH.exists():
            return joblib.load(ANOMALY_PATH)

        default_amounts = [120, 430, 230, 199, 510, 45, 88, 900, 130, 210, 480, 70, 1200]
        self.train_anomaly_model(default_amounts)
        return self.anomaly_model  # type: ignore[return-value]
