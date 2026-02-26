from __future__ import annotations

from pathlib import Path
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
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def train_classifier(self, descriptions: list[str], categories: list[str]) -> None:
        if len(descriptions) < 2:
            raise ValueError("Need at least 2 training samples")
        self.classifier = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=400)),
            ]
        )
        self.classifier.fit(descriptions, categories)
        joblib.dump(self.classifier, CLASSIFIER_PATH)

    def predict_category(self, description: str) -> str:
        if self.classifier is None:
            self.classifier = self._load_or_default_classifier()
        return str(self.classifier.predict([description])[0])

    def train_anomaly_model(self, amounts: list[float]) -> None:
        if len(amounts) < 5:
            raise ValueError("Need at least 5 amounts to train anomaly model")
        array = np.array(amounts, dtype=float).reshape(-1, 1)
        self.anomaly_model = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_estimators=200,
        )
        self.anomaly_model.fit(array)
        joblib.dump(self.anomaly_model, ANOMALY_PATH)

    def detect_anomalies(self, amounts: list[float]) -> tuple[list[bool], list[float]]:
        if self.anomaly_model is None:
            self.anomaly_model = self._load_or_default_anomaly_model()
        array = np.array(amounts, dtype=float).reshape(-1, 1)
        predictions = self.anomaly_model.predict(array)
        scores = self.anomaly_model.decision_function(array)
        is_anomaly = [p == -1 for p in predictions]
        return is_anomaly, [float(s) for s in scores]

    def _load_or_default_classifier(self) -> Pipeline:
        if CLASSIFIER_PATH.exists():
            return joblib.load(CLASSIFIER_PATH)

        starter = [
            ("ZOMATO ORDER 3345", "Food"),
            ("UBER TRIP 882", "Transport"),
            ("NETFLIX SUBSCRIPTION", "Entertainment"),
            ("BIG BAZAAR", "Groceries"),
            ("APOLLO PHARMACY", "Healthcare"),
            ("AMAZON PAYMENT", "Shopping"),
            ("SALARY CREDIT", "Income"),
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
