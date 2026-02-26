from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        res = client.get('/health')
        assert res.status_code == 200
        assert res.json()['status'] == 'ok'


def test_predict_and_anomaly() -> None:
    with TestClient(app) as client:
        pred = client.post('/predict', json={'description': 'ZOMATO ORDER 100'})
        assert pred.status_code == 200
        assert 'category' in pred.json()

        anom = client.post('/anomalies', json={'amounts': [120, 130, 200, 5000]})
        assert anom.status_code == 200
        assert len(anom.json()['points']) == 4


def test_transaction_persistence() -> None:
    with TestClient(app) as client:
        created = client.post('/transactions', json={'description': 'UBER TRIP', 'amount': 300})
        assert created.status_code == 200
        payload = created.json()
        assert payload['id'] >= 1

        listed = client.get('/transactions')
        assert listed.status_code == 200
        assert len(listed.json()['items']) >= 1
