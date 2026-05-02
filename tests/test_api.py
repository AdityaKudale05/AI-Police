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


def test_transaction_search_filters_and_pagination() -> None:
    with TestClient(app) as client:
        client.post('/transactions', json={'description': 'ZOMATO ORDER', 'amount': 150})
        client.post('/transactions', json={'description': 'AMAZON PAYMENT', 'amount': 1300})

        filtered = client.get('/transactions/search', params={'category': 'Food', 'max_amount': 300})
        assert filtered.status_code == 200
        payload = filtered.json()
        assert payload['total'] >= 1
        assert payload['limit'] == 50
        assert payload['offset'] == 0
        assert all(item['category'] == 'Food' for item in payload['items'])
        assert all(item['amount'] <= 300 for item in payload['items'])

        paged = client.get('/transactions/search', params={'limit': 1, 'offset': 0})
        assert paged.status_code == 200
        assert len(paged.json()['items']) == 1


def test_transaction_search_validation_errors() -> None:
    with TestClient(app) as client:
        bad_amount_range = client.get('/transactions/search', params={'min_amount': 500, 'max_amount': 100})
        assert bad_amount_range.status_code == 400
        assert bad_amount_range.json()['detail'] == 'min_amount cannot be greater than max_amount'

        bad_time_range = client.get(
            '/transactions/search',
            params={
                'start_time': '2026-01-03T00:00:00',
                'end_time': '2026-01-01T00:00:00',
            },
        )
        assert bad_time_range.status_code == 400
        assert bad_time_range.json()['detail'] == 'start_time cannot be after end_time'
