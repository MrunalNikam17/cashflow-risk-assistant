import io

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_upload_accepts_valid_csv():
    csv_data = """date,description,amount,direction,source_file,category
2024-01-01,Invoice 1001,25000,inflow,invoice.csv,Sales Revenue
2024-01-02,Office rent,45000,outflow,expenses.csv,Rent
2024-01-03,Payroll,70000,outflow,expenses.csv,Salary
"""
    response = client.post('/api/upload', files=[('files', ('sample.csv', csv_data, 'text/csv'))])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['rows'] == 3
    assert payload['session_id']


def test_upload_rejects_wrong_type():
    response = client.post('/api/upload', files=[('files', ('bad.txt', 'hello', 'text/plain'))])
    assert response.status_code == 400


def test_upload_rejects_empty_file():
    response = client.post('/api/upload', files=[('files', ('empty.csv', b'', 'text/csv'))])
    assert response.status_code == 400


def test_session_sample_and_summary():
    sample = client.post('/api/sessions/sample')
    assert sample.status_code == 200, sample.text
    session_id = sample.json()['session_id']
    summary = client.get(f'/api/sessions/{session_id}/summary')
    assert summary.status_code == 200, summary.text
    payload = summary.json()
    assert payload['current_balance'] >= 0
    assert 'risk_status' in payload


def test_transactions_and_patch():
    sample = client.post('/api/sessions/sample')
    session_id = sample.json()['session_id']
    response = client.get(f'/api/sessions/{session_id}/transactions', params={'page': 1, 'page_size': 10})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['total'] >= 1
    tx_id = 0 if data['items'] else None
    if tx_id is not None:
        patch = client.patch(f'/api/sessions/{session_id}/transactions/{tx_id}', json={'category': 'Adjusted Category'})
        assert patch.status_code == 200, patch.text
        assert patch.json()['category'] == 'Adjusted Category'


def test_forecast_and_anomalies():
    sample = client.post('/api/sessions/sample')
    session_id = sample.json()['session_id']
    forecast = client.get(f'/api/sessions/{session_id}/forecast', params={'horizon': 30, 'safety_buffer': 50000, 'include_receivables': True})
    assert forecast.status_code == 200, forecast.text
    payload = forecast.json()
    assert payload['lowest_balance'] is not None
    assert len(payload['forecast']) >= 1
    anomalies = client.get(f'/api/sessions/{session_id}/anomalies')
    assert anomalies.status_code == 200, anomalies.text
    assert isinstance(anomalies.json(), list)


def test_ask_and_scenario():
    sample = client.post('/api/sessions/sample')
    session_id = sample.json()['session_id']
    ask = client.post(f'/api/sessions/{session_id}/ask', json={'question': 'Can I afford a purchase of ₹80,000 in 45 days?', 'amount': 80000, 'safety_buffer': 50000})
    assert ask.status_code == 200, ask.text
    answer = ask.json()
    assert answer['verdict'] in {'SAFE', 'RISKY', 'NOT RECOMMENDED'}
    scenario = client.post(f'/api/sessions/{session_id}/scenario', json=[{'date': '2030-01-01', 'amount': 15000, 'direction': 'outflow'}])
    assert scenario.status_code == 200, scenario.text
    payload = scenario.json()
    assert 'new_lowest_balance' in payload


def test_export_csv_and_pdf():
    sample = client.post('/api/sessions/sample')
    session_id = sample.json()['session_id']
    csv_resp = client.get(f'/api/sessions/{session_id}/export', params={'format': 'csv'})
    assert csv_resp.status_code == 200
    pdf_resp = client.get(f'/api/sessions/{session_id}/export', params={'format': 'pdf'})
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers['content-type'].startswith('application/pdf')


def test_unknown_session_404():
    response = client.get('/api/sessions/does-not-exist/summary')
    assert response.status_code == 404
