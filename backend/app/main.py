import io
import logging
from datetime import datetime
from typing import Annotated, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.app.config import ALLOWED_ORIGINS, API_TITLE, API_VERSION, LOG_LEVEL, MAX_SESSION_COUNT, SESSION_TTL_SECONDS, UPLOAD_MAX_BYTES, UPLOAD_MAX_FILES
from backend.app.schemas import AskRequest, AskResponse, ScenarioItem, ScenarioResponse, SummaryResponse, TransactionListResponse, TransactionUpdateRequest, UploadResponse
from backend.app.store import SessionStore
from src.ingestion.loader import load_files

from backend.app.services.analysis import (
    build_anomalies,
    build_ask_result,
    build_forecast,
    build_scenario,
    build_summary,
    export_csv,
    export_pdf,
    generate_sample_data,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('cashflow_api')
limiter = Limiter(key_func=get_remote_address)
store = SessionStore(ttl_seconds=SESSION_TTL_SECONDS, max_sessions=MAX_SESSION_COUNT)

app = FastAPI(title=API_TITLE, version=API_VERSION, docs_url="/docs")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": "http_error", "message": exc.detail}})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception('Unhandled error', exc_info=exc)
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "An unexpected server error occurred."}})


@app.get('/api/health')
async def health():
    return {'status': 'ok', 'service': API_TITLE, 'timestamp': datetime.utcnow().isoformat() + 'Z'}


@app.post('/api/upload')
@limiter.limit('10/minute')
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    if len(files) > UPLOAD_MAX_FILES:
        raise HTTPException(status_code=400, detail=f'Maximum of {UPLOAD_MAX_FILES} files allowed.')
    if not files:
        raise HTTPException(status_code=400, detail='At least one file is required.')
    entries = []
    file_schemas = {}
    warnings = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail='Each uploaded file requires a filename.')
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail=f'Unsupported file type: {file.filename}')
        content = await file.read()
        if len(content) > UPLOAD_MAX_BYTES:
            raise HTTPException(status_code=400, detail=f'File {file.filename} exceeds the 5 MB limit.')
        if not content.strip():
            raise HTTPException(status_code=400, detail=f'File {file.filename} is empty.')
        buffer = io.BytesIO(content)
        buffer.name = file.filename
        entries.append(buffer)
        file_schemas[file.filename] = []
    try:
        df = load_files(entries)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if df.empty:
        raise HTTPException(status_code=400, detail='No valid rows were found in the uploaded files.')
    for key in file_schemas:
        file_schemas[key] = list(df.columns)
    duplicates_dropped = int(df.attrs.get('duplicates_dropped', 0))
    if duplicates_dropped:
        warnings.append(f'Removed {duplicates_dropped} duplicate transactions.')
    session = store.create(rows=df.to_dict(orient='records'), source='upload')
    dates = sorted(pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').unique().tolist())
    return UploadResponse(
        session_id=session['id'],
        rows=int(len(df)),
        date_range={'start': dates[0] if dates else None, 'end': dates[-1] if dates else None},
        schema=file_schemas,
        duplicates_dropped=duplicates_dropped,
        warnings=warnings,
    ).model_dump()


@app.post('/api/sessions/sample')
async def create_sample_session():
    rows = generate_sample_data()
    if not rows:
        raise HTTPException(status_code=500, detail='Sample data generation failed.')
    session = store.create(rows=rows, source='sample')
    return {'session_id': session['id'], 'rows': len(rows)}


@app.get('/api/sessions/{session_id}/summary')
async def session_summary(session_id: str):
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail='Session not found.')
    df = pd.DataFrame(entry['rows'])
    if df.empty:
        return SummaryResponse(current_balance=0.0, inflow_30d=0.0, outflow_30d=0.0, net_30d=0.0, burn_rate=0.0, runway_days=0.0, category_breakdown={}, monthly_series=[], risk_status='SAFE').model_dump()
    df['date'] = pd.to_datetime(df['date'])
    if 'current_balance' not in df.attrs:
        df.attrs['current_balance'] = float(df['amount'].sum())
    summary = build_summary(df)
    return SummaryResponse(**summary).model_dump()


@app.get('/api/sessions/{session_id}/transactions')
async def list_transactions(session_id: str, page: int = 1, page_size: int = 25, search: str | None = None, category: str | None = None, direction: str | None = None, from_date: str | None = None, to_date: str | None = None):
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail='Session not found.')
    df = pd.DataFrame(entry['rows'])
    if df.empty:
        return TransactionListResponse(items=[], page=page, page_size=page_size, total=0).model_dump()
    df['date'] = pd.to_datetime(df['date'])
    if search:
        df = df[df['description'].str.contains(search, case=False, na=False)]
    if category:
        df = df[df['category'].fillna('').str.contains(category, case=False, na=False)]
    if direction:
        df = df[df['direction'].str.lower() == direction.lower()]
    if from_date:
        df = df[df['date'] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df['date'] <= pd.to_datetime(to_date)]
    total = len(df)
    page = max(1, page)
    page_size = max(1, page_size)
    start = (page - 1) * page_size
    end = start + page_size
    items = df.iloc[start:end].to_dict(orient='records')
    for index, row in enumerate(items):
        row['id'] = start + index
    return TransactionListResponse(items=items, page=page, page_size=page_size, total=total).model_dump()


@app.patch('/api/sessions/{session_id}/transactions/{tx_id}')
async def patch_transaction(session_id: str, tx_id: int, payload: TransactionUpdateRequest):
    if not store.update_row_category(session_id, tx_id, payload.category):
        raise HTTPException(status_code=404, detail='Transaction not found.')
    return {'ok': True, 'tx_id': tx_id, 'category': payload.category}


@app.get('/api/sessions/{session_id}/forecast')
async def get_forecast(session_id: str, horizon: int = Query(30, ge=30, le=90), safety_buffer: float = 50000.0, include_receivables: bool = False):
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail='Session not found.')
    df = pd.DataFrame(entry['rows'])
    df['date'] = pd.to_datetime(df['date'])
    result = build_forecast(df, horizon=horizon, safety_buffer=safety_buffer, include_receivables=include_receivables)
    return result


@app.get('/api/sessions/{session_id}/anomalies')
async def get_anomalies(session_id: str):
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail='Session not found.')
    df = pd.DataFrame(entry['rows'])
    if df.empty:
        return []
    df['date'] = pd.to_datetime(df['date'])
    return build_anomalies(df)


@app.post('/api/sessions/{session_id}/ask')
@limiter.limit('10/minute')
async def ask_advisor(request: Request, session_id: str, payload: AskRequest):
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail='Session not found.')
    df = pd.DataFrame(entry['rows'])
    df['date'] = pd.to_datetime(df['date'])
    result = build_ask_result(df, payload.question, amount=payload.amount, safety_buffer=payload.safety_buffer)
    return AskResponse(**result).model_dump()


@app.post('/api/sessions/{session_id}/scenario')
async def scenario(request: Request, session_id: str, payload: list[ScenarioItem]):
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail='Session not found.')
    df = pd.DataFrame(entry['rows'])
    df['date'] = pd.to_datetime(df['date'])
    result = build_scenario(df, [item.model_dump() for item in payload])
    return ScenarioResponse(**result).model_dump()


@app.get('/api/sessions/{session_id}/export')
async def export_session(session_id: str, format: str = 'csv'):
    entry = store.get(session_id)
    if not entry:
        raise HTTPException(status_code=404, detail='Session not found.')
    df = pd.DataFrame(entry['rows'])
    if df.empty:
        raise HTTPException(status_code=404, detail='No data available to export.')
    df['date'] = pd.to_datetime(df['date'])
    if format == 'csv':
        csv_text = export_csv(df)
        return Response(content=csv_text, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=ledger.csv'})
    if format == 'pdf':
        pdf_bytes = export_pdf(df)
        return Response(content=pdf_bytes, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename=risk-report.pdf'})
    raise HTTPException(status_code=400, detail='Unsupported export format. Use csv or pdf.')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('backend.app.main:app', host='0.0.0.0', port=8000)
