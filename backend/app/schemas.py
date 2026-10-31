from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorPayload(BaseModel):
    code: str
    message: str


class StandardError(BaseModel):
    error: ErrorPayload


class UploadResponse(BaseModel):
    session_id: str
    rows: int
    date_range: Dict[str, Optional[str]]
    schema: Dict[str, List[str]]
    duplicates_dropped: int
    warnings: List[str] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    current_balance: float
    inflow_30d: float
    outflow_30d: float
    net_30d: float
    burn_rate: float
    runway_days: float
    category_breakdown: Dict[str, float]
    monthly_series: List[Dict[str, Any]]
    risk_status: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    amount: Optional[float] = None
    safety_buffer: Optional[float] = None


class AskResponse(BaseModel):
    verdict: str
    numbers: Dict[str, float]
    explanation: str
    explanation_source: Literal["llm", "template"]
    guardrail_passed: bool


class ScenarioItem(BaseModel):
    date: str
    amount: float
    direction: Literal["inflow", "outflow"]


class ScenarioResponse(BaseModel):
    new_lowest_balance: float
    lowest_balance_date: str
    base_lowest_balance: float
    adjusted_forecast: List[Dict[str, Any]]


class TransactionUpdateRequest(BaseModel):
    category: str = Field(..., min_length=1)


class TransactionListResponse(BaseModel):
    items: List[Dict[str, Any]]
    page: int
    page_size: int
    total: int


class SessionCreated(BaseModel):
    session_id: str
    rows: int
    created: str
    source: str
