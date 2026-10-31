from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    return {"status": "ok"}
