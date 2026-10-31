from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from backend.app.config import MAX_SESSION_COUNT, SESSION_TTL_SECONDS


class SessionStore:
    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS, max_sessions: int = MAX_SESSION_COUNT):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _cleanup(self) -> None:
        now = self._now()
        expired = [sid for sid, entry in self._sessions.items() if entry["expires_at"] <= now]
        for sid in expired:
            del self._sessions[sid]

    def _prune_oldest(self) -> None:
        while len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)

    def create(self, *, rows: list[dict], source: str) -> Dict[str, Any]:
        self._cleanup()
        self._prune_oldest()
        session_id = uuid4().hex[:8]
        now = self._now()
        payload = {
            "id": session_id,
            "source": source,
            "rows": rows,
            "created_at": now,
            "expires_at": now + timedelta(seconds=self.ttl_seconds),
        }
        self._sessions[session_id] = payload
        return payload

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._cleanup()
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        entry["expires_at"] = self._now() + timedelta(seconds=self.ttl_seconds)
        return entry

    def update_row_category(self, session_id: str, tx_id: int, category: str) -> bool:
        entry = self.get(session_id)
        if not entry:
            return False
        rows = entry["rows"]
        if 0 <= tx_id < len(rows):
            rows[tx_id]["category"] = category
            return True
        return False

    def list(self) -> list[Dict[str, Any]]:
        self._cleanup()
        return list(self._sessions.values())
