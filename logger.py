from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in ("token", "authorization", "api_key", "secret")):
            return "[REDACTED]"
        if len(value) > 2000:
            return value[:2000] + "...[truncated]"
        return value
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return value


@dataclass
class TraceEvent:
    name: str
    status: str
    timestamp: str = field(default_factory=_utc_now_iso)
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "timestamp": self.timestamp,
            "data": _safe_value(self.data),
        }


@dataclass
class TraceSpan:
    name: str
    status: str = "in_progress"
    started_at: str = field(default_factory=_utc_now_iso)
    started_at_monotonic: float = field(default_factory=time.monotonic)
    finished_at: str | None = None
    duration_ms: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    children: list["TraceSpan"] = field(default_factory=list)

    def event(self, name: str, status: str = "info", **data: Any) -> None:
        self.events.append(TraceEvent(name=name, status=status, data=data))

    def child(self, name: str, **data: Any) -> "TraceSpan":
        span = TraceSpan(name=name, data=data)
        self.children.append(span)
        return span

    def finish(self, status: str = "ok", **data: Any) -> None:
        if self.finished_at is not None:
            return
        self.status = status
        self.finished_at = _utc_now_iso()
        self.duration_ms = int((time.monotonic() - self.started_at_monotonic) * 1000)
        if data:
            self.data.update(data)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "data": _safe_value(self.data),
            "events": [event.as_dict() for event in self.events],
            "children": [child.as_dict() for child in self.children],
        }


class RequestTrace:
    def __init__(self, request_id: str, metadata: dict[str, Any] | None = None):
        self.request_id = request_id
        self.metadata = metadata or {}
        self.started_at = _utc_now_iso()
        self.root = TraceSpan(name="request.lifecycle")

    def event(self, name: str, status: str = "info", **data: Any) -> None:
        self.root.event(name=name, status=status, **data)

    def span(self, name: str, **data: Any) -> TraceSpan:
        return self.root.child(name, **data)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "started_at": self.started_at,
            "metadata": _safe_value(self.metadata),
            "trace": self.root.as_dict(),
        }


class TraceStore:
    def __init__(self):
        pass

    def create(self, metadata: dict[str, Any] | None = None) -> RequestTrace:
        return RequestTrace(request_id=str(uuid.uuid4()), metadata=metadata)

    def persist(self, trace: RequestTrace) -> None:
        if trace.root.finished_at is None:
            trace.root.finish(status=trace.root.status)
        payload = trace.as_dict()
        print("TRACE_START")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("TRACE_END")
