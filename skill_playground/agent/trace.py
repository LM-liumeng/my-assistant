from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TraceEntry:
    index: int
    stage: str
    message: str
    timestamp: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "stage": self.stage,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class TraceRecorder:
    def __init__(self) -> None:
        self._entries: List[TraceEntry] = []
        self._lock = Lock()

    def log(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            entry = TraceEntry(
                index=len(self._entries) + 1,
                stage=str(stage or "system"),
                message=str(message or ""),
                timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                details=dict(details or {}),
            )
            self._entries.append(entry)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [entry.to_dict() for entry in self._entries]
