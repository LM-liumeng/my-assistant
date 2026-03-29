"""EvidenceStore: logging and auditing facility.

The evidence store records significant events and tool invocations.  In this
demo it writes JSON lines to a log file.  A real system might forward
records to a central logging service or database.  Keeping a log of all
actions enables traceability and supports safety and auditing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


class EvidenceStore:
    """Persists logs of events and tool invocations."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.log_path = os.path.join(base_dir, "evidence.log")

    def log_event(self, data: Dict[str, Any]) -> None:
        """Write a single log record to the log file with a timestamp.

        Any IO errors are caught and ignored so that logging failures do
        not interrupt the main workflow.  In a real system you might
        forward errors to a monitoring system.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **data,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
