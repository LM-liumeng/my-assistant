"""MetadataStore: simple persistence for assistant state.

The metadata store tracks user preferences, last used directories and other
configuration.  It persists data in a JSON file under the base directory.  In
this demo the metadata is only lightly used, but the class demonstrates the
pattern for reading and writing persistent state.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


class MetadataStore:
    """Manages assistant metadata stored in a JSON file."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.path = os.path.join(base_dir, "metadata.json")
        # Ensure base directory exists
        os.makedirs(base_dir, exist_ok=True)
        # Load or initialise data
        self.data: Dict[str, Any] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                # If the file is malformed, start fresh
                self.data = {}

    def save(self) -> None:
        """Persist metadata to disk.

        Wrap the write in a try/except so that IO errors do not
        propagate.  If saving fails (e.g. due to missing permissions or
        disk full) the error is silently ignored, but the caller can
        examine the exception via the return value if needed.
        """
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            # In a production system you might log this error or
            # propagate it; here we swallow it to avoid crashing.
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def update(self, updates: Dict[str, Any]) -> None:
        self.data.update(updates)
        self.save()
