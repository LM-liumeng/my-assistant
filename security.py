"""Safety and governance layer.

Provides rudimentary checks and auditing to ensure that operations are
controlled and logged.  In a real assistant you would enforce fine‑grained
permissions, user confirmations and secure credential handling.  Here we
provide simple mechanisms that demonstrate how such checks can be structured.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from context.evidence_store import EvidenceStore


class SafetyLayer:
    """Implements basic safety checks, directory controls and confirmations.

    The SafetyLayer maintains a set of allowed directories for file
    operations, logs tool invocations and provides a configurable
    confirmation mechanism.  Confirmation behaviour is controlled via the
    ``AUTO_CONFIRM`` environment variable: if set to a truthy value
    ("1", "true", "yes"), confirmations are auto‑approved; otherwise
    ``confirm_action`` will return ``False`` so that the calling tool can
    request explicit user consent.
    """

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store
        # maintain a set instead of a list so that multiple tools can add
        # their own allowed directories without overwriting each other
        self._allowed_dirs: set[str] = set()

    def add_allowed_dir(self, directory: str) -> None:
        """Add a directory to the set of allowed directories for file writes."""
        abs_dir = os.path.abspath(directory)
        self._allowed_dirs.add(abs_dir)

    def is_path_allowed(self, path: str) -> bool:
        """Return True if the given path is within one of the allowed directories."""
        if not self._allowed_dirs:
            # no restrictions configured
            return True
        abs_path = os.path.abspath(path)
        for base in self._allowed_dirs:
            if abs_path.startswith(base + os.sep) or abs_path == base:
                return True
        return False

    def confirm_action(self, description: str) -> bool:
        """Decide whether to proceed with a sensitive action.

        By default this method checks the ``AUTO_CONFIRM`` environment
        variable.  If it is set to a truthy value (e.g. "1", "true",
        "yes") then confirmations are automatically approved.  Otherwise
        ``False`` is returned, indicating that the caller should
        prompt the user for explicit confirmation before proceeding.

        The method logs the fact that a confirmation was requested.
        """
        # Log the confirmation request for auditing
        try:
            self.evidence_store.log_event({"event": "confirmation_requested", "description": description})
        except Exception:
            # ignore logging errors to avoid blocking
            pass
        auto = os.environ.get("AUTO_CONFIRM", "").lower()
        return auto in {"1", "true", "yes"}

    def log_tool_call(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Record that a tool was invoked along with its parameters."""
        try:
            self.evidence_store.log_event({"event": "tool_call", "tool": tool_name, "params": params})
        except Exception:
            # ignore logging errors
            pass
