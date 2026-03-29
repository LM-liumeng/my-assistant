"""DocumentTool: create or append to text documents.

Writes plain text files into the workspace directory.  If the file already
exists, the content will be appended.  Before writing the file the tool
checks that the path is allowed and requests confirmation via the safety
layer.
"""

from __future__ import annotations

import os
from typing import Dict, Any

from context.evidence_store import EvidenceStore
from security import SafetyLayer


class DocumentTool:
    def __init__(self, base_dir: str, evidence_store: EvidenceStore, safety: SafetyLayer) -> None:
        self.base_dir = os.path.join(base_dir, "workspace")
        os.makedirs(self.base_dir, exist_ok=True)
        self.evidence_store = evidence_store
        self.safety = safety
        # By default, restrict writes to the workspace directory.  We add
        # instead of setting the list to avoid clobbering any existing
        # allowed directories configured by other tools.
        self.safety.add_allowed_dir(self.base_dir)

    def handle_document(self, filename: str = "", content: str = "") -> Dict[str, Any]:
        """Create or append to a text document and return updated contents.

        Writes the provided ``content`` to a file within the ``workspace``
        directory.  The method checks that the file path is allowed via the
        safety layer and requests confirmation before writing.  When
        confirmation is required, it returns both the proposed filename and
        content so that the front‑end can offer a confirmation button,
        similar to the email workflow.  After successfully writing, it reads
        the entire file back and returns it under the ``display_content``
        key so that the UI can show the updated document.  The
        ``display_filename`` mirrors the filename argument for convenience.
        """
        # Require a filename
        if not filename:
            return {"error": "Filename must be specified."}
        path = os.path.join(self.base_dir, filename)
        # Ensure the path is within an allowed directory
        if not self.safety.is_path_allowed(path):
            return {"error": f"Access to path '{path}' is not allowed."}
        # Ask the safety layer for confirmation
        description = f"Write to file {path}"
        if not self.safety.confirm_action(description):
            # Return a structured payload so the UI can show a confirmation
            # button and allow the user to confirm the write explicitly.
            return {
                "message": f"Confirmation required to write to {path}. Set AUTO_CONFIRM or confirm manually.",
                "display_content": content or "",
                "display_filename": filename,
            }
        try:
            mode = "a" if os.path.exists(path) else "w"
            # Write the content to the file (append if it exists)
            with open(path, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n")
                f.write(content or "")
            # Log the write event
            try:
                self.evidence_store.log_event(
                    {"event": "document_written", "path": path, "content": content}
                )
            except Exception:
                pass
            # Read back the full file contents to display to the user
            try:
                with open(path, "r", encoding="utf-8") as f:
                    full_contents = f.read()
            except Exception:
                full_contents = content or ""
            return {
                "message": f"Content written to {path}.",
                "display_content": full_contents,
                "display_filename": filename,
            }
        except Exception as exc:
            return {"error": f"Failed to write to {path}: {exc}"}

    def read_document(self, filename: str = "") -> Dict[str, Any]:
        """Read a text document from the workspace and return its contents.

        This helper is used when the user asks to "open" or "load" a file via
        a natural‑language command.  It does not modify the file, only reads
        it after verifying that the path is allowed.
        """
        if not filename:
            return {"error": "Filename must be specified."}
        path = os.path.join(self.base_dir, filename)
        if not self.safety.is_path_allowed(path):
            return {"error": f"Access to path '{path}' is not allowed."}
        if not os.path.exists(path):
            return {"error": f"File '{filename}' does not exist."}
        try:
            with open(path, "r", encoding="utf-8") as f:
                contents = f.read()
            try:
                self.evidence_store.log_event({"event": "document_read", "path": path})
            except Exception:
                pass
            return {
                "message": f"Loaded file {path}.",
                "display_content": contents,
                "display_filename": filename,
            }
        except Exception as exc:
            return {"error": f"Failed to read from {path}: {exc}"}
