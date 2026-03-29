"""FileSearchTool: search for files on disk.

Uses the retrieval store to find files by name.  The tool logs its
invocation via the safety layer.  The response is returned as a
human‑readable message.
"""

from __future__ import annotations

from typing import Dict, Any

from context.retrieval_store import RetrievalStore
from context.evidence_store import EvidenceStore
from security import SafetyLayer


class FileSearchTool:
    def __init__(self, retrieval_store: RetrievalStore, evidence_store: EvidenceStore, safety: SafetyLayer) -> None:
        self.retrieval_store = retrieval_store
        self.evidence_store = evidence_store
        self.safety = safety

    def search(self, query: str) -> Dict[str, Any]:
        """Search for files by name.

        If the query is empty or only whitespace, an error message is
        returned asking the user to provide a search term.  Otherwise
        the retrieval store is consulted and a summary message is
        constructed.  The search invocation is logged via the safety
        layer.
        """
        self.safety.log_tool_call("file_search", {"query": query})
        if not query or not query.strip():
            return {"error": "Search query cannot be empty."}
        results = self.retrieval_store.search_files(query)
        if not results:
            message = f"No files found matching '{query}'."
            return {"message": message, "files": []}
        # Build a summary string listing up to 5 results
        file_list = "\n".join(f"- {path}" for path in results[:5])
        if len(results) > 5:
            file_list += f"\n(and {len(results) - 5} more)"
        message = f"Found {len(results)} file(s) matching '{query}':\n{file_list}"
        # Provide display content of all result paths to allow the front end to show them
        display = "\n".join(results)
        return {"message": message, "files": results, "display_content": display, "display_filename": "search_results.txt"}
