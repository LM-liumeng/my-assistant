"""RetrievalStore: simplistic file indexer and searcher.

The retrieval store scans a base directory recursively and returns files
whose names contain a given substring.  It provides a lightweight way to
support file search without requiring an external search service.
"""

from __future__ import annotations

import os
from typing import List, Dict


class RetrievalStore:
    """Searches for files under the base directory."""

    def __init__(self, base_dir: str) -> None:
        # The base directory for searches; all file access is restricted
        self.base_dir = os.path.join(base_dir, "workspace")
        os.makedirs(self.base_dir, exist_ok=True)

    def search_files(self, query: str) -> List[str]:
        """Return a list of file paths whose names contain the query (case‑insensitive).

        If the query is empty or consists of whitespace, an empty list is
        returned rather than returning all files.  This avoids the
        accidental listing of very large directories when the user has
        forgotten to supply a search term.  Trim whitespace from the query
        before matching.
        """
        results: List[str] = []
        q = (query or "").strip().lower()
        if not q:
            return results
        for root, _, files in os.walk(self.base_dir):
            for fname in files:
                if q in fname.lower():
                    results.append(os.path.join(root, fname))
        return results
