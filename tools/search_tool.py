"""WebSearchTool: stub implementation for web searches.

In this demo no external network access is available, so the search tool
returns a placeholder result.  To integrate with a real search engine or
the browser tool, replace the ``search`` method with code that performs
the lookup and returns meaningful results.
"""

from __future__ import annotations

from typing import Dict, Any

from context.evidence_store import EvidenceStore


class WebSearchTool:
    """Performs a web search using an external API if configured.

    This tool attempts to call the Google Custom Search API when the
    environment variables ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID`` are
    provided.  Results are returned as a list of (title, link, snippet)
    dictionaries.  If no API credentials are configured, a placeholder
    response is returned instructing the user how to enable real search.
    """

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store
        import os
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.cse_id = os.environ.get("GOOGLE_CSE_ID") or os.environ.get("GOOGLE_CX")

    def search(self, query: str) -> Dict[str, Any]:
        # log invocation
        try:
            self.evidence_store.log_event({"event": "web_search", "query": query})
        except Exception:
            pass
        if not query:
            return {"error": "Search query must not be empty."}
        # Try using Google Custom Search API if configured
        if self.api_key and self.cse_id:
            try:
                import urllib.parse
                import urllib.request
                import json
                params = {
                    "key": self.api_key,
                    "cx": self.cse_id,
                    "q": query,
                }
                url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(params)
                with urllib.request.urlopen(url) as response:
                    data = json.load(response)
                items = data.get("items", [])
                results = []
                for item in items[:5]:
                    results.append({
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet"),
                    })
                # Build display text
                display_lines = []
                for idx, r in enumerate(results, 1):
                    display_lines.append(f"{idx}. {r['title']}\n{r['link']}\n{r['snippet']}")
                display_text = "\n\n".join(display_lines)
                return {
                    "message": f"Found {len(results)} result(s)",
                    "results": results,
                    "display_content": display_text,
                    "display_filename": "web_search_results.txt"
                }
            except Exception as exc:
                # fall back to stub with error message
                return {"error": f"Web search failed: {exc}"}
        # Otherwise return stub
        return {
            "message": (
                "Web search is not configured. To enable, set GOOGLE_API_KEY and "
                "GOOGLE_CSE_ID environment variables for Google Custom Search API."
            ),
            "results": [],
            "display_content": ("Web search is not configured. To enable, set GOOGLE_API_KEY and "
                                 "GOOGLE_CSE_ID environment variables."),
            "display_filename": "web_search_results.txt"
        }
