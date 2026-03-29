"""Knowledge-base retrieval tool with hybrid retrieval and reranking."""

from __future__ import annotations

from typing import Any, Dict

from context.evidence_store import EvidenceStore
from context.rag_store import HybridRAGStore
from security import SafetyLayer


class RAGTool:
    def __init__(self, rag_store: HybridRAGStore, evidence_store: EvidenceStore, safety: SafetyLayer) -> None:
        self.rag_store = rag_store
        self.evidence_store = evidence_store
        self.safety = safety

    def query(self, query: str, top_k: int = 6, knowledge_base: str = '') -> Dict[str, Any]:
        self.safety.log_tool_call('rag_query', {'query': query, 'top_k': top_k, 'knowledge_base': knowledge_base})
        result = self.rag_store.query(question=query, top_k=top_k, knowledge_base=knowledge_base or None)
        if 'error' not in result:
            try:
                self.evidence_store.log_event({
                    'event': 'rag_query_completed',
                    'query': query,
                    'knowledge_base': knowledge_base or result.get('knowledge_base', {}).get('id', ''),
                    'hits': len(result.get('evidence') or []),
                })
            except Exception:
                pass
        return result
