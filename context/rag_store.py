"""Runtime hybrid retrieval for local multi-knowledge-base RAG indexes."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from context.knowledge_registry import KnowledgeRegistry

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_path: str
    source_name: str
    section: str
    content: str
    score: float
    semantic_score: float
    lexical_score: float
    rerank_score: float
    topic_overlap: float
    matched_terms: List[str]
    metadata: Dict[str, Any]


class HybridRAGStore:
    ENGLISH_STOP_TERMS = {
        "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
        "which", "why", "how", "who", "whom", "whose", "about", "overall", "trend",
        "trends", "summary", "status", "situation", "please", "could", "would",
    }
    CHINESE_FILLER_PHRASES = (
        "请帮我", "请告诉我", "请问", "帮我", "根据", "关于", "整体", "情况", "一下",
        "是什么", "怎么样", "如何", "哪些", "多少", "能否", "是否", "什么",
    )
    CHINESE_STOP_TERMS = {
        "情况", "整体", "趋势", "什么", "如何", "哪些", "多少", "是否", "能否", "一下",
        "根据", "关于", "这个", "那个", "我们", "你们", "请问", "请帮", "告诉", "目前",
        "现在", "里面", "之间", "影响", "分析", "总结", "概括", "说明",
    }

    def __init__(self, base_dir: str, registry: Optional[KnowledgeRegistry] = None) -> None:
        self.base_dir = Path(base_dir)
        self.registry = registry or KnowledgeRegistry(base_dir)
        self._manifest_cache: Dict[str, Dict[str, Any]] = {}
        self._chunk_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._model_cache: Dict[str, Any] = {}

    def is_available(self, knowledge_base: Optional[str] = None) -> bool:
        _, manifest_path, chunks_path, embeddings_path = self._paths_for(knowledge_base)
        return manifest_path.exists() and chunks_path.exists() and embeddings_path.exists()

    def query(self, question: str, top_k: int = 8, initial_k: int = 24, knowledge_base: Optional[str] = None) -> Dict[str, Any]:
        query = (question or "").strip()
        kb = self.registry.get_base(knowledge_base)
        index_dir, manifest_path, chunks_path, embeddings_path = self._paths_for(kb["id"])
        if not query:
            return {"error": "RAG query cannot be empty."}
        if not (manifest_path.exists() and chunks_path.exists() and embeddings_path.exists()):
            return {"error": f"RAG index not found for knowledge base '{kb['id']}' in {index_dir}."}

        chunks = self._load_chunks(index_dir)
        embeddings = self._load_embeddings(index_dir)
        if not chunks or embeddings is None or len(chunks) != len(embeddings):
            return {"error": f"RAG index files are inconsistent or empty for knowledge base '{kb['id']}'."}

        lexical_scores = self._lexical_scores(query, chunks)
        semantic_scores = self._semantic_scores(query, embeddings, index_dir)
        combined_scores = [0.45 * lexical + 0.55 * semantic for lexical, semantic in zip(lexical_scores, semantic_scores)]
        ranked_indices = sorted(range(len(chunks)), key=lambda idx: combined_scores[idx], reverse=True)[: max(top_k, initial_k)]
        reranked = self._rerank(
            query,
            [chunks[idx] for idx in ranked_indices],
            [combined_scores[idx] for idx in ranked_indices],
            top_k=max(top_k, initial_k),
        )
        selected, answerable = self._select_answer_evidence(query, reranked, top_k=top_k)
        visible = selected if selected else reranked[: min(2, len(reranked))]

        return {
            "question": query,
            "knowledge_base": kb,
            "answerable": answerable,
            "evidence": [self._to_payload(item) for item in visible],
            "display_content": self._format_evidence(query, visible, kb),
            "display_filename": "rag_evidence.txt",
            "message": f"Retrieved {len(visible)} evidence chunk(s) from knowledge base '{kb['id']}'.",
        }

    def _paths_for(self, knowledge_base: Optional[str]) -> Tuple[Path, Path, Path, Path]:
        index_dir = self.registry.resolve_output_dir(knowledge_base)
        return (
            index_dir,
            index_dir / "ingestion_manifest.json",
            index_dir / "chunks.jsonl",
            index_dir / "embeddings.npy",
        )

    def _load_manifest(self, index_dir: Path) -> Dict[str, Any]:
        key = str(index_dir)
        if key not in self._manifest_cache:
            self._manifest_cache[key] = json.loads((index_dir / "ingestion_manifest.json").read_text(encoding="utf-8"))
        return self._manifest_cache[key]

    def _load_chunks(self, index_dir: Path) -> List[Dict[str, Any]]:
        key = str(index_dir)
        if key not in self._chunk_cache:
            rows: List[Dict[str, Any]] = []
            with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            self._chunk_cache[key] = rows
        return self._chunk_cache[key]

    def _load_embeddings(self, index_dir: Path) -> Optional[np.ndarray]:
        key = str(index_dir)
        if key not in self._embedding_cache and (index_dir / "embeddings.npy").exists():
            self._embedding_cache[key] = np.load(index_dir / "embeddings.npy")
        return self._embedding_cache.get(key)

    def _ensure_model(self, index_dir: Path) -> Any:
        if SentenceTransformer is None:
            return None
        model_name = self._load_manifest(index_dir).get("embedding_model") or "BAAI/bge-m3"
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        allow_remote = os.environ.get("RAG_ALLOW_REMOTE_MODEL_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}
        try:
            if allow_remote:
                self._model_cache[model_name] = SentenceTransformer(model_name)
            else:
                self._model_cache[model_name] = SentenceTransformer(model_name, local_files_only=True)
        except TypeError:
            if allow_remote:
                self._model_cache[model_name] = SentenceTransformer(model_name)
            else:
                self._model_cache[model_name] = None
        except Exception:
            self._model_cache[model_name] = None
        return self._model_cache[model_name]

    def _semantic_scores(self, query: str, embeddings: np.ndarray, index_dir: Path) -> List[float]:
        manifest = self._load_manifest(index_dir)
        model_name = str(manifest.get("embedding_model") or "")
        if model_name.startswith("hashing-fallback:"):
            dims = int(model_name.split(":", 1)[1] or embeddings.shape[1])
            query_vector = self._hash_query_vector(query, dims)
            scores = embeddings @ query_vector
            return [float(score) for score in scores.tolist()]
        model = self._ensure_model(index_dir)
        if model is None:
            return [0.0 for _ in range(len(embeddings))]
        vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector)
        query_vector = vector[0].astype(np.float32)
        scores = embeddings @ query_vector
        return [float(score) for score in scores.tolist()]

    def _lexical_scores(self, query: str, chunks: Sequence[Dict[str, Any]]) -> List[float]:
        query_terms = self._query_terms(query, keep_generic=True)
        specific_terms = [term for term in query_terms if not self._is_generic_term(term)]
        generic_terms = [term for term in query_terms if self._is_generic_term(term)]
        if not query_terms:
            return [0.0 for _ in chunks]
        scores: List[float] = []
        for chunk in chunks:
            haystack = " ".join([
                str(chunk.get("title", "")),
                str(chunk.get("section", "")),
                str(chunk.get("content", "")),
                " ".join(chunk.get("keywords", []) or []),
            ]).lower()
            matched_specific = sum(1 for term in specific_terms if self._term_matches(term, haystack))
            matched_generic = sum(1 for term in generic_terms if self._term_matches(term, haystack))
            specific_score = matched_specific / max(len(specific_terms), 1) if specific_terms else 0.0
            generic_score = matched_generic / max(len(generic_terms), 1) if generic_terms else 0.0
            phrase_bonus = 0.2 if query.lower() in haystack else 0.0
            negation_penalty = 0.35 if any(pattern in haystack for pattern in ["does not mention", "not mention"]) else 0.0
            score = (0.8 * specific_score) + (0.2 * generic_score) + phrase_bonus
            if specific_terms and matched_specific == 0:
                score *= 0.3
            scores.append(max(0.0, score - negation_penalty))
        return scores

    def _rerank(self, query: str, chunks: Sequence[Dict[str, Any]], base_scores: Sequence[float], top_k: int) -> List[RetrievedChunk]:
        query_terms = self._query_terms(query, keep_generic=True)
        specific_terms = [term for term in query_terms if not self._is_generic_term(term)]
        results: List[RetrievedChunk] = []
        for chunk, base_score in zip(chunks, base_scores):
            content = str(chunk.get("content", ""))
            section = str(chunk.get("section", ""))
            title = str(chunk.get("title", ""))
            lexical = self._lexical_scores(query, [chunk])[0]
            semantic = float(base_score)
            keywords = [str(item).lower() for item in (chunk.get("keywords") or [])]
            haystack = " ".join([title, section, content]).lower()
            matched_terms = [term for term in query_terms if self._term_matches(term, haystack)]
            matched_specific_terms = [term for term in specific_terms if self._term_matches(term, haystack)]
            topic_overlap = (
                len(matched_specific_terms) / max(len(specific_terms), 1)
                if specific_terms
                else len(matched_terms) / max(len(query_terms), 1)
            ) if query_terms else 0.0
            term_pool = specific_terms or query_terms
            title_bonus = 0.25 if any(self._term_matches(term, title.lower()) for term in term_pool) else 0.0
            section_bonus = 0.15 if any(self._term_matches(term, section.lower()) for term in term_pool) else 0.0
            keyword_bonus = 0.12 * sum(1 for term in term_pool if any(term in keyword or keyword in term for keyword in keywords))
            density_bonus = min(sum(haystack.count(term) for term in matched_specific_terms[:8]) * 0.05, 0.25) if matched_specific_terms else 0.0
            mismatch_penalty = 0.45 if specific_terms and not matched_specific_terms else 0.0
            rerank_score = semantic + lexical + (0.9 * topic_overlap) + title_bonus + section_bonus + keyword_bonus + density_bonus - mismatch_penalty
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk.get("chunk_id", "")),
                    source_path=str(chunk.get("source_path", "")),
                    source_name=str(chunk.get("source_name", "")),
                    section=section,
                    content=content,
                    score=rerank_score,
                    semantic_score=semantic,
                    lexical_score=lexical,
                    rerank_score=rerank_score,
                    topic_overlap=topic_overlap,
                    matched_terms=matched_terms[:12],
                    metadata=dict(chunk.get("metadata") or {}),
                )
            )
        results.sort(key=lambda item: item.rerank_score, reverse=True)
        return results[:top_k]

    def _select_answer_evidence(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int) -> Tuple[List[RetrievedChunk], bool]:
        if not chunks:
            return [], False
        top = chunks[0]
        if not self._is_answerable(query, top):
            return [], False

        selected = [top]
        for item in chunks[1:]:
            if len(selected) >= top_k:
                break
            if item.rerank_score < top.rerank_score * 0.72:
                continue
            if item.topic_overlap < max(0.3, top.topic_overlap * 0.6):
                continue
            if item.lexical_score < max(0.35, top.lexical_score * 0.6):
                continue
            selected.append(item)
        return selected, True

    def _is_answerable(self, query: str, top: RetrievedChunk) -> bool:
        specific_terms = [term for term in self._query_terms(query, keep_generic=True) if not self._is_generic_term(term)]
        if specific_terms:
            required_overlap = min(0.45, max(0.2, 1.0 / max(len(specific_terms), 1)))
            return top.topic_overlap >= required_overlap and top.lexical_score >= 0.35
        return top.lexical_score >= 0.9 and top.semantic_score >= 0.35

    def _format_evidence(self, question: str, evidence: Sequence[RetrievedChunk], kb: Dict[str, Any]) -> str:
        lines = [f"Knowledge base: {kb['id']} ({kb.get('name', '')})", f"RAG query: {question}", "Evidence:"]
        for idx, item in enumerate(evidence, start=1):
            lines.append(f"[{idx}] {item.source_name} | {item.section} | score={item.score:.4f}")
            lines.append(item.content[:800].strip())
            lines.append(f"Path: {item.source_path}")
            lines.append("")
        return "\n".join(lines).strip()

    def _to_payload(self, item: RetrievedChunk) -> Dict[str, Any]:
        return {
            "chunk_id": item.chunk_id,
            "source_path": item.source_path,
            "source_name": item.source_name,
            "section": item.section,
            "content": item.content,
            "score": item.score,
            "semantic_score": item.semantic_score,
            "lexical_score": item.lexical_score,
            "rerank_score": item.rerank_score,
            "topic_overlap": item.topic_overlap,
            "matched_terms": item.matched_terms,
            "metadata": item.metadata,
        }

    def _hash_query_vector(self, query: str, dims: int) -> np.ndarray:
        vector = np.zeros(dims, dtype=np.float32)
        terms = self._query_terms(query, keep_generic=True)
        if not terms:
            return vector
        for term in terms:
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def _term_matches(self, term: str, haystack: str) -> bool:
        term = term.lower()
        if term in haystack:
            return True
        if len(term) >= 5 and term[:5] in haystack:
            return True
        if term.endswith("ion") and term[:-3] in haystack:
            return True
        if term.endswith("ing") and term[:-3] in haystack:
            return True
        return False

    def _query_terms(self, text: str, keep_generic: bool = False) -> List[str]:
        lowered = (text or "").lower()
        terms: set[str] = set()

        for token in re.findall(r"[A-Za-z0-9_\-]{2,}", lowered):
            if keep_generic or not self._is_generic_term(token):
                terms.add(token)

        for seq in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
            cleaned = seq
            for phrase in self.CHINESE_FILLER_PHRASES:
                cleaned = cleaned.replace(phrase, " ")
            cleaned = re.sub(r"[的是了吗呢吧啊呀将对和与及在从向把给就都还很]", " ", cleaned)
            pieces = [piece.strip() for piece in cleaned.split() if len(piece.strip()) >= 2]
            if not pieces and len(seq) >= 2:
                pieces = [seq]
            for piece in pieces:
                piece = piece.lstrip("年")
                if len(piece) < 2:
                    continue
                if keep_generic or not self._is_generic_term(piece):
                    terms.add(piece)
                if len(piece) >= 4:
                    for gram_size in (2, 3, 4):
                        for start in range(0, len(piece) - gram_size + 1):
                            gram = piece[start:start + gram_size]
                            if keep_generic or not self._is_generic_term(gram):
                                terms.add(gram)

        ordered = sorted(terms, key=lambda item: (-len(item), item))
        return ordered[:32]

    def _is_generic_term(self, term: str) -> bool:
        if not term:
            return True
        if re.fullmatch(r"20\d{2}(?:年)?", term):
            return True
        if term in self.ENGLISH_STOP_TERMS or term in self.CHINESE_STOP_TERMS:
            return True
        if len(term) <= 1:
            return True
        return False
