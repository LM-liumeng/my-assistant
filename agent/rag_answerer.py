"""Answer synthesis for retrieved RAG evidence."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.skill_manager import SkillManager
from tools.chat_tool import ChatTool


_RAG_SYSTEM_PROMPT = """
You are a retrieval-augmented assistant. Answer only from the supplied evidence.
Rules:
1. Do not invent facts not supported by evidence snippets.
2. Prefer direct, specific answers with short source references like [1], [2].
3. If the evidence is insufficient, say so explicitly.
4. Write concise Chinese prose instead of long bullet lists unless the user explicitly asks for a list.
5. Keep key numbers, named entities, and trend statements from the evidence.
""".strip()


class RAGAnswerAgent:
    CHINESE_FILLER_PHRASES = (
        "请帮我", "请告诉我", "请问", "帮我", "根据", "关于", "整体", "情况", "一下",
        "是什么", "怎么样", "如何", "哪些", "多少", "能否", "是否", "什么",
    )
    GENERIC_TERMS = {
        "情况", "整体", "趋势", "什么", "如何", "哪些", "多少", "是否", "能否", "一下",
        "根据", "关于", "这个", "那个", "目前", "现在", "里面", "之间", "分析", "总结",
        "概括", "说明", "overall", "trend", "summary", "status", "situation", "what", "how",
    }

    def __init__(self, chat_tool: ChatTool, skill_manager: Optional[SkillManager] = None) -> None:
        self.chat_tool = chat_tool
        self.skill_manager = skill_manager
        self.enable_llm_synthesis = os.environ.get("RAG_USE_LLM_SYNTHESIS", "").strip().lower() in {"1", "true", "yes"}

    def answer(self, question: str, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        evidence = retrieval_result.get("evidence") or []
        knowledge_base = retrieval_result.get("knowledge_base")
        display_content = retrieval_result.get("display_content") or self._format_context(question, evidence, knowledge_base)
        display_filename = retrieval_result.get("display_filename", "rag_evidence.txt")
        answerable = bool(retrieval_result.get("answerable", bool(evidence)))
        local_answer = self._build_local_answer(question, evidence, knowledge_base, answerable)

        local_answer = self._postprocess(local_answer, user_query=question)

        if not self.enable_llm_synthesis or not self.chat_tool.is_configured() or not evidence or not answerable:
            return self._build_response(local_answer, display_content, display_filename, knowledge_base)

        prompt = (
            f"Question:\n{question}\n\n"
            f"Evidence:\n{display_content}\n\n"
            f"Base answer:\n{local_answer}\n\n"
            "Rewrite the base answer into concise Chinese prose with citations, but do not add unsupported facts."
        )
        result = self.chat_tool.complete(prompt, system_prompt=self._augment_prompt(_RAG_SYSTEM_PROMPT, scope="rag", user_query=question))
        candidate = (result.get("message") or "").strip()
        if result.get("error") or not self._is_grounded_answer(candidate, evidence):
            return self._build_response(local_answer, display_content, display_filename, knowledge_base)
        return self._build_response(self._postprocess(candidate, user_query=question), display_content, display_filename, knowledge_base)


    def _augment_prompt(self, base_prompt: str, scope: str, user_query: str = "") -> str:
        if self.skill_manager is None:
            return base_prompt
        return self.skill_manager.augment_prompt(base_prompt, scope=scope, user_query=user_query)

    def _postprocess(self, text: str, user_query: str = "") -> str:
        if self.skill_manager is None:
            return (text or "").strip()
        return self.skill_manager.postprocess_response(text, scope="rag", user_query=user_query)

    def can_answer_from_context(self, question: str, display_content: Optional[str], display_filename: Optional[str]) -> bool:
        if not display_content:
            return False
        if display_filename == "rag_evidence.txt":
            return True
        content = display_content.strip()
        return "RAG query:" in content and "Evidence:" in content

    def answer_from_context(self, question: str, display_content: str, display_filename: Optional[str]) -> Dict[str, Any]:
        retrieval_result = self.parse_context(question, display_content, display_filename)
        if retrieval_result is None:
            return {"error": "Unable to parse RAG evidence context.", "message": "I could not reuse the current RAG evidence."}
        return self.answer(question, retrieval_result)

    def parse_context(self, question: str, display_content: str, display_filename: Optional[str]) -> Optional[Dict[str, Any]]:
        content = (display_content or "").strip()
        if not content:
            return None
        lines = content.splitlines()
        kb: Dict[str, Any] = {}
        evidence: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        in_evidence = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    evidence.append(current)
                    current = None
                continue
            if stripped.startswith("Knowledge base: "):
                kb_text = stripped.split(":", 1)[1].strip()
                kb_match = re.match(r"^(.*?)\s+\((.*?)\)$", kb_text)
                if kb_match:
                    kb = {"id": kb_match.group(1).strip(), "name": kb_match.group(2).strip()}
                else:
                    kb = {"id": kb_text, "name": kb_text}
                continue
            if stripped == "Evidence:":
                in_evidence = True
                continue
            if not in_evidence:
                continue
            match = re.match(r"^\[(\d+)\]\s+(.+?)\s+\|\s+(.+?)\s+\|\s+score=([0-9.]+)$", stripped)
            if match:
                if current:
                    evidence.append(current)
                current = {
                    "source_name": match.group(2),
                    "section": match.group(3),
                    "score": float(match.group(4)),
                    "content": "",
                    "source_path": "",
                }
                continue
            if stripped.startswith("Path: ") and current is not None:
                current["source_path"] = stripped.split(":", 1)[1].strip()
                continue
            if current is not None:
                current["content"] = ((current.get("content") or "") + "\n" + stripped).strip()
        if current:
            evidence.append(current)
        if not evidence:
            return None
        return {
            "question": question,
            "knowledge_base": kb,
            "answerable": True,
            "evidence": evidence,
            "display_content": content,
            "display_filename": display_filename or "rag_evidence.txt",
        }

    def _build_response(self, message: str, display_content: str, display_filename: str, knowledge_base: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "message": message,
            "display_content": display_content,
            "display_filename": display_filename,
            "knowledge_base": knowledge_base,
            "active_knowledge_base": (knowledge_base or {}).get("id", ""),
        }

    def _build_local_answer(self, question: str, evidence: Sequence[Dict[str, Any]], knowledge_base: Optional[Dict[str, Any]], answerable: bool) -> str:
        if not evidence:
            return self._kb_prefix(knowledge_base) + "当前知识库里没有检索到足够证据，无法可靠回答这个问题。"
        if not answerable:
            top = evidence[0]
            source = top.get("source_name", "未知来源")
            return (
                self._kb_prefix(knowledge_base)
                + f"当前知识库里没有与这个问题直接对应的证据。最接近的材料来自 {source}[1]，但它不足以支持可靠结论。"
            )

        relevant = self._filter_relevant_evidence(question, evidence)
        top = relevant[0]
        conclusion, fallback_detail = self._summarize_evidence(question, str(top.get("content", "")))
        if not conclusion:
            conclusion = str(top.get("content", "")).strip()[:180]
        conclusion = self._ensure_sentence(conclusion)

        metrics = [
            metric for metric in self._extract_key_metrics(str(top.get("content", "")))
            if not self._is_redundant_detail(conclusion, metric)
        ]
        detail_parts: List[str] = []
        if metrics:
            detail_parts.append(self._ensure_sentence("关键数据包括：" + "；".join(metrics[:2])))
        elif fallback_detail and self._topic_overlap_ratio(question, fallback_detail) >= 0.35:
            detail_parts.append(self._ensure_sentence(fallback_detail))

        secondary = self._secondary_support(question, relevant[1:], top_score=float(top.get("score", 0.0) or 0.0))
        if secondary:
            detail_parts.append(self._ensure_sentence(secondary))

        source_parts = [f"{top.get('source_name', '未知来源')}[1]"]
        for idx, item in enumerate(relevant[1:], start=2):
            source_parts.append(f"{item.get('source_name', '未知来源')}[{idx}]")

        parts = [f"{self._kb_prefix(knowledge_base)}{conclusion}[1]"]
        if detail_parts:
            parts.append(" ".join(detail_parts))
        parts.append("来源：" + "；".join(source_parts) + "。")
        return "\n".join(parts).strip()

    def _filter_relevant_evidence(self, question: str, evidence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not evidence:
            return []
        top = evidence[0]
        top_score = float(top.get("score", 0.0) or 0.0)
        filtered = [top]
        for item in evidence[1:]:
            score = float(item.get("score", 0.0) or 0.0)
            if top_score and score < top_score * 0.72:
                continue
            if self._topic_overlap_ratio(question, str(item.get("content", ""))) < 0.25:
                continue
            filtered.append(item)
            if len(filtered) >= 3:
                break
        return filtered

    def _summarize_evidence(self, question: str, content: str) -> Tuple[str, str]:
        sentences = self._split_sentences(content)
        if not sentences:
            return "", ""
        query_terms = self._extract_query_terms(question)
        ranked = sorted(sentences, key=lambda sentence: self._sentence_score(sentence, query_terms), reverse=True)
        top_sentence = ranked[0].strip()
        fallback_detail = ""
        for sentence in ranked[1:4]:
            cleaned = sentence.strip()
            if not cleaned or cleaned == top_sentence:
                continue
            if self._topic_overlap_ratio(question, cleaned) < 0.25:
                continue
            fallback_detail = cleaned
            break
        return top_sentence, fallback_detail

    def _extract_key_metrics(self, content: str) -> List[str]:
        metrics: List[str] = []
        for sentence in self._split_sentences(content):
            if not re.search(r"\d", sentence):
                continue
            clauses = re.split(r"[；;]", sentence)
            for clause in clauses:
                clause = clause.strip().strip("，,；;：:")
                if not clause or not re.search(r"\d", clause):
                    continue
                if len(clause) > 110:
                    clause = clause[:107].rstrip() + "..."
                if clause not in metrics:
                    metrics.append(clause)
                if len(metrics) >= 3:
                    return metrics
        return metrics

    def _secondary_support(self, question: str, evidence: Sequence[Dict[str, Any]], top_score: float) -> str:
        if not evidence:
            return ""
        lines = []
        for idx, item in enumerate(evidence, start=2):
            score = float(item.get("score", 0.0) or 0.0)
            if top_score and score < top_score * 0.78:
                continue
            sentence, _ = self._summarize_evidence(question, str(item.get("content", "")))
            if sentence and self._topic_overlap_ratio(question, sentence) >= 0.35:
                lines.append(f"补充证据显示，{sentence}[{idx}]")
        return " ".join(lines[:1]).strip()

    def _is_redundant_detail(self, conclusion: str, detail: str) -> bool:
        short_conclusion = re.sub(r"\[[0-9]+\]", "", conclusion or "")
        short_conclusion = re.sub(r"\s+", "", short_conclusion)
        short_detail = re.sub(r"\s+", "", detail or "")
        if not short_detail:
            return True
        if short_detail in short_conclusion or short_conclusion[:16] in short_detail:
            return True
        return False

    def _sentence_score(self, sentence: str, query_terms: Sequence[str]) -> float:
        score = 0.0
        lowered = sentence.lower()
        for term in query_terms:
            if term and term.lower() in lowered:
                score += 1.0
        if re.search(r"\d", sentence):
            score += 0.35
        if any(keyword in sentence for keyword in ["趋势", "整体", "增长", "下降", "特征", "显示", "说明", "韧性", "回报"]):
            score += 0.35
        return score

    def _split_sentences(self, text: str) -> List[str]:
        raw = re.split(r"(?<=[。！？!?])\s+|(?<=[。！？!?])", str(text or "").replace("\n", " "))
        return [part.strip() for part in raw if part and part.strip()]

    def _extract_query_terms(self, question: str) -> List[str]:
        lowered = (question or "").lower()
        terms: set[str] = set(re.findall(r"[A-Za-z0-9_\-]{2,}", lowered))
        for seq in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
            cleaned = seq
            for phrase in self.CHINESE_FILLER_PHRASES:
                cleaned = cleaned.replace(phrase, " ")
            cleaned = re.sub(r"[的是了吗呢吧啊呀将对和与及在从向把给就都还很]", " ", cleaned)
            pieces = [piece.strip().lstrip("年") for piece in cleaned.split() if len(piece.strip()) >= 2]
            if not pieces and len(seq) >= 2:
                pieces = [seq]
            for piece in pieces:
                if len(piece) >= 2 and piece not in self.GENERIC_TERMS and not re.fullmatch(r"20\d{2}(?:年)?", piece):
                    terms.add(piece)
                if len(piece) >= 4:
                    for size in (2, 3, 4):
                        for start in range(0, len(piece) - size + 1):
                            gram = piece[start:start + size]
                            if gram not in self.GENERIC_TERMS and not re.fullmatch(r"20\d{2}(?:年)?", gram):
                                terms.add(gram)
        ordered = sorted((term for term in terms if term not in self.GENERIC_TERMS), key=lambda item: (-len(item), item))
        return ordered[:16]

    def _topic_overlap_ratio(self, question: str, content: str) -> float:
        terms = self._extract_query_terms(question)
        if not terms:
            return 0.0
        lowered = (content or "").lower()
        hits = sum(1 for term in terms if term in lowered)
        return hits / max(len(terms), 1)

    def _kb_prefix(self, knowledge_base: Optional[Dict[str, Any]]) -> str:
        kb_id = (knowledge_base or {}).get("id", "")
        return f"基于知识库 {kb_id}，" if kb_id else ""

    def _ensure_sentence(self, text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        if value[-1] not in "。！？!?":
            value += "。"
        return value

    def _is_grounded_answer(self, answer: str, evidence: Sequence[Dict[str, Any]]) -> bool:
        text = (answer or "").strip()
        if not text:
            return False
        if len(re.findall(r"\[[0-9]+\]", text)) == 0:
            return False
        evidence_terms = self._collect_distinctive_terms(evidence)
        matched = sum(1 for term in evidence_terms if term in text)
        return matched >= 2

    def _collect_distinctive_terms(self, evidence: Sequence[Dict[str, Any]]) -> List[str]:
        terms: List[str] = []
        for item in evidence[:2]:
            content = str(item.get("content", ""))
            numbers = re.findall(r"\d+(?:\.\d+)?%?|\d+(?:\.\d+)?个?百分点", content)
            terms.extend(numbers[:4])
            words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[\u4e00-\u9fff]{2,}", content)
            for word in words:
                if word not in terms:
                    terms.append(word)
                if len(terms) >= 12:
                    return terms
        return terms[:12]

    def _format_context(self, question: str, evidence: Sequence[Dict[str, Any]], knowledge_base: Optional[Dict[str, Any]]) -> str:
        lines = []
        if knowledge_base and knowledge_base.get("id"):
            lines.append(f"Knowledge base: {knowledge_base.get('id')} ({knowledge_base.get('name', '')})")
        lines.extend([f"RAG query: {question}", "Evidence:"])
        for idx, item in enumerate(evidence, start=1):
            lines.append(f"[{idx}] {item.get('source_name', '')} | {item.get('section', '')} | score={float(item.get('score', 0.0)):.4f}")
            lines.append(str(item.get("content", "")).strip())
            lines.append(f"Path: {item.get('source_path', '')}")
            lines.append("")
        return "\n".join(lines).strip()
