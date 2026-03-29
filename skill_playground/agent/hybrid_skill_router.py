from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from skill_playground.agent.skill_registry import LoadedSkill, SkillRegistry
from skill_playground.agent.trace import TraceRecorder


@dataclass(frozen=True)
class SkillMatch:
    skill_id: str
    base_score: float
    score: float
    priority: int
    source: str
    sources: Tuple[str, ...]
    matched_terms: Tuple[str, ...]


class HybridSkillRouter:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def route(self, user_query: str, surface: Optional[str] = None, phase: Optional[str] = None, trace: Optional[TraceRecorder] = None) -> Tuple[Optional[str], float]:
        matches = self.route_all(user_query, surface=surface, phase=phase, trace=trace)
        if not matches:
            return None, 0.0
        top = matches[0]
        return top.skill_id, top.score

    def route_all(self, user_query: str, surface: Optional[str] = None, phase: Optional[str] = None, trace: Optional[TraceRecorder] = None) -> List[SkillMatch]:
        if trace:
            trace.log("router", "Starting rule-based recall.", {"surface": surface or "", "phase": phase or "", "query": user_query})
        recalled = self.recall_candidates(user_query, surface=surface, phase=phase)
        if trace:
            trace.log(
                "router",
                "Recall stage completed.",
                {"candidates": [{"skill_id": item.skill_id, "score": round(item.base_score, 3), "source": item.source} for item in recalled]},
            )
        reranked = self.rerank_candidates(recalled)
        if trace:
            trace.log(
                "router",
                "Rerank stage completed.",
                {"candidates": [{"skill_id": item.skill_id, "score": round(item.score, 3), "source": item.source} for item in reranked]},
            )
        return reranked

    def recall_candidates(self, user_query: str, surface: Optional[str] = None, phase: Optional[str] = None, limit: int = 12) -> List[SkillMatch]:
        query = (user_query or "").strip()
        if not query:
            return []
        lowered = query.lower()
        candidates: List[SkillMatch] = []
        for skill in self.registry.list_skills(surface=surface, phase=phase):
            base_score, source, matched_terms, sources = self._score_skill(skill, query, lowered)
            if base_score <= 0.0:
                continue
            candidates.append(
                SkillMatch(
                    skill_id=skill.skill_id,
                    base_score=base_score,
                    score=base_score,
                    priority=skill.priority,
                    source=source,
                    sources=sources,
                    matched_terms=tuple(matched_terms),
                )
            )
        candidates.sort(key=lambda item: (-item.base_score, -item.priority, item.skill_id))
        return candidates[:limit]

    def rerank_candidates(self, candidates: List[SkillMatch]) -> List[SkillMatch]:
        reranked: List[SkillMatch] = []
        for match in candidates:
            skill = self.registry.get_skill(match.skill_id)
            if skill is None:
                continue
            score = match.base_score
            if skill.priority >= 80:
                score += 0.01
            if skill.risk_level == "high":
                score -= 0.03
            elif skill.risk_level == "medium":
                score -= 0.01
            if skill.cost_level == "high":
                score -= 0.01
            distinct_evidence = len({term.strip().lower() for term in match.matched_terms if term.strip()})
            if distinct_evidence >= 2:
                score += min(0.03, 0.01 * (distinct_evidence - 1))
            reranked.append(
                SkillMatch(
                    skill_id=match.skill_id,
                    base_score=match.base_score,
                    score=max(0.0, min(0.99, score)),
                    priority=match.priority,
                    source=match.source,
                    sources=match.sources,
                    matched_terms=match.matched_terms,
                )
            )
        reranked.sort(key=lambda item: (-item.score, -item.priority, item.skill_id))
        return reranked

    def _score_skill(self, skill: LoadedSkill, query: str, lowered: str) -> Tuple[float, str, List[str], Tuple[str, ...]]:
        evidence: List[Tuple[float, str, str]] = []
        matched_terms: List[str] = []
        matched_sources: List[str] = []
        seen_terms = set()
        seen_sources = set()

        for trigger in skill.triggers:
            keyword = trigger.strip()
            if not keyword:
                continue
            trigger_score, trigger_source = self._score_trigger(keyword, query, lowered)
            if trigger_score <= 0.0:
                continue
            evidence.append((trigger_score, trigger_source, keyword))
            normalized_term = keyword.lower()
            if normalized_term not in seen_terms:
                matched_terms.append(keyword)
                seen_terms.add(normalized_term)
            if trigger_source not in seen_sources:
                matched_sources.append(trigger_source)
                seen_sources.add(trigger_source)

        for pattern in skill.patterns:
            expr = pattern.strip()
            if not expr:
                continue
            try:
                match = re.search(expr, query, re.IGNORECASE)
            except re.error:
                continue
            if not match:
                continue
            pattern_score, pattern_source = self._score_pattern(expr, query, match)
            evidence.append((pattern_score, pattern_source, expr))
            normalized_term = expr.lower()
            if normalized_term not in seen_terms:
                matched_terms.append(expr)
                seen_terms.add(normalized_term)
            if pattern_source not in seen_sources:
                matched_sources.append(pattern_source)
                seen_sources.add(pattern_source)

        if not evidence:
            return 0.0, "", [], tuple()

        evidence.sort(key=lambda item: (-item[0], item[1], item[2]))
        best_score, primary_source, _ = evidence[0]
        distinct_evidence = len(seen_terms)
        bonus = min(0.06, 0.02 * max(0, distinct_evidence - 1))
        return min(0.99, best_score + bonus), primary_source or "router", matched_terms, tuple(matched_sources)

    def _score_trigger(self, keyword: str, query: str, lowered: str) -> Tuple[float, str]:
        keyword_lower = keyword.lower()
        if lowered == keyword_lower:
            return 0.99, f"trigger:exact:{keyword}"
        if self._starts_with_phrase(lowered, keyword_lower):
            return 0.94, f"trigger:prefix:{keyword}"
        if self._contains_phrase(lowered, keyword, keyword_lower):
            return 0.80, f"trigger:contains:{keyword}"
        return 0.0, ""

    def _score_pattern(self, expr: str, query: str, match: re.Match[str]) -> Tuple[float, str]:
        if match.start() == 0 and match.end() == len(query):
            return 0.97, f"pattern:fullmatch:{expr}"
        if match.start() == 0:
            return 0.93, f"pattern:prefix:{expr}"
        specificity = 0.86
        if expr.startswith("^"):
            specificity += 0.02
        if expr.endswith("$"):
            specificity += 0.02
        if any(token in expr for token in [r"\d", "[", "(", "{", "+", "?"]):
            specificity += 0.01
        if len(expr) >= 16:
            specificity += 0.01
        return min(0.95, specificity), f"pattern:search:{expr}"

    def _starts_with_phrase(self, lowered_query: str, lowered_keyword: str) -> bool:
        if not lowered_query.startswith(lowered_keyword):
            return False
        if len(lowered_query) == len(lowered_keyword):
            return True
        next_char = lowered_query[len(lowered_keyword)]
        return not next_char.isalnum()

    def _contains_phrase(self, lowered_query: str, keyword: str, lowered_keyword: str) -> bool:
        if self._is_ascii_word(keyword):
            if len(lowered_keyword) < 4:
                return False
            pattern = rf"(?<!\w){re.escape(lowered_keyword)}(?!\w)"
            return bool(re.search(pattern, lowered_query, re.IGNORECASE))
        if self._looks_cjk(keyword):
            if len(keyword) < 2:
                return False
            return lowered_keyword in lowered_query
        if len(keyword) < 4:
            return False
        return keyword.lower() in lowered_query

    def _is_ascii_word(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_\- ]+", value or ""))

    def _looks_cjk(self, value: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", value or ""))
