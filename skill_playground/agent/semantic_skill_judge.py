from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from skill_playground.agent.hybrid_skill_router import SkillMatch
from skill_playground.agent.skill_registry import SkillRegistry
from skill_playground.agent.trace import TraceRecorder
from skill_playground.tools.chat_tool import ChatTool


_SEMANTIC_JUDGE_SYSTEM_PROMPT = """
You are a semantic skill judge for an assistant skill system.
You must choose only from the provided candidate skills. Never invent a new skill id.

Return strict JSON with exactly these fields:
- primary_skill: string or null
- secondary_skills: array of strings
- rejected_skills: array of strings
- confidence: number between 0 and 1
- intent_summary: short string

Rules:
1. Choose one primary skill only if a candidate clearly fits the user's intent.
2. If no candidate clearly fits the user's intent, return primary_skill as null.
3. Secondary skills should be supportive overlays, not duplicates of the primary skill.
4. Reject skills that are clearly off-topic, too broad, only weak lexical matches, or not semantically suitable.
5. It is valid for both primary_skill and secondary_skills to be empty if no suitable skill exists.
6. Prefer concise intent summaries.
7. Do not output markdown, prose outside JSON, or comments.
""".strip()


@dataclass(frozen=True)
class SemanticSkillJudgment:
    primary_skill: Optional[str]
    secondary_skills: Tuple[str, ...]
    rejected_skills: Tuple[str, ...]
    confidence: float
    intent_summary: str
    source: str = "heuristic"

    def accepted_skills(self) -> Tuple[str, ...]:
        items: List[str] = []
        if self.primary_skill:
            items.append(self.primary_skill)
        for skill_id in self.secondary_skills:
            if skill_id and skill_id not in items:
                items.append(skill_id)
        return tuple(items)

    @property
    def primary_candidate(self) -> Optional[str]:
        return self.primary_skill

    @property
    def secondary_candidates(self) -> Tuple[str, ...]:
        return self.secondary_skills

    @property
    def rejected_candidates(self) -> Tuple[str, ...]:
        return self.rejected_skills

    def to_dict(self) -> Dict[str, object]:
        return {
            "primary_skill": self.primary_skill,
            "secondary_skills": list(self.secondary_skills),
            "rejected_skills": list(self.rejected_skills),
            "confidence": self.confidence,
            "intent_summary": self.intent_summary,
            "source": self.source,
        }


class SemanticSkillJudge:
    def __init__(
        self,
        registry: SkillRegistry,
        chat_tool: Optional[ChatTool] = None,
        top_k: int = 6,
        no_match_threshold: float = 0.55,
    ) -> None:
        self.registry = registry
        self.chat_tool = chat_tool
        self.top_k = top_k
        self.no_match_threshold = no_match_threshold

    def attach_chat_tool(self, chat_tool: ChatTool) -> None:
        self.chat_tool = chat_tool

    def judge(
        self,
        user_query: str,
        candidates: Sequence[SkillMatch],
        surface: Optional[str] = None,
        phase: Optional[str] = None,
        trace: Optional[TraceRecorder] = None,
    ) -> SemanticSkillJudgment:
        limited_candidates = list(candidates[: self.top_k])
        if trace:
            trace.log(
                "semantic_judge",
                "Evaluating semantic fit for recalled skills.",
                {"candidate_ids": [item.skill_id for item in limited_candidates], "surface": surface or "", "phase": phase or ""},
            )
        if not limited_candidates:
            judgment = SemanticSkillJudgment(None, (), (), 0.0, "No candidate skills were recalled.", "heuristic")
            if trace:
                trace.log("semantic_judge", "No candidates available for semantic judgment.")
            return judgment

        if self.chat_tool is None or not self.chat_tool.is_configured():
            if trace:
                trace.log("semantic_judge", "Falling back to heuristic semantic judgment because chat model is unavailable.")
            return self._heuristic_fallback(user_query, limited_candidates, trace=trace)

        prompt = self._build_prompt(user_query, limited_candidates, surface=surface, phase=phase)
        result = self.chat_tool.complete(prompt, system_prompt=_SEMANTIC_JUDGE_SYSTEM_PROMPT, trace=trace)
        if result.get("error"):
            if trace:
                trace.log("semantic_judge", "LLM semantic judgment failed; using heuristic fallback.", {"error": result.get("error")})
            return self._heuristic_fallback(user_query, limited_candidates, trace=trace)

        parsed = self._parse_json_result(str(result.get("message") or ""))
        if not parsed:
            if trace:
                trace.log("semantic_judge", "Model response was not valid JSON; using heuristic fallback.")
            return self._heuristic_fallback(user_query, limited_candidates, trace=trace)

        judgment = self._normalize_result(parsed, limited_candidates, user_query=user_query)
        if trace:
            trace.log(
                "semantic_judge",
                "Semantic judgment completed.",
                {
                    "primary_skill": judgment.primary_skill,
                    "secondary_skills": list(judgment.secondary_skills),
                    "rejected_skills": list(judgment.rejected_skills),
                    "confidence": judgment.confidence,
                    "source": judgment.source,
                },
            )
        return judgment

    def _heuristic_fallback(self, user_query: str, candidates: Sequence[SkillMatch], trace: Optional[TraceRecorder] = None) -> SemanticSkillJudgment:
        if not candidates:
            judgment = SemanticSkillJudgment(None, (), (), 0.0, "No candidate skills available.", "heuristic")
            if trace:
                trace.log("semantic_judge", "Heuristic fallback found no candidates.")
            return judgment
        top = candidates[0]
        if top.score >= self.no_match_threshold:
            secondary = tuple(
                match.skill_id
                for match in candidates[1:]
                if match.score >= self.no_match_threshold and match.skill_id != top.skill_id
            )
            accepted = {top.skill_id, *secondary}
            rejected = tuple(match.skill_id for match in candidates if match.skill_id not in accepted)
            judgment = SemanticSkillJudgment(
                primary_skill=top.skill_id,
                secondary_skills=secondary,
                rejected_skills=rejected,
                confidence=max(0.0, min(0.95, top.score)),
                intent_summary=self._build_fallback_summary(user_query, candidates, no_match=False),
                source="heuristic",
            )
            if trace:
                trace.log("semantic_judge", "Heuristic fallback selected skills.", {"primary_skill": top.skill_id, "secondary_skills": list(secondary)})
            return judgment
        judgment = SemanticSkillJudgment(
            primary_skill=None,
            secondary_skills=(),
            rejected_skills=tuple(match.skill_id for match in candidates),
            confidence=max(0.0, min(0.49, top.score)),
            intent_summary=self._build_fallback_summary(user_query, candidates, no_match=True),
            source="heuristic",
        )
        if trace:
            trace.log("semantic_judge", "Heuristic fallback rejected all recalled skills.", {"top_score": top.score})
        return judgment

    def _build_prompt(
        self,
        user_query: str,
        candidates: Sequence[SkillMatch],
        surface: Optional[str],
        phase: Optional[str],
    ) -> str:
        payload: List[Dict[str, object]] = []
        for index, match in enumerate(candidates, start=1):
            skill = self.registry.get_skill(match.skill_id)
            if skill is None:
                continue
            payload.append(
                {
                    "rank": index,
                    "skill_id": skill.skill_id,
                    "display_name": skill.display_name,
                    "type": skill.type,
                    "subtype": skill.subtype,
                    "router_score": round(match.score, 4),
                    "router_source": match.source,
                    "matched_terms": list(match.matched_terms),
                    "routing_card": skill.routing_card.to_prompt_payload(),
                }
            )
        body = {
            "user_query": user_query,
            "surface": surface or "",
            "phase": phase or "",
            "candidate_skills": payload,
            "constraints": {
                "allowed_skill_ids": [item["skill_id"] for item in payload],
                "select_only_from_candidates": True,
                "allow_no_suitable_skill": True,
            },
        }
        return json.dumps(body, ensure_ascii=False, indent=2)

    def _parse_json_result(self, text: str) -> Optional[Dict[str, object]]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            loaded = json.loads(cleaned)
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    def _normalize_result(
        self,
        parsed: Dict[str, object],
        candidates: Sequence[SkillMatch],
        user_query: str = "",
    ) -> SemanticSkillJudgment:
        candidate_ids = [match.skill_id for match in candidates]
        allowed = set(candidate_ids)
        primary = parsed.get("primary_skill")
        primary_skill = str(primary).strip() if isinstance(primary, str) and str(primary).strip() in allowed else None

        secondary_skills: List[str] = []
        raw_secondary = parsed.get("secondary_skills") or []
        if isinstance(raw_secondary, list):
            for item in raw_secondary:
                skill_id = str(item).strip()
                if skill_id in allowed and skill_id != primary_skill and skill_id not in secondary_skills:
                    secondary_skills.append(skill_id)

        explicit_rejected: List[str] = []
        raw_rejected = parsed.get("rejected_skills") or []
        if isinstance(raw_rejected, list):
            for item in raw_rejected:
                skill_id = str(item).strip()
                if skill_id in allowed and skill_id != primary_skill and skill_id not in secondary_skills and skill_id not in explicit_rejected:
                    explicit_rejected.append(skill_id)

        accepted = [skill_id for skill_id in ([primary_skill] if primary_skill else []) + secondary_skills if skill_id]
        rejected_skills = list(explicit_rejected)
        for skill_id in candidate_ids:
            if skill_id not in accepted and skill_id not in rejected_skills:
                rejected_skills.append(skill_id)

        confidence = self._parse_confidence(parsed.get("confidence"))
        if confidence <= 0.0:
            confidence = max(0.0, min(0.95, candidates[0].score if accepted else min(0.49, candidates[0].score)))

        intent_summary = str(parsed.get("intent_summary") or "").strip()
        if not intent_summary:
            intent_summary = self._build_fallback_summary(user_query, candidates, no_match=not bool(accepted))

        return SemanticSkillJudgment(
            primary_skill=primary_skill,
            secondary_skills=tuple(secondary_skills),
            rejected_skills=tuple(rejected_skills),
            confidence=confidence,
            intent_summary=intent_summary,
            source="llm",
        )

    def _parse_confidence(self, raw_value: object) -> float:
        try:
            value = float(raw_value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, value))

    def _build_fallback_summary(self, user_query: str, candidates: Sequence[SkillMatch], no_match: bool = False) -> str:
        if not candidates:
            return "No candidate skills available."
        if no_match:
            return f"No suitable skill identified for query: {user_query.strip()[:80]}" if user_query else "No suitable skill identified from recalled candidates."
        return f"Fallback semantic judgment for query: {user_query.strip()[:80]}" if user_query else f"Fallback semantic judgment favoring {candidates[0].skill_id}."
