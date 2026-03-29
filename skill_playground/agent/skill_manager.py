from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from skill_playground.agent.hybrid_skill_router import HybridSkillRouter
from skill_playground.agent.semantic_skill_judge import SemanticSkillJudge
from skill_playground.agent.skill_compiler import SkillCompiler
from skill_playground.agent.skill_policy_engine import PhaseSkillPlan, SkillPolicyEngine
from skill_playground.agent.skill_registry import LoadedSkill, SkillRegistry
from skill_playground.agent.trace import TraceRecorder


@dataclass(frozen=True)
class ResolvedSkill:
    skill: LoadedSkill
    score: float
    source: str
    role: str = "modifier"
    phase: str = "generation"


class SkillManager:
    def __init__(
        self,
        registry: SkillRegistry,
        router: Optional[HybridSkillRouter] = None,
        semantic_judge: Optional[SemanticSkillJudge] = None,
        threshold: float = 0.75,
        max_dynamic_skills: int = 3,
        environment: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.router = router or HybridSkillRouter(registry)
        self.semantic_judge = semantic_judge or SemanticSkillJudge(registry)
        self.threshold = threshold
        self.max_dynamic_skills = max_dynamic_skills
        self._manual_skill_ids = self._parse_manual_skill_ids(os.environ.get("ASSISTANT_ACTIVE_SKILL") or "")
        self._plan_cache: "OrderedDict[Tuple[str, str, str], PhaseSkillPlan]" = OrderedDict()
        self._cache_limit = 48
        self.policy_engine = SkillPolicyEngine(
            registry,
            self.router,
            threshold=threshold,
            max_dynamic_skills=max_dynamic_skills,
            environment=environment,
        )
        self.compiler = SkillCompiler()

    def attach_chat_tool(self, chat_tool: object) -> None:
        if hasattr(self.semantic_judge, "attach_chat_tool"):
            self.semantic_judge.attach_chat_tool(chat_tool)  # type: ignore[arg-type]
        self._plan_cache.clear()

    def active_skill(self) -> Optional[LoadedSkill]:
        plan = self.resolve_plan(user_query="", scope="chat", phase="generation")
        if plan.primary is not None:
            return plan.primary.skill
        if plan.selected:
            return plan.selected[0].skill
        return self.registry.get_default_skill(surface="chat", phase="generation")

    def resolve_plan(self, user_query: str, scope: str, phase: str = "generation", trace: Optional[TraceRecorder] = None) -> PhaseSkillPlan:
        cache_key = ((user_query or "").strip(), (scope or "").strip(), (phase or "").strip())
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            self._plan_cache.move_to_end(cache_key)
            if trace:
                trace.log("skill_manager", "Using cached skill plan.", {"scope": scope, "phase": phase, "selected": [item.skill.skill_id for item in cached.selected]})
            return cached

        if trace:
            trace.log("skill_manager", "Resolving skill plan.", {"scope": scope, "phase": phase, "query": user_query})
        allowed_types = ("prompt", "hybrid")
        recalled_matches = self.router.route_all(user_query, surface=scope, phase=phase, trace=trace)
        candidate_matches = self._prepare_dynamic_candidates(recalled_matches, allowed_types=allowed_types)
        if trace:
            trace.log(
                "skill_manager",
                "Prepared dynamic candidates for semantic judgment.",
                {"candidate_ids": [item.skill_id for item in candidate_matches], "allowed_types": list(allowed_types)},
            )

        semantic_judgment = None
        if candidate_matches:
            semantic_judgment = self.semantic_judge.judge(
                user_query=user_query,
                candidates=candidate_matches,
                surface=scope,
                phase=phase,
                trace=trace,
            )

        plan = self.policy_engine.build_plan(
            user_query=user_query,
            surface=scope,
            phase=phase,
            manual_skill_ids=self._manual_skill_ids,
            allowed_types=allowed_types,
            candidate_matches=candidate_matches,
            semantic_judgment=semantic_judgment,
            trace=trace,
        )
        self._remember_plan(cache_key, plan)
        return plan

    def resolve_skills(self, user_query: str, scope: str, phase: str = "generation", trace: Optional[TraceRecorder] = None) -> List[ResolvedSkill]:
        plan = self.resolve_plan(user_query=user_query, scope=scope, phase=phase, trace=trace)
        return [
            ResolvedSkill(skill=item.skill, score=item.score, source=item.source, role=item.role, phase=phase)
            for item in plan.selected
        ]

    def resolved_skill_ids(self, user_query: str, scope: str, phase: str = "generation", trace: Optional[TraceRecorder] = None) -> List[str]:
        return [item.skill.skill_id for item in self.resolve_skills(user_query, scope, phase=phase, trace=trace)]

    def augment_prompt(self, base_prompt: str, scope: str, user_query: str = "", trace: Optional[TraceRecorder] = None) -> str:
        plan = self.resolve_plan(user_query=user_query, scope=scope, phase="generation", trace=trace)
        return self.compiler.compile_prompt(base_prompt, plan)

    def postprocess_response(self, text: str, scope: str, user_query: str = "", trace: Optional[TraceRecorder] = None) -> str:
        _ = self.resolve_plan(user_query=user_query, scope=scope, phase="postprocess", trace=trace)
        cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not cleaned:
            return cleaned
        cleaned = re.sub(r"(?m)^\s*(?:---+|\*\*\*+|___+)\s*$", "", cleaned)
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
        cleaned = re.sub(r"(?m)^\s*[-*]{3,}\s*$", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        if trace:
            trace.log("skill_manager", "Applied postprocess skill cleanup.", {"length": len(cleaned)})
        return cleaned.strip()

    def _parse_manual_skill_ids(self, raw_value: str) -> List[str]:
        return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]

    def _prepare_dynamic_candidates(self, matches, allowed_types: Tuple[str, ...]):
        allowed = {item.strip().lower() for item in allowed_types if str(item).strip()}
        filtered = []
        for match in matches:
            skill = self.registry.get_skill(match.skill_id)
            if skill is None:
                continue
            if allowed and skill.type not in allowed:
                continue
            filtered.append(match)
        top_k = getattr(self.semantic_judge, "top_k", 0) or 0
        return filtered[:top_k] if top_k > 0 else filtered

    def _remember_plan(self, cache_key: Tuple[str, str, str], plan: PhaseSkillPlan) -> None:
        self._plan_cache[cache_key] = plan
        self._plan_cache.move_to_end(cache_key)
        while len(self._plan_cache) > self._cache_limit:
            self._plan_cache.popitem(last=False)
