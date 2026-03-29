from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from skill_playground.agent.hybrid_skill_router import HybridSkillRouter, SkillMatch
from skill_playground.agent.semantic_skill_judge import SemanticSkillJudgment
from skill_playground.agent.skill_registry import LoadedSkill, SkillRegistry
from skill_playground.agent.trace import TraceRecorder


@dataclass(frozen=True)
class SkillDecision:
    skill_id: str
    phase: str
    decision: str
    source: str
    reason: str
    score: float = 0.0


@dataclass(frozen=True)
class SkillSelection:
    skill: LoadedSkill
    score: float
    source: str
    phase: str
    role: str = "modifier"
    reason: str = ""


@dataclass(frozen=True)
class PhaseSkillPlan:
    surface: str
    phase: str
    primary: Optional[SkillSelection]
    modifiers: Tuple[SkillSelection, ...]
    selected: Tuple[SkillSelection, ...]
    decisions: Tuple[SkillDecision, ...]
    semantic_judgment: Optional[SemanticSkillJudgment] = None

    def selected_ids(self) -> Tuple[str, ...]:
        return tuple(item.skill.skill_id for item in self.selected)


class SkillPolicyEngine:
    SOURCE_PRIORITY = {"manual": 4, "dependency": 3, "dynamic": 2, "auto_apply": 1}
    RISK_PRIORITY = {"low": 3, "medium": 2, "high": 1}

    def __init__(
        self,
        registry: SkillRegistry,
        router: HybridSkillRouter,
        threshold: float = 0.75,
        max_dynamic_skills: int = 3,
        environment: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.threshold = threshold
        self.max_dynamic_skills = max_dynamic_skills
        self.environment = (
            environment
            or os.environ.get("ASSISTANT_ENV")
            or os.environ.get("APP_ENV")
            or "dev"
        ).strip().lower()
        self.default_thresholds = {
            "prompt": {"generation": threshold, "postprocess": 0.0, "planning": max(threshold, 0.78)},
            "handler": {"execution": 0.9, "planning": max(threshold, 0.82)},
            "tool": {"execution": 0.9, "planning": max(threshold, 0.82)},
            "hybrid": {"generation": threshold, "postprocess": 0.0, "planning": max(threshold, 0.8), "execution": 0.88},
        }

    def build_plan(
        self,
        user_query: str,
        surface: str,
        phase: str,
        manual_skill_ids: Optional[Sequence[str]] = None,
        allowed_types: Optional[Sequence[str]] = None,
        candidate_matches: Optional[Sequence[SkillMatch]] = None,
        semantic_judgment: Optional[SemanticSkillJudgment] = None,
        trace: Optional[TraceRecorder] = None,
    ) -> PhaseSkillPlan:
        if trace:
            trace.log(
                "policy_engine",
                "Building final skill plan.",
                {"surface": surface, "phase": phase, "manual_skill_ids": list(manual_skill_ids or []), "allowed_types": list(allowed_types or [])},
            )
        allowed_type_set = {item.strip().lower() for item in (allowed_types or []) if str(item).strip()}
        selected: Dict[str, SkillSelection] = {}
        decisions: List[SkillDecision] = []

        for skill_id in manual_skill_ids or []:
            skill = self.registry.get_skill(skill_id)
            if skill is None:
                decisions.append(SkillDecision(skill_id, phase, "rejected", "manual", "skill_not_found"))
                continue
            self._admit_static(selected, decisions, skill, "manual", surface, phase, user_query, allowed_type_set)

        for skill in self.registry.list_auto_apply_skills(surface=surface, phase=phase, environment=self.environment, rollout_key=user_query):
            self._admit_static(selected, decisions, skill, "auto_apply", surface, phase, user_query, allowed_type_set)

        routed_matches = list(candidate_matches) if candidate_matches is not None else self.router.route_all(user_query, surface=surface, phase=phase)
        if semantic_judgment is not None and not semantic_judgment.accepted_skills() and routed_matches:
            decisions.append(SkillDecision("*", phase, "rejected", "semantic", "no_suitable_skill_for_phase", semantic_judgment.confidence))

        dynamic_matches = 0
        for match in self._filtered_dynamic_matches(routed_matches, semantic_judgment, decisions, phase):
            skill = self.registry.get_skill(match.skill_id)
            if skill is None:
                continue
            if not self._supports_type(skill, allowed_type_set):
                decisions.append(SkillDecision(skill.skill_id, phase, "rejected", match.source, "type_not_allowed", match.score))
                continue
            allowed, reason = self._is_eligible(skill, surface=surface, phase=phase, user_query=user_query)
            if not allowed:
                decisions.append(SkillDecision(skill.skill_id, phase, "rejected", match.source, reason, match.score))
                continue
            threshold = self._threshold_for(skill, phase)
            if match.score < threshold:
                decisions.append(SkillDecision(skill.skill_id, phase, "rejected", match.source, f"below_threshold:{threshold:.2f}", match.score))
                continue
            admitted = self._merge_selection(selected, decisions, skill=skill, score=match.score, source="dynamic", phase=phase)
            if admitted:
                dynamic_matches += 1
            if dynamic_matches >= self.max_dynamic_skills:
                break

        self._inject_dependencies(selected, decisions, surface=surface, phase=phase, user_query=user_query, allowed_types=allowed_type_set)
        self._remove_unmet_dependencies(selected, decisions, phase=phase)

        ordered = self._ordered(selected.values())
        preferred_primary_skill_id = self._qualified_primary_skill_id(
            semantic_judgment.primary_skill if semantic_judgment is not None else None,
            ordered,
            phase=phase,
        )
        semantic_no_match = semantic_judgment is not None and not semantic_judgment.accepted_skills()
        primary, modifiers, finalized = self._assign_roles(
            ordered,
            phase=phase,
            preferred_primary_skill_id=preferred_primary_skill_id,
            semantic_no_match=semantic_no_match,
        )
        plan = PhaseSkillPlan(surface, phase, primary, tuple(modifiers), tuple(finalized), tuple(decisions), semantic_judgment)
        if trace:
            trace.log(
                "policy_engine",
                "Final skill plan created.",
                {
                    "selected": [item.skill.skill_id for item in finalized],
                    "primary": primary.skill.skill_id if primary else None,
                    "modifier_count": len(modifiers),
                    "decision_count": len(decisions),
                },
            )
        return plan

    def _filtered_dynamic_matches(
        self,
        matches: Sequence[SkillMatch],
        semantic_judgment: Optional[SemanticSkillJudgment],
        decisions: List[SkillDecision],
        phase: str,
    ) -> List[SkillMatch]:
        if semantic_judgment is None:
            return list(matches)
        accepted_ids = set(semantic_judgment.accepted_skills())
        if not accepted_ids:
            for match in matches:
                decisions.append(SkillDecision(match.skill_id, phase, "rejected", "semantic", "no_suitable_skill", match.score))
            return []
        filtered: List[SkillMatch] = []
        for match in matches:
            if match.skill_id in accepted_ids:
                filtered.append(match)
            else:
                decisions.append(SkillDecision(match.skill_id, phase, "rejected", "semantic", "semantic_rejected", match.score))
        return filtered

    def _admit_static(
        self,
        selected: Dict[str, SkillSelection],
        decisions: List[SkillDecision],
        skill: LoadedSkill,
        source: str,
        surface: str,
        phase: str,
        user_query: str,
        allowed_types: Sequence[str],
    ) -> None:
        if not self._supports_type(skill, allowed_types):
            decisions.append(SkillDecision(skill.skill_id, phase, "rejected", source, "type_not_allowed"))
            return
        allowed, reason = self._is_eligible(skill, surface=surface, phase=phase, user_query=user_query)
        if not allowed:
            decisions.append(SkillDecision(skill.skill_id, phase, "rejected", source, reason))
            return
        self._merge_selection(selected, decisions, skill=skill, score=1.0, source=source, phase=phase)

    def _inject_dependencies(
        self,
        selected: Dict[str, SkillSelection],
        decisions: List[SkillDecision],
        surface: str,
        phase: str,
        user_query: str,
        allowed_types: Sequence[str],
    ) -> None:
        changed = True
        while changed:
            changed = False
            for item in list(selected.values()):
                missing = [dep_id for dep_id in item.skill.depends_on if dep_id not in selected]
                if not missing:
                    continue
                dep_id = missing[0]
                dep_skill = self.registry.get_skill(dep_id)
                if dep_skill is None:
                    selected.pop(item.skill.skill_id, None)
                    decisions.append(SkillDecision(item.skill.skill_id, phase, "rejected", item.source, f"missing_dependency:{dep_id}", item.score))
                    changed = True
                    break
                if not self._supports_type(dep_skill, allowed_types):
                    selected.pop(item.skill.skill_id, None)
                    decisions.append(SkillDecision(item.skill.skill_id, phase, "rejected", item.source, f"dependency_type_not_allowed:{dep_id}", item.score))
                    changed = True
                    break
                allowed, reason = self._is_eligible(dep_skill, surface=surface, phase=phase, user_query=user_query)
                if not allowed:
                    selected.pop(item.skill.skill_id, None)
                    decisions.append(SkillDecision(item.skill.skill_id, phase, "rejected", item.source, f"dependency_unavailable:{dep_id}:{reason}", item.score))
                    changed = True
                    break
                self._merge_selection(selected, decisions, dep_skill, 1.0, "dependency", phase)
                changed = True
                break

    def _remove_unmet_dependencies(self, selected: Dict[str, SkillSelection], decisions: List[SkillDecision], phase: str) -> None:
        changed = True
        while changed:
            changed = False
            for item in list(selected.values()):
                if all(dep_id in selected for dep_id in item.skill.depends_on):
                    continue
                selected.pop(item.skill.skill_id, None)
                decisions.append(SkillDecision(item.skill.skill_id, phase, "rejected", item.source, "dependency_not_selected", item.score))
                changed = True
                break

    def _merge_selection(
        self,
        selected: Dict[str, SkillSelection],
        decisions: List[SkillDecision],
        skill: LoadedSkill,
        score: float,
        source: str,
        phase: str,
    ) -> bool:
        existing = selected.get(skill.skill_id)
        if existing is not None:
            selected[skill.skill_id] = SkillSelection(
                skill=existing.skill,
                score=max(existing.score, score),
                source=self._merge_source(existing.source, source),
                phase=phase,
                role=existing.role,
                reason=existing.reason,
            )
            return False

        candidate = SkillSelection(skill=skill, score=score, source=source, phase=phase)
        for existing_id, existing_item in list(selected.items()):
            if not self._skills_conflict(candidate.skill, existing_item.skill):
                continue
            if self._compare_precedence(candidate, existing_item) > 0:
                selected.pop(existing_id, None)
                decisions.append(SkillDecision(existing_item.skill.skill_id, phase, "rejected", existing_item.source, f"conflict_with:{candidate.skill.skill_id}", existing_item.score))
                continue
            decisions.append(SkillDecision(candidate.skill.skill_id, phase, "rejected", source, f"conflict_with:{existing_item.skill.skill_id}", score))
            return False

        selected[skill.skill_id] = candidate
        return True

    def _is_eligible(self, skill: LoadedSkill, surface: str, phase: str, user_query: str) -> Tuple[bool, str]:
        if not skill.enabled:
            return False, "disabled"
        if not skill.supports_surface(surface):
            return False, "surface_mismatch"
        if not skill.supports_phase(phase):
            return False, "phase_mismatch"
        if not skill.rollout.allows_env(self.environment):
            return False, f"env_mismatch:{self.environment}"
        if not self._passes_rollout(skill, user_query, surface, phase):
            return False, "rollout_filtered"
        return True, "selected"

    def _passes_rollout(self, skill: LoadedSkill, user_query: str, surface: str, phase: str) -> bool:
        traffic = max(0, min(100, int(skill.rollout.traffic_percent)))
        if traffic >= 100:
            return True
        if traffic <= 0:
            return False
        key = f"{skill.skill_id}|{surface}|{phase}|{self.environment}|{user_query or ''}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < traffic

    def _threshold_for(self, skill: LoadedSkill, phase: str) -> float:
        phase_key = (phase or "generation").strip().lower()
        configured = skill.thresholds.get(phase_key)
        if configured is not None:
            return configured
        type_defaults = self.default_thresholds.get(skill.type, {})
        base = float(type_defaults.get(phase_key, type_defaults.get("generation", self.threshold)))
        if skill.threshold > 0:
            return max(base, skill.threshold)
        return base

    def _supports_type(self, skill: LoadedSkill, allowed_types: Sequence[str]) -> bool:
        if not allowed_types:
            return True
        return skill.type in allowed_types

    def _skills_conflict(self, left: LoadedSkill, right: LoadedSkill) -> bool:
        return right.skill_id in left.conflicts_with or left.skill_id in right.conflicts_with

    def _compare_precedence(self, left: SkillSelection, right: SkillSelection) -> int:
        left_key = self._precedence_tuple(left)
        right_key = self._precedence_tuple(right)
        if left_key > right_key:
            return 1
        if left_key < right_key:
            return -1
        return 0

    def _precedence_tuple(self, item: SkillSelection) -> Tuple[int, int, float, int, str]:
        return (
            self._source_rank(item.source),
            item.skill.priority,
            item.score,
            self.RISK_PRIORITY.get(item.skill.risk_level, 0),
            item.skill.skill_id,
        )

    def _ordered(self, items: Sequence[SkillSelection]) -> List[SkillSelection]:
        return sorted(
            items,
            key=lambda item: (
                -item.skill.priority,
                -self._source_rank(item.source),
                -item.score,
                -self.RISK_PRIORITY.get(item.skill.risk_level, 0),
                item.skill.skill_id,
            ),
        )

    def _qualified_primary_skill_id(
        self,
        preferred_primary_skill_id: Optional[str],
        items: Sequence[SkillSelection],
        phase: str,
    ) -> Optional[str]:
        if not preferred_primary_skill_id:
            return None
        for item in items:
            if item.skill.skill_id != preferred_primary_skill_id:
                continue
            return preferred_primary_skill_id if self._can_be_primary(item, phase=phase) else None
        return None

    def _can_be_primary(self, item: SkillSelection, phase: str) -> bool:
        if phase != "generation":
            return False
        if not item.skill.is_prompt_skill():
            return False
        if item.skill.auto_apply:
            return False
        if item.skill.subtype in {"style", "postprocess"}:
            return False
        return True

    def _assign_roles(
        self,
        items: Sequence[SkillSelection],
        phase: str,
        preferred_primary_skill_id: Optional[str] = None,
        semantic_no_match: bool = False,
    ) -> Tuple[Optional[SkillSelection], List[SkillSelection], List[SkillSelection]]:
        primary_id: Optional[str] = None
        if not semantic_no_match:
            if preferred_primary_skill_id and any(item.skill.skill_id == preferred_primary_skill_id for item in items):
                primary_id = preferred_primary_skill_id
            elif phase == "generation":
                primary_candidates = [item for item in items if self._can_be_primary(item, phase=phase) and item.source != "auto_apply"]
                if primary_candidates:
                    primary_id = primary_candidates[0].skill.skill_id

        finalized: List[SkillSelection] = []
        primary: Optional[SkillSelection] = None
        modifiers: List[SkillSelection] = []
        for item in items:
            if primary_id and item.skill.skill_id == primary_id:
                updated = replace(item, role="primary", reason="semantic-or-policy primary")
                primary = updated
                finalized.append(updated)
                continue
            reason = "modifier retained after semantic no-match" if semantic_no_match else ("always-on modifier" if item.skill.auto_apply else "phase-compatible modifier")
            updated = replace(item, role="modifier", reason=reason)
            modifiers.append(updated)
            finalized.append(updated)

        if primary is not None:
            finalized = [primary] + [item for item in finalized if item.skill.skill_id != primary.skill.skill_id]
        return primary, modifiers, finalized

    def _source_rank(self, source: str) -> int:
        ranks = [self.SOURCE_PRIORITY.get(part.strip(), 0) for part in str(source or "").split("+") if part.strip()]
        return max(ranks) if ranks else 0

    def _merge_source(self, left: str, right: str) -> str:
        items = [item.strip() for item in f"{left}+{right}".split("+") if item.strip()]
        merged: List[str] = []
        for item in items:
            if item not in merged:
                merged.append(item)
        return "+".join(merged)
