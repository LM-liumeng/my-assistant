"""Phase-aware skill policy resolution for prompt and future handler skills.

技能策略引擎（Skill Policy Engine）
核心职责：在不同的 surface（交互界面）和 phase（处理阶段）下，
根据用户查询智能地决定应该使用哪些技能（Skill），并为它们分配角色（primary / modifier），
同时处理手动指定、自动应用、动态路由、依赖注入、冲突解决、灰度发布等复杂策略。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from agent.hybrid_skill_router import HybridSkillRouter, SkillMatch
from agent.semantic_skill_judge import SemanticSkillJudgment
from agent.skill_registry import LoadedSkill, SkillRegistry


@dataclass(frozen=True)
class SkillDecision:
    """记录一次技能决策的过程（用于审计和调试）。"""
    skill_id: str
    phase: str
    decision: str          # "selected" / "rejected"
    source: str            # manual / auto_apply / dynamic / dependency / semantic
    reason: str            # 拒绝或选择的具体原因
    score: float = 0.0     # 匹配分数（仅 dynamic 时有意义）


@dataclass(frozen=True)
class SkillSelection:
    """表示一个被选中（或候选）的技能及其元信息。"""
    skill: LoadedSkill
    score: float
    source: str
    phase: str
    role: str = "modifier"      # "primary" 或 "modifier"
    reason: str = ""            # 为什么被选为该角色


@dataclass(frozen=True)
class PhaseSkillPlan:
    """一次技能策略决策的最终输出结果（Phase-aware Skill Plan）。"""
    surface: str
    phase: str
    primary: Optional[SkillSelection]           # 生成阶段的主提示技能（可选）
    modifiers: Tuple[SkillSelection, ...]       # 所有修改器技能
    selected: Tuple[SkillSelection, ...]        # 最终选中的所有技能（有序）
    decisions: Tuple[SkillDecision, ...]        # 完整的决策轨迹（可审计）
    semantic_judgment: Optional[SemanticSkillJudgment] = None

    def selected_ids(self) -> Tuple[str, ...]:
        """返回最终选中的所有 skill_id，方便下游使用。"""
        return tuple(item.skill.skill_id for item in self.selected)


class SkillPolicyEngine:
    """阶段感知的技能策略引擎（核心决策大脑）。

    它整合了多种来源的技能：
        - manual（手动强制指定）
        - auto_apply（自动应用）
        - dynamic（通过 HybridSkillRouter 语义匹配）
        - dependency（依赖自动注入）

    并进行冲突解决、依赖管理、灰度控制、阈值过滤、角色分配等一系列策略决策。
    """

    # 来源优先级（数字越大优先级越高）
    SOURCE_PRIORITY = {
        "manual": 4,
        "dependency": 3,
        "dynamic": 2,
        "auto_apply": 1,
    }

    # 风险等级优先级（low 风险的技能更优先保留）
    RISK_PRIORITY = {
        "low": 3,
        "medium": 2,
        "high": 1,
    }

    def __init__(
        self,
        registry: SkillRegistry,
        router: HybridSkillRouter,
        threshold: float = 0.75,
        max_dynamic_skills: int = 3,
        environment: Optional[str] = None,
    ) -> None:
        """初始化技能策略引擎。"""
        self.registry = registry
        self.router = router
        self.threshold = threshold
        self.max_dynamic_skills = max_dynamic_skills

        # 环境识别（支持 dev / prod / staging 等）
        self.environment = (
            environment
            or os.environ.get("ASSISTANT_ENV")
            or os.environ.get("APP_ENV")
            or "dev"
        ).strip().lower()

        # 不同技能类型 + 不同阶段的默认阈值配置
        self.default_thresholds = {
            "prompt": {
                "planning": max(threshold, 0.78),
                "generation": threshold,
                "postprocess": 0.0,
            },
            "handler": {
                "planning": max(threshold, 0.82),
                "execution": 0.90,
            },
            "tool": {
                "planning": max(threshold, 0.82),
                "execution": 0.90,
            },
            "hybrid": {
                "planning": max(threshold, 0.80),
                "generation": threshold,
                "execution": 0.88,
                "postprocess": 0.0,
            },
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
    ) -> PhaseSkillPlan:
        """构建当前阶段的技能执行计划（核心入口方法）。

        处理流程：
        1. 手动指定的技能
        2. 自动应用的技能
        3. 动态路由匹配 + 语义过滤
        4. 依赖注入与清理
        5. 冲突解决与排序
        6. Primary / Modifier 角色分配
        """
        allowed_type_set = {item.strip().lower() for item in (allowed_types or []) if str(item).strip()}

        selected: Dict[str, SkillSelection] = {}   # 当前已选中的技能（去重）
        decisions: List[SkillDecision] = []        # 全程决策记录

        # ==================== 1. 处理手动指定技能 ====================
        for skill_id in manual_skill_ids or []:
            skill = self.registry.get_skill(skill_id)
            if skill is None:
                decisions.append(SkillDecision(skill_id=skill_id, phase=phase, decision="rejected", source="manual", reason="skill_not_found"))
                continue
            self._admit_static(selected, decisions, skill, source="manual", surface=surface, phase=phase, user_query=user_query, allowed_types=allowed_type_set)

        # ==================== 2. 处理自动应用技能 ====================
        for skill in self.registry.list_auto_apply_skills(surface=surface, phase=phase):
            self._admit_static(selected, decisions, skill, source="auto_apply", surface=surface, phase=phase, user_query=user_query, allowed_types=allowed_type_set)

        # ==================== 3. 处理动态路由匹配 ====================
        routed_matches = list(candidate_matches) if candidate_matches is not None else self.router.route_all(
            user_query, surface=surface, phase=phase
        )

        # 语义法官拒绝所有技能时的特殊处理
        if semantic_judgment is not None and not semantic_judgment.accepted_skills() and routed_matches:
            decisions.append(
                SkillDecision(
                    skill_id="*",
                    phase=phase,
                    decision="rejected",
                    source="semantic",
                    reason="no_suitable_skill_for_phase",
                    score=semantic_judgment.confidence,
                )
            )

        # 动态匹配过滤 + 准入检查
        dynamic_matches = 0
        for match in self._filtered_dynamic_matches(routed_matches, semantic_judgment, decisions, phase):
            skill = self.registry.get_skill(match.skill_id)
            if skill is None:
                continue

            if not self._supports_type(skill, allowed_type_set):
                decisions.append(SkillDecision(skill_id=skill.skill_id, phase=phase, decision="rejected", source=match.source, reason="type_not_allowed", score=match.score))
                continue

            allowed, reason = self._is_eligible(skill, surface=surface, phase=phase, user_query=user_query)
            if not allowed:
                decisions.append(SkillDecision(skill_id=skill.skill_id, phase=phase, decision="rejected", source=match.source, reason=reason, score=match.score))
                continue

            threshold = self._threshold_for(skill, phase)
            if match.score < threshold:
                decisions.append(SkillDecision(skill_id=skill.skill_id, phase=phase, decision="rejected", source=match.source, reason=f"below_threshold:{threshold:.2f}", score=match.score))
                continue

            admitted = self._merge_selection(selected, decisions, skill=skill, score=match.score, source="dynamic", phase=phase)
            if admitted:
                dynamic_matches += 1
            if dynamic_matches >= self.max_dynamic_skills:
                break

        # ==================== 4. 依赖管理 ====================
        self._inject_dependencies(selected, decisions, surface=surface, phase=phase, user_query=user_query, allowed_types=allowed_type_set)
        self._remove_unmet_dependencies(selected, decisions, phase=phase)

        # ==================== 5. 排序与角色分配 ====================
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

        return PhaseSkillPlan(
            surface=surface,
            phase=phase,
            primary=primary,
            modifiers=tuple(modifiers),
            selected=tuple(finalized),
            decisions=tuple(decisions),
            semantic_judgment=semantic_judgment,
        )

    # ====================== 以下为内部辅助方法 ======================

    def _filtered_dynamic_matches(
            self,
            matches: Sequence[SkillMatch],
            semantic_judgment: Optional[SemanticSkillJudgment],
            decisions: List[SkillDecision],
            phase: str,
    ) -> List[SkillMatch]:
        """根据语义法官结果过滤动态匹配的技能。"""
        if semantic_judgment is None:
            return list(matches)

        accepted_ids = set(semantic_judgment.accepted_skills())
        filtered: List[SkillMatch] = []

        if not accepted_ids:
            for match in matches:
                decisions.append(
                    SkillDecision(
                        skill_id=match.skill_id,
                        phase=phase,
                        decision="rejected",
                        source="semantic",
                        reason="no_suitable_skill",
                        score=match.score,
                    )
                )
            return []

        for match in matches:
            if match.skill_id in accepted_ids:
                filtered.append(match)
            else:
                decisions.append(
                    SkillDecision(
                        skill_id=match.skill_id,
                        phase=phase,
                        decision="rejected",
                        source="semantic",
                        reason="semantic_rejected",
                        score=match.score,
                    )
                )
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
        """静态技能（manual / auto_apply）的准入检查与录取。"""
        if not self._supports_type(skill, allowed_types):
            decisions.append(SkillDecision(skill_id=skill.skill_id, phase=phase, decision="rejected", source=source, reason="type_not_allowed"))
            return

        allowed, reason = self._is_eligible(skill, surface=surface, phase=phase, user_query=user_query)
        if not allowed:
            decisions.append(SkillDecision(skill_id=skill.skill_id, phase=phase, decision="rejected", source=source, reason=reason))
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
        """自动注入缺失的依赖技能（迭代直到稳定）。"""
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
                    decisions.append(SkillDecision(skill_id=item.skill.skill_id, phase=phase, decision="rejected", source=item.source, reason=f"missing_dependency:{dep_id}", score=item.score))
                    changed = True
                    break

                if not self._supports_type(dep_skill, allowed_types):
                    selected.pop(item.skill.skill_id, None)
                    decisions.append(SkillDecision(skill_id=item.skill.skill_id, phase=phase, decision="rejected", source=item.source, reason=f"dependency_type_not_allowed:{dep_id}", score=item.score))
                    changed = True
                    break

                allowed, reason = self._is_eligible(dep_skill, surface=surface, phase=phase, user_query=user_query)
                if not allowed:
                    selected.pop(item.skill.skill_id, None)
                    decisions.append(SkillDecision(skill_id=item.skill.skill_id, phase=phase, decision="rejected", source=item.source, reason=f"dependency_unavailable:{dep_id}:{reason}", score=item.score))
                    changed = True
                    break

                self._merge_selection(selected, decisions, skill=dep_skill, score=1.0, source="dependency", phase=phase)
                changed = True
                break

    def _remove_unmet_dependencies(
        self,
        selected: Dict[str, SkillSelection],
        decisions: List[SkillDecision],
        phase: str,
    ) -> None:
        """移除那些依赖没有被满足的技能（连锁清理）。"""
        changed = True
        while changed:
            changed = False
            for item in list(selected.values()):
                if all(dep_id in selected for dep_id in item.skill.depends_on):
                    continue
                selected.pop(item.skill.skill_id, None)
                decisions.append(SkillDecision(skill_id=item.skill.skill_id, phase=phase, decision="rejected", source=item.source, reason="dependency_not_selected", score=item.score))
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
        """将一个技能加入已选中集合，处理已存在、同名冲突、互斥冲突等情况。"""
        existing = selected.get(skill.skill_id)
        if existing is not None:
            # 已存在则取更高分数，并合并来源
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

        # 检查是否与已选中技能冲突
        for existing_id, existing_item in list(selected.items()):
            if not self._skills_conflict(candidate.skill, existing_item.skill):
                continue

            if self._compare_precedence(candidate, existing_item) > 0:
                # 新技能优先级更高，踢掉旧的
                selected.pop(existing_id, None)
                decisions.append(SkillDecision(skill_id=existing_item.skill.skill_id, phase=phase, decision="rejected", source=existing_item.source, reason=f"conflict_with:{candidate.skill.skill_id}", score=existing_item.score))
                continue
            else:
                # 旧技能优先级更高，拒绝新技能
                decisions.append(SkillDecision(skill_id=candidate.skill.skill_id, phase=phase, decision="rejected", source=source, reason=f"conflict_with:{existing_item.skill.skill_id}", score=score))
                return False

        selected[skill.skill_id] = candidate
        return True

    def _is_eligible(self, skill: LoadedSkill, surface: str, phase: str, user_query: str) -> Tuple[bool, str]:
        """检查技能是否符合当前环境、表面、阶段、灰度等准入条件。"""
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
        """灰度发布控制：基于哈希的分桶流量控制（确定性）。"""
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
        """获取当前技能在指定阶段应该使用的匹配阈值。"""
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
        """类型过滤（prompt / handler / tool / hybrid 等）。"""
        if not allowed_types:
            return True
        return skill.type in allowed_types

    def _skills_conflict(self, left: LoadedSkill, right: LoadedSkill) -> bool:
        """判断两个技能是否互斥。"""
        return right.skill_id in left.conflicts_with or left.skill_id in right.conflicts_with

    def _compare_precedence(self, left: SkillSelection, right: SkillSelection) -> int:
        """比较两个技能的优先级，返回 1 表示 left 更高，-1 表示 right 更高。"""
        left_key = self._precedence_tuple(left)
        right_key = self._precedence_tuple(right)
        if left_key > right_key:
            return 1
        if left_key < right_key:
            return -1
        return 0

    def _precedence_tuple(self, item: SkillSelection) -> Tuple[int, int, float, int, str]:
        """生成用于优先级比较的元组（从高到低重要性）。"""
        return (
            self._source_rank(item.source),
            item.skill.priority,
            item.score,
            self.RISK_PRIORITY.get(item.skill.risk_level, 0),
            item.skill.skill_id,
        )

    def _ordered(self, items: Sequence[SkillSelection]) -> List[SkillSelection]:
        """对选中的技能进行排序（优先级从高到低）。"""
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
        """判断语义法官推荐的 primary 是否真正合格。"""
        if not preferred_primary_skill_id:
            return None
        for item in items:
            if item.skill.skill_id != preferred_primary_skill_id:
                continue
            if self._can_be_primary(item, phase=phase):
                return preferred_primary_skill_id
            return None
        return None

    def _can_be_primary(self, item: SkillSelection, phase: str) -> bool:
        """判断一个技能是否可以作为 primary（主生成技能）。"""
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
        """为所有选中的技能分配角色：primary（最多一个）或 modifier（可以多个）。"""
        primary_id: Optional[str] = None

        if not semantic_no_match:
            if preferred_primary_skill_id and any(item.skill.skill_id == preferred_primary_skill_id for item in items):
                primary_id = preferred_primary_skill_id
            elif phase == "generation":
                primary_candidates = [
                    item
                    for item in items
                    if self._can_be_primary(item, phase=phase) and item.source != "auto_apply"
                ]
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

            reason = (
                "modifier retained after semantic no-match"
                if semantic_no_match
                else ("always-on modifier" if item.skill.auto_apply else "phase-compatible modifier")
            )
            updated = replace(item, role="modifier", reason=reason)
            modifiers.append(updated)
            finalized.append(updated)

        # 把 primary 排到最前面
        if primary is not None:
            finalized = [primary] + [item for item in finalized if item.skill.skill_id != primary.skill.skill_id]

        return primary, modifiers, finalized

    def _source_rank(self, source: str) -> int:
        """计算来源的优先级分数（支持合并来源如 dynamic+dependency）。"""
        ranks = [self.SOURCE_PRIORITY.get(part.strip(), 0) for part in str(source or "").split("+") if part.strip()]
        return max(ranks) if ranks else 0

    def _merge_source(self, left: str, right: str) -> str:
        """合并两个来源字符串，去重并保持顺序。"""
        items = [item.strip() for item in f"{left}+{right}".split("+") if item.strip()]
        merged: List[str] = []
        for item in items:
            if item not in merged:
                merged.append(item)
        return "+".join(merged)