"""Skill selection, policy resolution, and prompt compilation.

技能管理器（SkillManager）
这是 Agent 系统中技能层面的“总指挥”和“胶水层”。
它负责：
1. 协调 SkillPolicyEngine 进行智能技能决策
2. 管理技能解析结果的缓存
3. 将选中的技能编译成 Prompt（通过 SkillCompiler）
4. 提供技能后处理（postprocess）能力
5. 对外暴露简洁的高层 API，供 ContextAgent、聊天流程等使用
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.hybrid_skill_router import HybridSkillRouter
from agent.skill_compiler import SkillCompiler
from agent.semantic_skill_judge import SemanticSkillJudge
from agent.skill_policy_engine import PhaseSkillPlan, SkillPolicyEngine
from agent.skill_registry import LoadedSkill, SkillRegistry


@dataclass(frozen=True)
class ResolvedSkill:
    """解析后的技能最终表示形式（对外暴露的简化结构）。

    与 PhaseSkillPlan 中的 SkillSelection 相比，去掉了部分内部字段，
    更适合上层业务代码使用。
    """
    skill: LoadedSkill
    score: float
    source: str
    role: str = "modifier"      # "primary" 或 "modifier"
    phase: str = "generation"


class SkillManager:
    """技能管理器（Skill Manager）—— 技能系统的对外统一入口。

    核心职责：
    - 封装 SkillPolicyEngine 的复杂决策逻辑
    - 提供缓存机制（提升性能）
    - 负责将技能计划编译成最终 Prompt
    - 支持手动指定技能（环境变量控制）
    - 提供技能后处理（postprocess_response）
    """

    def __init__(
        self,
        registry: SkillRegistry,
        router: Optional[HybridSkillRouter] = None,
        semantic_judge: Optional[SemanticSkillJudge] = None,
        threshold: float = 0.75,
        max_dynamic_skills: int = 3,
        environment: Optional[str] = None,
    ) -> None:
        """初始化 SkillManager 并组装核心组件。"""
        self.registry = registry
        # 如果没有传入 router，则自动创建默认的 HybridSkillRouter
        self.router = router or HybridSkillRouter(registry)
        # 如果没有传入 semantic_judge，则自动创建默认的 SemanticSkillJudge
        self.semantic_judge = semantic_judge or SemanticSkillJudge(registry)

        self.threshold = threshold
        self.max_dynamic_skills = max_dynamic_skills

        # 支持通过环境变量强制激活某些技能（用于调试或特定场景）
        self._manual_skill_ids = self._parse_manual_skill_ids(os.environ.get("ASSISTANT_ACTIVE_SKILL") or "")

        # 使用 OrderedDict 实现 LRU 风格的计划缓存（最近使用的排在末尾）
        self._plan_cache: "OrderedDict[Tuple[str, str, str], PhaseSkillPlan]" = OrderedDict()
        self._cache_limit = 48                     # 最多缓存 48 个不同的 (query, scope, phase) 组合

        # 核心策略引擎：负责复杂的技能选择、依赖、冲突、角色分配等逻辑
        self.policy_engine = SkillPolicyEngine(
            registry,
            self.router,
            threshold=threshold,
            max_dynamic_skills=max_dynamic_skills,
            environment=environment,
        )

        # 技能编译器：负责把选中的技能组合成最终的 Prompt 叠加层
        self.compiler = SkillCompiler()

    def attach_chat_tool(self, chat_tool: object) -> None:
        """将 ChatTool 附加到语义法官（用于语义判断时可能需要调用 LLM）。"""
        if hasattr(self.semantic_judge, "attach_chat_tool"):
            self.semantic_judge.attach_chat_tool(chat_tool)  # type: ignore[arg-type]
        # 附加工具后清除缓存，避免使用旧的语义判断结果
        self._plan_cache.clear()

    def active_skill(self) -> Optional[LoadedSkill]:
        """获取当前聊天场景下的“活跃技能”（主要用于生成阶段）。

        优先级：primary → 第一个 selected → 默认技能
        """
        plan = self.resolve_plan(user_query="", scope="chat", phase="generation")
        if plan.primary is not None:
            return plan.primary.skill
        if plan.selected:
            return plan.selected[0].skill
        return self.registry.get_default_skill(surface="chat", phase="generation")

    def resolve_plan(self, user_query: str, scope: str, phase: str = "generation") -> PhaseSkillPlan:
        """解析当前用户查询 + scope + phase 下的完整技能执行计划（核心方法）。

        流程：
        1. 检查缓存
        2. 路由获取候选匹配
        3. 进行精细语义判断
        4. 调用 PolicyEngine 生成最终计划
        5. 缓存结果
        """
        # 生成缓存键（user_query, scope, phase）
        cache_key = ((user_query or "").strip(), (scope or "").strip(), (phase or "").strip())

        # 命中缓存则直接返回，并将该项移到最近使用位置（LRU 机制）
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            self._plan_cache.move_to_end(cache_key)
            return cached

        # 只允许 prompt 和 hybrid 类型的技能参与生成阶段
        allowed_types = ("prompt", "hybrid")

        # 通过路由器获取所有可能的动态匹配
        recalled_matches = self.router.route_all(user_query, surface=scope, phase=phase)

        # 过滤并准备动态候选（按类型过滤 + 限制数量）
        candidate_matches = self._prepare_dynamic_candidates(recalled_matches, allowed_types=allowed_types)

        # 如果有候选，则让语义法官进行更智能的判断
        semantic_judgment = None
        if candidate_matches:
            semantic_judgment = self.semantic_judge.judge(
                user_query=user_query,
                candidates=candidate_matches,
                surface=scope,
                phase=phase,
            )

        # 调用策略引擎生成最终计划（这是最核心的决策步骤）
        plan = self.policy_engine.build_plan(
            user_query=user_query,
            surface=scope,
            phase=phase,
            manual_skill_ids=self._manual_skill_ids,
            allowed_types=allowed_types,
            candidate_matches=candidate_matches,
            semantic_judgment=semantic_judgment,
        )

        # 记住本次计划（放入缓存）
        self._remember_plan(cache_key, plan)
        return plan

    def resolve_skills(self, user_query: str, scope: str, phase: str = "generation") -> List[ResolvedSkill]:
        """将 PhaseSkillPlan 转换为对外友好的 ResolvedSkill 列表。"""
        plan = self.resolve_plan(user_query=user_query, scope=scope, phase=phase)
        return [
            ResolvedSkill(
                skill=item.skill,
                score=item.score,
                source=item.source,
                role=item.role,
                phase=phase,
            )
            for item in plan.selected
        ]

    def resolved_skill_ids(self, user_query: str, scope: str, phase: str = "generation") -> List[str]:
        """便捷方法：返回本次解析中最终选中的所有 skill_id 列表。"""
        return [item.skill.skill_id for item in self.resolve_skills(user_query, scope, phase=phase)]

    def augment_prompt(self, base_prompt: str, scope: str, user_query: str = "") -> str:
        """在基础 Prompt 上叠加当前选中的技能（最常用接口）。

        这是 SkillManager 最核心的“Prompt 增强”能力。
        """
        plan = self.resolve_plan(user_query=user_query, scope=scope, phase="generation")
        return self.compiler.compile_prompt(base_prompt, plan)

    def postprocess_response(self, text: str, scope: str, user_query: str = "") -> str:
        """对模型生成的最终回复进行后处理（postprocess 阶段）。

        当前主要进行格式清理（移除多余的分隔线、标题、加粗等）。
        未来可扩展为使用 postprocess 类型的技能进行更智能的润色。
        """
        plan = self.resolve_plan(user_query=user_query, scope=scope, phase="postprocess")
        if not plan.selected:
            return (text or "").strip()

        cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not cleaned:
            return cleaned

        # 清理常见的 Markdown 格式残留
        cleaned = re.sub(r"(?m)^\s*(?:---+|\*\*\*+|___+)\s*$", "", cleaned)   # 移除分隔线
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)               # 移除标题
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)                    # 移除加粗
        cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)                        # 移除下划线强调
        cleaned = re.sub(r"(?m)^\s*[-*]{3,}\s*$", "", cleaned)                # 移除水平线
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)                          # 压缩多余空行

        return cleaned.strip()

    def _parse_manual_skill_ids(self, raw_value: str) -> List[str]:
        """解析环境变量中手动指定的技能 ID 列表（逗号分隔）。"""
        return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]

    def _prepare_dynamic_candidates(self, matches, allowed_types: Tuple[str, ...]):
        """对路由器返回的匹配结果进行过滤和裁剪，准备给语义法官使用。"""
        allowed = {item.strip().lower() for item in allowed_types if str(item).strip()}
        filtered = []
        for match in matches:
            skill = self.registry.get_skill(match.skill_id)
            if skill is None:
                continue
            if allowed and skill.type not in allowed:
                continue
            filtered.append(match)

        # 如果语义判断设置了 top_k，则只保留前 N 个候选
        top_k = getattr(self.semantic_judge, "top_k", 0) or 0
        if top_k > 0:
            return filtered[:top_k]
        return filtered

    def _remember_plan(self, cache_key: Tuple[str, str, str], plan: PhaseSkillPlan) -> None:
        """将计划存入缓存，并维护 LRU 淘汰策略。"""
        self._plan_cache[cache_key] = plan
        self._plan_cache.move_to_end(cache_key)          # 移到最近使用位置
        # 超出缓存上限时，删除最旧的条目
        while len(self._plan_cache) > self._cache_limit:
            self._plan_cache.popitem(last=False)