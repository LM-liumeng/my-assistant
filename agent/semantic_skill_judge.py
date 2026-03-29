"""LLM-backed semantic judgment over router-recalled skill candidates.

语义技能判断
这是 Agent 技能系统中的“二次精炼”层。
它只在 HybridSkillRouter 已经召回的候选技能集合内进行语义判断，
避免 LLM 幻觉出不存在的技能，同时支持“没有合适技能”这一合法结果。

核心设计目标：
1. 严格限定在 router 召回的候选范围内（安全性）
2. 支持 LLM 判断 + 保守的启发式 fallback
3. 输出结构化的 SemanticSkillJudgment，供 SkillPolicyEngine 使用
4. 保持与原有 primary_skill / secondary_skills 的兼容性
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agent.hybrid_skill_router import SkillMatch
from agent.skill_registry import SkillRegistry
from tools.chat_tool import ChatTool


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
"""语义法官专用的 System Prompt。
要求 LLM 严格输出 JSON，且只能从提供的候选技能中选择，绝不能编造新 skill_id。
"""


@dataclass(frozen=True)
class SemanticSkillJudgment:
    """语义判断的最终结构化结果（Semantic Skill Judgment）。

    这是 SkillPolicyEngine 和 SkillManager 后续决策的重要输入。
    """

    primary_skill: Optional[str]          # 主技能（最多一个），可能为 None
    secondary_skills: Tuple[str, ...]     # 辅助技能（modifiers）
    rejected_skills: Tuple[str, ...]      # 被明确拒绝的技能
    confidence: float                     # 判断置信度（0.0~1.0）
    intent_summary: str                   # 对用户意图的简短总结
    source: str = "heuristic"             # 判断来源：llm 或 heuristic

    def accepted_skills(self) -> Tuple[str, ...]:
        """返回所有被接受的技能 ID（primary + secondary，去重）。"""
        items: List[str] = []
        if self.primary_skill:
            items.append(self.primary_skill)
        for skill_id in self.secondary_skills:
            if skill_id and skill_id not in items:
                items.append(skill_id)
        return tuple(items)

    @property
    def primary_candidate(self) -> Optional[str]:
        """兼容旧代码的属性：主技能候选。"""
        return self.primary_skill

    @property
    def secondary_candidates(self) -> Tuple[str, ...]:
        """兼容旧代码的属性：辅助技能候选。"""
        return self.secondary_skills

    @property
    def rejected_candidates(self) -> Tuple[str, ...]:
        """兼容旧代码的属性：被拒绝的技能。"""
        return self.rejected_skills

    def to_dict(self) -> Dict[str, object]:
        """转换为可序列化的字典，常用于日志、调试或 API 返回。"""
        return {
            "primary_skill": self.primary_skill,
            "secondary_skills": list(self.secondary_skills),
            "rejected_skills": list(self.rejected_skills),
            "confidence": self.confidence,
            "intent_summary": self.intent_summary,
            "source": self.source,
        }


class SemanticSkillJudge:
    """语义技能法官（Semantic Skill Judge）。

    职责：
    - 对 HybridSkillRouter 召回的候选技能进行二次语义精炼
    - 优先使用 LLM（ChatTool）进行判断，失败时回退到保守的启发式策略
    - 明确支持“没有合适技能”的合法结果，避免 LLM 强制选一个
    """

    def __init__(
        self,
        registry: SkillRegistry,
        chat_tool: Optional[ChatTool] = None,
        top_k: int = 6,
        no_match_threshold: float = 0.55,
    ) -> None:
        """
        Args:
            registry: 技能注册表（用于获取技能的 routing_card 等元数据）
            chat_tool: 用于 LLM 判断的聊天工具（可延迟注入）
            top_k: 最多只把前 N 个候选传给 LLM（控制成本和上下文长度）
            no_match_threshold: 启发式 fallback 时，只有 router 分数高于此阈值才认为有匹配
        """
        self.registry = registry
        self.chat_tool = chat_tool
        self.top_k = top_k
        self.no_match_threshold = no_match_threshold

    def attach_chat_tool(self, chat_tool: ChatTool) -> None:
        """运行时注入 ChatTool（支持 SkillManager 动态附加）。"""
        self.chat_tool = chat_tool

    def judge(
        self,
        user_query: str,
        candidates: Sequence[SkillMatch],
        surface: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> SemanticSkillJudgment:
        """对候选技能进行语义判断（核心入口方法）。

        流程：
        1. 限制候选数量（top_k）
        2. 如果没有 ChatTool 或未配置 → 直接走 heuristic fallback
        3. 构建结构化 Prompt → 调用 LLM
        4. 解析 JSON 结果并规范化
        5. LLM 失败时自动 fallback 到启发式判断
        """
        # 只取前 top_k 个候选，减少 Prompt 长度和成本
        limited_candidates = list(candidates[: self.top_k])
        if not limited_candidates:
            return SemanticSkillJudgment(
                primary_skill=None,
                secondary_skills=(),
                rejected_skills=(),
                confidence=0.0,
                intent_summary="No candidate skills were recalled.",
                source="heuristic",
            )

        # 如果没有可用的聊天工具，则直接使用保守的启发式判断
        if self.chat_tool is None or not self.chat_tool.is_configured():
            return self._heuristic_fallback(user_query, limited_candidates)

        # 构建给 LLM 的结构化 Prompt（JSON 格式）
        prompt = self._build_prompt(user_query, limited_candidates, surface=surface, phase=phase)

        # 调用 LLM 进行判断
        result = self.chat_tool.complete(prompt, system_prompt=_SEMANTIC_JUDGE_SYSTEM_PROMPT)

        # LLM 调用出错时回退
        if result.get("error"):
            return self._heuristic_fallback(user_query, limited_candidates)

        # 尝试解析 LLM 返回的 JSON
        parsed = self._parse_json_result(str(result.get("message") or ""))
        if not parsed:
            return self._heuristic_fallback(user_query, limited_candidates)

        # 规范化并返回最终判断结果
        return self._normalize_result(parsed, limited_candidates, user_query=user_query)

    def _heuristic_fallback(
        self,
        user_query: str,
        candidates: Sequence[SkillMatch],
    ) -> SemanticSkillJudgment:
        """保守的启发式 fallback 策略（当 LLM 不可用或解析失败时使用）。

        设计原则：宁可少选、也不乱选。
        只有 router 分数足够高时才接受，否则明确返回“no suitable skill”。
        """
        if not candidates:
            return SemanticSkillJudgment(
                primary_skill=None,
                secondary_skills=(),
                rejected_skills=(),
                confidence=0.0,
                intent_summary="No candidate skills available.",
                source="heuristic",
            )

        top = candidates[0]

        # router 分数足够高时，才认为有匹配
        if top.score >= self.no_match_threshold:
            primary_skill: Optional[str] = top.skill_id
            secondary = tuple(
                match.skill_id
                for match in candidates[1:]
                if match.score >= self.no_match_threshold and match.skill_id != top.skill_id
            )
            accepted = {primary_skill, *secondary}
            rejected = tuple(match.skill_id for match in candidates if match.skill_id not in accepted)

            confidence = max(0.0, min(0.95, top.score))
            return SemanticSkillJudgment(
                primary_skill=primary_skill,
                secondary_skills=secondary,
                rejected_skills=rejected,
                confidence=confidence,
                intent_summary=self._build_fallback_summary(user_query, candidates, no_match=False),
                source="heuristic",
            )

        # 否则明确表示“没有合适技能”
        return SemanticSkillJudgment(
            primary_skill=None,
            secondary_skills=(),
            rejected_skills=tuple(match.skill_id for match in candidates),
            confidence=max(0.0, min(0.49, top.score)),
            intent_summary=self._build_fallback_summary(user_query, candidates, no_match=True),
            source="heuristic",
        )

    def _build_prompt(
        self,
        user_query: str,
        candidates: Sequence[SkillMatch],
        surface: Optional[str],
        phase: Optional[str],
    ) -> str:
        """构建给 LLM 的结构化 JSON Prompt（包含候选技能的完整元数据）。"""
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
                    "routing_card": skill.routing_card.to_prompt_payload(),   # 包含 when_to_use、examples 等丰富信息
                }
            )

        body = {
            "user_query": user_query,
            "surface": surface or "",
            "phase": phase or "",
            "candidate_skills": payload,
            "constraints": {
                "allowed_skill_ids": [item["skill_id"] for item in payload],
                "select_only_from_candidates": True,      # 强制 LLM 只能从候选里选
                "allow_no_suitable_skill": True,          # 明确允许返回无合适技能
            },
        }
        return json.dumps(body, ensure_ascii=False, indent=2)

    def _parse_json_result(self, text: str) -> Optional[Dict[str, object]]:
        """从 LLM 返回的文本中提取并解析 JSON（支持 ```json 代码块）。"""
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        # 支持被 markdown 代码块包裹的 JSON
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
        """规范化 LLM 返回的结果，确保只使用合法的 candidate skill_id，并处理边缘情况。"""
        candidate_ids = [match.skill_id for match in candidates]
        allowed = set(candidate_ids)

        # primary_skill 必须在候选集合中
        primary = parsed.get("primary_skill")
        primary_skill = str(primary).strip() if isinstance(primary, str) and str(primary).strip() in allowed else None

        # secondary_skills 过滤
        secondary_skills: List[str] = []
        raw_secondary = parsed.get("secondary_skills") or []
        if isinstance(raw_secondary, list):
            for item in raw_secondary:
                skill_id = str(item).strip()
                if skill_id in allowed and skill_id != primary_skill and skill_id not in secondary_skills:
                    secondary_skills.append(skill_id)

        # rejected_skills 过滤
        explicit_rejected: List[str] = []
        raw_rejected = parsed.get("rejected_skills") or []
        if isinstance(raw_rejected, list):
            for item in raw_rejected:
                skill_id = str(item).strip()
                if (skill_id in allowed
                    and skill_id != primary_skill
                    and skill_id not in secondary_skills
                    and skill_id not in explicit_rejected):
                    explicit_rejected.append(skill_id)

        accepted = [skill_id for skill_id in ([primary_skill] if primary_skill else []) + secondary_skills if skill_id]

        # 所有未被明确接受的候选都被视为 rejected（保持完整性）
        rejected_skills = list(explicit_rejected)
        for skill_id in candidate_ids:
            if skill_id not in accepted and skill_id not in rejected_skills:
                rejected_skills.append(skill_id)

        confidence = self._parse_confidence(parsed.get("confidence"))

        # 如果 LLM 没有给出合理置信度，则保守估计
        if confidence <= 0.0:
            if accepted and candidates:
                confidence = max(0.0, min(0.95, candidates[0].score))
            elif candidates:
                confidence = max(0.0, min(0.49, candidates[0].score))
            else:
                confidence = 0.0

        intent_summary = str(parsed.get("intent_summary") or "").strip()
        if not intent_summary:
            if accepted:
                intent_summary = self._build_fallback_summary(user_query, candidates, no_match=False)
            else:
                intent_summary = self._build_fallback_summary(user_query, candidates, no_match=True)

        return SemanticSkillJudgment(
            primary_skill=primary_skill,
            secondary_skills=tuple(secondary_skills),
            rejected_skills=tuple(rejected_skills),
            confidence=confidence,
            intent_summary=intent_summary,
            source="llm",
        )

    def _parse_confidence(self, raw_value: object) -> float:
        """安全解析 confidence，确保在 [0.0, 1.0] 范围内。"""
        try:
            value = float(raw_value)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, value))

    def _build_fallback_summary(
        self,
        user_query: str,
        candidates: Sequence[SkillMatch],
        no_match: bool = False,
    ) -> str:
        """为 heuristic fallback 生成简洁的意图总结。"""
        if not candidates:
            return "No candidate skills available."
        if no_match:
            if user_query:
                return f"No suitable skill identified for query: {user_query.strip()[:80]}"
            return "No suitable skill identified from recalled candidates."
        if user_query:
            return f"Fallback semantic judgment for query: {user_query.strip()[:80]}"
        return f"Fallback semantic judgment favoring {candidates[0].skill_id}."