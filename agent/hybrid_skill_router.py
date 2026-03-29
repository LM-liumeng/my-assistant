"""Rule-based two-stage skill routing with recall and rerank.

本模块实现了一个**纯规则驱动的两阶段技能路由器**（HybridSkillRouter）。
它 intentionally 不依赖 embedding 或大模型语义搜索，而是完全基于技能预定义的 triggers（触发词）和 patterns（正则）进行召回，
再通过轻量策略提示进行重排序。

设计理念：
- 配置驱动：新增技能只需在 skill.json 中填写 triggers/patterns，无需改代码
- 可解释性极高：每个匹配都有明确的 matched_terms 和 source
- 生产友好：延迟低、成本低、易调试、适合技能数量不多（几十~上百个）的场景
- 不是语义意图模型（not a semantic intent model）

主要接口：
- route_all()：生产环境主要接口，返回排序后的 SkillMatch 列表
- route()：便捷接口，只返回 top-1 技能（供简单场景使用）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.skill_registry import LoadedSkill, SkillRegistry


@dataclass(frozen=True)
class SkillMatch:
    """单次技能匹配结果（路由召回阶段的输出单元）。

    包含原始分数、最终分数、优先级、匹配来源和具体命中的词语，便于后续审计和重排序。
    """
    skill_id: str                    # 技能唯一 ID
    base_score: float                # 原始匹配分数（仅来自 trigger/pattern）
    score: float                     # 经过 rerank 后的最终分数（0.0~0.99）
    priority: int                    # 技能自身优先级（用于 tie-breaker）
    source: str                      # 主匹配来源（"trigger:exact:xxx" 或 "pattern:..."）
    sources: Tuple[str, ...]         # 所有匹配来源（支持多个）
    matched_terms: Tuple[str, ...]   # 实际命中的触发词/正则（用于解释和去重）


class HybridSkillRouter:
    """混合技能路由器（Hybrid Skill Router）—— 纯规则驱动的两阶段路由。

    工作流程：
    1. recall_candidates：基于 triggers 和 patterns 快速召回候选（最多 12 个）
    2. rerank_candidates：根据优先级、风险、成本、证据充分度进行轻量重排序
    """

    def __init__(self, registry: SkillRegistry):
        """初始化路由器，只依赖技能注册表。"""
        self.registry = registry

    def route(self, user_query: str, surface: Optional[str] = None, phase: Optional[str] = None) -> Tuple[Optional[str], float]:
        """便捷接口：仅返回置信度最高的单个技能（供只需要 top-1 的调用方使用）。

        返回 (skill_id, score) 元组。
        """
        matches = self.route_all(user_query, surface=surface, phase=phase)
        if not matches:
            return None, 0.0
        top = matches[0]
        return top.skill_id, top.score

    def route_all(self, user_query: str, surface: Optional[str] = None, phase: Optional[str] = None) -> List[SkillMatch]:
        """生产环境主要接口：完整两阶段路由。

        先召回候选，再重排序，返回最终排序后的 SkillMatch 列表。
        """
        recalled = self.recall_candidates(user_query, surface=surface, phase=phase)
        return self.rerank_candidates(recalled, user_query=user_query, surface=surface, phase=phase)

    def recall_candidates(self, user_query: str, surface: Optional[str] = None, phase: Optional[str] = None, limit: int = 12) -> List[SkillMatch]:
        """第一阶段：规则召回（Recall）。

        从当前可见技能中，根据 triggers 和 patterns 计算匹配分数，
        只保留分数 > 0 的候选，并限制最多 limit 个（默认 12）。
        """
        query = (user_query or "").strip()
        if not query:
            return []

        lowered = query.lower()                     # 转为小写用于快速匹配
        candidates: List[SkillMatch] = []

        # 只遍历当前 surface 和 phase 可见的技能（已包含 enabled/rollout 过滤）
        for skill in self.registry.list_skills(surface=surface, phase=phase):
            base_score, source, matched_terms, sources = self._score_skill(skill, query, lowered)
            if base_score <= 0.0:
                continue

            candidates.append(
                SkillMatch(
                    skill_id=skill.skill_id,
                    base_score=base_score,
                    score=base_score,               # 此时 score 等于 base_score
                    priority=skill.priority,
                    source=source,
                    sources=sources,
                    matched_terms=tuple(matched_terms),
                )
            )

        # 按 base_score（降序）→ priority（降序）→ skill_id（升序）排序
        candidates.sort(key=lambda item: (-item.base_score, -item.priority, item.skill_id))
        return candidates[:limit]

    def rerank_candidates(
        self,
        candidates: List[SkillMatch],
        user_query: str,
        surface: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[SkillMatch]:
        """第二阶段：轻量重排序（Rerank）。

        在召回的候选上，根据技能的优先级、风险等级、成本等级、
        证据多样性等策略提示进行微调分数，实现更合理的排序。
        """
        reranked: List[SkillMatch] = []

        for match in candidates:
            skill = self.registry.get_skill(match.skill_id)
            if skill is None:
                continue

            score = match.base_score

            # 正向加分：高优先级技能略微提升
            if skill.priority >= 80:
                score += 0.01

            # 负向惩罚：高风险/高成本技能降低分数
            if skill.risk_level == "high":
                score -= 0.03
            elif skill.risk_level == "medium":
                score -= 0.01
            if skill.cost_level == "high":
                score -= 0.01

            # 证据多样性奖励：命中多个不同 term 时加分
            distinct_evidence = len({term.strip().lower() for term in match.matched_terms if term.strip()})
            if distinct_evidence >= 2:
                score += min(0.03, 0.01 * (distinct_evidence - 1))

            reranked.append(
                SkillMatch(
                    skill_id=match.skill_id,
                    base_score=match.base_score,
                    score=max(0.0, min(0.99, score)),   # 最终分数限制在 [0.0, 0.99]
                    priority=skill.priority,
                    source=match.source,
                    sources=match.sources,
                    matched_terms=match.matched_terms,
                )
            )

        # 最终排序：score（降序）→ priority（降序）→ skill_id（升序）
        reranked.sort(key=lambda item: (-item.score, -item.priority, item.skill_id))
        return reranked

    def _score_skill(self, skill: LoadedSkill, query: str, lowered: str) -> Tuple[float, str, List[str], Tuple[str, ...]]:
        """对单个技能进行打分（核心评分函数）。

        综合 triggers 和 patterns 的所有证据，取最高分 + 多样性 bonus。
        """
        evidence: List[Tuple[float, str, str]] = []      # (score, source, term)
        matched_terms: List[str] = []
        matched_sources: List[str] = []
        seen_terms = set()
        seen_sources = set()

        # ==================== 1. Triggers 匹配 ====================
        for trigger in skill.triggers:
            keyword = trigger.strip()
            if not keyword:
                continue
            trigger_score, trigger_source = self._score_trigger(keyword, query, lowered)
            if trigger_score <= 0.0:
                continue

            evidence.append((trigger_score, trigger_source, keyword))

            # 记录匹配词（去重）
            normalized_term = keyword.lower()
            if normalized_term not in seen_terms:
                matched_terms.append(keyword)
                seen_terms.add(normalized_term)

            if trigger_source not in seen_sources:
                matched_sources.append(trigger_source)
                seen_sources.add(trigger_source)

        # ==================== 2. Patterns（正则）匹配 ====================
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

        # 取证据中分数最高的一项作为 primary
        evidence.sort(key=lambda item: (-item[0], item[1], item[2]))
        best_score, primary_source, _ = evidence[0]

        # 证据多样性奖励
        distinct_evidence = len(seen_terms)
        bonus = min(0.06, 0.02 * max(0, distinct_evidence - 1))

        return min(0.99, best_score + bonus), primary_source or "router", matched_terms, tuple(matched_sources)

    def _score_trigger(self, keyword: str, query: str, lowered: str) -> Tuple[float, str]:
        """触发词（keyword）匹配打分。

        精确匹配 > 前缀匹配 > 包含匹配。
        """
        keyword_lower = keyword.lower()

        if lowered == keyword_lower:                                   # 完全相等
            return 0.99, f"trigger:exact:{keyword}"
        if self._starts_with_phrase(lowered, keyword_lower):           # 前缀 + 词边界
            return 0.94, f"trigger:prefix:{keyword}"
        if self._contains_phrase(lowered, keyword, keyword_lower):     # 包含（词边界保护）
            return 0.80, f"trigger:contains:{keyword}"

        return 0.0, ""

    def _score_pattern(self, expr: str, query: str, match: re.Match[str]) -> Tuple[float, str]:
        """正则模式匹配打分。

        根据匹配位置和正则复杂度给出不同分数。
        """
        if match.start() == 0 and match.end() == len(query):           # 全匹配
            return 0.97, f"pattern:fullmatch:{expr}"
        if match.start() == 0:                                         # 前缀匹配
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
        """判断是否为词边界前缀匹配（避免“hello”匹配“hell”）。"""
        if not lowered_query.startswith(lowered_keyword):
            return False
        if len(lowered_query) == len(lowered_keyword):
            return True
        next_char = lowered_query[len(lowered_keyword)]
        return not next_char.isalnum()

    def _contains_phrase(self, lowered_query: str, keyword: str, lowered_keyword: str) -> bool:
        """安全的包含匹配，支持 ASCII 词边界保护和 CJK 语言。"""
        if self._is_ascii_word(keyword):
            if len(lowered_keyword) < 4:
                return False
            # 词边界正则：前后不是字母/数字
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
        """判断是否为纯 ASCII 单词（英文/数字/下划线/空格）。"""
        return bool(re.fullmatch(r"[A-Za-z0-9_\- ]+", value or ""))

    def _looks_cjk(self, value: str) -> bool:
        """判断是否包含中文/日文/韩文字符（CJK）。"""
        return bool(re.search(r"[\u4e00-\u9fff]", value or ""))