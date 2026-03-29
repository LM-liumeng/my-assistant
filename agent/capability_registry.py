"""Dynamic capability descriptions derived from the tool registry.

本模块负责根据 ToolRegistry 中已注册的工具，动态生成各种提示词（Prompt）和能力描述。
核心目标是让 LLM “知道”当前系统真实具备哪些能力，避免幻觉，同时为不同场景提供合适的提示和回复。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from agent.tool_registry import ToolRegistry


TOOL_CALLING_PROMPT = """
You are the planning layer for a local desktop assistant.
Decide whether a registered tool should be called.

Rules:
1. Call a tool only when it would materially help answer or execute the user's request.
2. Do not call tools for small talk, casual conversation, or pure capability questions.
3. Use only the provided registered tools. Never invent a tool name.
4. If no tool is needed, answer normally without any tool call.
5. When a tool can satisfy the request directly, prefer the tool over vague generic prose.
""".strip()
"""工具调用决策专用的系统提示词（Planning Layer Prompt）。
用于指导 LLM 在 Function Calling 阶段是否应该触发工具。
"""


CAPABILITY_ANSWER_RULES = """
You are answering a user's question about what this assistant can do.
Use only the real capabilities registered by the host application.

Rules:
1. Answer in the same language as the user.
2. Treat the question as a capability explanation, not as an instruction to execute a tool.
3. If the user asks about one capability, answer that directly and mention prerequisites briefly.
4. If the user asks broadly, summarize the implemented features in natural prose.
5. Keep the answer concise and conversational.
6. Do not ask for missing parameters such as email address, file path, or search query when the user is only asking whether the capability exists.
""".strip()
"""用户询问“助手能做什么”时的专用回答规则提示词。
强调只能基于真实注册的能力回答，不能诱导执行工具。
"""


class CapabilityRegistry:
    """能力注册与描述生成器（Capability Registry）。

    作用：
    - 从 ToolRegistry 中读取所有已注册且可用的工具元数据
    - 动态构建不同场景下需要的系统提示词（Prompt）
    - 为用户询问能力时提供自然、准确的回复（包括模糊匹配）
    - 避免 LLM 幻觉不存在的能力
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """初始化能力注册器，依赖 ToolRegistry 作为数据源。"""
        self.tool_registry = tool_registry

    def list_capabilities(self) -> List[Dict[str, Any]]:
        """返回当前所有可用（available=True）的工具能力快照。

        用于构建提示词和生成回复，只包含运行时可用的能力。
        """
        rows: List[Dict[str, Any]] = []
        for item in self.tool_registry.snapshot():
            status = item.get("status") or {}
            if not bool(status.get("available", False)):
                continue
            rows.append(item)
        return rows

    def build_conversation_prompt(self) -> str:
        """构建普通对话场景的系统提示词（Conversation Layer Prompt）。

        该提示词会告诉 LLM 当前真实具备哪些能力，以及对话时的行为规则。
        通常在每次对话开始或上下文重置时使用。
        """
        lines = [
            "You are the conversation layer for a local desktop assistant.",
            "These are the real registered capabilities currently exposed by the host application:",
        ]
        lines.extend(self._capability_lines())
        lines.extend(
            [
                "",
                "Rules:",
                "1. Do not deny capabilities that are listed above.",
                "2. If active artifact context is supplied, use it as the primary source of truth.",
                "3. If you infer scene meaning from detected objects, clearly say it is an inference based on detections.",
                "4. Do not claim to have direct filesystem, network, or plugin abilities beyond the registered capabilities above.",
                "5. Prefer using the current conversation context over generic disclaimer text.",
            ]
        )
        return "\n".join(lines).strip()

    def build_tool_calling_prompt(self) -> str:
        """构建工具调用决策专用的提示词（Tool Calling / Planning Prompt）。

        包含 TOOL_CALLING_PROMPT + 当前所有可用工具的详细列表。
        用于 Function Calling 阶段，让 LLM 决定是否调用工具以及调用哪个。
        """
        lines = [TOOL_CALLING_PROMPT, "", "Registered tools:"]
        lines.extend(self._tool_lines())
        return "\n".join(lines).strip()

    def build_capability_answer_prompt(self) -> str:
        """构建用户询问“助手能力”时的专用提示词。

        包含 CAPABILITY_ANSWER_RULES + 当前能力列表。
        用于处理类似“你能做什么？”、“你支持发送邮件吗？”这类问题。
        """
        lines = [CAPABILITY_ANSWER_RULES, "", "Registered capabilities:"]
        lines.extend(self._capability_lines())
        return "\n".join(lines).strip()

    def fallback_answer(self, command: str) -> str:
        """当用户直接询问能力但未触发工具调用时，提供自然的回退回答（中文版）。

        支持：
        - 精确匹配单个能力
        - 模糊匹配多个能力
        - 完全不匹配时给出总体能力概述
        """
        query = (command or "").strip()
        matches = self._match_capabilities(query)
        if matches:
            if len(matches) == 1:
                capability = matches[0]
                detail = self._status_suffix(capability)
                return f"可以。我支持{capability['capability_summary']}。{detail}".strip()

            # 多个匹配时列出前4个
            summaries = []
            for capability in matches[:4]:
                text = capability["capability_summary"]
                if text not in summaries:
                    summaries.append(text)
            return "可以。我支持：" + "；".join(summaries) + "。"

        # 无匹配时给出宽泛总结
        return "除了普通对话，我目前还支持：" + "；".join(self._broad_summaries()) + "。"

    def unavailable_chat_reply(self) -> str:
        """当聊天模型不可用时，返回的友好提示（仍会展示已注册的工具能力）。"""
        return (
            "当前未配置聊天模型，或聊天模型暂时不可用，所以开放式对话现在无法正常工作。"
            "已实现的工具能力仍包括：" + "；".join(self._broad_summaries()) + "。"
        )

    # ==================== 内部辅助方法 ====================

    def _tool_lines(self) -> List[str]:
        """生成工具列表，每行格式为：`- name: description [status]`"""
        lines: List[str] = []
        for capability in self.list_capabilities():
            status = capability["status"]
            suffix = f" [{status['status']}]" if status.get("status") else ""
            lines.append(
                f"- {capability['name']}: {capability['description']}{suffix}"
            )
        return lines or ["- No registered tools."]

    def _capability_lines(self) -> List[str]:
        """生成能力摘要列表，每行格式为：`- capability_summary (detail)`"""
        lines: List[str] = []
        for capability in self.list_capabilities():
            status = capability["status"]
            detail = f" ({status['detail']})" if status.get("detail") else ""
            lines.append(f"- {capability['capability_summary']}{detail}")
        return lines or ["- No registered capabilities."]

    def _broad_summaries(self) -> List[str]:
        """提取所有可用能力的简短摘要（去重），用于宽泛回答。"""
        seen: List[str] = []
        for capability in self.list_capabilities():
            summary = str(capability.get("capability_summary")
                         or capability.get("description") or "").strip()
            if summary and summary not in seen:
                seen.append(summary)
        return seen or ["当前没有可用的已注册能力"]

    def _status_suffix(self, capability: Dict[str, Any]) -> str:
        """返回能力的状态附加信息（如果有 detail 则显示）。"""
        status = capability.get("status") or {}
        detail = str(status.get("detail") or "").strip()
        return detail if detail else ""

    def _match_capabilities(self, command: str) -> List[Dict[str, Any]]:
        """对用户查询进行模糊匹配，返回最相关的能力列表（按匹配度排序）。"""
        query = (command or "").strip().lower()
        if not query:
            return []

        scored: List[tuple[int, Dict[str, Any]]] = []
        for capability in self.list_capabilities():
            score = self._score_capability(query, capability)
            if score > 0:
                scored.append((score, capability))

        scored.sort(key=lambda item: (-item[0], item[1]["name"]))
        return [capability for score, capability in scored if score > 0]

    def _score_capability(self, query: str, capability: Dict[str, Any]) -> int:
        """计算用户查询与某个能力的匹配分数（简单关键词 + 子串匹配）。"""
        haystacks = [
            str(capability.get("name") or "").lower(),
            str(capability.get("description") or "").lower(),
            str(capability.get("capability_summary") or "").lower(),
            str(capability.get("category") or "").lower(),
        ]
        haystacks.extend(str(tag).lower() for tag in capability.get("tags") or [])

        joined = " \n".join(haystacks)
        score = 0

        if query and query in joined:
            score += 8
        for haystack in haystacks:
            if not haystack:
                continue
            if haystack in query:
                score += 6
        for token in self._tokens(query):
            if len(token) < 2:
                continue
            if token in joined:
                score += 2

        return score

    def _tokens(self, text: str) -> List[str]:
        """将文本拆分成单词/标记，用于模糊匹配。"""
        raw = re.split(r"[\s,.:;!?(){}\[\]<>/\\|]+", text)
        tokens = [item.strip().lower() for item in raw if item.strip()]
        return tokens