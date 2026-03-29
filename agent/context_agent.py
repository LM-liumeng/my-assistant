"""Context selection and prompt assembly for the assistant.

本模块负责 Agent 的上下文管理与 Prompt 构建，是连接用户输入、历史记忆、当前 artifact（媒体/RAG/聊天结果）和技能系统的核心枢纽。
主要解决“当前应该给 LLM 看什么上下文？”和“如何智能地补充缺失参数？”这两个关键问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.conversation_memory import ConversationMemory
from agent.media_answerer import MediaAnswerAgent
from agent.rag_answerer import RAGAnswerAgent
from agent.skill_manager import SkillManager


@dataclass
class ContextRequest:
    """一次上下文请求的统一数据结构。

    用于在意图识别后，向 ContextAgent 传递用户命令及可能的显示内容、文件名、知识库等信息。
    """
    command: str                                   # 用户原始输入命令
    display_content: Optional[str] = None          # 当前显示的内容（例如图片描述、文档内容等）
    display_filename: Optional[str] = None         # 当前显示的文件名
    knowledge_base: Optional[str] = None           # 用户指定的知识库 ID


class PromptContextBuilder:
    """Prompt 构建器，专门负责组装 system prompt 和 messages 列表。"""

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    def build_messages(self, user_command: str, limit: int = 8) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表（messages）。

        包含：最近 N 轮对话历史 + 当前用户输入。
        """
        messages: List[Dict[str, str]] = []
        for turn in self.memory.get_recent_turns(limit=limit):
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_command})
        return messages

    def build_system_prompt(self, capability_prompt: str) -> str:
        """构建完整的 System Prompt。

        组合以下内容：
        1. 能力描述（来自 CapabilityRegistry）
        2. 当前激活的知识库
        3. 当前活跃的 Artifact（媒体、RAG、聊天结果等）
        4. 最近对话摘要
        """
        sections = [capability_prompt]

        # 当前激活的知识库
        if self.memory.active_knowledge_base:
            sections.append(f"Current active knowledge base: {self.memory.active_knowledge_base}")

        # 当前活跃的 Artifact（最重要上下文）
        artifact = self.memory.active_artifact
        if artifact and artifact.display_content:
            sections.append(
                "Current active artifact context:\n"
                f"Artifact kind: {artifact.kind}\n"
                f"Display filename: {artifact.display_filename or ''}\n"
                f"Artifact content:\n{artifact.display_content}"
            )

        # 最近对话摘要（帮助保持连贯性）
        recent_turns = self.memory.get_recent_turns(limit=6)
        if recent_turns:
            sections.append(
                "Recent conversation summary:\n"
                + "\n".join(f"{turn.role}: {turn.content}" for turn in recent_turns)
            )

        return "\n\n".join(sections)


class ContextAgent:
    """上下文代理（Context Agent）—— Agent 系统中的上下文管理中枢。

    主要职责：
    1. 提供当前活跃的显示内容、文件名、知识库等上下文信息
    2. 根据意图智能补充缺失的参数（例如 email body、document content 等）
    3. 处理媒体分析、RAG 查询的后续对话（follow-up）
    4. 构建最终发送给 LLM 的 messages 和 system prompt
    5. 记忆助手返回的结果（artifact、knowledge_base 等）
    """

    def __init__(
        self,
        memory: ConversationMemory,
        media_answer_agent: MediaAnswerAgent,
        rag_answer_agent: RAGAnswerAgent,
        skill_manager: Optional[SkillManager] = None,
    ) -> None:
        self.memory = memory
        self.media_answer_agent = media_answer_agent
        self.rag_answer_agent = rag_answer_agent
        self.skill_manager = skill_manager
        self.prompt_builder = PromptContextBuilder(memory)

    def active_display_content(self) -> Optional[str]:
        """返回当前活跃 Artifact 的显示内容（最高优先级上下文）。"""
        artifact = self.memory.active_artifact
        return artifact.display_content if artifact else None

    def active_display_filename(self) -> Optional[str]:
        """返回当前活跃 Artifact 的显示文件名。"""
        artifact = self.memory.active_artifact
        return artifact.display_filename if artifact else None

    def active_knowledge_base(self) -> Optional[str]:
        """获取当前活跃的知识库 ID（优先从 Artifact 中提取，其次从内存）。"""
        artifact = self.memory.active_artifact
        if artifact:
            kb = str((artifact.payload or {}).get('knowledge_base', '') or '').strip()
            if kb:
                return kb
            kb_payload = artifact.payload.get('knowledge_base') if artifact.payload else None
            if isinstance(kb_payload, dict):
                return str(kb_payload.get('id', '')).strip() or self.memory.active_knowledge_base
        return self.memory.active_knowledge_base

    def preferred_content(self, request: ContextRequest) -> str:
        """返回当前最优的内容片段（优先级从高到低）。"""
        for candidate in [
            request.display_content,           # 本次请求携带的内容
            self.active_display_content(),     # 当前内存中的活跃 Artifact
            self.memory.last_assistant_message,# 上一次助手回复
        ]:
            text = (candidate or '').strip()
            if text:
                return text
        return ''

    def normalize_request(
        self,
        intent: str,
        params: Dict[str, Any],
        request: ContextRequest,
        recognizer: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """根据意图智能规范化请求参数，自动填充缺失的关键字段。

        这是一个非常实用的“上下文感知参数补全”逻辑。
        """
        normalized = dict(params)
        selected_kb = (request.knowledge_base or self.active_knowledge_base() or '').strip()
        if selected_kb:
            self.memory.remember_knowledge_base(selected_kb)

        # ==================== Email 意图 ====================
        if intent == 'email':
            if not (normalized.get('body') or '').strip():
                fallback = self.preferred_content(request)
                if fallback:
                    normalized['body'] = fallback
            if not normalized.get('subject'):
                normalized['subject'] = self.derive_email_subject(request.command, normalized.get('body') or '')
            return intent, normalized

        # ==================== Document 意图 ====================
        if intent == 'document':
            if not (normalized.get('content') or '').strip():
                fallback = self.preferred_content(request)
                if fallback:
                    normalized['content'] = fallback
            return intent, normalized

        # ==================== Media Analysis 意图 ====================
        if intent == 'media_analysis':
            if not (normalized.get('source') or '').strip():
                active_filename = request.display_filename or self.active_display_filename()
                if active_filename:
                    normalized['source'] = active_filename.strip()
            return intent, normalized

        # ==================== RAG 相关意图 ====================
        if intent == 'rag_query' or intent == 'rag_manage':
            if selected_kb and not normalized.get('knowledge_base'):
                normalized['knowledge_base'] = selected_kb
            return intent, normalized

        # ==================== run_model 意图的智能路由 ====================
        if intent != 'run_model':
            return intent, normalized

        text_value = (normalized.get('text') or '').strip()
        active_filename = (request.display_filename or self.active_display_filename() or '').strip()

        # 如果输入是媒体路径 → 转为 media_analysis
        if recognizer.is_media_path(text_value):
            return 'media_analysis', {'source': text_value}

        # 如果当前有媒体文件且用户说“分析”“模型”等模糊词 → 转为 media_analysis
        if (active_filename and recognizer.is_media_path(active_filename) and
            (not text_value or text_value in {'分析', '模型', 'analyze', 'run model'})):
            return 'media_analysis', {'source': active_filename}

        # 兜底：把当前最佳内容填入 text 参数
        if not text_value:
            fallback = self.preferred_content(request)
            if fallback:
                normalized['text'] = fallback

        return intent, normalized

    def can_answer_media_followup(self, request: ContextRequest) -> bool:
        """判断当前上下文是否足够让 MediaAnswerAgent 直接回答（无需再次调用模型）。"""
        display_content = request.display_content or self.active_display_content()
        display_filename = request.display_filename or self.active_display_filename()
        return self.media_answer_agent.can_answer_from_context(request.command, display_content, display_filename)

    def answer_media_followup(self, request: ContextRequest) -> Dict[str, Any]:
        """使用 MediaAnswerAgent 直接从上下文回答媒体相关后续问题。"""
        display_content = request.display_content or self.active_display_content() or ''
        display_filename = request.display_filename or self.active_display_filename()
        return self.media_answer_agent.answer_from_context(request.command, display_content, display_filename)

    def can_answer_rag_followup(self, request: ContextRequest) -> bool:
        """判断是否可以直接用 RAGAnswerAgent 从上下文回答。"""
        display_content = request.display_content or self.active_display_content()
        display_filename = request.display_filename or self.active_display_filename()
        return self.rag_answer_agent.can_answer_from_context(request.command, display_content, display_filename)

    def answer_rag_followup(self, request: ContextRequest) -> Dict[str, Any]:
        """使用 RAGAnswerAgent 直接从上下文回答 RAG 相关后续问题。"""
        display_content = request.display_content or self.active_display_content() or ''
        display_filename = request.display_filename or self.active_display_filename()
        return self.rag_answer_agent.answer_from_context(request.command, display_content, display_filename)

    def build_chat_messages(self, user_command: str) -> List[Dict[str, str]]:
        """构建聊天用的消息列表（供 LLM 调用）。"""
        return self.prompt_builder.build_messages(user_command)

    def build_chat_system_prompt(self, capability_prompt: str, user_command: str = "", scope: str = "chat") -> str:
        """构建完整的聊天 System Prompt，支持 SkillManager 进一步增强。"""
        prompt = capability_prompt
        if self.skill_manager is not None:
            prompt = self.skill_manager.augment_prompt(prompt, scope=scope, user_query=user_command)
        return self.prompt_builder.build_system_prompt(prompt)

    def prepare_chat_result(
        self,
        result: Dict[str, Any],
        request: ContextRequest,
        scope: str = 'chat'
    ) -> Dict[str, Any]:
        """对聊天结果进行后处理，补充 display 信息和已解析技能列表。"""
        prepared = dict(result)
        message = str(prepared.get('message') or '').strip()

        if self.skill_manager is not None and message:
            message = self.skill_manager.postprocess_response(message, scope=scope, user_query=request.command)
            prepared['active_skills'] = self.skill_manager.resolved_skill_ids(request.command, scope)

        prepared['message'] = message
        prepared['display_content'] = message or self.preferred_content(request)
        prepared['display_filename'] = request.display_filename or self.active_display_filename()

        if request.knowledge_base:
            prepared['active_knowledge_base'] = request.knowledge_base

        return prepared

    def remember_response(
        self,
        intent: str,
        response: Dict[str, Any],
        raw_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """将助手的回复记录到 ConversationMemory 中，并根据意图保存对应的 Artifact。"""
        self.memory.add_turn('assistant', response.get('message', ''))

        display_content = response.get('display_content')
        display_filename = response.get('display_filename')
        knowledge_base = response.get('active_knowledge_base')

        if not knowledge_base and isinstance(response.get('knowledge_base'), dict):
            knowledge_base = response['knowledge_base'].get('id')

        if knowledge_base:
            self.memory.remember_knowledge_base(str(knowledge_base))

        # ==================== 不同意图的 Artifact 记忆逻辑 ====================
        if intent == 'media_analysis':
            source_result = raw_results[0] if raw_results else response
            self.memory.remember_artifact(
                kind='media_analysis',
                display_filename=display_filename,
                display_content=display_content,
                payload=dict(source_result),
            )
            return

        if intent == 'rag_query':
            source_result = raw_results[0] if raw_results else response
            payload = dict(source_result)
            if knowledge_base and 'knowledge_base' not in payload:
                payload['knowledge_base'] = knowledge_base
            self.memory.remember_artifact(
                kind='rag_evidence',
                display_filename=display_filename or 'rag_evidence.txt',
                display_content=display_content,
                payload=payload,
            )
            return

        if intent == 'chat':
            artifact_content = display_content or response.get('message')
            if artifact_content:
                self.memory.remember_artifact(
                    kind='chat_reply',
                    display_filename=display_filename,
                    display_content=artifact_content,
                    payload=dict(response),
                )
            return

        # 兜底：普通意图也记录 artifact
        if display_content or display_filename:
            self.memory.remember_artifact(
                kind=intent,
                display_filename=display_filename,
                display_content=display_content,
                payload=dict(response),
            )

    def derive_email_subject(self, command: str, body: str) -> str:
        """从邮件正文智能生成邮件主题（简洁友好）。"""
        text = (body or '').strip()
        if text:
            first_line = text.split('\n', 1)[0].strip()
            if len(first_line) <= 40:
                return first_line
            return first_line[:37].rstrip() + '...'

        normalized_command = (command or '').strip()
        return normalized_command[:40] if normalized_command else '(no subject)'

    def clear_display_context(self) -> None:
        """清除当前显示的上下文（通常在用户明确要求重置时调用）。"""
        self.memory.clear_active_context()