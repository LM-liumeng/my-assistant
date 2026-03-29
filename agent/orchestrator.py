"""Agent orchestration and request handling."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.capability_registry import CapabilityRegistry
from agent.context_agent import ContextAgent, ContextRequest
from agent.mcp_discovery import MCPDiscoveryService
from agent.rag_answerer import RAGAnswerAgent
from agent.conversation_memory import ConversationMemory
from agent.hybrid_skill_router import HybridSkillRouter
from agent.media_answerer import MediaAnswerAgent
from agent.skill_manager import SkillManager
from agent.skill_registry import SkillRegistry
from agent.tool_registry import ToolRegistry, ToolSpec
from context.evidence_store import EvidenceStore
from context.metadata_store import MetadataStore
from context.rag_store import HybridRAGStore
from context.retrieval_store import RetrievalStore
from security import SafetyLayer
from tools.chat_tool import ChatTool
from tools.document_tool import DocumentTool
from tools.email_tool import EmailTool
from tools.file_tool import FileSearchTool
from tools.intent_tool import IntentTool
from tools.model_tool import ModelTool
from tools.rag_management_tool import RAGManagementTool
from tools.rag_tool import RAGTool
from tools.search_tool import WebSearchTool
from tools.video_tool import VideoAnalysisTool


@dataclass
class UserRequest:
    command: str
    display_content: Optional[str] = None
    display_filename: Optional[str] = None
    knowledge_base: Optional[str] = None


class IntentRecognizer:
    """Categorise user input into supported assistant intents."""

    MEDIA_EXTENSIONS = (
        "jpg", "jpeg", "png", "bmp", "webp", "gif", "tif", "tiff",
        "mp4", "avi", "mov", "mkv", "wmv", "flv", "m4v", "webm", "mpeg", "mpg",
    )
    EMAIL_PATTERN = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

    def __init__(self, intent_tool: Optional[IntentTool] = None) -> None:
        self.intent_tool = intent_tool

    def recognise(self, command: str) -> Tuple[str, Dict[str, Any]]:
        cmd = (command or "").strip()
        lower = cmd.lower()

        if self.is_capability_question(cmd, lower):
            return "chat", {"prompt": command}
        if self._looks_like_clear_display(cmd, lower):
            return "clear_display", {}

        if self._looks_like_web_search(cmd, lower):
            return "web_search", {"query": self._extract_web_query(cmd, lower)}
        if self._looks_like_rag_management(cmd, lower):
            return "rag_manage", self._extract_rag_management_params(cmd, lower)
        if self._looks_like_rag_query(cmd, lower):
            return "rag_query", {"query": self._extract_rag_query(cmd, lower)}
        if self._looks_like_email(cmd, lower):
            return "email", self._extract_email_params(cmd, lower)

        media_source = self._extract_media_source(cmd)
        if self._looks_like_media_analysis(cmd, lower, media_source):
            return "media_analysis", {"source": media_source or ""}

        if self._looks_like_document_write(cmd, lower):
            return "document", self._extract_document_params(cmd, lower)
        if self._looks_like_save_as(cmd, lower):
            return "document", {"filename": self._extract_filename_after_phrase(cmd, lower, ["save as", "save file"]), "content": ""}
        if self._looks_like_document_read(cmd, lower):
            return "document_read", {"filename": self._extract_filename_after_phrase(cmd, lower, ["open file", "load file"])}
        if self._looks_like_file_search(cmd, lower):
            return "file_search", {"query": self._extract_file_query(cmd, lower)}
        if self._looks_like_model_analysis(cmd, lower):
            return "run_model", {"text": self._extract_model_text(cmd, lower)}

        if self.intent_tool is not None:
            parsed = self._parse_with_llm(cmd)
            if parsed is not None:
                return parsed

        return "chat", {"prompt": command}

    def _parse_with_llm(self, command: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        try:
            parsed = self.intent_tool.parse_intent(command) if self.intent_tool is not None else None
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            return None
        raw_intent = str(parsed.get("intent", "")).strip().lower()
        args = dict(parsed.get("args") or {})
        mapping = {
            "email": "email",
            "send_email": "email",
            "document": "document",
            "document_write": "document",
            "write_document": "document",
            "document_read": "document_read",
            "read_document": "document_read",
            "open_document": "document_read",
            "file_search": "file_search",
            "search_file": "file_search",
            "web_search": "web_search",
            "search_web": "web_search",
            "run_model": "run_model",
            "analysis": "run_model",
            "video_analysis": "media_analysis",
            "image_analysis": "media_analysis",
            "media_analysis": "media_analysis",
            "chat": "chat",
            "rag_manage": "rag_manage",
            "rag_status": "rag_manage",
        }
        intent = mapping.get(raw_intent)
        if not intent:
            return None
        if intent == "email":
            args = self._normalize_email_args(args)
        if intent == "web_search" and not self._looks_like_web_search(command, command.lower()):
            return None
        if intent == "media_analysis" and not args.get("source"):
            args["source"] = self._extract_media_source(command) or ""
        if intent == "chat" and "prompt" not in args:
            args = {"prompt": command}
        return intent, args

    def _normalize_email_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(args)
        if not normalized.get("to"):
            for candidate in ["recipient", "address", "email", "email_address", "to_addr", "to_address"]:
                if normalized.get(candidate):
                    normalized["to"] = normalized[candidate]
                    break
        if not normalized.get("subject"):
            for candidate in ["title", "subject_line"]:
                if normalized.get(candidate):
                    normalized["subject"] = normalized[candidate]
                    break
        if not normalized.get("body"):
            for candidate in ["content", "text", "message"]:
                if normalized.get(candidate):
                    normalized["body"] = normalized[candidate]
                    break
        return normalized

    def is_capability_question(self, command: str, lower: Optional[str] = None) -> bool:
        cmd = (command or "").strip()
        lowered = lower or cmd.lower()
        if not cmd:
            return False

        english_prefixes = (
            "what can you do",
            "what else can you do",
            "can you ",
            "are you able to",
            "do you support",
            "what features",
            "what capabilities",
        )
        if any(lowered.startswith(prefix) for prefix in english_prefixes):
            return True

        chinese_keywords = [
            "有什么功能", "有哪些功能", "还有什么功能", "都有什么功能", "都能做什么",
            "能做什么", "可以做什么", "支持什么功能", "支持哪些功能",
            "除了对话之外", "除了聊天之外", "除了聊天还有什么", "除了对话还有什么",
        ]
        if any(keyword in cmd for keyword in chinese_keywords):
            return True

        patterns = [
            "^(你|助手|这个助手).{0,8}(可以|能|会|支持).{0,30}(吗|么|什么|哪些|否|不)$",
            "^(你|助手|这个助手).{0,8}(可以|能|会|支持).{0,30}(吗|么|什么|哪些|否|不)[？?]$",
        ]
        return any(re.search(pattern, cmd) for pattern in patterns)

    def is_media_path(self, value: str) -> bool:
        text = (value or "").strip().strip("\"'")
        return bool(text) and any(text.lower().endswith("." + ext) for ext in self.MEDIA_EXTENSIONS)

    def _looks_like_clear_display(self, cmd: str, lower: str) -> bool:
        english_phrases = [
            "clear display",
            "clear display window",
            "clear background info",
            "clear context window",
            "clear editor",
        ]
        chinese_phrases = [
            "清空背景信息窗口",
            "清空背景信息",
            "清空展示区",
            "清空右侧窗口",
            "清空编辑区",
            "清空当前内容",
        ]
        return any(lower == phrase or lower.startswith(phrase + " ") for phrase in english_phrases) or any(cmd == phrase or cmd.startswith(phrase) for phrase in chinese_phrases)

    def _looks_like_web_search(self, cmd: str, lower: str) -> bool:
        english_prefixes = ["google", "search web", "search the web", "web search"]
        chinese_prefixes = ["\u7f51\u9875\u641c\u7d22", "\u641c\u7d22\u7f51\u9875", "\u4e0a\u7f51\u641c\u7d22"]
        if any(lower == prefix or lower.startswith(prefix + " ") or lower.startswith(prefix + ":") for prefix in english_prefixes):
            return True
        return any(cmd == prefix or cmd.startswith(prefix) for prefix in chinese_prefixes)

    def _extract_web_query(self, cmd: str, lower: str) -> str:
        for prefix in ["google", "search web", "search the web"]:
            if lower.startswith(prefix):
                return cmd[len(prefix):].strip()
        for prefix in ["\u7f51\u9875\u641c\u7d22", "\u641c\u7d22\u7f51\u9875", "\u4e0a\u7f51\u641c\u7d22"]:
            if cmd.startswith(prefix):
                return cmd[len(prefix):].strip()
        return cmd

    def _looks_like_rag_management(self, cmd: str, lower: str) -> bool:
        management_prefixes = [
            "rag status",
            "rag ingest",
            "rag rebuild",
            "rag index status",
            "rebuild rag",
            "ingest knowledge",
        ]
        if any(lower.startswith(prefix) for prefix in management_prefixes):
            return True
        chinese_keywords = [
            "查看知识库状态",
            "查看RAG状态",
            "知识库状态",
            "增量入库",
            "知识库入库",
            "重建索引",
            "重建知识库索引",
            "重建RAG索引",
            "更新知识库",
        ]
        return any(keyword in cmd for keyword in chinese_keywords)

    def _extract_rag_management_params(self, cmd: str, lower: str) -> Dict[str, str]:
        action = "status"
        if any(keyword in lower for keyword in ["rebuild rag", "rag rebuild", "rebuild index"]):
            action = "rebuild"
        elif any(keyword in lower for keyword in ["rag ingest", "ingest knowledge"]):
            action = "ingest"
        elif any(keyword in cmd for keyword in ["重建", "重新入库"]):
            action = "rebuild"
        elif any(keyword in cmd for keyword in ["增量入库", "入库", "更新知识库"]):
            action = "ingest"

        input_dir = ""
        windows_path = re.search(r'([A-Za-z]:\\[^\r\n]+)', cmd)
        relative_path = re.search(r'((?:\.\.?[\\/])[^\s]+)', cmd)
        absolute_path = re.search(r'([\\/][^\s]+)', cmd)
        if windows_path:
            input_dir = windows_path.group(1).strip('"')
        elif relative_path:
            input_dir = relative_path.group(1).strip('"')
        elif absolute_path:
            input_dir = absolute_path.group(1).strip('"')
        return {"action": action, "input_dir": input_dir}

    def _looks_like_rag_query(self, cmd: str, lower: str) -> bool:
        if lower.startswith(("knowledge ", "knowledge base", "rag ", "retrieve ", "retrieval ")):
            return True
        if "knowledge base" in lower:
            return True
        chinese_keywords = [
            "知识库",
            "知识检索",
            "资料库",
            "根据资料",
            "根据知识库",
            "查知识库",
        ]
        return any(keyword in cmd for keyword in chinese_keywords)

    def _extract_rag_query(self, cmd: str, lower: str) -> str:
        for prefix in ["knowledge base", "knowledge", "rag", "retrieve", "retrieval"]:
            if lower.startswith(prefix):
                return cmd[len(prefix):].strip(" :")
        for prefix in [
            "知识库",
            "知识检索",
            "资料库",
            "查知识库",
            "根据资料",
            "根据知识库",
        ]:
            if cmd.startswith(prefix):
                return cmd[len(prefix):].strip(" ：:")
        return cmd

    def _looks_like_email(self, cmd: str, lower: str) -> bool:
        if self.is_capability_question(cmd, lower):
            return False

        email_keywords = ["send email", "draft email", "email to", "mail to"]
        has_email_address = bool(self.EMAIL_PATTERN.search(cmd))
        has_english_action = any(keyword in lower for keyword in email_keywords)
        has_chinese_email_action = any(
            keyword in cmd
            for keyword in ["发送邮件", "发邮件", "写邮件", "邮件给", "发给", "发送给", "寄给", "收件人"]
        )
        has_chinese_email_fields = any(keyword in cmd for keyword in ["主题", "正文", "内容"]) and "邮件" in cmd
        return has_email_address or has_english_action or has_chinese_email_action or has_chinese_email_fields

    def _extract_email_params(self, cmd: str, lower: str) -> Dict[str, str]:
        params = {"to": "", "subject": "", "body": ""}

        email_match = self.EMAIL_PATTERN.search(cmd)
        if email_match:
            params["to"] = email_match.group(0)

        english_to = re.search(r"\bto\s+([A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)", cmd, re.IGNORECASE)
        if english_to:
            params["to"] = english_to.group(1).strip()

        chinese_to = re.search(r"(?:\u6536\u4ef6\u4eba|\u53d1\u7ed9|\u53d1\u9001\u7ed9|\u5bc4\u7ed9|\u90ae\u4ef6\u7ed9|\u53d1\u90ae\u4ef6\u7ed9|\u53d1\u9001\u90ae\u4ef6\u7ed9)\s*[:\uff1a]?\s*([A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)", cmd)
        if chinese_to:
            params["to"] = chinese_to.group(1).strip()

        english_subject = re.search(r"\bsubject\s+(.+?)(?=\s+\bbody\b\s+|$)", cmd, re.IGNORECASE)
        if english_subject:
            params["subject"] = english_subject.group(1).strip().strip('"')

        chinese_subject = re.search(r"(?:\u4e3b\u9898|\u6807\u9898)\s*[:\uff1a]?\s*(.+?)(?=\s*(?:\u6b63\u6587|\u5185\u5bb9|\u544a\u8bc9\u4ed6|\u544a\u8bc9\u5979|\u544a\u8bc9\u5bf9\u65b9|\u8bf4|\u8bf4\u4e00\u58f0)\s*[:\uff1a]?|$)", cmd)
        if chinese_subject:
            params["subject"] = chinese_subject.group(1).strip().strip('"')

        english_body = re.search(r"\bbody\s+(.+)$", cmd, re.IGNORECASE)
        if english_body:
            params["body"] = english_body.group(1).strip().strip('"')

        chinese_body = re.search(r"(?:\u6b63\u6587|\u5185\u5bb9|\u544a\u8bc9\u4ed6|\u544a\u8bc9\u5979|\u544a\u8bc9\u5bf9\u65b9|\u8bf4|\u8bf4\u4e00\u58f0)\s*[:\uff1a\uff0c,]?\s*(.+)$", cmd)
        if chinese_body:
            params["body"] = chinese_body.group(1).strip().strip('"')

        if not params["body"] and params["to"]:
            tail = cmd.split(params["to"], 1)[1].strip()
            tail = re.sub(r"^(?:\uff0c|,|\u3002|\s)*(?:\u53d1\u9001\u90ae\u4ef6\u7ed9|\u53d1\u90ae\u4ef6\u7ed9|\u90ae\u4ef6\u7ed9|\u53d1\u9001\u7ed9|\u53d1\u7ed9|\u5bc4\u7ed9)\s*", "", tail)
            tail = re.sub(r"^(?:\uff0c|,|\u3002|\s)*(?:\u544a\u8bc9\u4ed6|\u544a\u8bc9\u5979|\u544a\u8bc9\u5bf9\u65b9|\u8bf4|\u8bf4\u4e00\u58f0)\s*[:\uff1a\uff0c,]?", "", tail)
            if tail and not re.match(r"^(?:\u4e3b\u9898|\u6807\u9898|subject|body|\u6b63\u6587|\u5185\u5bb9)\b", tail, re.IGNORECASE):
                params["body"] = tail.strip(" \uff0c,\u3002")

        if not params["to"] and "\u90ae\u4ef6" in cmd and email_match:
            params["to"] = email_match.group(0)
        return params

    def _looks_like_media_analysis(self, cmd: str, lower: str, media_source: Optional[str]) -> bool:
        media_keywords = ["analyze video", "analyse video", "video analysis", "analyze image", "analyse image", "image analysis", "analyze media", "analyse media", "detect objects", "object detection", "what is in", "what's in"]
        chinese_keywords = ["\u5206\u6790\u89c6\u9891", "\u89c6\u9891\u5206\u6790", "\u5206\u6790\u56fe\u7247", "\u56fe\u7247\u5206\u6790", "\u5206\u6790\u56fe\u50cf", "\u56fe\u50cf\u5206\u6790", "\u5a92\u4f53\u5206\u6790", "\u8bc6\u522b\u76ee\u6807", "\u68c0\u6d4b\u76ee\u6807", "\u6709\u4ec0\u4e48\u76ee\u6807"]
        if any(keyword in lower for keyword in media_keywords):
            return True
        if any(keyword in cmd for keyword in chinese_keywords):
            return True
        if media_source and any(keyword in lower for keyword in ["analyze", "analyse", "detect", "object", "video", "image", "media"]):
            return True
        if media_source and any(keyword in cmd for keyword in ["\u5206\u6790", "\u76ee\u6807", "\u89c6\u9891", "\u56fe\u7247", "\u56fe\u50cf", "\u5a92\u4f53"]):
            return True
        return bool(media_source and cmd.strip() == media_source)

    def _extract_media_source(self, cmd: str) -> Optional[str]:
        ext_group = "|".join(self.MEDIA_EXTENSIONS)
        patterns = [
            rf'"([^"\r\n]+?\.(?:{ext_group}))"',
            rf"'([^'\r\n]+?\.(?:{ext_group}))'",
            rf"([A-Za-z]:\\.*?\.(?:{ext_group}))",
            rf"((?:\.{{1,2}}[\\/]|[\\/])?[^\s\"']+\.(?:{ext_group}))",
        ]
        for pattern in patterns:
            match = re.search(pattern, cmd, re.IGNORECASE)
            if match:
                candidate = self._normalize_media_source_candidate(match.group(1).strip())
                if candidate:
                    return candidate
        return None

    def _normalize_media_source_candidate(self, value: str) -> str:
        candidate = (value or "").strip().strip("\"'")
        prefixes = ["\u5206\u6790", "\u68c0\u6d4b", "\u8bc6\u522b", "\u67e5\u770b", "\u6253\u5f00", "\u770b\u770b", "\u8bf7\u5206\u6790", "\u56fe\u7247", "\u89c6\u9891", "\u56fe\u50cf"]
        suffixes = ["\u6587\u4ef6", "\u56fe\u7247", "\u56fe\u50cf", "\u89c6\u9891"]
        changed = True
        while candidate and changed:
            changed = False
            for prefix in prefixes:
                if candidate.startswith(prefix) and len(candidate) > len(prefix):
                    candidate = candidate[len(prefix):].strip()
                    changed = True
            for suffix in suffixes:
                if candidate.endswith(suffix) and len(candidate) > len(suffix):
                    candidate = candidate[:-len(suffix)].strip()
                    changed = True
        return candidate

    def _looks_like_document_write(self, cmd: str, lower: str) -> bool:
        return any(keyword in lower for keyword in ["create document", "append document", "document"]) or any(keyword in cmd for keyword in ["\u6587\u6863", "\u521b\u5efa\u6587\u6863", "\u5199\u6587\u6863", "\u8ffd\u52a0\u6587\u6863"])

    def _extract_document_params(self, cmd: str, lower: str) -> Dict[str, str]:
        match = re.match(r"(?:create|append)?\s*document\s+(\S+)(?:.*content\s+(.*))?", lower)
        if match:
            content = cmd[lower.find("content") + len("content") :].strip() if match.group(2) else ""
            return {"filename": match.group(1), "content": content}
        parts = cmd.split()
        filename = ""
        content = ""
        try:
            idx = next(i for i, part in enumerate(parts) if "\u6587\u6863" in part)
            if idx + 1 < len(parts):
                filename = parts[idx + 1]
            rest = cmd
            for keyword in ["\u5185\u5bb9", "\u8ffd\u52a0", "\u521b\u5efa", "\u6587\u6863", filename]:
                if keyword:
                    rest = rest.replace(keyword, "", 1)
            content = rest.strip()
        except StopIteration:
            pass
        return {"filename": filename, "content": content}

    def _looks_like_save_as(self, cmd: str, lower: str) -> bool:
        return any(keyword in lower for keyword in ["save as", "save file"]) or any(keyword in cmd for keyword in ["\u4fdd\u5b58\u4e3a\u6587\u4ef6", "\u4fdd\u5b58\u4e3a", "\u4fdd\u5b58\u5230"])

    def _looks_like_document_read(self, cmd: str, lower: str) -> bool:
        return any(keyword in lower for keyword in ["open file", "load file"]) or any(keyword in cmd for keyword in ["\u6253\u5f00\u6587\u4ef6", "\u52a0\u8f7d\u6587\u4ef6", "\u67e5\u770b\u6587\u4ef6"])

    def _extract_filename_after_phrase(self, cmd: str, lower: str, english_prefixes: List[str]) -> str:
        for prefix in english_prefixes:
            match = re.search(rf"{re.escape(prefix)}\s+(\S+)", lower)
            if match:
                return cmd[match.start(1) : match.end(1)]
        for keyword in ["\u4fdd\u5b58\u4e3a\u6587\u4ef6", "\u4fdd\u5b58\u4e3a", "\u4fdd\u5b58\u5230", "\u6253\u5f00\u6587\u4ef6", "\u52a0\u8f7d\u6587\u4ef6", "\u67e5\u770b\u6587\u4ef6"]:
            if keyword in cmd:
                after = cmd.split(keyword, 1)[1].strip()
                if after:
                    return after.split()[0]
        return ""

    def _looks_like_file_search(self, cmd: str, lower: str) -> bool:
        return any(keyword in lower for keyword in ["search for", "find"]) or any(keyword in cmd for keyword in ["\u641c\u7d22", "\u67e5\u627e", "\u627e"])

    def _extract_file_query(self, cmd: str, lower: str) -> str:
        if any(keyword in lower for keyword in ["web", "internet"]) or any(keyword in cmd for keyword in ["\u7f51\u9875", "\u7f51\u7ad9", "\u4e0a\u7f51"]):
            return cmd
        for prefix in ["search for", "find"]:
            if lower.startswith(prefix):
                return cmd[len(prefix) :].strip()
        for prefix in ["\u641c\u7d22", "\u67e5\u627e", "\u627e"]:
            if cmd.startswith(prefix):
                return cmd[len(prefix) :].strip()
        return cmd

    def _looks_like_model_analysis(self, cmd: str, lower: str) -> bool:
        explicit_english = [
            'analyze sentiment',
            'sentiment analysis',
            'run sentiment model',
            'classify sentiment',
        ]
        explicit_chinese = [
            '\u60c5\u611f\u5206\u6790',
            '\u60c5\u7eea\u5206\u6790',
            '\u6587\u672c\u60c5\u611f',
            '\u60c5\u611f\u5224\u65ad',
            '\u60c5\u7eea\u5224\u65ad',
        ]
        generic_exact = {'\u5206\u6790', 'run model', 'sentiment'}

        if lower in generic_exact or cmd in {'\u5206\u6790'}:
            return True
        if any(keyword in lower for keyword in explicit_english):
            return True
        return any(keyword in cmd for keyword in explicit_chinese)

    def _extract_model_text(self, cmd: str, lower: str) -> str:
        if ':' in cmd:
            return cmd.split(':', 1)[1].strip()
        for keyword in ['analyze sentiment', 'sentiment analysis', 'run sentiment model', 'classify sentiment']:
            if keyword in lower:
                return cmd[lower.find(keyword) + len(keyword) :].strip()
        for keyword in [
            '\u60c5\u611f\u5206\u6790',
            '\u60c5\u7eea\u5206\u6790',
            '\u6587\u672c\u60c5\u611f',
            '\u60c5\u611f\u5224\u65ad',
            '\u60c5\u7eea\u5224\u65ad',
        ]:
            if keyword in cmd:
                return cmd.split(keyword, 1)[1].strip()
        if lower in {'run model', 'sentiment'} or cmd == '\u5206\u6790':
            return ''
        return cmd


class Planner:
    def create_plan(self, intent: str, params: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        if intent in {"file_search", "email", "document", "document_read", "web_search", "run_model", "media_analysis", "rag_query", "rag_manage", "chat", "clear_display"}:
            return [(intent, params)]
        return [("unknown", params)]


class ToolRouter:
    def __init__(
        self,
        file_tool: FileSearchTool,
        email_tool: EmailTool,
        document_tool: DocumentTool,
        search_tool: WebSearchTool,
        model_tool: ModelTool,
        chat_tool: ChatTool,
        video_tool: VideoAnalysisTool,
        rag_tool: RAGTool,
        rag_management_tool: RAGManagementTool,
    ) -> None:
        self.file_tool = file_tool
        self.email_tool = email_tool
        self.document_tool = document_tool
        self.search_tool = search_tool
        self.model_tool = model_tool
        self.chat_tool = chat_tool
        self.video_tool = video_tool
        self.rag_tool = rag_tool
        self.rag_management_tool = rag_management_tool

    def route(self, step: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
        name, params = step
        if name == "file_search":
            return self.file_tool.search(**params)
        if name == "email":
            return self.email_tool.handle_email(to=params.get("to", ""), subject=params.get("subject", ""), body=params.get("body", ""))
        if name == "document":
            return self.document_tool.handle_document(**params)
        if name == "document_read":
            return self.document_tool.read_document(**params)
        if name == "web_search":
            return self.search_tool.search(**params)
        if name == "run_model":
            return self.model_tool.run(**params)
        if name == "media_analysis":
            return self.video_tool.analyze(**params)
        if name == "rag_query":
            return self.rag_tool.query(**params)
        if name == "rag_manage":
            return self.rag_management_tool.manage(**params)
        if name == "chat":
            return self.chat_tool.chat(prompt=params.get("prompt", ""))
        return {"error": f"Unknown command: {params.get('command', '')}"}


class ResultIntegrator:
    def integrate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        message_lines: List[str] = []
        display_content: Optional[str] = None
        display_filename: Optional[str] = None
        for result in results:
            if result is None:
                continue
            message = result.get("message")
            error = result.get("error")
            if error:
                message_lines.append(f"Error: {error}")
            elif message:
                message_lines.append(message)
            else:
                message_lines.append(str(result))
            if display_content is None and result.get("display_content") is not None:
                display_content = result.get("display_content")
                display_filename = result.get("display_filename")
        return {"message": "\n".join(message_lines), "display_content": display_content, "display_filename": display_filename}


class Agent:
    """High-level coordinator with explicit recognize -> route -> context persistence flow."""

    def __init__(self, base_dir: str, skill_threshold: float = 0.75, max_dynamic_skills: int = 3) -> None:
        self.project_root = Path(base_dir).resolve().parent
        self.skill_registry = SkillRegistry(str(self.project_root))
        self.skill_router = HybridSkillRouter(self.skill_registry)
        self.skill_manager = SkillManager(
            self.skill_registry,
            self.skill_router,
            threshold=skill_threshold,
            max_dynamic_skills=max_dynamic_skills,
        )
        self.metadata_store = MetadataStore(base_dir)
        self.retrieval_store = RetrievalStore(base_dir)
        self.evidence_store = EvidenceStore(base_dir)
        self.safety_layer = SafetyLayer(self.evidence_store)
        self.memory = ConversationMemory()

        self.file_tool = FileSearchTool(self.retrieval_store, self.evidence_store, self.safety_layer)
        self.email_tool = EmailTool(self.evidence_store, self.safety_layer)
        self.document_tool = DocumentTool(base_dir, self.evidence_store, self.safety_layer)
        self.search_tool = WebSearchTool(self.evidence_store)
        self.model_tool = ModelTool(self.evidence_store)
        self.chat_tool = ChatTool(self.evidence_store)
        self.skill_manager.attach_chat_tool(self.chat_tool)
        self.intent_tool = IntentTool(self.evidence_store)
        self.video_tool = VideoAnalysisTool(self.evidence_store, self.safety_layer, base_dir=base_dir)
        self.rag_management_tool = RAGManagementTool(base_dir, self.evidence_store, self.safety_layer)
        self.rag_store = HybridRAGStore(base_dir, registry=self.rag_management_tool.registry)
        self.rag_tool = RAGTool(self.rag_store, self.evidence_store, self.safety_layer)
        self.media_answer_agent = MediaAnswerAgent(self.chat_tool, self.skill_manager)
        self.rag_answer_agent = RAGAnswerAgent(self.chat_tool, self.skill_manager)
        self.context_agent = ContextAgent(self.memory, self.media_answer_agent, self.rag_answer_agent, self.skill_manager)
        self.tool_registry = ToolRegistry()
        self.capability_registry = CapabilityRegistry(self.tool_registry)
        self.mcp_discovery = MCPDiscoveryService(str(self.project_root))

        self.recogniser = IntentRecognizer(self.intent_tool)
        self.planner = Planner()
        self.router = ToolRouter(
            self.file_tool,
            self.email_tool,
            self.document_tool,
            self.search_tool,
            self.model_tool,
            self.chat_tool,
            self.video_tool,
            self.rag_tool,
            self.rag_management_tool,
        )
        self.integrator = ResultIntegrator()
        self._register_local_tools()
        self.refresh_dynamic_tools()

    def handle(
        self,
        command: str,
        display_content: Optional[str] = None,
        display_filename: Optional[str] = None,
        knowledge_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        effective_knowledge_base = knowledge_base or self.rag_management_tool.registry.get_active_id()
        request = UserRequest(
            command=command,
            display_content=display_content,
            display_filename=display_filename,
            knowledge_base=effective_knowledge_base,
        )
        context_request = ContextRequest(**request.__dict__)
        self.memory.add_turn("user", request.command)
        self.evidence_store.log_event({"event": "command_received", "command": request.command})

        tool_call_payload = self._attempt_tool_calling(context_request)
        if tool_call_payload is not None:
            tool_intent, response, raw_results = tool_call_payload
            if tool_intent == "clear_display":
                self.evidence_store.log_event({"event": "response_generated", "response": response.get("message")})
                return response
            self.context_agent.remember_response(tool_intent, response, raw_results=raw_results)
            self.evidence_store.log_event({"event": "response_generated", "response": response.get("message")})
            return response

        if self.recogniser.is_capability_question(request.command):
            response = self._answer_capability_question(context_request)
            self.context_agent.remember_response("chat", response)
            self.evidence_store.log_event({"event": "response_generated", "response": response.get("message")})
            return response

        intent, params = self.recogniser.recognise(request.command)
        intent, params = self.context_agent.normalize_request(intent, params, context_request, self.recogniser)

        if intent == "clear_display":
            response = self.clear_display_context()
            self.evidence_store.log_event({"event": "response_generated", "response": response.get("message")})
            return response

        auto_rag_payload = None
        if intent == "chat" and not self.recogniser.is_capability_question(request.command) and self._should_use_rag_for_chat(context_request):
            auto_rag_payload = self._attempt_auto_rag(request)
        if auto_rag_payload is not None:
            auto_rag_response, retrieval_result = auto_rag_payload
            self.context_agent.remember_response("rag_query", auto_rag_response, raw_results=[retrieval_result])
            self.evidence_store.log_event({"event": "response_generated", "response": auto_rag_response.get("message")})
            return auto_rag_response

        if intent == "chat":
            response = self._answer_chat(context_request)
            self.context_agent.remember_response(intent, response)
            self.evidence_store.log_event({"event": "response_generated", "response": response.get("message")})
            return response

        plan = self.planner.create_plan(intent, params)
        results = [self.router.route(step) for step in plan]
        response = self._finalize_response(intent, request, results)
        self.context_agent.remember_response(intent, response, raw_results=results)
        self.evidence_store.log_event({"event": "response_generated", "response": response.get("message")})
        return response

    def _attempt_auto_rag(self, request: UserRequest) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        knowledge_base = request.knowledge_base or self.context_agent.active_knowledge_base() or ""
        retrieval_result = self.rag_tool.query(query=request.command, knowledge_base=knowledge_base)
        if retrieval_result.get("error") or not retrieval_result.get("answerable"):
            return None
        response = self.rag_answer_agent.answer(request.command, retrieval_result)
        return response, retrieval_result

    def _finalize_response(self, intent: str, request: UserRequest, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if intent == "media_analysis" and len(results) == 1 and "error" not in results[0]:
            return self.media_answer_agent.answer(request.command, results[0])
        if intent == "rag_query" and len(results) == 1 and "error" not in results[0]:
            return self.rag_answer_agent.answer(request.command, results[0])
        return self.integrator.integrate(results)

    def _register_local_tools(self) -> None:
        self.tool_registry.unregister_source("local")
        self.tool_registry.register(
            ToolSpec(
                tool_id="workspace.file_search",
                name="search_files",
                description="Search for files inside the local workspace.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "File name or keyword to search for."},
                    },
                    "required": ["query"],
                },
                source="local",
                category="workspace",
                tags=("file", "search", "文件", "搜索", "查找"),
                capability_summary="搜索工作区中的文件",
                legacy_intent="file_search",
            ),
            handler=lambda args, context=None: self.file_tool.search(query=str(args.get("query", "") or "")),
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="workspace.document_write",
                name="write_document",
                description="Create, append, or save a text document in the workspace.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Target file name in the workspace."},
                        "content": {"type": "string", "description": "Text content to write."},
                    },
                    "required": ["filename"],
                },
                source="local",
                category="workspace",
                requires_confirmation=True,
                tags=("document", "write", "save", "文档", "保存", "写入"),
                capability_summary="创建、追加和保存工作区中的文本文件",
                legacy_intent="document",
            ),
            handler=lambda args, context=None: self.document_tool.handle_document(
                filename=str(args.get("filename", "") or ""),
                content=str(args.get("content", "") or ""),
            ),
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="workspace.document_read",
                name="read_document",
                description="Load a text document from the workspace.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "File name to load from the workspace."},
                    },
                    "required": ["filename"],
                },
                source="local",
                category="workspace",
                tags=("document", "read", "open", "load", "文档", "打开", "加载"),
                capability_summary="加载工作区中的文本文件",
                legacy_intent="document_read",
            ),
            handler=lambda args, context=None: self.document_tool.read_document(filename=str(args.get("filename", "") or "")),
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="communication.email_send",
                name="send_email",
                description="Draft or send an email message.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email address."},
                        "subject": {"type": "string", "description": "Email subject line."},
                        "body": {"type": "string", "description": "Email body content."},
                    },
                    "required": ["to"],
                },
                source="local",
                category="communication",
                requires_confirmation=True,
                tags=("email", "mail", "邮件", "发送", "收件人"),
                capability_summary="起草并发送邮件；未配置 SMTP 时会生成本地邮件草稿",
                legacy_intent="email",
            ),
            handler=lambda args, context=None: self.email_tool.handle_email(
                to=str(args.get("to", "") or ""),
                subject=str(args.get("subject", "") or ""),
                body=str(args.get("body", "") or ""),
            ),
            availability=self._email_tool_status,
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="web.search",
                name="search_web",
                description="Search the web using the configured search provider.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                    },
                    "required": ["query"],
                },
                source="local",
                category="search",
                tags=("web", "search", "google", "网页", "搜索", "上网"),
                capability_summary="执行网页搜索；配置 Google Custom Search 后可返回真实搜索结果",
                legacy_intent="web_search",
            ),
            handler=lambda args, context=None: self.search_tool.search(query=str(args.get("query", "") or "")),
            availability=self._web_search_status,
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="analysis.sentiment",
                name="analyze_sentiment",
                description="Run a simple local sentiment analysis model over text.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze for sentiment."},
                    },
                },
                source="local",
                category="analysis",
                tags=("sentiment", "analysis", "情感", "情绪", "文本分析"),
                capability_summary="对文本执行本地情感分析",
                legacy_intent="run_model",
            ),
            handler=lambda args, context=None: self.model_tool.run(text=str(args.get("text", "") or "")),
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="media.analysis",
                name="analyze_media",
                description="Analyze an image or video for detected objects.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Image or video path."},
                        "weights": {"type": "string", "description": "Optional YOLO weights path."},
                        "max_frames": {"type": "integer", "description": "Maximum frames to inspect for video input."},
                    },
                    "required": ["source"],
                },
                source="local",
                category="analysis",
                tags=("media", "image", "video", "detect", "图片", "视频", "媒体", "目标检测"),
                capability_summary="分析图片和视频中的目标，并支持基于检测结果继续追问",
                legacy_intent="media_analysis",
            ),
            handler=lambda args, context=None: self.video_tool.analyze(
                source=str(args.get("source", "") or ""),
                weights=str(args.get("weights", "") or "") or None,
                max_frames=int(args.get("max_frames", 100) or 100),
            ),
            availability=self._media_tool_status,
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="knowledge.rag_query",
                name="query_knowledge_base",
                description="Retrieve evidence from the selected local knowledge base.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Knowledge question to search for."},
                        "top_k": {"type": "integer", "description": "Maximum number of evidence chunks to retrieve."},
                        "knowledge_base": {"type": "string", "description": "Knowledge-base id to query."},
                    },
                    "required": ["query"],
                },
                source="local",
                category="knowledge",
                tags=("rag", "knowledge", "retrieval", "知识库", "检索", "RAG"),
                capability_summary="查询本地知识库并返回相关证据",
                legacy_intent="rag_query",
            ),
            handler=lambda args, context=None: self.rag_tool.query(
                query=str(args.get("query", "") or ""),
                top_k=int(args.get("top_k", 6) or 6),
                knowledge_base=str(args.get("knowledge_base", "") or ""),
            ),
            availability=self._rag_query_status,
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="knowledge.rag_manage",
                name="manage_knowledge_base",
                description="Inspect or update local knowledge-base indexing state.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Management action such as status, contents, ingest, rebuild, list, save, or select.",
                        },
                        "input_dir": {"type": "string", "description": "Optional knowledge source directory."},
                        "knowledge_base": {"type": "string", "description": "Knowledge-base id."},
                        "knowledge_base_name": {"type": "string", "description": "Display name when creating a knowledge base."},
                        "description": {"type": "string", "description": "Optional knowledge-base description."},
                    },
                    "required": ["action"],
                },
                source="local",
                category="knowledge",
                tags=("rag", "index", "knowledge", "知识库", "入库", "索引"),
                capability_summary="查看知识库状态、预览内容、增量入库并重建索引",
                legacy_intent="rag_manage",
            ),
            handler=lambda args, context=None: self.rag_management_tool.manage(
                action=str(args.get("action", "") or ""),
                input_dir=str(args.get("input_dir", "") or ""),
                knowledge_base=str(args.get("knowledge_base", "") or ""),
                knowledge_base_name=str(args.get("knowledge_base_name", "") or ""),
                description=str(args.get("description", "") or ""),
            ),
        )
        self.tool_registry.register(
            ToolSpec(
                tool_id="context.clear_display",
                name="clear_display_context",
                description="Clear the background information or display panel context.",
                parameters_schema={"type": "object", "properties": {}},
                source="local",
                category="context",
                tags=("clear", "display", "context", "清空", "背景信息", "展示区"),
                capability_summary="清空右侧展示区和当前背景上下文",
                legacy_intent="clear_display",
            ),
            handler=lambda args, context=None: self.clear_display_context(),
        )

    def refresh_dynamic_tools(self) -> Dict[str, Any]:
        return self.mcp_discovery.refresh(self.tool_registry)

    def list_registered_tools(self) -> Dict[str, Any]:
        rows = self.tool_registry.snapshot()
        lines = ["Registered tools:"]
        for row in rows:
            status = row["status"]
            detail = f" ({status['detail']})" if status.get("detail") else ""
            lines.append(f"- {row['name']} [{row['source']}/{row['category']}] : {row['description']}{detail}")
        return {
            "message": "Tool registry loaded.",
            "tools": rows,
            "display_content": "\n".join(lines),
            "display_filename": "tools_registry.txt",
        }

    def list_capabilities(self) -> Dict[str, Any]:
        rows = self.capability_registry.list_capabilities()
        lines = ["Capabilities:"]
        for row in rows:
            status = row["status"]
            detail = f" ({status['detail']})" if status.get("detail") else ""
            lines.append(f"- {row['capability_summary']}{detail}")
        return {
            "message": "Capabilities loaded.",
            "capabilities": rows,
            "display_content": "\n".join(lines),
            "display_filename": "capabilities.txt",
        }

    def _attempt_tool_calling(
        self,
        request: ContextRequest,
    ) -> Optional[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
        if not self.chat_tool.is_configured():
            return None
        if not hasattr(self.chat_tool, "complete_with_tools"):
            return None

        tools = self.tool_registry.list_openai_tools()
        if not tools:
            return None

        messages = self.context_agent.build_chat_messages(request.command)
        planner_prompt = self.context_agent.build_chat_system_prompt(
            self.capability_registry.build_tool_calling_prompt(),
            user_command=request.command,
            scope="planning",
        )
        result = self.chat_tool.complete_with_tools(
            messages=messages,
            tools=tools,
            system_prompt=planner_prompt,
            tool_choice="auto",
        )
        if result.get("error"):
            return None

        raw_calls = list(result.get("tool_calls") or [])
        if not raw_calls:
            return None

        prepared_calls: List[Tuple[str, Dict[str, Any], Optional[ToolSpec]]] = []
        for call in raw_calls:
            tool_ref = str(call.get("name", "") or "").strip()
            arguments = dict(call.get("arguments") or {})
            normalized_ref, normalized_args, spec = self._normalize_tool_call(tool_ref, arguments, request)
            if not normalized_ref:
                continue
            prepared_calls.append((normalized_ref, normalized_args, spec))
        if not prepared_calls:
            return None

        raw_results = self.tool_registry.execute_many(
            [(tool_ref, arguments) for tool_ref, arguments, _spec in prepared_calls],
            context={"request": request, "agent": self},
        )
        if raw_results and all("error" in result for result in raw_results):
            return None
        response = self._finalize_tool_calling_response(request, prepared_calls, raw_results)
        tool_intent = (prepared_calls[0][2].legacy_intent if prepared_calls[0][2] is not None else "") or "chat"
        response["used_tools"] = [spec.name if spec is not None else tool_ref for tool_ref, _arguments, spec in prepared_calls]
        return tool_intent, response, raw_results

    def _normalize_tool_call(
        self,
        tool_ref: str,
        arguments: Dict[str, Any],
        request: ContextRequest,
    ) -> Tuple[str, Dict[str, Any], Optional[ToolSpec]]:
        spec = self.tool_registry.get_spec(tool_ref)
        if spec is None:
            return tool_ref, arguments, None

        normalized_ref = spec.tool_id
        normalized_args = dict(arguments or {})
        if spec.legacy_intent:
            normalized_intent, normalized_args = self.context_agent.normalize_request(
                spec.legacy_intent,
                normalized_args,
                request,
                self.recogniser,
            )
            if normalized_intent != spec.legacy_intent:
                redirected_spec = self.tool_registry.find_by_legacy_intent(normalized_intent)
                if redirected_spec is not None:
                    spec = redirected_spec
                    normalized_ref = redirected_spec.tool_id
        return normalized_ref, normalized_args, spec

    def _finalize_tool_calling_response(
        self,
        request: ContextRequest,
        prepared_calls: List[Tuple[str, Dict[str, Any], Optional[ToolSpec]]],
        raw_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        primary_spec = prepared_calls[0][2]
        primary_intent = primary_spec.legacy_intent if primary_spec is not None else "chat"
        if primary_intent == "media_analysis" and len(raw_results) == 1 and "error" not in raw_results[0]:
            return self.media_answer_agent.answer(request.command, raw_results[0])
        if primary_intent == "rag_query" and len(raw_results) == 1 and "error" not in raw_results[0]:
            return self.rag_answer_agent.answer(request.command, raw_results[0])
        response = self.integrator.integrate(raw_results)
        if request.knowledge_base:
            response["active_knowledge_base"] = request.knowledge_base
        return response

    def _email_tool_status(self) -> Dict[str, Any]:
        smtp_ready = all(
            [
                bool(os.environ.get("SMTP_SERVER")),
                bool(os.environ.get("SMTP_PORT")),
                bool(os.environ.get("SMTP_USERNAME")),
                bool(os.environ.get("SMTP_PASSWORD")),
            ]
        )
        if smtp_ready:
            return {"available": True, "status": "available", "detail": "已配置 SMTP，可实际发送邮件。"}
        return {"available": True, "status": "partial", "detail": "当前可起草本地邮件草稿；配置 SMTP 后可实际发送。"}

    def _web_search_status(self) -> Dict[str, Any]:
        configured = bool(self.search_tool.api_key and self.search_tool.cse_id)
        if configured:
            return {"available": True, "status": "available", "detail": "已配置 Google Custom Search。"}
        return {"available": True, "status": "partial", "detail": "当前未配置 Google Custom Search，工具会返回配置提示。"}

    def _media_tool_status(self) -> Dict[str, Any]:
        if hasattr(self.video_tool, "_model_cache"):
            try:
                from tools.video_tool import YOLO  # type: ignore
            except Exception:
                YOLO = None  # type: ignore
            if YOLO is None:
                return {"available": True, "status": "partial", "detail": "当前未安装 ultralytics，调用时会返回依赖提示。"}
        return {"available": True, "status": "available", "detail": "可分析图片和视频中的目标。"}

    def _rag_query_status(self) -> Dict[str, Any]:
        active_id = self.rag_management_tool.registry.get_active_id()
        if active_id and self.rag_store.is_available(active_id):
            return {"available": True, "status": "available", "detail": f"当前激活知识库为 {active_id}。"}
        return {"available": True, "status": "partial", "detail": "当前知识库未完成索引时，查询会返回入库提示。"}

    def _answer_chat(self, request: ContextRequest) -> Dict[str, Any]:
        if self.recogniser.is_capability_question(request.command):
            return self._answer_capability_question(request)

        if self.context_agent.can_answer_media_followup(request):
            contextual_answer = self.context_agent.answer_media_followup(request)
            if "error" not in contextual_answer:
                return contextual_answer

        if self.context_agent.can_answer_rag_followup(request):
            contextual_answer = self.context_agent.answer_rag_followup(request)
            if "error" not in contextual_answer:
                return contextual_answer

        if not self.chat_tool.is_configured():
            fallback_message = self._build_chat_unavailable_reply()
            return {
                "message": fallback_message,
                "display_content": fallback_message,
                "display_filename": request.display_filename or self.context_agent.active_display_filename(),
            }

        messages = self.context_agent.build_chat_messages(request.command)
        result = self.chat_tool.complete_messages(
            messages=messages,
            system_prompt=self.context_agent.build_chat_system_prompt(
                self.capability_registry.build_conversation_prompt(),
                user_command=request.command,
                scope="chat",
            ),
        )
        if result.get("error"):
            fallback_message = self._build_chat_unavailable_reply()
            return {
                "message": fallback_message,
                "display_content": fallback_message,
                "display_filename": request.display_filename or self.context_agent.active_display_filename(),
            }
        return self.context_agent.prepare_chat_result(result, request, scope='chat')

    def _should_use_rag_for_chat(self, request: ContextRequest) -> bool:
        knowledge_base = (request.knowledge_base or self.context_agent.active_knowledge_base() or "").strip()
        if not knowledge_base or not self.rag_store.is_available(knowledge_base):
            return False
        command = (request.command or "").strip()
        if not command or len(command) < 4:
            return False
        lowered = command.lower()
        if self.recogniser.is_capability_question(command):
            return False
        small_talk = ["hello", "hi", "thanks", "thank you", "你好", "您好", "谢谢", "再见", "拜拜"]
        if lowered in small_talk or any(lowered.startswith(item + " ") for item in small_talk if " " not in item):
            return False

        non_rag_tasks = [
            "翻译", "润色", "改写", "写一封", "写封", "写邮件", "发送邮件", "发邮件",
            "save as", "open file", "load file", "search web", "web search",
            "讲个笑话", "夸夸我", "写诗", "创作", "生成图片",
        ]
        if any(marker in command for marker in non_rag_tasks) or any(marker in lowered for marker in non_rag_tasks):
            return False

        question_markers = [
            "?", "？", "where", "what", "why", "how", "when", "which",
            "什么", "为什么", "如何", "怎么样", "哪些", "多少", "是否", "能否",
            "介绍", "说明", "总结", "概括", "趋势", "情况", "原因", "影响", "区别", "对比", "是什么",
        ]
        if any(marker in command or marker in lowered for marker in question_markers):
            return True

        if re.search(r"20\d{2}年", command) or re.search(r"\b20\d{2}\b", lowered):
            return True

        return False

    def _build_chat_unavailable_reply(self) -> str:
        return self.capability_registry.unavailable_chat_reply()

    def _answer_capability_question(self, request: ContextRequest) -> Dict[str, Any]:
        fallback_message = self._build_capability_answer(request.command)
        if not self.chat_tool.is_configured():
            return {
                "message": fallback_message,
                "display_content": fallback_message,
                "display_filename": None,
            }

        messages = self.context_agent.build_chat_messages(request.command)
        result = self.chat_tool.complete_messages(
            messages=messages,
            system_prompt=self.context_agent.build_chat_system_prompt(
                self.capability_registry.build_capability_answer_prompt(),
                user_command=request.command,
                scope="capability",
            ),
        )
        if result.get("error"):
            return {
                "message": fallback_message,
                "display_content": fallback_message,
                "display_filename": None,
            }

        message = self.skill_manager.postprocess_response((result.get("message") or "").strip() or fallback_message, scope='capability', user_query=request.command)
        response = {
            "message": message,
            "display_content": message,
            "display_filename": None,
        }
        if request.knowledge_base:
            response["active_knowledge_base"] = request.knowledge_base
        return response

    def _build_local_capability_reply(self, request: ContextRequest) -> str:
        return self.capability_registry.fallback_answer(request.command)

    def _build_capability_answer(self, command: str) -> str:
        return self.capability_registry.fallback_answer(command)

    def clear_display_context(self) -> Dict[str, Any]:
        self.context_agent.clear_display_context()
        return {
            "message": "已清空背景信息窗口。",
            "display_content": "",
            "display_filename": "",
            "active_knowledge_base": self.rag_management_tool.registry.get_active_id() or None,
        }

