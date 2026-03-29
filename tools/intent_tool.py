"""IntentTool: use an LLM to parse user commands into intents and arguments.

This tool delegates natural‑language understanding to a large language model.
Given a free‑form user command it asks the model to return a small JSON object
of the form:

    {
      "intent": "email" | "document" | "document_read" | "file_search"
                | "web_search" | "run_model" | "chat",
      "args": { ... }   # free‑form key/value parameters
    }

If the API call fails or the response cannot be parsed as JSON, ``None`` is
returned so that the caller can fall back to heuristic rules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import json
import os

from context.evidence_store import EvidenceStore


class IntentTool:
    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store
        # Reuse the same configuration pattern as ChatTool so that users can
        # configure endpoints in one place.
        self.api_key = (
            os.environ.get("MY_DEEPSEEK_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.api_base = (
            os.environ.get("DEEPSEEK_API_BASE")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.deepseek.com"
        )
        self.model = os.environ.get("DEEPSEEK_MODEL") or os.environ.get("LLM_MODEL", "deepseek-chat")

    def _log(self, data: Dict[str, Any]) -> None:
        try:
            self.evidence_store.log_event(data)
        except Exception:
            pass

    def parse_intent(self, command: str) -> Optional[Dict[str, Any]]:
        """Return a parsed intent dict or ``None`` on failure.

        The returned dictionary has at least an ``intent`` key and may contain
        an ``args`` dictionary.  On any error (missing API key, HTTP failure,
        invalid JSON) this method returns ``None`` so that the caller can use
        a fallback recogniser.
        """
        text = (command or "").strip()
        if not text:
            return None

        # If chat credentials are not configured, skip LLM parsing.
        if not self.api_key:
            return None

        self._log({"event": "intent_llm_invoked", "command": text})

        # System prompt constrains the model to output STRICT JSON only.
        system_prompt = (
            "你是一个指令解析器，只负责把用户的自然语言指令解析成结构化 JSON。"
            "你可以识别以下几类操作：\n"
            "- email: 发送邮件\n"
            "- document: 创建或写入文档\n"
            "- document_read: 打开/读取文档\n"
            "- file_search: 在本地文件中搜索\n"
            "- web_search: 上网搜索\n"
            "- run_model: 对文本做分析或模型推理\n"
            "- chat: 与助手闲聊或问答\n\n"
            "请根据用户输入，返回一个 JSON 对象，格式如下：\n"
            '{\"intent\": \"email\" | \"document\" | \"document_read\" | '
            '\"file_search\" | \"web_search\" | \"run_model\" | \"chat\", '
            '\"args\": { ... 任意键值对 ... }}\n'
            "如果你不确定要调用哪种工具，就使用 intent \"chat\"，并把原始用户输入放在 "
            "args.prompt 里。不要输出任何解释性文字，也不要输出 Markdown，只输出 JSON。"
        )

        try:
            import urllib.error
            import urllib.request

            api_url = f"{self.api_base.rstrip('/')}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                api_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req) as resp:
                resp_data = json.load(resp)
        except Exception as exc:
            self._log({"event": "intent_llm_error", "error": str(exc)})
            return None

        try:
            choices = resp_data.get("choices", [])
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content", "")
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                return None
            return parsed
        except Exception as exc:
            self._log({"event": "intent_llm_parse_error", "error": str(exc)})
            return None

