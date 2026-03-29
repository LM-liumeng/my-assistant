"""Chat tool and optional OpenAI-compatible completion wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import os
import urllib.request

from context.evidence_store import EvidenceStore


DEFAULT_SYSTEM_PROMPT = (
    "You are a local office assistant. Use only the capabilities that are actually "
    "implemented by the host application. Do not claim access to arbitrary files, "
    "websites, shells, or private systems unless the host explicitly provides that tool."
)


class ChatTool:
    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store
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
        self.system_prompt = os.environ.get("LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        return self.complete_messages(
            messages=[{"role": "user", "content": (user_prompt or "").strip()}],
            system_prompt=system_prompt,
        )

    def complete_messages(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> Dict[str, Any]:
        cleaned_messages = [m for m in messages if (m.get("content") or "").strip()]
        if not cleaned_messages:
            return {"error": "Empty prompt.", "message": "Please provide a prompt."}
        if not self.api_key:
            return {"error": "Chat service is not configured.", "message": "Chat service is not configured."}

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt or self.system_prompt}, *cleaned_messages],
        }
        return self._request_completion(payload, event_name="chat_complete")

    def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: Any = "auto",
    ) -> Dict[str, Any]:
        cleaned_messages = [m for m in messages if (m.get("content") or "").strip()]
        if not cleaned_messages:
            return {"error": "Empty prompt.", "message": "Please provide a prompt."}
        if not self.api_key:
            return {"error": "Chat service is not configured.", "message": "Chat service is not configured."}

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt or self.system_prompt}, *cleaned_messages],
            "tools": list(tools or []),
            "tool_choice": tool_choice,
        }
        return self._request_completion(payload, event_name="chat_complete_with_tools")

    def _request_completion(self, payload: Dict[str, Any], event_name: str) -> Dict[str, Any]:
        try:
            self.evidence_store.log_event({"event": f"{event_name}_invoked", "payload": payload})
        except Exception:
            pass

        req = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.load(resp)
        except Exception as exc:
            try:
                self.evidence_store.log_event({"event": f"{event_name}_error", "error": str(exc)})
            except Exception:
                pass
            return {"error": f"Chat service call failed: {exc}", "message": "Chat service is temporarily unavailable."}

        choices = body.get("choices", [])
        if not choices:
            return {"error": "No completion choices returned.", "message": "The language model returned no answer."}
        first_message = choices[0].get("message", {}) or {}
        tool_calls = []
        for call in first_message.get("tool_calls", []) or []:
            function = dict(call.get("function") or {})
            raw_arguments = function.get("arguments", "")
            parsed_arguments: Dict[str, Any]
            if isinstance(raw_arguments, str) and raw_arguments.strip():
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except Exception:
                    parsed_arguments = {"_raw": raw_arguments}
            elif isinstance(raw_arguments, dict):
                parsed_arguments = raw_arguments
            else:
                parsed_arguments = {}
            tool_calls.append(
                {
                    "id": call.get("id"),
                    "type": call.get("type", "function"),
                    "name": function.get("name", ""),
                    "arguments": parsed_arguments,
                    "raw_arguments": raw_arguments,
                }
            )

        message = str(first_message.get("content") or "").strip()
        if tool_calls:
            return {"message": message, "tool_calls": tool_calls, "used_llm": True}
        if not message:
            return {"error": "Empty completion message.", "message": "The language model returned an empty answer."}
        return {"message": message, "used_llm": True}

    def chat(self, prompt: str) -> Dict[str, Any]:
        text = (prompt or "").strip()
        if not text:
            return {"message": "Please enter a message."}
        if not self.api_key:
            return {
                "message": (
                    "Chat is not enabled. Set MY_DEEPSEEK_KEY or DEEPSEEK_API_KEY "
                    "(and optionally DEEPSEEK_API_BASE, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, "
                    "OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_BASE_URL, LLM_MODEL, LLM_SYSTEM_PROMPT) to enable it."
                )
            }
        return self.complete(text)
