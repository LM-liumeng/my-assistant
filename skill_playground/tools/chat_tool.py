from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import os
import urllib.request

from skill_playground.agent.trace import TraceRecorder


DEFAULT_SYSTEM_PROMPT = (
    "You are the chat engine of a skill testing playground. "
    "Use only the skills and constraints provided by the host application."
)


class _NullEvidenceStore:
    def log_event(self, payload: Dict[str, Any]) -> None:
        return None


class ChatTool:
    def __init__(self, evidence_store: Optional[object] = None) -> None:
        self.evidence_store = evidence_store or _NullEvidenceStore()
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

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None, trace: Optional[TraceRecorder] = None) -> Dict[str, Any]:
        return self.complete_messages(
            messages=[{"role": "user", "content": (user_prompt or "").strip()}],
            system_prompt=system_prompt,
            trace=trace,
        )

    def complete_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        trace: Optional[TraceRecorder] = None,
    ) -> Dict[str, Any]:
        cleaned_messages = [m for m in messages if (m.get("content") or "").strip()]
        if not cleaned_messages:
            if trace:
                trace.log("chat_tool", "Rejected empty prompt.")
            return {"error": "Empty prompt.", "message": "Please provide a prompt."}
        if not self.api_key:
            if trace:
                trace.log("chat_tool", "Chat service is not configured.")
            return {
                "error": "Chat service is not configured.",
                "message": "Chat service is not configured. Set MY_DEEPSEEK_KEY or DEEPSEEK_API_KEY to enable the playground.",
            }

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt or self.system_prompt}, *cleaned_messages],
        }
        if trace:
            trace.log(
                "chat_tool",
                "Sending chat completion request.",
                {"model": self.model, "message_count": len(cleaned_messages)},
            )
        return self._request_completion(payload, trace=trace)

    def complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tool_choice: Any = "auto",
        trace: Optional[TraceRecorder] = None,
    ) -> Dict[str, Any]:
        cleaned_messages = [m for m in messages if (m.get("content") or "").strip()]
        if not cleaned_messages:
            if trace:
                trace.log("chat_tool", "Rejected empty prompt for tool calling.")
            return {"error": "Empty prompt.", "message": "Please provide a prompt."}
        if not self.api_key:
            if trace:
                trace.log("chat_tool", "Chat service is not configured for tool calling.")
            return {
                "error": "Chat service is not configured.",
                "message": "Chat service is not configured. Set MY_DEEPSEEK_KEY or DEEPSEEK_API_KEY to enable the playground.",
            }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt or self.system_prompt}, *cleaned_messages],
            "tools": list(tools or []),
            "tool_choice": tool_choice,
        }
        if trace:
            trace.log(
                "chat_tool",
                "Sending tool-enabled completion request.",
                {"model": self.model, "message_count": len(cleaned_messages), "tool_count": len(tools or [])},
            )
        return self._request_completion(payload, trace=trace)

    def _request_completion(self, payload: Dict[str, Any], trace: Optional[TraceRecorder] = None) -> Dict[str, Any]:
        try:
            self.evidence_store.log_event({"event": "playground_chat_invoked", "payload": payload})
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
            if trace:
                trace.log("chat_tool", "Chat completion request failed.", {"error": str(exc)})
            try:
                self.evidence_store.log_event({"event": "playground_chat_error", "error": str(exc)})
            except Exception:
                pass
            return {"error": f"Chat service call failed: {exc}", "message": "Chat service is temporarily unavailable."}

        choices = body.get("choices", [])
        if not choices:
            if trace:
                trace.log("chat_tool", "Model returned no choices.")
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
            if trace:
                trace.log("chat_tool", "Model returned tool calls.", {"tool_call_count": len(tool_calls)})
            return {"message": message, "tool_calls": tool_calls, "used_llm": True}
        if not message:
            if trace:
                trace.log("chat_tool", "Model returned an empty message.")
            return {"error": "Empty completion message.", "message": "The language model returned an empty answer."}
        if trace:
            trace.log("chat_tool", "Received chat completion response.", {"content_preview": message[:120]})
        return {"message": message, "used_llm": True}
