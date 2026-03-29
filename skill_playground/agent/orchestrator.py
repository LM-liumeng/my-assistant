"""Minimal chat-only orchestrator for the skill playground."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from skill_playground.agent.skill_manager import SkillManager
from skill_playground.agent.skill_registry import SkillRegistry
from skill_playground.agent.trace import TraceRecorder
from skill_playground.tools.chat_tool import ChatTool


BASE_SYSTEM_PROMPT = (
    "You are a local skill playground assistant. "
    "This playground is only for testing how skills change planning, tone, structure, and response quality. "
    "Do not claim tools or host capabilities that are not explicitly present in this playground."
)


class PlaygroundOrchestrator:
    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root)
        self.registry = SkillRegistry(str(self.project_root))
        self.chat_tool = ChatTool()
        self.skill_manager = SkillManager(self.registry)
        self.skill_manager.attach_chat_tool(self.chat_tool)

    def refresh_skills(self) -> None:
        self.registry.reload()
        self.skill_manager = SkillManager(self.registry)
        self.skill_manager.attach_chat_tool(self.chat_tool)

    def chat(self, message: str, history: Optional[List[Dict[str, str]]] = None, trace: Optional[TraceRecorder] = None) -> Dict[str, Any]:
        user_text = (message or "").strip()
        if not user_text:
            return {"error": "Empty message.", "message": "Please enter a message."}

        if trace:
            trace.log("orchestrator", "Starting playground chat request.", {"message": user_text})

        generation_plan = self.skill_manager.resolve_plan(user_text, scope="chat", phase="generation", trace=trace)
        system_prompt = self.skill_manager.compiler.compile_prompt(BASE_SYSTEM_PROMPT, generation_plan)
        if trace:
            trace.log(
                "orchestrator",
                "Generation prompt prepared.",
                {
                    "selected_skills": [item.skill.skill_id for item in generation_plan.selected],
                    "primary_skill": generation_plan.primary.skill.skill_id if generation_plan.primary else None,
                },
            )
        messages = self._build_messages(history or [], user_text)
        if trace:
            trace.log("orchestrator", "Chat messages prepared.", {"history_turns": len(messages) - 1})

        completion = self.chat_tool.complete_messages(messages=messages, system_prompt=system_prompt, trace=trace)
        reply = str(completion.get("message") or "").strip()
        if not reply:
            reply = "The chat model did not return a response."
            if trace:
                trace.log("orchestrator", "Chat completion returned no message; using fallback text.")

        final_text = self.skill_manager.postprocess_response(reply, scope="chat", user_query=user_text, trace=trace)
        if trace:
            trace.log("orchestrator", "Playground chat request completed.", {"response_preview": final_text[:160]})
        return {
            "message": final_text,
            "active_skills": [item.skill.display_name for item in generation_plan.selected],
            "active_skill_ids": [item.skill.skill_id for item in generation_plan.selected],
            "primary_skill": generation_plan.primary.skill.display_name if generation_plan.primary else None,
            "semantic_judgment": generation_plan.semantic_judgment.to_dict() if generation_plan.semantic_judgment else None,
            "used_llm": bool(completion.get("used_llm")),
            "error": completion.get("error"),
        }

    def _build_messages(self, history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
        trimmed: List[Dict[str, str]] = []
        for item in history[-12:]:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            trimmed.append({"role": role, "content": content})
        trimmed.append({"role": "user", "content": user_text})
        return trimmed
