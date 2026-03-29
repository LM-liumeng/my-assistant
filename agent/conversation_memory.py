"""Thread-scoped short-term memory for the demo assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AssistantArtifact:
    kind: str
    display_filename: Optional[str] = None
    display_content: Optional[str] = None
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.active_artifact: Optional[AssistantArtifact] = None
        self.last_assistant_message: Optional[str] = None
        self.active_knowledge_base: Optional[str] = None

    def add_turn(self, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        self.turns.append(ConversationTurn(role=role, content=text))
        if role == "assistant":
            self.last_assistant_message = text
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def remember_artifact(
        self,
        kind: str,
        display_filename: Optional[str],
        display_content: Optional[str],
        payload: Optional[Dict[str, object]] = None,
    ) -> None:
        self.active_artifact = AssistantArtifact(
            kind=kind,
            display_filename=display_filename,
            display_content=display_content,
            payload=dict(payload or {}),
        )

    def remember_knowledge_base(self, knowledge_base: Optional[str]) -> None:
        text = (knowledge_base or '').strip()
        if text:
            self.active_knowledge_base = text

    def clear_artifact(self) -> None:
        self.active_artifact = None

    def clear_active_context(self) -> None:
        self.active_artifact = None
        self.last_assistant_message = None

    def get_recent_turns(self, limit: int = 8) -> List[ConversationTurn]:
        return self.turns[-limit:]
