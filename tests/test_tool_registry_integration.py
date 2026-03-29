import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.capability_registry import CapabilityRegistry
from agent.mcp_discovery import MCPDiscoveryService
from agent.orchestrator import Agent
from agent.tool_registry import ToolRegistry, ToolSpec


def test_capability_registry_builds_fallback_from_registered_tools():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            tool_id="communication.email_send",
            name="send_email",
            description="Draft or send an email message.",
            parameters_schema={"type": "object", "properties": {"to": {"type": "string"}}},
            category="communication",
            tags=("email", "mail", "smtp"),
            capability_summary="draft and send email, with local draft fallback when SMTP is not configured",
        ),
        handler=lambda args, context=None: {"message": "ok"},
    )
    capability_registry = CapabilityRegistry(registry)

    answer = capability_registry.fallback_answer("can you send email")

    assert "email" in answer.lower()
    assert "smtp" in answer.lower()


def test_agent_prefers_tool_calling_before_legacy_intent_flow(tmp_path: Path):
    project_root = tmp_path / "project"
    base_dir = project_root / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    agent = Agent(str(base_dir))

    class StubChatTool:
        def is_configured(self):
            return True

        def complete_with_tools(self, messages, tools, system_prompt=None, tool_choice="auto"):
            return {
                "message": "",
                "tool_calls": [
                    {
                        "name": "clear_display_context",
                        "arguments": {},
                    }
                ],
            }

        def complete_messages(self, messages, system_prompt=None):
            return {"message": "fallback chat should not be used"}

    stub = StubChatTool()
    agent.chat_tool = stub
    agent.skill_manager.attach_chat_tool(stub)

    response = agent.handle("clear the background info window")

    assert response["message"] == "已清空背景信息窗口。"
    assert response["display_content"] == ""
    assert response["used_tools"] == ["clear_display_context"]


def test_mcp_discovery_can_hot_register_local_module(tmp_path: Path):
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    module_path = project_root / "mcp_demo.py"
    module_path.write_text(
        """
class DummyMCP:
    def tool(self):
        def decorator(func):
            return func
        return decorator

mcp = DummyMCP()

@mcp.tool()
def get_status(name: str = 'demo'):
    return f"status:{name}"
""".strip(),
        encoding="utf-8",
    )

    registry = ToolRegistry()
    discovery = MCPDiscoveryService(str(project_root))

    result = discovery.refresh(registry)
    tools = registry.snapshot()
    tool_names = [item["name"] for item in tools]

    assert result["registered"] == 1
    assert "mcp_mcp_demo_get_status" in tool_names
    execution = registry.execute("mcp_mcp_demo_get_status", {"name": "ok"})
    assert execution["message"] == "status:ok"
