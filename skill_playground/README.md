# Skill Playground

This folder is a minimal isolated playground for testing skill routing and skill-guided chat output.

Included:
- `agent/skill_registry.py`
- `agent/hybrid_skill_router.py`
- `agent/semantic_skill_judge.py`
- `agent/skill_policy_engine.py`
- `agent/skill_compiler.py`
- `agent/skill_manager.py`
- `agent/orchestrator.py`
- `tools/chat_tool.py`
- `skills/` (copied local skills)
- a minimal Flask frontend and backend

Not included:
- file tools
- document tools
- email tools
- RAG
- media analysis
- MCP runtime
- main app routing logic

Run:

```bash
python -m skill_playground.app
```

Then open [http://127.0.0.1:5051](http://127.0.0.1:5051).
