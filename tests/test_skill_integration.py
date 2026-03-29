from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.hybrid_skill_router import HybridSkillRouter
from agent.orchestrator import Agent, IntentRecognizer
from agent.semantic_skill_judge import SemanticSkillJudge, SemanticSkillJudgment
from agent.skill_manager import SkillManager
from agent.skill_policy_engine import SkillPolicyEngine
from agent.skill_registry import SkillRegistry


def _write_skill(project_root: Path, skill_id: str, description: str, body: str, config: dict) -> None:
    skill_dir = project_root / "skills" / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    (skill_dir / "skill.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    routing_card = {
        "summary": description,
        "when_to_use": [description] if description else [],
        "when_not_to_use": [],
        "examples": [],
        "input_signals": list(config.get("triggers") or []),
        "output_role": str(config.get("subtype") or "generation"),
    }
    (skill_dir / "routing_card.json").write_text(json.dumps(routing_card, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_prompt_skill_config(name: str, description: str, priority: int, *, auto_apply: bool = False, scope=None, surfaces=None, triggers=None, patterns=None, depends_on=None, conflicts_with=None, threshold=None):
    config = {
        "skill_id": name,
        "name": name.replace("-", " ").title(),
        "version": "1.0.0",
        "enabled": True,
        "type": "prompt",
        "subtype": "generation",
        "description": description,
        "auto_apply": auto_apply,
        "priority": priority,
        "scope": scope or ["generation"],
        "surfaces": surfaces or ["chat", "rag"],
        "triggers": triggers or [],
        "patterns": patterns or [],
        "input_schema": {},
        "output_schema": {},
        "depends_on": depends_on or [],
        "conflicts_with": conflicts_with or [],
        "fail_policy": "fail_open",
        "timeout_ms": 1000,
        "owner": "agent-team",
        "tags": ["test"],
        "cost_level": "low",
        "risk_level": "low",
        "rollout": {"env": ["all"], "traffic_percent": 100},
    }
    if threshold is not None:
        config["threshold"] = threshold
    return config


def _build_multi_skill_project(project_root: Path) -> None:
    _write_skill(
        project_root,
        "comfortable-response",
        "Keep responses natural and readable.",
        "Use this skill as a response-style overlay only.\n\nCore rules:\n1. Avoid decorative symbols and markdown dividers.\n2. Keep the answer natural and readable.",
        {
            "skill_id": "comfortable-response",
            "name": "Comfortable Response",
            "version": "1.0.0",
            "enabled": True,
            "type": "prompt",
            "subtype": "style",
            "description": "Keep responses natural and readable.",
            "auto_apply": True,
            "priority": 40,
            "scope": ["generation", "postprocess"],
            "surfaces": ["chat", "capability", "media", "rag"],
            "triggers": [],
            "patterns": [],
            "input_schema": {},
            "output_schema": {},
            "depends_on": [],
            "conflicts_with": [],
            "fail_policy": "fail_open",
            "timeout_ms": 1000,
            "owner": "agent-team",
            "tags": ["style", "clean-output"],
            "cost_level": "low",
            "risk_level": "low",
            "rollout": {"env": ["all"], "traffic_percent": 100},
        },
    )
    _write_skill(
        project_root,
        "summarize-document",
        "Prefer concise summaries when the user asks for summary-style output.",
        "When this skill applies, compress the answer into a short summary first.",
        _base_prompt_skill_config(
            "summarize-document",
            "Prefer concise summaries when the user asks for summary-style output.",
            60,
            triggers=["summary", "abstract", "overview"],
            threshold=0.75,
        ),
    )
    _write_skill(
        project_root,
        "technical-report-writer",
        "Generate formal technical reports.",
        "When this skill applies, structure the answer around conclusion, tradeoffs, and next steps.",
        _base_prompt_skill_config(
            "technical-report-writer",
            "Generate formal technical reports.",
            85,
            triggers=["report", "plan", "feasibility"],
            depends_on=["comfortable-response"],
            conflicts_with=["casual-style"],
            threshold=0.75,
        ),
    )
    _write_skill(
        project_root,
        "casual-style",
        "Use a casual tone for low-stakes chat.",
        "When this skill applies, keep the tone casual and playful.",
        _base_prompt_skill_config(
            "casual-style",
            "Use a casual tone for low-stakes chat.",
            20,
            triggers=["report"],
            conflicts_with=["technical-report-writer"],
            threshold=0.70,
        ),
    )


def test_skill_registry_loads_extended_metadata(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    skill = registry.get_skill("technical-report-writer")

    assert skill is not None
    assert skill.skill_id == "technical-report-writer"
    assert skill.name == "Technical Report Writer"
    assert skill.display_name == "Technical Report Writer"
    assert skill.version == "1.0.0"
    assert skill.type == "prompt"
    assert skill.subtype == "generation"
    assert skill.scope == ("generation",)
    assert skill.surfaces == ("chat", "rag")
    assert skill.depends_on == ("comfortable-response",)
    assert skill.conflicts_with == ("casual-style",)


def test_skill_registry_filters_disabled_skills(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)
    _write_skill(
        project_root,
        "disabled-skill",
        "Disabled skill should stay hidden.",
        "Do not use this skill.",
        {
            "skill_id": "disabled-skill",
            "name": "Disabled Skill",
            "version": "1.0.0",
            "enabled": False,
            "type": "prompt",
            "subtype": "generation",
            "description": "Disabled skill should stay hidden.",
            "auto_apply": True,
            "priority": 99,
            "scope": ["generation"],
            "surfaces": ["chat"],
            "triggers": ["disabled"],
            "patterns": [],
            "input_schema": {},
            "output_schema": {},
            "depends_on": [],
            "conflicts_with": [],
            "fail_policy": "fail_open",
            "timeout_ms": 1000,
            "owner": "agent-team",
            "tags": ["test"],
            "cost_level": "low",
            "risk_level": "low",
            "rollout": {"env": ["all"], "traffic_percent": 100},
        },
    )

    registry = SkillRegistry(str(project_root))
    visible_ids = [skill.skill_id for skill in registry.list_skills(surface="chat", phase="generation")]

    assert "disabled-skill" not in visible_ids
    assert registry.get_skill("disabled-skill") is not None


def test_skill_registry_can_filter_by_environment(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)
    _write_skill(
        project_root,
        "prod-only-skill",
        "Only visible in prod.",
        "Prod only skill.",
        {
            "skill_id": "prod-only-skill",
            "name": "Prod Only Skill",
            "version": "1.0.0",
            "enabled": True,
            "type": "prompt",
            "subtype": "generation",
            "description": "Only visible in prod.",
            "auto_apply": False,
            "priority": 70,
            "scope": ["generation"],
            "surfaces": ["chat"],
            "triggers": ["prod"],
            "patterns": [],
            "input_schema": {},
            "output_schema": {},
            "depends_on": [],
            "conflicts_with": [],
            "fail_policy": "fail_open",
            "timeout_ms": 1000,
            "owner": "agent-team",
            "tags": ["test"],
            "cost_level": "low",
            "risk_level": "low",
            "rollout": {"env": ["prod"], "traffic_percent": 100},
        },
    )

    registry = SkillRegistry(str(project_root))
    dev_ids = [skill.skill_id for skill in registry.list_skills(surface="chat", phase="generation", environment="dev")]
    prod_ids = [skill.skill_id for skill in registry.list_skills(surface="chat", phase="generation", environment="prod")]

    assert "prod-only-skill" not in dev_ids
    assert "prod-only-skill" in prod_ids


def test_skill_registry_prefers_explicit_default_skill(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)
    _write_skill(
        project_root,
        "default-summary",
        "Explicit default summary skill.",
        "Use this as the explicit default.",
        {
            "skill_id": "default-summary",
            "name": "Default Summary",
            "version": "1.0.0",
            "enabled": True,
            "type": "prompt",
            "subtype": "generation",
            "description": "Explicit default summary skill.",
            "auto_apply": False,
            "priority": 55,
            "scope": ["generation"],
            "surfaces": ["chat"],
            "triggers": [],
            "patterns": [],
            "input_schema": {},
            "output_schema": {},
            "depends_on": [],
            "conflicts_with": [],
            "fail_policy": "fail_open",
            "timeout_ms": 1000,
            "owner": "agent-team",
            "tags": ["test"],
            "cost_level": "low",
            "risk_level": "low",
            "rollout": {"env": ["all"], "traffic_percent": 100},
            "is_default": True,
            "default_for_surfaces": ["chat"],
            "default_for_phases": ["generation"],
        },
    )

    registry = SkillRegistry(str(project_root))
    default_skill = registry.get_default_skill(surface="chat", phase="generation")

    assert default_skill is not None
    assert default_skill.skill_id == "default-summary"


def test_hybrid_skill_router_returns_highest_scoring_skill(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)

    skill_name, score = router.route("Please create a feasibility report summary", surface="chat", phase="generation")

    assert skill_name == "technical-report-writer"
    assert score >= 0.80


def test_hybrid_skill_router_route_all_preserves_multiple_candidates(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)

    matches = router.route_all("Please create a feasibility report summary", surface="chat", phase="generation")

    assert [item.skill_id for item in matches[:3]] == [
        "technical-report-writer",
        "summarize-document",
        "casual-style",
    ]


def test_hybrid_skill_router_deduplicates_evidence_and_keeps_all_sources(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)

    matches = router.route_all("Please create a feasibility report summary", surface="chat", phase="generation")
    top = matches[0]

    assert top.source.startswith("trigger:")
    assert any(source.startswith("trigger:") for source in top.sources)
    assert len(top.sources) >= 1
    assert len(top.matched_terms) == len({term.lower() for term in top.matched_terms})


def test_hybrid_skill_router_avoids_short_ascii_substring_false_positive(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)
    _write_skill(
        project_root,
        "short-trigger-skill",
        "Skill with too-short ascii trigger.",
        "Do not overmatch short ascii substrings.",
        _base_prompt_skill_config(
            "short-trigger-skill",
            "Skill with too-short ascii trigger.",
            75,
            triggers=["an"],
            threshold=0.75,
        ),
    )

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)

    matches = router.route_all("analysis workflow", surface="chat", phase="generation")

    assert "short-trigger-skill" not in [item.skill_id for item in matches]


def test_hybrid_skill_router_pattern_strength_varies_by_match_shape(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)
    _write_skill(
        project_root,
        "pattern-heavy-skill",
        "Skill driven by regex patterns.",
        "Pattern-heavy skill.",
        _base_prompt_skill_config(
            "pattern-heavy-skill",
            "Skill driven by regex patterns.",
            70,
            triggers=[],
            patterns=[r"^feasibility report summary$"],
            threshold=0.75,
        ),
    )

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)

    exact = router.route_all("feasibility report summary", surface="chat", phase="generation")
    partial = router.route_all("please draft a feasibility report summary now", surface="chat", phase="generation")

    exact_skill = next(item for item in exact if item.skill_id == "pattern-heavy-skill")
    partial_skill = next(item for item in partial if item.skill_id == "technical-report-writer")

    assert exact_skill.base_score >= 0.97
    assert partial_skill.base_score < exact_skill.base_score


def test_skill_registry_loads_routing_card_metadata(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    skill = registry.get_skill("technical-report-writer")

    assert skill is not None
    assert skill.routing_card.summary == "Generate formal technical reports."
    assert "report" in skill.routing_card.input_signals


class _SemanticJudgeChatTool:
    def __init__(self, payload: dict):
        self.payload = payload

    def is_configured(self):
        return True

    def complete(self, user_prompt: str, system_prompt=None):
        return {"message": json.dumps(self.payload, ensure_ascii=False)}


def test_semantic_skill_judge_selects_only_recalled_candidates(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)
    matches = router.route_all("Please create a feasibility report summary", surface="chat", phase="generation")

    judge = SemanticSkillJudge(
        registry,
        chat_tool=_SemanticJudgeChatTool({
            "primary_skill": "technical-report-writer",
            "secondary_skills": ["summarize-document"],
            "rejected_skills": ["casual-style", "not-in-candidates"],
            "confidence": 0.91,
            "intent_summary": "The user wants a formal feasibility-style summary."
        }),
    )

    judgment = judge.judge("Please create a feasibility report summary", matches, surface="chat", phase="generation")

    assert judgment.primary_skill == "technical-report-writer"
    assert judgment.secondary_skills == ("summarize-document",)
    assert "casual-style" in judgment.rejected_skills
    assert "not-in-candidates" not in judgment.rejected_skills
    assert judgment.source == "llm"


def test_semantic_skill_judgment_to_dict_uses_contract_field_names():
    judgment = SemanticSkillJudge(
        registry=None,  # type: ignore[arg-type]
        chat_tool=None,
    )
    result = judgment._heuristic_fallback(
        "summarize this report",
        [
            type("Match", (), {"skill_id": "summarize-document", "score": 0.9})(),
        ],
    ).to_dict()

    assert "primary_skill" in result
    assert "secondary_skills" in result
    assert "rejected_skills" in result
    assert "primary_candidate" not in result


def test_skill_manager_limits_semantic_candidates_to_allowed_types_and_top_k(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)
    _write_skill(
        project_root,
        "report-executor",
        "Execute report workflows as a handler skill.",
        "Run the report workflow.",
        {
            **_base_prompt_skill_config(
                "report-executor",
                "Execute report workflows as a handler skill.",
                95,
                triggers=["report"],
                threshold=0.75,
            ),
            "type": "handler",
            "subtype": "execution",
        },
    )

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)

    class RecordingSemanticJudge:
        def __init__(self):
            self.top_k = 2
            self.candidate_ids = []

        def judge(self, user_query, candidates, surface=None, phase=None):
            self.candidate_ids = [item.skill_id for item in candidates]
            return SemanticSkillJudgment(
                primary_skill="technical-report-writer",
                secondary_skills=("summarize-document",),
                rejected_skills=(),
                confidence=0.88,
                intent_summary="The user wants a report-style summary.",
                source="test",
            )

    semantic = RecordingSemanticJudge()
    manager = SkillManager(registry, router, semantic_judge=semantic, threshold=0.75, max_dynamic_skills=4)

    plan = manager.resolve_plan("Please create a feasibility report summary", scope="chat", phase="generation")

    assert semantic.candidate_ids == ["technical-report-writer", "summarize-document"]
    assert "report-executor" not in plan.selected_ids()
    semantic_rejections = [item.skill_id for item in plan.decisions if item.source == "semantic"]
    assert "casual-style" not in semantic_rejections


def test_skill_manager_uses_semantic_judgment_before_policy_engine(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)
    semantic = SemanticSkillJudge(
        registry,
        chat_tool=_SemanticJudgeChatTool({
            "primary_skill": "summarize-document",
            "secondary_skills": ["technical-report-writer"],
            "rejected_skills": ["casual-style"],
            "confidence": 0.87,
            "intent_summary": "The user primarily wants a summary with report support."
        }),
    )
    manager = SkillManager(registry, router, semantic_judge=semantic, threshold=0.75, max_dynamic_skills=4)

    plan = manager.resolve_plan("Please create a feasibility report summary", scope="chat", phase="generation")

    assert plan.semantic_judgment is not None
    assert plan.primary is not None
    assert plan.primary.skill.skill_id == "summarize-document"
    assert plan.selected_ids() == ("summarize-document", "technical-report-writer", "comfortable-response")
    rejected_ids = [item.skill_id for item in plan.decisions if item.source == "semantic"]
    assert "casual-style" in rejected_ids


def test_policy_engine_does_not_promote_style_skill_to_primary(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)
    semantic = SemanticSkillJudge(
        registry,
        chat_tool=_SemanticJudgeChatTool({
            "primary_skill": "comfortable-response",
            "secondary_skills": ["technical-report-writer", "summarize-document"],
            "rejected_skills": ["casual-style"],
            "confidence": 0.86,
            "intent_summary": "The user wants a clean report-like answer."
        }),
    )
    manager = SkillManager(registry, router, semantic_judge=semantic, threshold=0.75, max_dynamic_skills=4)

    plan = manager.resolve_plan("Please create a feasibility report summary", scope="chat", phase="generation")

    assert plan.primary is not None
    assert plan.primary.skill.skill_id == "technical-report-writer"
    assert plan.selected_ids() == ("technical-report-writer", "summarize-document", "comfortable-response")


def test_policy_engine_assigns_primary_and_modifiers_and_resolves_conflicts(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    router = HybridSkillRouter(registry)
    engine = SkillPolicyEngine(registry, router, threshold=0.75, max_dynamic_skills=4, environment="dev")

    plan = engine.build_plan(
        user_query="Please create a feasibility report summary",
        surface="chat",
        phase="generation",
        allowed_types=("prompt", "hybrid"),
    )

    assert plan.primary is not None
    assert plan.primary.skill.skill_id == "technical-report-writer"
    assert plan.selected_ids() == ("technical-report-writer", "summarize-document", "comfortable-response")
    assert "casual-style" not in plan.selected_ids()


def test_policy_engine_filters_by_phase(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    engine = SkillPolicyEngine(registry, HybridSkillRouter(registry), threshold=0.75, max_dynamic_skills=3, environment="dev")

    plan = engine.build_plan(
        user_query="Please create a feasibility report summary",
        surface="chat",
        phase="postprocess",
        allowed_types=("prompt", "hybrid"),
    )

    assert plan.primary is None
    assert plan.selected_ids() == ("comfortable-response",)


def test_skill_manager_resolves_generation_skills_in_policy_order(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    manager = SkillManager(registry, HybridSkillRouter(registry), threshold=0.75, max_dynamic_skills=4)

    resolved = manager.resolve_skills("Please create a feasibility report summary", scope="chat")
    ids = [item.skill.skill_id for item in resolved]

    assert ids == ["technical-report-writer", "summarize-document", "comfortable-response"]
    assert resolved[0].role == "primary"


def test_agent_chat_prompt_uses_compiled_skill_policy(tmp_path: Path):
    project_root = tmp_path / "project"
    base_dir = project_root / "data"
    base_dir.mkdir(parents=True, exist_ok=True)
    _build_multi_skill_project(project_root)

    agent = Agent(str(base_dir), skill_threshold=0.75, max_dynamic_skills=4)

    class StubChatTool:
        def __init__(self):
            self.system_prompt = ""

        def is_configured(self):
            return True

        def complete_messages(self, messages, system_prompt=None):
            self.system_prompt = system_prompt or ""
            return {"message": "This is a smoother answer."}

    stub = StubChatTool()
    agent.chat_tool = stub

    response = agent.handle("Please create a feasibility report summary")

    assert response["message"] == "This is a smoother answer."
    assert response["active_skills"] == ["technical-report-writer", "summarize-document", "comfortable-response"]
    assert "Compiled skill policy for surface=chat, phase=generation." in stub.system_prompt
    assert "Primary skill:" in stub.system_prompt
    assert "Modifier skills:" in stub.system_prompt


def test_skill_postprocess_cleans_markdown_noise_when_postprocess_skill_is_present(tmp_path: Path):
    project_root = tmp_path / "project"
    _build_multi_skill_project(project_root)

    registry = SkillRegistry(str(project_root))
    manager = SkillManager(registry, HybridSkillRouter(registry))

    cleaned = manager.postprocess_response("### Title\n\n---\n\n**Content**", scope="chat", user_query="hello")
    assert cleaned == "Title\n\nContent"


def test_intent_does_not_route_large_model_research_question_to_sentiment():
    recognizer = IntentRecognizer(intent_tool=None)
    query = "What are the research hotspots for large models in 2026?"
    intent, params = recognizer.recognise(query)
    assert intent == "chat"
    assert params["prompt"] == query

