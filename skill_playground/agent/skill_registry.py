from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SkillRollout:
    env: Tuple[str, ...] = ("all",)
    traffic_percent: int = 100

    def allows_env(self, environment: str) -> bool:
        values = {item.strip().lower() for item in self.env if str(item).strip()}
        if not values or "all" in values:
            return True
        return environment.strip().lower() in values


@dataclass(frozen=True)
class SkillRoutingCard:
    summary: str = ""
    when_to_use: Tuple[str, ...] = ()
    when_not_to_use: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()
    input_signals: Tuple[str, ...] = ()
    output_role: str = ""

    def to_prompt_payload(self) -> Dict[str, object]:
        return {
            "summary": self.summary,
            "when_to_use": list(self.when_to_use),
            "when_not_to_use": list(self.when_not_to_use),
            "examples": list(self.examples),
            "input_signals": list(self.input_signals),
            "output_role": self.output_role,
        }


@dataclass(frozen=True)
class LoadedSkill:
    skill_id: str
    name: str
    display_name: str
    description: str
    version: str
    enabled: bool
    type: str
    subtype: str
    auto_apply: bool
    priority: int
    scope: Tuple[str, ...]
    surfaces: Tuple[str, ...]
    triggers: Tuple[str, ...]
    patterns: Tuple[str, ...]
    depends_on: Tuple[str, ...]
    conflicts_with: Tuple[str, ...]
    prompt: str
    path: str
    threshold: float
    thresholds: Mapping[str, float]
    cost_level: str
    risk_level: str
    rollout: SkillRollout
    is_default: bool
    default_for_surfaces: Tuple[str, ...]
    default_for_phases: Tuple[str, ...]
    routing_card: SkillRoutingCard

    def supports_surface(self, surface: Optional[str]) -> bool:
        if not surface:
            return True
        values = {item.strip().lower() for item in self.surfaces if item.strip()}
        return not values or "all" in values or surface.strip().lower() in values

    def supports_phase(self, phase: Optional[str]) -> bool:
        if not phase:
            return True
        values = {item.strip().lower() for item in self.scope if item.strip()}
        return not values or "all" in values or phase.strip().lower() in values

    def is_prompt_skill(self) -> bool:
        return self.type in {"prompt", "hybrid"}

    def is_handler_skill(self) -> bool:
        return self.type in {"handler", "tool", "hybrid"}

    def is_default_for(self, surface: Optional[str], phase: Optional[str]) -> bool:
        if not self.is_default:
            return False
        surface_values = {item.strip().lower() for item in self.default_for_surfaces if item.strip()}
        phase_values = {item.strip().lower() for item in self.default_for_phases if item.strip()}
        surface_ok = not surface_values or "all" in surface_values or not surface or surface.strip().lower() in surface_values
        phase_ok = not phase_values or "all" in phase_values or not phase or phase.strip().lower() in phase_values
        return surface_ok and phase_ok


class SkillRegistry:
    def __init__(self, project_root: str) -> None:
        self.project_root = Path(project_root)
        self.skills_dir = self.project_root / "skills"
        self._skills: Dict[str, LoadedSkill] = {}
        self.reload()

    def reload(self) -> None:
        skills: Dict[str, LoadedSkill] = {}
        if self.skills_dir.exists():
            for skill_dir in sorted(path for path in self.skills_dir.iterdir() if path.is_dir()):
                loaded = self._load_skill(skill_dir)
                if loaded is not None:
                    skills[loaded.skill_id] = loaded
        self._skills = skills

    def list_skills(
        self,
        surface: Optional[str] = None,
        phase: Optional[str] = None,
        environment: Optional[str] = None,
        rollout_key: Optional[str] = None,
    ) -> List[LoadedSkill]:
        return sorted(
            [
                skill
                for skill in self._skills.values()
                if self._is_visible(skill, surface=surface, phase=phase, environment=environment, rollout_key=rollout_key)
            ],
            key=lambda item: (-item.priority, item.skill_id),
        )

    def list_auto_apply_skills(
        self,
        surface: Optional[str] = None,
        phase: Optional[str] = None,
        environment: Optional[str] = None,
        rollout_key: Optional[str] = None,
    ) -> List[LoadedSkill]:
        return [
            skill
            for skill in self.list_skills(surface=surface, phase=phase, environment=environment, rollout_key=rollout_key)
            if skill.auto_apply
        ]

    def get_skill(self, skill_id: str) -> Optional[LoadedSkill]:
        return self._skills.get((skill_id or "").strip())

    def get_default_skill(
        self,
        surface: Optional[str] = None,
        phase: Optional[str] = "generation",
        environment: Optional[str] = None,
        rollout_key: Optional[str] = None,
    ) -> Optional[LoadedSkill]:
        visible = self.list_skills(surface=surface, phase=phase, environment=environment, rollout_key=rollout_key)
        explicit = [skill for skill in visible if skill.is_default_for(surface, phase)]
        if explicit:
            return explicit[0]
        auto = [skill for skill in visible if skill.auto_apply]
        return auto[0] if auto else None

    def _load_skill(self, skill_dir: Path) -> Optional[LoadedSkill]:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return None

        metadata, prompt = self._split_frontmatter(skill_file.read_text(encoding="utf-8"))
        config = self._load_json(skill_dir / "skill.json")

        skill_id = str(config.get("skill_id") or config.get("id") or skill_dir.name).strip() or skill_dir.name
        display_name = str(config.get("display_name") or config.get("name") or metadata.get("name") or skill_id).strip() or skill_id
        description = str(config.get("description") or metadata.get("description") or "").strip()
        version = str(config.get("version") or "1.0.0").strip() or "1.0.0"
        skill_type = str(config.get("type") or config.get("mount_mode") or "prompt").strip().lower() or "prompt"
        subtype = str(config.get("subtype") or "generation").strip().lower() or "generation"
        scope = tuple(self._parse_list(config.get("scope") or config.get("phases") or [subtype]))
        surfaces = tuple(self._parse_list(config.get("surfaces") or config.get("applies_to") or ["all"]))
        triggers = tuple(self._parse_list(config.get("triggers") or []))
        patterns = tuple(self._parse_list(config.get("patterns") or []))
        depends_on = tuple(self._parse_list(config.get("depends_on") or config.get("dependencies") or []))
        conflicts_with = tuple(self._parse_list(config.get("conflicts_with") or config.get("conflicts") or []))
        routing_card = self._load_routing_card(skill_dir, config, description, triggers)

        return LoadedSkill(
            skill_id=skill_id,
            name=skill_id,
            display_name=display_name,
            description=description,
            version=version,
            enabled=self._parse_bool(config.get("enabled"), default=True),
            type=skill_type,
            subtype=subtype,
            auto_apply=self._parse_bool(config.get("auto_apply"), default=False),
            priority=self._parse_int(config.get("priority"), default=50),
            scope=scope,
            surfaces=surfaces,
            triggers=triggers,
            patterns=patterns,
            depends_on=depends_on,
            conflicts_with=conflicts_with,
            prompt=prompt.strip(),
            path=str(skill_dir),
            threshold=self._parse_float(config.get("threshold"), default=0.0),
            thresholds=self._parse_thresholds(config.get("thresholds")),
            cost_level=str(config.get("cost_level") or "low").strip().lower() or "low",
            risk_level=str(config.get("risk_level") or "low").strip().lower() or "low",
            rollout=self._parse_rollout(config.get("rollout")),
            is_default=self._parse_bool(config.get("is_default"), default=False),
            default_for_surfaces=tuple(self._parse_list(config.get("default_for_surfaces") or [])),
            default_for_phases=tuple(self._parse_list(config.get("default_for_phases") or [])),
            routing_card=routing_card,
        )

    def _load_routing_card(
        self,
        skill_dir: Path,
        config: Mapping[str, object],
        description: str,
        triggers: Tuple[str, ...],
    ) -> SkillRoutingCard:
        raw = self._load_json(skill_dir / "routing_card.json")
        if not raw:
            raw = {
                "summary": description,
                "when_to_use": [description] if description else [],
                "when_not_to_use": [],
                "examples": [],
                "input_signals": list(triggers[:8]),
                "output_role": str(config.get("subtype") or "").strip(),
            }
        return SkillRoutingCard(
            summary=str(raw.get("summary") or description or "").strip(),
            when_to_use=tuple(self._parse_list(raw.get("when_to_use") or [])),
            when_not_to_use=tuple(self._parse_list(raw.get("when_not_to_use") or [])),
            examples=tuple(self._parse_list(raw.get("examples") or [])),
            input_signals=tuple(self._parse_list(raw.get("input_signals") or raw.get("signals") or [])),
            output_role=str(raw.get("output_role") or config.get("subtype") or "").strip(),
        )

    def _load_json(self, path: Path) -> Dict[str, object]:
        if not path.exists():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _split_frontmatter(self, text: str) -> Tuple[Dict[str, str], str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text
        metadata: Dict[str, str] = {}
        body_start = 0
        for idx in range(1, len(lines)):
            line = lines[idx]
            if line.strip() == "---":
                body_start = idx + 1
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata, "\n".join(lines[body_start:])

    def _is_visible(
        self,
        skill: LoadedSkill,
        surface: Optional[str],
        phase: Optional[str],
        environment: Optional[str],
        rollout_key: Optional[str],
    ) -> bool:
        if not skill.enabled:
            return False
        if not skill.supports_surface(surface) or not skill.supports_phase(phase):
            return False
        if environment and not skill.rollout.allows_env(environment):
            return False
        if environment and rollout_key and not self._passes_rollout(skill, rollout_key):
            return False
        return True

    def _passes_rollout(self, skill: LoadedSkill, rollout_key: str) -> bool:
        traffic = max(0, min(100, int(skill.rollout.traffic_percent)))
        if traffic >= 100:
            return True
        if traffic <= 0:
            return False
        digest = hashlib.sha1(f"{skill.skill_id}|{rollout_key}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < traffic

    def _parse_list(self, raw_value: object) -> List[str]:
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        value = str(raw_value or "").strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return [item.strip() for item in value.split(",") if item.strip()]

    def _parse_bool(self, raw_value: object, default: bool = False) -> bool:
        if raw_value is None:
            return default
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def _parse_int(self, raw_value: object, default: int) -> int:
        try:
            return int(raw_value)
        except Exception:
            return default

    def _parse_float(self, raw_value: object, default: float) -> float:
        try:
            return float(raw_value)
        except Exception:
            return default

    def _parse_rollout(self, raw_value: object) -> SkillRollout:
        if not isinstance(raw_value, dict):
            return SkillRollout()
        env = tuple(self._parse_list(raw_value.get("env") or ["all"])) or ("all",)
        traffic_percent = max(0, min(100, self._parse_int(raw_value.get("traffic_percent"), default=100)))
        return SkillRollout(env=env, traffic_percent=traffic_percent)

    def _parse_thresholds(self, raw_value: object) -> Mapping[str, float]:
        if not isinstance(raw_value, dict):
            return {}
        parsed: Dict[str, float] = {}
        for key, value in raw_value.items():
            try:
                parsed[str(key).strip().lower()] = float(value)
            except Exception:
                continue
        return parsed
