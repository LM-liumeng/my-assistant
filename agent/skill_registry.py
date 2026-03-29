"""Project-local skill registry with production-oriented metadata.

项目本地技能注册表（Skill Registry）
这是整个 Agent 技能系统的核心“技能资产管理中心”。
它负责从项目目录 `skills/` 下以声明式方式（declarative）加载所有技能，
每个技能以文件夹形式存在，包含 SKILL.md、skill.json、routing_card.json 等文件。
支持生产级特性：灰度发布（rollout）、依赖/冲突管理、路由卡片（routing card）、多环境、多表面（surface）、多阶段（phase）等。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SkillRollout:
    """技能灰度发布配置（生产环境必备特性）。

    支持按环境（dev/prod/staging 等）和流量百分比进行灰度控制。
    """

    env: Tuple[str, ...] = ("all",)           # 允许的环境列表，"all" 表示所有环境
    traffic_percent: int = 100                # 灰度流量百分比（0-100）

    def allows_env(self, environment: str) -> bool:
        """判断当前环境是否允许该技能运行。"""
        envs = {item.strip().lower() for item in self.env if str(item).strip()}
        if not envs or "all" in envs:
            return True
        return environment.strip().lower() in envs


@dataclass(frozen=True)
class SkillRoutingCard:
    """技能路由卡片（Routing Card）—— 为 LLM 提供结构化的“什么时候用这个技能”信息。

    这是生产级 Agent 中非常重要的“可解释性”与“提示工程”特性，
    可被 SkillPolicyEngine 或 CapabilityRegistry 转换为 prompt 片段。
    """

    summary: str = ""                         # 技能一句话总结
    when_to_use: Tuple[str, ...] = ()         # 什么情况下应该使用
    when_not_to_use: Tuple[str, ...] = ()     # 什么情况下不应该使用
    examples: Tuple[str, ...] = ()            # 使用示例
    input_signals: Tuple[str, ...] = ()       # 输入信号/触发词
    output_role: str = ""                     # 输出角色（primary / modifier 等）

    def to_prompt_payload(self) -> Dict[str, object]:
        """转换为可直接塞入 Prompt 的结构化字典。"""
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
    """已加载的技能完整元数据（生产级技能描述对象）。

    所有字段均来自 skill.json + SKILL.md + routing_card.json 的解析结果。
    """

    skill_id: str
    name: str
    display_name: str
    version: str
    enabled: bool
    type: str                              # prompt / handler / tool / hybrid
    subtype: str                           # generation / style / postprocess 等
    description: str
    auto_apply: bool                       # 是否在对应 surface/phase 自动应用
    priority: int                          # 优先级（越高越优先）
    scope: Tuple[str, ...]                 # 支持的阶段（phases）
    surfaces: Tuple[str, ...]              # 支持的表面（chat / desktop / mobile 等）
    triggers: Tuple[str, ...]              # 触发词
    patterns: Tuple[str, ...]              # 正则匹配模式
    input_schema: Mapping[str, Any]        # 输入参数 JSON Schema
    output_schema: Mapping[str, Any]       # 输出参数 JSON Schema
    depends_on: Tuple[str, ...]            # 依赖的其他技能
    conflicts_with: Tuple[str, ...]        # 互斥的技能
    fail_policy: str                       # 失败策略（fail_open / fail_fast）
    timeout_ms: int
    owner: str
    tags: Tuple[str, ...]
    cost_level: str                        # low / medium / high
    risk_level: str                        # low / medium / high
    rollout: SkillRollout                  # 灰度配置
    prompt: str                            # 技能实际的 Prompt 模板（来自 SKILL.md）
    path: str                              # 技能所在文件夹路径
    threshold: float                       # 全局匹配阈值
    thresholds: Mapping[str, float]        # 按 phase 细分的阈值
    is_default: bool                       # 是否为默认技能
    default_for_surfaces: Tuple[str, ...]
    default_for_phases: Tuple[str, ...]
    routing_card: SkillRoutingCard         # 路由卡片（给 LLM 的决策依据）

    def supports_surface(self, surface: Optional[str]) -> bool:
        """判断技能是否支持指定的 surface（表面）。"""
        if not surface:
            return True
        values = {item.strip().lower() for item in self.surfaces if item.strip()}
        return not values or "all" in values or surface.strip().lower() in values

    def supports_phase(self, phase: Optional[str]) -> bool:
        """判断技能是否支持指定的 phase（阶段）。"""
        if not phase:
            return True
        values = {item.strip().lower() for item in self.scope if item.strip()}
        return not values or "all" in values or phase.strip().lower() in values

    def is_prompt_skill(self) -> bool:
        """是否为 Prompt 类技能（可作为 primary）。"""
        return self.type in {"prompt", "hybrid"}

    def is_handler_skill(self) -> bool:
        """是否为 Handler/Tool 类技能（可被 Function Calling 调用）。"""
        return self.type in {"handler", "tool", "hybrid"}

    def is_default_for(self, surface: Optional[str], phase: Optional[str]) -> bool:
        """判断当前技能是否是指定 surface + phase 的默认技能。"""
        if not self.is_default:
            return False
        surface_values = {item.strip().lower() for item in self.default_for_surfaces if item.strip()}
        phase_values = {item.strip().lower() for item in self.default_for_phases if item.strip()}
        surface_ok = not surface_values or "all" in surface_values or not surface or surface.strip().lower() in surface_values
        phase_ok = not phase_values or "all" in phase_values or not phase or phase.strip().lower() in phase_values
        return surface_ok and phase_ok


class SkillRegistry:
    """项目本地技能注册表（Skill Registry）—— 技能系统的单点数据源。

    设计目标：
    - 声明式加载：技能即文件夹，无需修改代码即可新增技能
    - 生产级特性：支持灰度、依赖、冲突、路由卡片、动态重载
    - 高性能：所有技能一次性加载到内存，支持快速查询
    """

    def __init__(self, project_root: str) -> None:
        """初始化技能注册表并立即加载所有技能。"""
        self.project_root = Path(project_root)
        self.skills_dir = self.project_root / "skills"   # 技能存放目录：project_root/skills/
        self._skills: Dict[str, LoadedSkill] = {}
        self.reload()

    def reload(self) -> None:
        """重新加载所有技能（支持热更新）。"""
        skills: Dict[str, LoadedSkill] = {}
        if not self.skills_dir.exists():
            self._skills = skills
            return

        for skill_dir in sorted(path for path in self.skills_dir.iterdir() if path.is_dir()):
            skill = self._load_skill(skill_dir)
            if skill is not None:
                skills[skill.skill_id] = skill

        self._skills = skills

    def list_skills(
        self,
        surface: Optional[str] = None,
        phase: Optional[str] = None,
        environment: Optional[str] = None,
        rollout_key: Optional[str] = None,
    ) -> List[LoadedSkill]:
        """列出符合当前上下文的所有可见技能（按优先级排序）。"""
        rows = [
            skill
            for skill in self._skills.values()
            if self._is_visible(skill, surface=surface, phase=phase, environment=environment, rollout_key=rollout_key)
        ]
        return sorted(rows, key=lambda item: (-item.priority, item.skill_id))

    def list_auto_apply_skills(
        self,
        surface: Optional[str] = None,
        phase: Optional[str] = None,
        environment: Optional[str] = None,
        rollout_key: Optional[str] = None,
    ) -> List[LoadedSkill]:
        """列出当前上下文下所有需要自动应用的技能。"""
        rows = [
            skill
            for skill in self.list_skills(surface=surface, phase=phase, environment=environment, rollout_key=rollout_key)
            if skill.auto_apply
        ]
        return sorted(rows, key=lambda item: (-item.priority, item.skill_id))

    def get_skill(self, skill_id: str) -> Optional[LoadedSkill]:
        """通过 skill_id 获取单个技能。"""
        return self._skills.get((skill_id or "").strip())

    def get_default_skill(
        self,
        surface: Optional[str] = None,
        phase: Optional[str] = "generation",
        environment: Optional[str] = None,
        rollout_key: Optional[str] = None,
    ) -> Optional[LoadedSkill]:
        """获取当前上下文下的默认技能（优先 explicit default → auto_apply）。"""
        visible = self.list_skills(surface=surface, phase=phase, environment=environment, rollout_key=rollout_key)
        explicit_defaults = [skill for skill in visible if skill.is_default_for(surface, phase)]
        if explicit_defaults:
            return sorted(explicit_defaults, key=lambda item: (-item.priority, item.skill_id))[0]

        auto = [skill for skill in visible if skill.auto_apply]
        if auto:
            return sorted(auto, key=lambda item: (-item.priority, item.skill_id))[0]
        return None

    # ====================== 内部加载逻辑 ======================

    def _load_skill(self, skill_dir: Path) -> Optional[LoadedSkill]:
        """从单个技能文件夹加载完整 LoadedSkill 对象。"""
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return None

        text = skill_file.read_text(encoding="utf-8")
        metadata, body = self._split_frontmatter(text)   # 解析 YAML frontmatter
        config = self._load_config(skill_dir)            # 读取 skill.json
        prompt = body.strip()

        # 兼容旧版 prompt.md 文件
        if not prompt:
            legacy_prompt = skill_dir / "prompt.md"
            if legacy_prompt.exists():
                prompt = legacy_prompt.read_text(encoding="utf-8").strip()

        # ==================== 解析各项配置 ====================
        skill_id = str(config.get("skill_id") or config.get("id") or skill_dir.name).strip() or skill_dir.name
        display_name = str(config.get("name") or metadata.get("name") or skill_id).strip() or skill_id
        description = str(config.get("description") or metadata.get("description") or "").strip()
        version = str(config.get("version") or "1.0.0").strip() or "1.0.0"
        enabled = self._parse_bool(config.get("enabled"), default=True)
        skill_type = str(config.get("type") or config.get("mount_mode") or "prompt").strip().lower() or "prompt"
        subtype = str(config.get("subtype") or "generation").strip().lower() or "generation"
        scope = self._parse_list(config.get("scope") or config.get("phases") or [subtype])
        surfaces = self._parse_list(config.get("surfaces") or config.get("applies_to") or ["all"])
        auto_apply = self._parse_bool(config.get("auto_apply"), default=False)
        priority = self._parse_int(config.get("priority"), default=50)
        triggers = self._parse_list(config.get("triggers") or [])
        patterns = self._parse_list(config.get("patterns") or [])
        input_schema = self._parse_mapping(config.get("input_schema"))
        output_schema = self._parse_mapping(config.get("output_schema"))
        depends_on = self._parse_list(config.get("depends_on") or config.get("dependencies") or [])
        conflicts_with = self._parse_list(config.get("conflicts_with") or config.get("conflicts") or [])
        fail_policy = str(config.get("fail_policy") or "fail_open").strip().lower() or "fail_open"
        timeout_ms = self._parse_int(config.get("timeout_ms"), default=1000)
        owner = str(config.get("owner") or "").strip()
        tags = self._parse_list(config.get("tags") or [])
        cost_level = str(config.get("cost_level") or "low").strip().lower() or "low"
        risk_level = str(config.get("risk_level") or "low").strip().lower() or "low"
        rollout = self._parse_rollout(config.get("rollout"))
        threshold = self._parse_float(config.get("threshold"), default=0.0)
        thresholds = self._parse_thresholds(config.get("thresholds"))
        is_default = self._parse_bool(config.get("is_default"), default=False)
        default_for_surfaces = self._parse_list(config.get("default_for_surfaces") or [])
        default_for_phases = self._parse_list(config.get("default_for_phases") or [])
        routing_card = self._load_routing_card(skill_dir, config, metadata, description, triggers, patterns, tags)

        return LoadedSkill(
            skill_id=skill_id,
            name=display_name,
            display_name=display_name,
            version=version,
            enabled=enabled,
            type=skill_type,
            subtype=subtype,
            description=description,
            auto_apply=auto_apply,
            priority=priority,
            scope=tuple(scope),
            surfaces=tuple(surfaces),
            triggers=tuple(triggers),
            patterns=tuple(patterns),
            input_schema=input_schema,
            output_schema=output_schema,
            depends_on=tuple(depends_on),
            conflicts_with=tuple(conflicts_with),
            fail_policy=fail_policy,
            timeout_ms=timeout_ms,
            owner=owner,
            tags=tuple(tags),
            cost_level=cost_level,
            risk_level=risk_level,
            rollout=rollout,
            prompt=prompt,
            path=str(skill_dir),
            threshold=threshold,
            thresholds=thresholds,
            is_default=is_default,
            default_for_surfaces=tuple(default_for_surfaces),
            default_for_phases=tuple(default_for_phases),
            routing_card=routing_card,
        )

    def _load_config(self, skill_dir: Path) -> Dict[str, object]:
        """读取 skill.json 配置文件（如果存在）。"""
        config_path = skill_dir / "skill.json"
        if not config_path.exists():
            return {}
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_routing_card(
        self,
        skill_dir: Path,
        config: Mapping[str, object],
        metadata: Mapping[str, str],
        description: str,
        triggers: List[str],
        patterns: List[str],
        tags: List[str],
    ) -> SkillRoutingCard:
        """加载 routing_card.json，若不存在则自动生成 fallback 版本。"""
        card_path = skill_dir / "routing_card.json"
        raw_card: Mapping[str, object] = {}

        if card_path.exists():
            try:
                loaded = json.loads(card_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw_card = loaded
            except Exception:
                raw_card = {}

        if not raw_card:
            raw_card = self._build_fallback_routing_card(
                config=config,
                metadata=metadata,
                description=description,
                triggers=triggers,
                patterns=patterns,
                tags=tags,
            )

        return SkillRoutingCard(
            summary=str(raw_card.get("summary") or description or "").strip(),
            when_to_use=tuple(self._parse_list(raw_card.get("when_to_use") or [])),
            when_not_to_use=tuple(self._parse_list(raw_card.get("when_not_to_use") or [])),
            examples=tuple(self._parse_list(raw_card.get("examples") or [])),
            input_signals=tuple(self._parse_list(raw_card.get("input_signals") or raw_card.get("signals") or [])),
            output_role=str(raw_card.get("output_role") or config.get("subtype") or "").strip(),
        )

    def _build_fallback_routing_card(
        self,
        config: Mapping[str, object],
        metadata: Mapping[str, str],
        description: str,
        triggers: List[str],
        patterns: List[str],
        tags: List[str],
    ) -> Mapping[str, object]:
        """当 routing_card.json 不存在时，自动构造一个合理的 fallback 卡片。"""
        summary = str(description or metadata.get("description") or config.get("description") or "").strip()
        when_to_use = []
        if summary:
            when_to_use.append(summary)
        if triggers:
            when_to_use.append("Lexical triggers: " + ", ".join(triggers[:6]))

        input_signals = triggers[:8]
        if not input_signals and tags:
            input_signals = tags[:6]

        return {
            "summary": summary,
            "when_to_use": when_to_use,
            "when_not_to_use": [],
            "examples": [],
            "input_signals": input_signals,
            "output_role": str(config.get("subtype") or "").strip(),
        }

    def _split_frontmatter(self, text: str) -> Tuple[Dict[str, str], str]:
        """解析 SKILL.md 文件的 YAML frontmatter（--- 开头）。"""
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
        body = "\n".join(lines[body_start:])
        return metadata, body

    # ====================== 解析工具方法 ======================

    def _parse_list(self, raw_value: object) -> List[str]:
        """统一解析列表类型配置（支持 JSON 数组或逗号分隔字符串）。"""
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        value = str(raw_value or "").strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return [item.strip() for item in value.split(",") if item.strip()]

    def _parse_bool(self, raw_value: object, default: bool = False) -> bool:
        """解析布尔值配置。"""
        if raw_value is None:
            return default
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def _parse_int(self, raw_value: object, default: int) -> int:
        """解析整数配置。"""
        try:
            return int(raw_value)
        except Exception:
            return default

    def _parse_float(self, raw_value: object, default: float) -> float:
        """解析浮点数配置。"""
        try:
            return float(raw_value)
        except Exception:
            return default

    def _parse_mapping(self, raw_value: object) -> Mapping[str, Any]:
        """解析字典类型配置。"""
        if isinstance(raw_value, dict):
            return {str(key): value for key, value in raw_value.items()}
        return {}

    def _parse_rollout(self, raw_value: object) -> SkillRollout:
        """解析灰度发布配置。"""
        if not isinstance(raw_value, dict):
            return SkillRollout()
        env = tuple(self._parse_list(raw_value.get("env") or ["all"])) or ("all",)
        traffic_percent = self._parse_int(raw_value.get("traffic_percent"), default=100)
        traffic_percent = max(0, min(100, traffic_percent))
        return SkillRollout(env=env, traffic_percent=traffic_percent)

    def _is_visible(
        self,
        skill: LoadedSkill,
        surface: Optional[str] = None,
        phase: Optional[str] = None,
        environment: Optional[str] = None,
        rollout_key: Optional[str] = None,
    ) -> bool:
        """判断技能在当前上下文下是否可见（核心过滤逻辑）。"""
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
        """针对特定 rollout_key 的灰度分桶检查（确定性哈希）。"""
        traffic = max(0, min(100, int(skill.rollout.traffic_percent)))
        if traffic >= 100:
            return True
        if traffic <= 0:
            return False
        import hashlib
        digest = hashlib.sha1(f"{skill.skill_id}|{rollout_key}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < traffic

    def _parse_thresholds(self, raw_value: object) -> Mapping[str, float]:
        """解析按阶段细分的阈值配置。"""
        if not isinstance(raw_value, dict):
            return {}
        parsed: Dict[str, float] = {}
        for key, value in raw_value.items():
            try:
                parsed[str(key).strip().lower()] = float(value)
            except Exception:
                continue
        return parsed