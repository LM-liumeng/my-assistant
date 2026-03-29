"""Structured tool registration and execution for the assistant.

本模块是整个 Agent 系统的工具核心注册中心（Tool Registry）。
它负责统一管理所有工具的元数据（ToolSpec）、执行函数（handler）以及可用性检查，
并为 Function Calling / Tool Use 提供标准化的接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# 类型别名定义
ToolHandler = Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]]
"""
工具执行函数的类型签名。
参数：
    arguments: 用户/LLM 传入的工具参数（dict）
    context:   可选的上下文信息（如 conversation_id、user_id、safety_context 等）
返回值：必须是 dict，包含执行结果
"""

AvailabilityHandler = Callable[[], Dict[str, Any]]
"""
工具可用性检查函数的类型签名。
用于运行时动态判断工具是否可用（例如：依赖服务是否在线、权限是否足够、MCP 模块是否可加载等）
返回值示例：{"available": True, "status": "available", "detail": "..." }
"""


@dataclass(frozen=True)
class ToolSpec:
    """工具的静态元数据描述（Tool Specification）。

    这是注册一个工具时必须提供的最核心信息，类似于 OpenAI/Anthropic 的 tool 定义，
    但增加了更多生产级特性（来源、分类、权限、依赖、前置条件等）。
    """

    tool_id: str                          # 全局唯一标识符（推荐格式：source.category.name）
    name: str                             # 工具名称（供 LLM 在 function calling 中使用，必须唯一）
    description: str                      # 详细描述，LLM 会看到此内容，质量直接影响调用准确率
    parameters_schema: Dict[str, Any]     # JSON Schema，定义工具接受的参数结构（OpenAI 兼容）

    # 可选字段（生产环境常用）
    source: str = "local"                 # 工具来源：local、mcp、skill、plugin 等
    category: str = "general"             # 分类：search、file、email、media、rag 等，便于管理
    requires_confirmation: bool = False   # 是否需要用户二次确认（高风险操作如发送邮件、删除文件等）
    enabled: bool = True                  # 是否启用，可动态开关
    tags: Tuple[str, ...] = ()            # 标签，用于搜索、过滤、权限控制
    prerequisites: Tuple[str, ...] = ()   # 前置依赖工具（暂未使用，可扩展为自动检查）
    capability_summary: str = ""          # 简短能力摘要，用于列表展示或 capability 页面
    legacy_intent: Optional[str] = None   # 兼容旧版意图识别系统（可逐步废弃）
    hidden: bool = False                  # 是否在列表中隐藏（内部工具、调试工具等）

    def to_openai_tool(self) -> Dict[str, Any]:
        """将 ToolSpec 转换为 OpenAI / Anthropic 兼容的 tool 定义格式。

        在 Function Calling 时会把 list_specs() 得到的结果通过此方法转为 tools 参数。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass(frozen=True)
class ToolStatus:
    """工具当前的运行时状态。"""
    available: bool                       # 是否可用
    status: str = "available"             # 状态码：available / unavailable / disabled / error 等
    detail: str = ""                      # 详细说明（出错原因、不可用原因等）

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典，常用于 API 返回或前端展示。"""
        return {
            "available": self.available,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ToolRegistration:
    """工具的完整注册记录（内部使用），将元数据、执行函数和可用性检查绑定在一起。"""
    spec: ToolSpec                        # 工具静态描述
    handler: ToolHandler                  # 实际执行函数
    availability: Optional[AvailabilityHandler] = None   # 可选的运行时可用性检查函数


class ToolRegistry:
    """Assistant 的工具注册中心（核心单例或全局实例）。

    职责：
    1. 统一注册、管理所有工具（Skill、MCP、传统 Tool 等）
    2. 提供按名称、ID、来源、状态的查询能力
    3. 为 LLM Function Calling 动态生成 tools 列表
    4. 负责工具的执行与异常处理
    5. 支持运行时热注册、注销、状态检查（配合 MCPDiscoveryService 等使用）
    """

    def __init__(self) -> None:
        """初始化空的工具注册表。"""
        # 主存储：tool_id → ToolRegistration
        self._registrations_by_id: Dict[str, ToolRegistration] = {}
        # 辅助索引：name → tool_id（支持通过 name 查找）
        self._tool_id_by_name: Dict[str, str] = {}

    def register(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        availability: Optional[AvailabilityHandler] = None,
    ) -> None:
        """注册一个工具（核心注册方法）。

        如果 tool_id 或 name 已存在，会覆盖原有注册（热更新场景常用）。
        """
        registration = ToolRegistration(spec=spec, handler=handler, availability=availability)
        self._registrations_by_id[spec.tool_id] = registration
        self._tool_id_by_name[spec.name] = spec.tool_id

    def unregister(self, tool_ref: str) -> None:
        """通过 tool_id 或 name 注销单个工具。"""
        registration = self.get(tool_ref)
        if registration is None:
            return
        self._tool_id_by_name.pop(registration.spec.name, None)
        self._registrations_by_id.pop(registration.spec.tool_id, None)

    def unregister_source(self, source: str) -> None:
        """批量注销某个来源的所有工具（例如：刷新 MCP 工具时先清空旧的）。"""
        for spec in list(self.list_specs(source=source, include_hidden=True, enabled_only=False)):
            self.unregister(spec.tool_id)

    def get(self, tool_ref: str) -> Optional[ToolRegistration]:
        """通过 tool_id 或 name 获取完整的 ToolRegistration。"""
        ref = (tool_ref or "").strip()
        if not ref:
            return None
        # 先按 tool_id 查找
        if ref in self._registrations_by_id:
            return self._registrations_by_id[ref]
        # 再按 name 查找
        tool_id = self._tool_id_by_name.get(ref)
        if tool_id:
            return self._registrations_by_id.get(tool_id)
        return None

    def get_spec(self, tool_ref: str) -> Optional[ToolSpec]:
        """便捷方法：只获取 ToolSpec（不需要 handler 时使用）。"""
        registration = self.get(tool_ref)
        return registration.spec if registration else None

    def find_by_legacy_intent(self, legacy_intent: str) -> Optional[ToolSpec]:
        """兼容旧版意图识别系统，根据 legacy_intent 查找工具。"""
        normalized = (legacy_intent or "").strip()
        if not normalized:
            return None
        for spec in self.list_specs(include_hidden=True, enabled_only=False):
            if spec.legacy_intent == normalized:
                return spec
        return None

    def list_specs(
        self,
        *,
        source: Optional[str] = None,
        include_hidden: bool = False,
        enabled_only: bool = True,
    ) -> List[ToolSpec]:
        """列出符合条件的工具规格列表（支持过滤）。"""
        items: List[ToolSpec] = []
        for registration in self._registrations_by_id.values():
            spec = registration.spec
            if source and spec.source != source:
                continue
            if enabled_only and not spec.enabled:
                continue
            if not include_hidden and spec.hidden:
                continue
            items.append(spec)
        # 按分类和名称排序，便于展示
        return sorted(items, key=lambda item: (item.category, item.name, item.tool_id))

    def list_openai_tools(self) -> List[Dict[str, Any]]:
        """返回当前可用的、适合传入 LLM 的 OpenAI 格式 tools 列表。

        这是在 Function Calling 流程中最常调用的方法。
        只会返回 available = True 的工具。
        """
        tools: List[Dict[str, Any]] = []
        for spec in self.list_specs():
            status = self.status_for(spec.tool_id)
            if not status.available:
                continue
            tools.append(spec.to_openai_tool())
        return tools

    def status_for(self, tool_ref: str) -> ToolStatus:
        """获取某个工具当前的运行时状态（支持动态 availability 检查）。"""
        registration = self.get(tool_ref)
        if registration is None:
            return ToolStatus(available=False, status="missing", detail="Tool is not registered.")

        spec = registration.spec
        if not spec.enabled:
            return ToolStatus(available=False, status="disabled", detail="Tool is disabled.")

        # 如果没有提供 availability 检查函数，则默认可用
        if registration.availability is None:
            return ToolStatus(available=True, status="available", detail="")

        # 执行动态可用性检查
        try:
            payload = dict(registration.availability() or {})
        except Exception as exc:
            return ToolStatus(available=False, status="error", detail=str(exc))

        available = bool(payload.get("available", True))
        status = str(payload.get("status") or ("available" if available else "unavailable")).strip() or "available"
        detail = str(payload.get("detail") or "").strip()

        return ToolStatus(available=available, status=status, detail=detail)

    def snapshot(self) -> List[Dict[str, Any]]:
        """生成当前所有工具的完整快照，常用于管理后台、调试或 API 返回。"""
        rows: List[Dict[str, Any]] = []
        for spec in self.list_specs():
            status = self.status_for(spec.tool_id)
            rows.append(
                {
                    "tool_id": spec.tool_id,
                    "name": spec.name,
                    "description": spec.description,
                    "category": spec.category,
                    "source": spec.source,
                    "requires_confirmation": spec.requires_confirmation,
                    "capability_summary": spec.capability_summary or spec.description,
                    "tags": list(spec.tags),
                    "prerequisites": list(spec.prerequisites),
                    "legacy_intent": spec.legacy_intent,
                    "status": status.to_dict(),
                }
            )
        return rows

    def execute(
        self,
        tool_ref: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行指定工具（统一入口，包含状态检查和异常捕获）。"""
        registration = self.get(tool_ref)
        if registration is None:
            return {"error": f"Unknown tool: {tool_ref}"}

        # 先检查工具是否可用
        status = self.status_for(registration.spec.tool_id)
        if not status.available:
            message = status.detail or f"Tool '{registration.spec.name}' is unavailable."
            return {"error": message}

        # 执行工具
        try:
            result = registration.handler(dict(arguments or {}), context)
        except Exception as exc:
            return {"error": f"Tool '{registration.spec.name}' execution failed: {exc}"}

        # 统一返回格式
        if isinstance(result, dict):
            return result
        return {"message": str(result)}

    def execute_many(
        self,
        calls: Iterable[Tuple[str, Dict[str, Any]]],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """批量执行多个工具调用（支持并行工具调用场景）。"""
        return [self.execute(tool_ref, arguments, context=context)
                for tool_ref, arguments in calls]