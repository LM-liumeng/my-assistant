"""Discover MCP-style tools and register them into the local tool registry.

本模块实现了对本地 MCP（Model Context Protocol 风格）工具的自动发现与注册。
支持在不重启服务的情况下动态加载项目中标记为 MCP 工具的函数。
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tool_registry import ToolRegistry, ToolSpec


@dataclass(frozen=True)
class MCPDiscoveredTool:
    """表示一个从本地文件中发现的 MCP 工具的元数据。"""
    file_path: str                    # 工具所在文件的完整路径
    callable_name: str                # 函数/方法名称
    description: str                  # 从函数文档字符串中提取的第一行描述
    parameters_schema: Dict[str, Any] # 根据函数参数自动生成的 JSON Schema（简化版）


class MCPDiscoveryService:
    """MCP 风格工具的自动发现与注册服务（Best-effort 实现）。

    核心功能：
    1. 扫描项目中符合命名规则的 Python 文件（mcp_tool.py 或文件名包含 mcp 的文件）
    2. 通过 AST 解析找出被 @tool 装饰器标记的函数
    3. 自动提取描述和参数 schema
    4. 将发现的工具动态注册到 ToolRegistry 中，支持热更新（无需重启）
    """

    def __init__(self, project_root: str) -> None:
        """初始化发现服务。

        Args:
            project_root: 项目根目录路径，用于扫描 MCP 工具文件
        """
        self.project_root = Path(project_root)

    def refresh(self, registry: ToolRegistry) -> Dict[str, Any]:
        """刷新 MCP 工具：重新扫描、注销旧 MCP 工具，并重新注册所有发现的工具。

        该方法设计为可重复调用，支持运行时动态更新工具。

        Returns:
            包含发现数量、注册成功数量及错误的统计信息
        """
        discovered = self.discover()
        # 先清除之前通过 "mcp" 来源注册的所有工具，避免重复或过期工具残留
        registry.unregister_source("mcp")

        registered = 0
        errors: List[str] = []

        for item in discovered:
            try:
                registry.register(
                    spec=ToolSpec(
                        tool_id=f"mcp.{Path(item.file_path).stem}.{item.callable_name}",
                        name=f"mcp_{Path(item.file_path).stem}_{item.callable_name}",
                        description=item.description or f"MCP tool {item.callable_name}",
                        parameters_schema=item.parameters_schema,
                        source="mcp",                    # 标记来源，便于后续统一管理
                        category="plugin",
                        tags=("mcp", Path(item.file_path).stem, item.callable_name),
                        capability_summary=item.description
                                         or f"调用 MCP 工具 {item.callable_name}",
                    ),
                    # 为每个发现的工具动态构建一个可执行的 handler
                    handler=self._build_handler(item.file_path, item.callable_name),
                    # 提供运行时可用性检查函数（懒加载检查）
                    availability=lambda file_path=item.file_path: self._module_status(file_path),
                )
                registered += 1
            except Exception as exc:
                errors.append(f"{item.callable_name}: {exc}")

        return {
            "message": f"Discovered {len(discovered)} MCP tool(s); registered {registered}.",
            "discovered": len(discovered),
            "registered": registered,
            "errors": errors,
        }

    def discover(self) -> List[MCPDiscoveredTool]:
        """扫描项目并发现所有符合条件的 MCP 工具。"""
        items: List[MCPDiscoveredTool] = []
        for path in self._candidate_files():
            items.extend(self._discover_in_file(path))
        return items

    def _candidate_files(self) -> List[Path]:
        """返回所有可能包含 MCP 工具的候选文件。

        目前支持两种命名模式：
        - 精确文件名：mcp_tool.py
        - 模糊匹配：文件名中包含 "mcp" 的任意 .py 文件
        """
        candidates: List[Path] = []
        for pattern in ("mcp_tool.py", "*mcp*.py"):
            for path in self.project_root.glob(pattern):
                if path.is_file() and path.suffix == ".py" and path not in candidates:
                    candidates.append(path)
        return sorted(candidates)

    def _discover_in_file(self, path: Path) -> List[MCPDiscoveredTool]:
        """对单个 Python 文件进行 AST 解析，找出所有被 @tool 装饰的函数。"""
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        items: List[MCPDiscoveredTool] = []

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not self._is_mcp_tool(node):
                continue

            # 提取函数文档字符串的第一行作为描述
            description = ""
            docstring = ast.get_docstring(node)
            if docstring:
                description = docstring.strip().splitlines()[0].strip()

            items.append(
                MCPDiscoveredTool(
                    file_path=str(path),
                    callable_name=node.name,
                    description=description,
                    parameters_schema=self._schema_from_ast(node),
                )
            )
        return items

    def _is_mcp_tool(self, node: ast.AST) -> bool:
        """判断一个函数是否被标记为 MCP Tool（通过检查是否有 @tool 装饰器）。"""
        decorators = getattr(node, "decorator_list", [])
        for decorator in decorators:
            # 处理 @tool 和 @tool(...) 两种写法
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                return True
        return False

    def _schema_from_ast(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, Any]:
        """通过 AST 简单推断函数参数，生成一个基础的 JSON Schema。

        注意：当前实现较为简化，仅支持位置参数 + 类型推断为 string。
        未来可扩展为使用 inspect + 类型注解生成更精确的 schema。
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        defaults = len(node.args.defaults)
        positional = node.args.args
        required_count = max(0, len(positional) - defaults)

        for index, arg in enumerate(positional):
            if arg.arg in {"self", "cls"}:
                continue

            properties[arg.arg] = {"type": "string"}   # 简化处理，默认为 string
            if index < required_count:
                required.append(arg.arg)

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        return schema

    def _build_handler(self, file_path: str, callable_name: str):
        """为发现的 MCP 工具动态构建一个可被 ToolRegistry 调用的 handler。

        该 handler 负责：
        1. 动态加载模块
        2. 执行对应的函数（支持同步和异步）
        3. 处理参数类型转换
        4. 统一返回格式
        """
        def _handler(arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            # 动态加载模块
            module = self._load_module(Path(file_path))
            func = getattr(module, callable_name, None)
            if func is None:
                return {"error": f"MCP callable '{callable_name}' is not available."}

            # 参数类型强制转换（根据函数签名）
            kwargs = self._coerce_arguments(func, dict(arguments or {}))

            # 执行函数（支持 async）
            if inspect.iscoroutinefunction(func):
                result = asyncio.run(func(**kwargs))
            else:
                result = func(**kwargs)

            # 统一返回格式
            if isinstance(result, dict):
                return result

            text = str(result)
            return {
                "message": text,
                "display_content": text,
                "display_filename": f"{callable_name}.txt",
            }

        return _handler

    def _module_status(self, file_path: str) -> Dict[str, Any]:
        """检查某个 MCP 模块当前是否可用（运行时可用性检查）。"""
        try:
            self._load_module(Path(file_path))
        except Exception as exc:
            return {
                "available": False,
                "status": "unavailable",
                "detail": f"MCP module is discoverable but not executable: {exc}",
            }
        return {
            "available": True,
            "status": "available",
            "detail": "Discovered from local MCP module.",
        }

    def _load_module(self, path: Path):
        """动态加载 Python 模块（支持热重载）。"""
        module_name = f"dynamic_mcp_{path.stem}"
        if module_name in sys.modules:
            return sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load MCP module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _coerce_arguments(self, func: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """根据函数签名对传入参数进行简单类型转换。

        当前支持 int、float、bool 的转换，其余保持原样。
        """
        try:
            signature = inspect.signature(func)
        except Exception:
            return arguments

        coerced: Dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if name not in arguments:
                continue

            raw_value = arguments[name]
            annotation = parameter.annotation

            if annotation in {int, "int"}:
                coerced[name] = int(raw_value)
            elif annotation in {float, "float"}:
                coerced[name] = float(raw_value)
            elif annotation in {bool, "bool"}:
                if isinstance(raw_value, bool):
                    coerced[name] = raw_value
                else:
                    coerced[name] = str(raw_value).strip().lower() in {"1", "true", "yes", "on"}
            else:
                coerced[name] = raw_value

        return coerced