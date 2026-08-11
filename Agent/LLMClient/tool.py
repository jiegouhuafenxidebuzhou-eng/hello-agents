import json
from typing import Any, Callable, Dict, List, Optional


class Tool:
    """工具基类。子类需实现 name / description / parameters / run。

    parameters 需返回一个 JSON Schema 字典，用于 OpenAI function calling。
    """

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def description(self) -> str:
        raise NotImplementedError

    @property
    def parameters(self) -> Dict[str, Any]:
        # 默认无参数
        return {"type": "object", "properties": {}}

    def run(self, **kwargs) -> str:
        """执行工具，返回字符串结果。"""
        raise NotImplementedError

    def spec(self) -> Dict[str, Any]:
        """生成 OpenAI function calling 所需的工具描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"


class FunctionTool(Tool):
    """把一个普通可调用对象快速包装成工具。"""

    def __init__(self, name: str, description: str, func: Callable, parameters: Dict[str, Any] = None):
        self._name = name
        self._description = description
        self._func = func
        self._parameters = parameters or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    def run(self, **kwargs) -> str:
        try:
            return str(self._func(**kwargs))
        except TypeError:
            # 工具函数不接受关键字参数时，退化为按位置取第一个参数
            return str(self._func(*kwargs.values()))


class ListToolsTool(Tool):
    """内置工具：让 LLM 自主查询当前注册了哪些工具及其用法。"""

    def __init__(self, registry: "ToolRegistry"):
        self._registry = registry

    @property
    def name(self) -> str:
        return "list_tools"

    @property
    def description(self) -> str:
        return "查询当前可用的所有工具列表，返回每个工具的名称、描述和参数说明。当你不确定有哪些工具可用时调用它。"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def run(self, **kwargs) -> str:
        return self._registry.describe_all()


class ToolRegistry:
    """工具注册中心：注册、查询、执行，并输出 OpenAI 工具描述。"""

    def __init__(self, with_builtin: bool = True):
        self._tools: Dict[str, Tool] = {}
        if with_builtin:
            self.register(ListToolsTool(self))

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            print(f"⚠️ 工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def specs(self) -> List[Dict[str, Any]]:
        """生成传给 LLM 的工具描述列表。"""
        return [t.spec() for t in self._tools.values()]

    def describe_all(self) -> str:
        """供 list_tools 工具返回的可读描述。"""
        if not self._tools:
            return "当前没有可用工具。"
        lines = []
        for t in self._tools.values():
            params = json.dumps(t.parameters, ensure_ascii=False)
            lines.append(f"- {t.name}: {t.description}\n  参数: {params}")
        return "\n".join(lines)

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """按名称执行工具，参数为解析后的字典。"""
        tool = self.get(name)
        if not tool:
            return f"错误：未找到名为 '{name}' 的工具。"
        try:
            return tool.run(**(arguments or {}))
        except Exception as e:
            return f"工具 '{name}' 执行出错: {e}"
