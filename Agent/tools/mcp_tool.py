"""MCP 工具：把 MCP 服务器作为 Tool 暴露给 LLM 自主调用。

设计参照 hello_agents protocol_tools 的 MCPTool，适配到本项目
Agent.LLMClient.tool.Tool 基类（name/description/parameters 属性 +
run(**kwargs)，OpenAI function calling JSON Schema 风格），并复用
本目录 Agent.MCP.client.MCPClient。

核心能力：
- 连接 MCP 服务器（stdio/http/sse/内存），列出并调用其工具、资源、提示词
- 三级环境变量优先级：env > env_keys > 自动检测
- auto_expand：把服务器里每个工具展开成独立 Tool，挂到 Agent 上
- 无任何参数时创建内置演示服务器（加减乘除/greet/get_system_info）

用法：
    from Agent.MCP import MCPTool

    # 1. 内置演示服务器
    tool = MCPTool()
    agent.add_tool(tool)

    # 2. 外部 MCP 服务器
    tool = MCPTool(name="github",
                   server_command=["npx", "-y", "@modelcontextprotocol/server-github"],
                   env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"})
    agent.add_tool(tool)
"""

import asyncio
import concurrent.futures
import os
from typing import Any, Dict, List, Optional

from Agent.LLMClient.tool import Tool
from Agent.MCP.client import MCPClient

# 常见 MCP 服务器需要的环境变量映射表，用于自动检测
MCP_SERVER_ENV_MAP = {
    "server-github": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    "server-slack": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
    "server-google-drive": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
    "server-postgres": ["POSTGRES_CONNECTION_STRING"],
    "server-sqlite": [],
    "server-filesystem": [],
}


class MCPTool(Tool):
    """MCP (Model Context Protocol) 工具。

    连接 MCP 服务器，调用其工具/资源/提示词。auto_expand=True 时，
    服务器里的每个工具会被展开成独立 Tool（见 get_expanded_tools），
    便于 Agent 像调用普通工具一样直接调用，而不必走 action 分发。
    """

    def __init__(
        self,
        name: str = "mcp",
        description: Optional[str] = None,
        server_command: Optional[List[str]] = None,
        server_args: Optional[List[str]] = None,
        server: Optional[Any] = None,
        auto_expand: bool = True,
        env: Optional[Dict[str, str]] = None,
        env_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            name: 工具名称，建议为不同服务器指定不同名称避免冲突（默认 "mcp"）。
            description: 工具描述，None 时按发现结果自动生成。
            server_command: 服务器启动命令，如 ["python", "server.py"] 或
                ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]。
            server_args: 服务器额外参数列表。
            server: FastMCP 服务器实例，用于内存传输（测试）。
            auto_expand: 是否自动展开为独立工具（默认 True）。
            env: 直接传递的环境变量字典（优先级最高）。
            env_keys: 从系统环境变量加载的 key 列表（优先级中等）。

        环境变量优先级（从高到低）：
            1. env 参数
            2. env_keys 指定的系统环境变量
            3. 根据 server_command 自动检测（见 MCP_SERVER_ENV_MAP）

        若 server_command 和 server 都为空，则创建内置演示服务器。
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self.server = server
        self._client_cls = MCPClient
        self._available_tools: List[Dict[str, Any]] = []
        self.auto_expand = auto_expand
        self.prefix = f"{name}_" if auto_expand else ""

        # 环境变量处理
        self.env = self._prepare_env(env, env_keys, server_command)

        # 没有指定任何服务器 → 创建内置演示服务器
        if not server_command and not server:
            self.server = self._create_builtin_server()

        # 自动发现工具（失败不影响初始化）
        self._discover_tools()

        # 名称 / 描述
        self._name = name
        self._description = description or self._generate_description()

    # ---- 属性 ----
    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        """统一 action 式调用的参数定义（auto_expand=False 时主要使用）。

        auto_expand=True 时一般用 get_expanded_tools() 拆成独立工具，
        各自有精确参数；此 parameters 仅作兜底。
        """
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_tools", "call_tool", "list_resources",
                             "read_resource", "list_prompts", "get_prompt"],
                    "description": "操作类型；未指定但有 tool_name 时自动推断为 call_tool",
                },
                "tool_name": {"type": "string", "description": "工具名称（call_tool 需要）"},
                "arguments": {"type": "object", "description": "工具参数（call_tool 需要）"},
                "uri": {"type": "string", "description": "资源 URI（read_resource 需要）"},
                "prompt_name": {"type": "string", "description": "提示词名称（get_prompt 需要）"},
                "prompt_arguments": {"type": "object", "description": "提示词参数（get_prompt 可选）"},
            },
            "required": ["action"],
        }

    # ---- 环境变量 ----
    def _prepare_env(
        self,
        env: Optional[Dict[str, str]],
        env_keys: Optional[List[str]],
        server_command: Optional[List[str]],
    ) -> Dict[str, str]:
        """合并环境变量，优先级：env > env_keys > 自动检测。"""
        result: Dict[str, str] = {}

        # 1. 自动检测（优先级最低）
        if server_command:
            server_name = None
            for part in server_command:
                if "server-" in part:
                    server_name = part.split("/")[-1] if "/" in part else part
                    break
            if server_name and server_name in MCP_SERVER_ENV_MAP:
                for key in MCP_SERVER_ENV_MAP[server_name]:
                    value = os.getenv(key)
                    if value:
                        result[key] = value
                        print(f"🔑 自动加载环境变量: {key}")

        # 2. env_keys（优先级中等）
        if env_keys:
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    result[key] = value
                    print(f"🔑 从 env_keys 加载环境变量: {key}")
                else:
                    print(f"⚠️  环境变量 {key} 未设置")

        # 3. 直接传递的 env（优先级最高）
        if env:
            result.update(env)
            for key in env.keys():
                print(f"🔑 使用直接传递的环境变量: {key}")

        return result

    # ---- 内置演示服务器 ----
    def _create_builtin_server(self):
        """创建内置演示服务器（无外部依赖时使用）。"""
        try:
            from fastmcp import FastMCP
        except ImportError as e:
            raise ImportError("创建内置 MCP 服务器需要 fastmcp 库: pip install fastmcp") from e

        server = FastMCP("HelloAgents-BuiltinServer")

        @server.tool()
        def add(a: float, b: float) -> float:
            """加法计算器"""
            return a + b

        @server.tool()
        def subtract(a: float, b: float) -> float:
            """减法计算器"""
            return a - b

        @server.tool()
        def multiply(a: float, b: float) -> float:
            """乘法计算器"""
            return a * b

        @server.tool()
        def divide(a: float, b: float) -> float:
            """除法计算器"""
            if b == 0:
                raise ValueError("除数不能为零")
            return a / b

        @server.tool()
        def greet(name: str = "World") -> str:
            """友好问候"""
            return f"Hello, {name}! 欢迎使用 MCP 工具！"

        @server.tool()
        def get_system_info() -> dict:
            """获取系统信息"""
            import platform
            import sys
            return {
                "platform": platform.system(),
                "python_version": sys.version,
                "server_name": "HelloAgents-BuiltinServer",
                "tools_count": 6,
            }

        return server

    # ---- 异步辅助 ----
    def _run_async(self, coro_factory):
        """同步运行异步协程，兼容“已有事件循环”的情况。

        coro_factory 是个无参 callable，返回新协程（避免协程跨线程复用）。
        """
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if not running:
            return asyncio.run(coro_factory())

        # 已有循环（如 Jupyter）→ 在新线程里跑新循环
        def _in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro_factory())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(_in_thread).result()

    def _client_source(self):
        """选择客户端连接源：内置服务器用 server，否则用命令。"""
        return self.server if self.server else self.server_command

    # ---- 工具发现 ----
    def _discover_tools(self):
        """发现 MCP 服务器提供的所有工具。"""

        async def discover():
            async with MCPClient(self._client_source(), self.server_args, env=self.env) as client:
                return await client.list_tools()

        try:
            self._available_tools = self._run_async(discover)
        except Exception as e:
            print(f"⚠️ MCP 工具发现失败: {e}")
            self._available_tools = []

    def _generate_description(self) -> str:
        """按发现结果生成工具描述。"""
        if not self._available_tools:
            return "连接到 MCP 服务器，调用工具、读取资源和获取提示词。支持内置服务器和外部服务器。"

        if self.auto_expand:
            return (f"MCP工具服务器，包含 {len(self._available_tools)} 个工具，"
                    "会自动展开为独立工具供 Agent 使用。")

        lines = [f"MCP工具服务器，提供 {len(self._available_tools)} 个工具："]
        for tool in self._available_tools:
            tool_name = tool.get("name", "unknown")
            tool_desc = tool.get("description", "无描述") or "无描述"
            short_desc = tool_desc.split(".")[0] if tool_desc else "无描述"
            lines.append(f"  • {tool_name}: {short_desc}")
        lines.append("\n调用格式：通过 action 指定操作，如 "
                     '{"action": "call_tool", "tool_name": "工具名", "arguments": {...}}')
        return "\n".join(lines)

    # ---- auto_expand ----
    def get_expanded_tools(self) -> List["MCPWrappedTool"]:
        """把服务器里的每个工具包装成独立 Tool。"""
        if not self.auto_expand or not self._available_tools:
            return []
        return [
            MCPWrappedTool(mcp_tool=self, tool_info=info, prefix=self.prefix)
            for info in self._available_tools
        ]

    # ---- 执行 ----
    def run(self, **kwargs) -> str:
        """执行 MCP 操作（action 分发）。

        action: list_tools / call_tool / list_resources / read_resource /
                list_prompts / get_prompt；未指定但有 tool_name 时自动推断为 call_tool。
        """
        action = (kwargs.get("action") or "").lower()
        if not action and kwargs.get("tool_name"):
            action = "call_tool"
        if not action:
            return "错误：必须指定 action 参数或 tool_name 参数"

        async def run_op():
            async with MCPClient(self._client_source(), self.server_args, env=self.env) as client:
                if action == "list_tools":
                    tools = await client.list_tools()
                    if not tools:
                        return "没有找到可用的工具"
                    out = f"找到 {len(tools)} 个工具:\n"
                    for t in tools:
                        out += f"- {t['name']}: {t.get('description', '')}\n"
                    return out

                if action == "call_tool":
                    tool_name = kwargs.get("tool_name")
                    if not tool_name:
                        return "错误：必须指定 tool_name 参数"
                    arguments = kwargs.get("arguments") or {}
                    result = await client.call_tool(tool_name, arguments)
                    return f"工具 '{tool_name}' 执行结果:\n{result}"

                if action == "list_resources":
                    resources = await client.list_resources()
                    if not resources:
                        return "没有找到可用的资源"
                    out = f"找到 {len(resources)} 个资源:\n"
                    for r in resources:
                        out += f"- {r['uri']}: {r.get('name', '')}\n"
                    return out

                if action == "read_resource":
                    uri = kwargs.get("uri")
                    if not uri:
                        return "错误：必须指定 uri 参数"
                    content = await client.read_resource(uri)
                    return f"资源 '{uri}' 内容:\n{content}"

                if action == "list_prompts":
                    prompts = await client.list_prompts()
                    if not prompts:
                        return "没有找到可用的提示词"
                    out = f"找到 {len(prompts)} 个提示词:\n"
                    for p in prompts:
                        out += f"- {p['name']}: {p.get('description', '')}\n"
                    return out

                if action == "get_prompt":
                    prompt_name = kwargs.get("prompt_name")
                    if not prompt_name:
                        return "错误：必须指定 prompt_name 参数"
                    prompt_arguments = kwargs.get("prompt_arguments") or {}
                    messages = await client.get_prompt(prompt_name, prompt_arguments)
                    out = f"提示词 '{prompt_name}':\n"
                    for msg in messages:
                        out += f"[{msg.get('role', '?')}] {msg.get('content', '')}\n"
                    return out

                return f"错误：不支持的操作 '{action}'"

        try:
            return self._run_async(run_op)
        except Exception as e:
            return f"MCP 操作失败: {e}"

    def __repr__(self) -> str:
        return f"<MCPTool {self._name} tools={len(self._available_tools)}>"


class MCPWrappedTool(Tool):
    """把 MCP 服务器里的单个工具包装成独立 Tool（auto_expand 用）。

    name 带前缀（如 "mcp_add"）避免与其它工具冲突；run 时回到所属
    MCPTool 走 call_tool，参数即原始工具的 input_schema。
    """

    def __init__(self, mcp_tool: "MCPTool", tool_info: Dict[str, Any], prefix: str = ""):
        self._mcp_tool = mcp_tool
        self._tool_info = tool_info
        raw_name = tool_info.get("name", "unknown")
        self._name = f"{prefix}{raw_name}"
        self._raw_name = raw_name
        self._description = tool_info.get("description") or f"MCP 工具: {raw_name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        """直接复用服务器给出的 input_schema（已是 JSON Schema）。"""
        schema = self._tool_info.get("input_schema") or self._tool_info.get("inputSchema")
        if isinstance(schema, dict) and schema:
            return schema
        return {"type": "object", "properties": {}}

    def run(self, **kwargs) -> str:
        """调用所属 MCPTool 执行该工具。"""
        return self._mcp_tool.run(action="call_tool", tool_name=self._raw_name, arguments=kwargs)

    def __repr__(self) -> str:
        return f"<MCPWrappedTool {self._name}>"
