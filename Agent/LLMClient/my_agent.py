import json
from typing import List, Optional

from Agent.LLMClient.agent import Agent
from Agent.LLMClient.llm_client import HelloAgentsLLM
from Agent.LLMClient.tool import Tool, ToolRegistry

DEFAULT_SYSTEM_PROMPT = """你是一个能够自主调用外部工具的智能助手。

你可以通过 function calling 调用工具来完成任务；如果不确定有哪些工具可用，
可以调用 list_tools 查询当前工具列表及其参数说明。

工作流程：
1. 分析用户问题，判断是否需要调用工具。
2. 如需调用工具，使用对应的 function；拿到结果后继续思考。
3. 当信息足够时，直接用自然语言给出最终答案（不要调用工具）。"""


class MyAgent(Agent):
    """封装 LLM + 工具的自主 Agent，继承 Agent 基类。

    设计参照 MyAgent/chapter7/core/SimpleAgent.py：在基类的 name/llm/
    system_prompt/history 之上，增加工具注册表与 add_tool/remove_tool/
    list_tools/has_tools 便利方法、_get_enhanced_system_prompt() 提示词增强。

    与 chapter7 的差异：底层用 OpenAI 原生 function calling（而非
    [TOOL_CALL:...] 文本协议），更稳；提示词增强仅作为可读工具清单做
    grounding，不参与调用解析。
    """

    def __init__(
        self,
        name: str = "MyAgent",
        llm: Optional[HelloAgentsLLM] = None,
        llm_client: Optional[HelloAgentsLLM] = None,
        tools: Optional[List[Tool]] = None,
        max_steps: int = 10,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        config=None,
        enable_history: bool = True,
    ):
        # 兼容旧关键字 llm_client
        llm = llm or llm_client
        if llm is None:
            raise ValueError("必须提供 llm 或 llm_client。")
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)

        self.registry = ToolRegistry(with_builtin=True)
        for tool in (tools or []):
            self.registry.register(tool)
        self.max_steps = max_steps
        self.enable_history = enable_history

    # ---- 提示词增强 ----
    def _get_enhanced_system_prompt(self) -> str:
        """在系统提示词后追加可用工具清单，作为模型的 grounding。

        原生 function calling 已通过 tools 参数告知工具，这里只补充一份
        可读说明，便于模型在不确定时主动调用 list_tools 或选择合适工具。
        """
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"
        if not self.registry.all():
            return base_prompt

        tools_description = self.registry.describe_all()
        if not tools_description:
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题（通过 function calling 调用）：\n"
        tools_section += tools_description + "\n"
        tools_section += "\n如不确定有哪些工具或参数含义，可调用 list_tools 查询。"
        return base_prompt + tools_section

    # ---- 工具便利方法（仿 SimpleAgent）----
    def add_tool(self, tool: Tool) -> None:
        """添加工具到 Agent（便利方法）。

        若工具带 auto_expand（如 MCPTool），会自动展开为多个独立工具
        分别注册，而非注册一个 action 式的统一工具。
        """
        if getattr(tool, "auto_expand", False):
            expanded = tool.get_expanded_tools()
            if expanded:
                for t in expanded:
                    self.registry.register(t)
                print(f"✅ MCP 工具 '{tool.name}' 已展开为 {len(expanded)} 个独立工具。")
                return
        self.registry.register(tool)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法），不存在返回 False。"""
        return self.registry.unregister(tool_name)

    def list_tools(self) -> List[str]:
        """列出所有可用工具名。"""
        return self.registry.list_tools()

    def has_tools(self) -> bool:
        """检查是否有可用工具。"""
        return len(self.registry.all()) > 0

    # 向后兼容别名
    def register_tool(self, tool: Tool) -> None:
        self.add_tool(tool)

    # ---- 运行循环 ----
    def run(self, question: str, **kwargs) -> Optional[str]:
        """处理用户问题，自动完成 工具调用 → 执行 → 回填 的循环。

        若 enable_history=True，本轮 user 输入与最终 assistant 回答会
        追加进 _history，供下一轮带入（中间的 tool 消息不持久化）。
        """
        messages: List[dict] = [
            {"role": "system", "content": self._get_enhanced_system_prompt()},
        ]
        # 带入跨轮历史
        if self.enable_history:
            for msg in self._history:
                messages.append(msg.to_dict())
        messages.append({"role": "user", "content": question})

        tool_specs = self.registry.specs()
        final_content: Optional[str] = None

        for step in range(1, self.max_steps + 1):
            print(f"\n===== 第 {step} 步 =====")
            assistant = self.llm.chat(messages=messages, tools=tool_specs)
            if assistant is None:
                print("错误：LLM 未能返回有效响应，流程终止。")
                break

            # 没有 tool_calls，说明 LLM 已经得出最终答案
            if not assistant.tool_calls:
                print("💬 最终回答:")
                print(assistant.content)
                final_content = assistant.content
                break

            # 先把 assistant 这条（含 tool_calls）追加到历史
            messages.append(assistant.model_dump(exclude_none=True))

            # 依次执行 LLM 决定调用的工具
            for tc in assistant.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"🔧 调用工具: {name}({args})")
                result = self.registry.execute(name, args)
                print(f"   ↳ 结果: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result,
                })
        else:
            print("⚠️ 已达最大步数，流程终止。")

        # 记录跨轮历史：仅存用户输入与最终回答
        if self.enable_history and final_content is not None:
            self.add_message("user", question)
            self.add_message("assistant", final_content)

        return final_content
