import json
from typing import List, Optional

from Agent.LLMClient.llm_client import HelloAgentsLLM
from Agent.LLMClient.tool import Tool, ToolRegistry

DEFAULT_SYSTEM_PROMPT = """你是一个能够自主调用外部工具的智能助手。

你可以通过 function calling 调用工具来完成任务；如果不确定有哪些工具可用，
可以调用 list_tools 查询当前工具列表及其参数说明。

工作流程：
1. 分析用户问题，判断是否需要调用工具。
2. 如需调用工具，使用对应的 function；拿到结果后继续思考。
3. 当信息足够时，直接用自然语言给出最终答案（不要调用工具）。"""


class MyAgent:
    """封装 LLM + 工具的自主 Agent。

    LLM 自主决定是否调用工具、调用哪个工具；Agent 负责执行工具
    并把结果回填，循环直到 LLM 给出最终文本答案或达到最大步数。
    """

    def __init__(self, llm_client: HelloAgentsLLM, tools: Optional[List[Tool]] = None,
                 max_steps: int = 10, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.llm_client = llm_client
        self.registry = ToolRegistry(with_builtin=True)
        for tool in (tools or []):
            self.registry.register(tool)
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def register_tool(self, tool: Tool) -> None:
        self.registry.register(tool)

    def run(self, question: str) -> Optional[str]:
        """处理用户问题，自动完成 工具调用 → 执行 → 回填 的循环。"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        tool_specs = self.registry.specs()

        for step in range(1, self.max_steps + 1):
            print(f"\n===== 第 {step} 步 =====")
            assistant = self.llm_client.chat(messages=messages, tools=tool_specs)
            if assistant is None:
                print("错误：LLM 未能返回有效响应，流程终止。")
                return None

            # 没有 tool_calls，说明 LLM 已经得出最终答案
            if not assistant.tool_calls:
                print("💬 最终回答:")
                print(assistant.content)
                return assistant.content

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

        print("⚠️ 已达最大步数，流程终止。")
        return None

