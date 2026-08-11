"""会话级 Agent 工厂。

方案 A：每个会话构造一个全新的 MyAgent，内含该会话专属的
WorkingMemory / MemoryManager / MemoryTool，身份 (user_id, session_id)
在构造时烙入。LLMClient 是无状态 HTTP 客户端，全局共享，不重复建。

不同会话 = 不同 Python 对象 → 物理隔离，无需过滤、无需加锁。
"""

from typing import Optional

from Agent.LLMClient.llm_client import HelloAgentsLLM
from Agent.LLMClient.my_agent import MyAgent
from Agent.LLMClient.tool import Tool
from Agent.Memory.manager import MemoryManager
from Agent.Memory.memory_tool import MemoryTool
from Agent.Memory.working import WorkingMemory


def build_session_agent(llm_client: HelloAgentsLLM,
                        user_id: str,
                        session_id: str,
                        extra_tools: Optional[list] = None,
                        max_steps: int = 10) -> MyAgent:
    """为一个会话构造专属 Agent。

    Args:
        llm_client: 全局共享的 LLM 客户端（无状态）。
        user_id: 用户标识，服务端注入，LLM 不可指定。
        session_id: 会话标识，服务端注入，LLM 不可指定。
        extra_tools: 除记忆工具外，该会话还需要挂载的工具。
    """
    # 本会话专属的工作记忆，身份烙入
    working = WorkingMemory(user_id=user_id, session_id=session_id)
    manager = MemoryManager(working=working, user_id=user_id, session_id=session_id)
    mem_tool = MemoryTool(manager, user_id=user_id, session_id=session_id)

    tools = [mem_tool] + list(extra_tools or [])
    return MyAgent(llm_client=llm_client, tools=tools, max_steps=max_steps)
