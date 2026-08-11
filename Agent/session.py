"""会话级 Agent 工厂。

方案 A：每个会话构造一个全新的 MyAgent，内含该会话专属的
WorkingMemory / MemoryManager / MemoryTool，身份 (user_id, session_id)
在构造时烙入。LLMClient 是无状态 HTTP 客户端，全局共享，不重复建。

不同会话 = 不同 Python 对象 → 物理隔离，无需过滤、无需加锁。

情景记忆为可选启用（enable_episodic=True）：SQLite/Qdrant 虽全局共享单例，
但所有读写都带 user_id 过滤，跨会话可召回长期记忆而互不串读。
"""

from typing import Optional

from Agent.LLMClient.llm_client import HelloAgentsLLM
from Agent.LLMClient.my_agent import MyAgent
from Agent.Memory.types.episodic import EpisodicMemory
from Agent.Memory.manager import MemoryManager
from Agent.tools.memory_tool import MemoryTool
from Agent.Memory.types.working import WorkingMemory


def build_session_agent(llm_client: HelloAgentsLLM,
                        user_id: str,
                        session_id: str,
                        extra_tools: Optional[list] = None,
                        max_steps: int = 10,
                        enable_episodic: bool = False,
                        episodic_db_path: Optional[str] = None,
                        enable_rag: bool = False,
                        rag_namespace: str = "default") -> MyAgent:
    """为一个会话构造专属 Agent。

    Args:
        llm_client: 全局共享的 LLM 客户端（无状态）。
        user_id: 用户标识，服务端注入，LLM 不可指定。
        session_id: 会话标识，服务端注入，LLM 不可指定。
        extra_tools: 除记忆工具外，该会话还需要挂载的工具。
        enable_episodic: 是否启用情景记忆（SQLite 权威层 + 可选 Qdrant 向量层）。
        episodic_db_path: 情景记忆 SQLite 路径，None 用默认。
        enable_rag: 是否挂载 RAG 工具（外部知识库检索增强生成）。默认关闭；
            开启后 LLM 可在 ReAct 循环里自主调 `rag` 工具。无 Qdrant 时自动
            降级为内存向量库，无 LLM 时 ask 降级为返回原文片段。
        rag_namespace: RAG 知识库命名空间（做多知识库隔离），构造时烙入，
            LLM 不可指定。默认 "default"。
    """
    # 本会话专属的工作记忆，身份烙入
    working = WorkingMemory(user_id=user_id, session_id=session_id)
    # 情景记忆（可选）：身份烙入；SQLite/Qdrant 全局共享但读写带 user_id 过滤
    episodic = (EpisodicMemory(user_id=user_id, session_id=session_id,
                               db_path=episodic_db_path)
                if enable_episodic else None)
    manager = MemoryManager(working=working, episodic=episodic,
                            user_id=user_id, session_id=session_id)
    mem_tool = MemoryTool(manager, user_id=user_id, session_id=session_id)

    tools = [mem_tool] + list(extra_tools or [])
    if enable_rag:
        from Agent.tools.rag_tool import RAGTool
        # RAG 是外部共享知识，namespace 在构造时烙入；llm_client 复用全局实例
        tools.append(RAGTool(rag_namespace=rag_namespace, llm_client=llm_client))
    return MyAgent(llm_client=llm_client, tools=tools, max_steps=max_steps)
