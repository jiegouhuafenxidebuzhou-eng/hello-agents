"""记忆管理器：统一管理工作记忆 / 情景记忆 / 语义记忆。

所有记忆后端遵循同一接口：
- add(content, importance, metadata) -> id
- retrieve(query, limit) -> List[MemoryItem]
- forget(memory_id) -> bool
- stats() -> dict

已接入工作记忆、情景记忆、语义记忆三类后端；语义记忆默认关闭，
enable_semantic=True 时按本会话身份构造（图谱/LLM 按 env/参数自动降级）。
"""

from typing import Any, Dict, List, Optional

from Agent.Memory.memory_item import MemoryItem
from Agent.Memory.types.working import WorkingMemory


class MemoryManager:
    def __init__(self,
                 working: Optional[WorkingMemory] = None,
                 episodic=None,
                 semantic=None,
                 user_id: str = "default",
                 session_id: str = "default",
                 enable_episodic: bool = False,
                 episodic_db_path: Optional[str] = None,
                 enable_semantic: bool = False,
                 semantic_db_path: Optional[str] = None,
                 enable_graph: bool = True,
                 enable_llm: bool = False):
        self.user_id = user_id
        self.session_id = session_id
        # 未显式提供 working 时，按本会话身份构造一个
        if working is None:
            working = WorkingMemory(user_id=user_id, session_id=session_id)
        # 未显式提供 episodic 但显式开启时，按本会话身份构造一个
        if episodic is None and enable_episodic:
            from Agent.Memory.types.episodic import EpisodicMemory
            episodic = EpisodicMemory(user_id=user_id, session_id=session_id,
                                     db_path=episodic_db_path)
        # 未显式提供 semantic 但显式开启时，按本会话身份构造一个
        if semantic is None and enable_semantic:
            from Agent.Memory.types.semantic import SemanticMemory
            llm_client = None
            if enable_llm:
                try:
                    from Agent.LLMClient.llm_client import HelloAgentsLLM
                    llm_client = HelloAgentsLLM()  # 读 .env；未配置抛 ValueError
                except Exception as e:
                    # graceful：未配 LLM 不阻断语义记忆，仅关掉 LLM 三元组抽取（回退规则）
                    print(f"[semantic] LLM 客户端未启用: {e}")
                    llm_client = None
            # db_path 默认复用 episodic_db_path（共享 SQLite 单例 + memory_type 分区）
            sem_db = semantic_db_path or episodic_db_path
            semantic = SemanticMemory(user_id=user_id, session_id=session_id,
                                      db_path=sem_db, enable_graph=enable_graph,
                                      llm_client=llm_client)
        self.backends: Dict[str, Any] = {
            "working": working,
            "episodic": episodic,      # 默认 None 占位；enable_episodic=True 时自动构造
            "semantic": semantic,      # 默认 None 占位；enable_semantic=True 时自动构造
        }

    @property
    def enabled_types(self) -> List[str]:
        return [t for t, b in self.backends.items() if b is not None]

    def _backend(self, memory_type: str):
        b = self.backends.get(memory_type)
        if b is None:
            raise ValueError(f"记忆类型 '{memory_type}' 未启用。当前可用: {self.enabled_types}")
        return b

    # ---------- 统一操作 ----------
    def add(self, content: str, memory_type: str = "working",
            importance: float = 0.5, metadata: Dict[str, Any] = None) -> str:
        """添加记忆，返回 id。"""
        return self._backend(memory_type).add(content, importance=importance, metadata=metadata or {})

    def retrieve(self, query: str = None, memory_type: str = "working",
                 limit: int = 5) -> List[MemoryItem]:
        """检索记忆。memory_type='all' 时跨所有已启用类型检索。"""
        if memory_type == "all":
            results: List[MemoryItem] = []
            for t in self.enabled_types:
                results.extend(self.backends[t].retrieve(query=query, limit=limit))
            # 跨类型按时间倒序统一排序
            results.sort(key=lambda m: m.timestamp, reverse=True)
            return results[:limit] if limit else results
        return self._backend(memory_type).retrieve(query=query, limit=limit)

    def forget(self, memory_id: str, memory_type: str = "working") -> bool:
        """遗忘指定 id 的记忆。"""
        return self._backend(memory_type).forget(memory_id)

    def stats(self, memory_type: str = None) -> Dict[str, Any]:
        """统计信息。memory_type=None 返回全部类型汇总。"""
        if memory_type:
            return {memory_type: self._backend(memory_type).get_stats()}
        return {t: b.get_stats() for t, b in self.backends.items() if b is not None}

    def clear(self, memory_type: str = "working") -> None:
        """清空指定类型记忆。"""
        self._backend(memory_type).clear()
