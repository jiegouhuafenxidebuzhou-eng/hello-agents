"""记忆管理器：统一管理工作记忆 / 情景记忆 / 语义记忆。

所有记忆后端遵循同一接口：
- add(content, importance, metadata) -> id
- retrieve(query, limit) -> List[MemoryItem]
- forget(memory_id) -> bool
- stats() -> dict

目前只接入了工作记忆；情景记忆、语义记忆后端留作插口（为 None 表示未启用），
后续实现同接口即可直接挂上，MemoryTool 无需改动。
"""

from typing import Any, Dict, List, Optional

from Agent.Memory.memory_item import MemoryItem
from Agent.Memory.working import WorkingMemory


class MemoryManager:
    def __init__(self,
                 working: Optional[WorkingMemory] = None,
                 episodic=None,
                 semantic=None,
                 user_id: str = "default",
                 session_id: str = "default"):
        self.user_id = user_id
        self.session_id = session_id
        # 未显式提供 working 时，按本会话身份构造一个
        if working is None:
            working = WorkingMemory(user_id=user_id, session_id=session_id)
        self.backends: Dict[str, Any] = {
            "working": working,
            "episodic": episodic,      # 暂未实现，留插口
            "semantic": semantic,      # 暂未实现，留插口
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
