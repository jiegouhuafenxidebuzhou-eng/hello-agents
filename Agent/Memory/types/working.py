"""工作记忆：纯内存、短期、小容量。

三大能力：
- 添加记忆 add()
- 检索记忆 retrieve()  —— 时间近度 + 关键词过滤
- 遗忘     forget() / 清理过期 / 容量驱逐

遗忘是混合策略：
1. TTL 过期：超过 ttl_minutes 的记忆在访问/添加时惰性清除。
2. 容量/Tokens 驱逐：超出上限时，按「优先级 = 重要性 × 时间衰减」驱逐最低者。
3. 主动遗忘：forget(id) 显式删除某条。
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from Agent.Memory.memory_item import MemoryItem


def _default_token_counter(text: str) -> int:
    """默认 token 估算：按字符数计。

    工作记忆只是粗略预算，不需要精确分词；后续可换成 tiktoken。
    """
    return len(text)


class WorkingMemory:
    """工作记忆，纯内存实现。"""

    def __init__(self,
                 max_capacity: int = 20,
                 max_tokens: int = 2000,
                 ttl_minutes: int = 120,
                 decay_hours: float = 6.0,
                 min_decay: float = 0.1,
                 token_counter: Callable[[str], int] = _default_token_counter,
                 user_id: str = "default",
                 session_id: str = "default"):
        # 上限配置
        self.max_capacity = max_capacity
        self.max_tokens = max_tokens
        self.ttl_minutes = ttl_minutes

        # 衰减参数
        self.decay_hours = decay_hours          # 每隔 decay_hours 衰减一轮
        self.min_decay = min_decay              # 衰减下限，保证最小权重

        # 会话身份：烙入本实例，add 时盖到每条记忆上
        self.user_id = user_id
        self.session_id = session_id

        self._counter = token_counter
        self._items: List[MemoryItem] = []      # 按插入顺序
        self._index: Dict[str, MemoryItem] = {} # id -> item，O(1) 查找
        self.current_tokens: int = 0

    # ---------- 添加 ----------
    def add(self, content: str, importance: float = 0.5,
            metadata: Dict[str, Any] = None) -> str:
        """添加一条工作记忆，返回其 id。

        会先惰性清理过期记忆，再加入新条目，最后按容量/Token 上限驱逐最低优先级。
        """
        self._expire_old()

        item = MemoryItem(
            content=content,
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {},
            user_id=self.user_id,
            session_id=self.session_id,
        )
        self._items.append(item)
        self._index[item.id] = item
        self.current_tokens += self._counter(content)

        self._evict()
        return item.id

    # ---------- 检索 ----------
    def retrieve(self, query: str = None, limit: int = None) -> List[MemoryItem]:
        """检索工作记忆：时间近度优先 + 可选关键词过滤。

        Args:
            query: 若提供，仅保留 content 中包含该关键词的记忆（大小写不敏感）。
            limit: 最多返回条数，默认全部返回。

        Returns:
            按时间倒序（最近在前）的记忆列表。被命中的记忆会更新 last_accessed。
        """
        self._expire_old()

        candidates = list(self._items)
        if query:
            q = query.lower()
            candidates = [m for m in candidates if q in m.content.lower()]

        # 时间近度优先：timestamp 倒序
        candidates.sort(key=lambda m: m.timestamp, reverse=True)

        if limit is not None:
            candidates = candidates[:limit]

        now = datetime.now()
        for m in candidates:
            m.touch(now)
        return candidates

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """按 id 取单条。"""
        return self._index.get(memory_id)

    # ---------- 遗忘 ----------
    def forget(self, memory_id: str) -> bool:
        """主动遗忘：按 id 删除一条。"""
        item = self._index.pop(memory_id, None)
        if item is None:
            return False
        try:
            self._items.remove(item)
        except ValueError:
            pass
        self.current_tokens = max(0, self.current_tokens - self._counter(item.content))
        return True

    def cleanup(self) -> int:
        """主动清理一次过期记忆，返回清理条数。"""
        return self._expire_old()

    def clear(self) -> None:
        """清空全部工作记忆。"""
        self._items.clear()
        self._index.clear()
        self.current_tokens = 0

    # ---------- 统计 ----------
    def get_stats(self) -> Dict[str, Any]:
        self._expire_old()
        n = len(self._items)
        return {
            "count": n,
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "capacity_usage": n / self.max_capacity if self.max_capacity else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens else 0.0,
            "ttl_minutes": self.ttl_minutes,
            "avg_importance": sum(m.importance for m in self._items) / n if n else 0.0,
            "memory_type": "working",
        }

    # ---------- 内部：遗忘相关 ----------
    def _expire_old(self) -> int:
        """TTL 过期清理，返回清理条数。"""
        if not self._items:
            return 0
        cutoff = datetime.now() - timedelta(minutes=self.ttl_minutes)
        kept: List[MemoryItem] = []
        removed_tokens = 0
        removed = 0
        for m in self._items:
            if m.timestamp >= cutoff:
                kept.append(m)
            else:
                removed_tokens += self._counter(m.content)
                removed += 1
        if removed == 0:
            return 0
        self._items = kept
        self._index = {m.id: m for m in kept}
        self.current_tokens = max(0, self.current_tokens - removed_tokens)
        return removed

    def _evict(self) -> int:
        """容量 / Token 超限时驱逐最低优先级记忆，返回驱逐条数。"""
        evicted = 0
        while self._over_capacity() and self._items:
            victim = min(self._items, key=self._priority)
            self.forget(victim.id)
            evicted += 1
        return evicted

    def _over_capacity(self) -> bool:
        return len(self._items) > self.max_capacity or self.current_tokens > self.max_tokens

    def _priority(self, m: MemoryItem) -> float:
        """优先级 = 重要性 × 时间衰减。值越大越该保留。"""
        return m.importance * self._time_decay(m.timestamp)

    def _time_decay(self, timestamp: datetime) -> float:
        """时间衰减因子：随时间指数下降，但不低于 min_decay。"""
        hours = (datetime.now() - timestamp).total_seconds() / 3600.0
        # 每经过 decay_hours，权重乘以 1/e 级别衰减；这里用 0.5 简化
        decay = 0.5 ** (hours / self.decay_hours)
        return max(self.min_decay, decay)


if __name__ == '__main__':
    wm = WorkingMemory(max_capacity=5, max_tokens=200, ttl_minutes=60)

    ids = []
    ids.append(wm.add("用户叫张三", importance=0.9))
    ids.append(wm.add("用户在咨询退货流程", importance=0.7))
    ids.append(wm.add("订单号是 A12345", importance=0.8))
    ids.append(wm.add("今天天气不错", importance=0.2))
    print("stats:", wm.get_stats())

    print("\n--- 检索全部（按近度）---")
    for m in wm.retrieve():
        print(m, m.content)

    print("\n--- 关键词检索 '订单' ---")
    for m in wm.retrieve(query="订单"):
        print(m, m.content)

    print("\n--- 主动遗忘一条 ---")
    wm.forget(ids[0])
    print("遗忘后数量:", len(wm.retrieve()))
