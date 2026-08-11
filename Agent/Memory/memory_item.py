"""工作记忆数据结构。

轻量实现，不依赖 pydantic；只用标准库 dataclass。
工作记忆放在内存中，生命周期短、容量小。
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class MemoryItem:
    """单条记忆项。"""

    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0.0 ~ 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[datetime] = None  # 最近一次被检索命中的时间，用于辅助遗忘
    # 身份隔离：方案 A 下由服务端在构造时烙入，LLM 不可指定
    user_id: str = "default"
    session_id: str = "default"

    def touch(self, now: Optional[datetime] = None) -> None:
        """更新最近访问时间。"""
        self.last_accessed = now or datetime.now()

    def __repr__(self) -> str:
        return f"<MemoryItem id={self.id[:8]} imp={self.importance:.2f} ts={self.timestamp:%H:%M:%S}>"


def make_item(content: str, importance: float = 0.5, metadata: Dict[str, Any] = None) -> MemoryItem:
    """快捷构造一条记忆。"""
    return MemoryItem(content=content, importance=importance, metadata=metadata or {})
