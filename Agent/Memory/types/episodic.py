"""情景记忆：本地缓存 + SQLite 权威存储 + Qdrant 向量检索（三层）。

适配 Agent/Memory/manager.py 的 MemoryManager 插口接口：
    add(content, importance, metadata) -> id
    retrieve(query, limit) -> List[MemoryItem]
    forget(memory_id) -> bool
    get_stats() -> dict
    clear()

三层职责：
1. 本地缓存（内存，有界 FIFO）：热路径单条命中 + stats；写时同步写 SQLite。
2. SQLite 权威存储：持久化、跨会话、结构化过滤；向量召回后回查完整原文。
3. Qdrant 向量存储（可选）：语义召回；未装 qdrant-client / 未配真嵌入模型时降级关闭，
   retrieve 自动回退到 SQLite 关键词检索。

身份隔离（方案 A）：user_id/session_id 在构造时烙入，所有读写都带 user_id 过滤。
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from Agent.Memory.memory_item import MemoryItem, make_item
from Agent.Memory.embedding import get_embedder, embedding_available
from Agent.Memory.storage.sqlite_store import EpisodicSQLiteStore
from Agent.Memory.storage.qdrant_store import QdrantStore, qdrant_from_env

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """情景记忆，三层存储。"""

    MEMORY_TYPE = "episodic"

    def __init__(self,
                 user_id: str = "default",
                 session_id: str = "default",
                 db_path: Optional[str] = None,
                 vector_store: Optional[QdrantStore] = None,
                 use_embedder: bool = True,
                 max_cache: int = 200):
        self.user_id = user_id
        self.session_id = session_id
        self.max_cache = max_cache

        # 1) 权威存储（单例，身份烙入本实例用于过滤）
        self.store = EpisodicSQLiteStore(db_path=db_path)

        # 2) 嵌入模型（单例）
        self.embedder = get_embedder() if use_embedder else None

        # 3) 向量存储（可选）：传入优先，否则按 env 构造
        if vector_store is None and use_embedder:
            vector_store = qdrant_from_env()
        self.vector_store = vector_store

        # 仅当 Qdrant 可用 且 装了真语义嵌入（非 hashing 兜底）时才走向量路径
        self._vector_ready = bool(
            self.vector_store is not None
            and getattr(self.vector_store, "available", False)
            and self.embedder is not None
            and self.embedder.available
        )

        # 本地缓存（有界 FIFO）+ 会话索引
        self._cache: List[MemoryItem] = []
        self._cache_index: Dict[str, MemoryItem] = {}
        self._sessions: Dict[str, List[str]] = {}

    # ---------- 添加 ----------
    def add(self, content: str, importance: float = 0.5,
            metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加一条情景记忆，返回 id。

        写本地缓存 + SQLite 权威层；向量层可用时同步 upsert（失败不影响权威存储）。
        """
        item = make_item(content=content, importance=importance, metadata=metadata)
        # 身份烙入（覆盖 metadata 里可能携带的身份字段）
        item.user_id = self.user_id
        item.session_id = self.session_id

        # 1) 本地缓存
        self._cache_append(item)

        # 2) SQLite 权威存储
        self.store.add_memory(
            memory_id=item.id,
            user_id=self.user_id,
            content=item.content,
            memory_type=self.MEMORY_TYPE,
            timestamp=int(item.timestamp.timestamp()),
            importance=item.importance,
            session_id=self.session_id,
            properties=item.metadata,
        )

        # 3) 向量索引（可选）
        if self._vector_ready:
            try:
                vec = self.embedder.encode(item.content)
                self.vector_store.add_vectors(
                    vectors=[vec],
                    payloads=[{
                        "memory_id": item.id,
                        "user_id": self.user_id,
                        "session_id": self.session_id,
                        "memory_type": self.MEMORY_TYPE,
                        "importance": item.importance,
                        "content": item.content,
                        "timestamp": int(item.timestamp.timestamp()),
                    }],
                    ids=[item.id],
                )
            except Exception as e:
                logger.warning(f"情景记忆向量入库失败 (id={item.id[:8]}): {e}")

        return item.id

    # ---------- 检索 ----------
    def retrieve(self, query: Optional[str] = None,
                 limit: int = 5) -> List[MemoryItem]:
        """检索情景记忆。

        - query 为空：按时间倒序取该用户最近 limit 条。
        - 有真语义向量层：向量召回 → 回 SQLite 取原文 → 综合评分（向量0.8+近因0.2）×重要性权重。
        - 无向量层 / 向量无结果：回退 SQLite 关键词（content LIKE）检索。
        """
        if not query or not query.strip():
            rows = self.store.search_memories(
                user_id=self.user_id, memory_type=self.MEMORY_TYPE, limit=limit,
            )
            return [self._touch(self._row_to_item(r)) for r in rows]

        # 主路径：语义向量检索
        if self._vector_ready:
            items = self._retrieve_vector(query, limit)
            if items:
                return items

        # 回退：SQLite 关键词
        rows = self.store.search_memories(
            user_id=self.user_id, memory_type=self.MEMORY_TYPE,
            keyword=query, limit=limit,
        )
        return [self._touch(self._row_to_item(r)) for r in rows]

    def _retrieve_vector(self, query: str, limit: int) -> List[MemoryItem]:
        """向量召回 + 综合评分。"""
        try:
            qvec = self.embedder.encode(query)
        except Exception as e:
            logger.warning(f"查询嵌入失败，回退关键词: {e}")
            return []

        hits = self.vector_store.search_similar(
            query_vector=qvec,
            limit=max(limit * 5, 20),
            where={"memory_type": self.MEMORY_TYPE, "user_id": self.user_id},
        )
        if not hits:
            return []

        now_ts = int(datetime.now().timestamp())
        scored: List[tuple] = []
        seen = set()
        for hit in hits:
            payload = hit.get("payload") or {}
            mid = payload.get("memory_id")
            if not mid or mid in seen:
                continue
            row = self.store.get_memory(mid)
            if not row:
                continue
            seen.add(mid)

            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(row["timestamp"])) / 86400.0)
            recency = 1.0 / (1.0 + age_days)
            imp = float(row.get("importance", 0.5))
            base = vec_score * 0.8 + recency * 0.2
            combined = base * (0.8 + imp * 0.4)

            item = self._row_to_item(row)
            item.metadata = {**(item.metadata or {}),
                             "relevance_score": round(combined, 4),
                             "vector_score": round(vec_score, 4),
                             "recency_score": round(recency, 4)}
            scored.append((combined, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._touch(it) for _, it in scored[:limit]]

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """按 id 取单条：先查本地缓存，再查 SQLite。"""
        cached = self._cache_index.get(memory_id)
        if cached is not None:
            return self._touch(cached)
        row = self.store.get_memory(memory_id)
        if not row:
            return None
        return self._row_to_item(row)

    # ---------- 遗忘 ----------
    def forget(self, memory_id: str) -> bool:
        """主动遗忘：缓存 + SQLite + Qdrant 三处清除。任一成功即返回 True。"""
        removed_cache = self._cache_remove(memory_id)
        removed_db = self.store.delete_memory(memory_id)
        if self._vector_ready:
            try:
                self.vector_store.delete_by_memory_ids([memory_id])
            except Exception as e:
                logger.warning(f"向量删除失败 (id={memory_id[:8]}): {e}")
        return removed_cache or removed_db

    # ---------- 清空 ----------
    def clear(self) -> None:
        """清空本 user_id 的全部情景记忆（隔离语义，不清全局）。"""
        # 先取本用户 episodic 的全部 id（用于 Qdrant 按需删除）
        rows = self.store.search_memories(
            user_id=self.user_id, memory_type=self.MEMORY_TYPE, limit=100000,
        )
        ids = [r["memory_id"] for r in rows]
        self.store.delete_by_user(self.user_id, memory_type=self.MEMORY_TYPE)
        if self._vector_ready and ids:
            try:
                self.vector_store.delete_by_memory_ids(ids)
            except Exception as e:
                logger.warning(f"清空向量层失败: {e}")
        self._cache.clear()
        self._cache_index.clear()
        self._sessions.clear()

    # ---------- 统计 ----------
    def get_stats(self) -> Dict[str, Any]:
        db_stats = self.store.get_stats(
            memory_type=self.MEMORY_TYPE, user_id=self.user_id,
        )
        vec_stats = (self.vector_store.get_stats()
                     if self._vector_ready else {"enabled": False})
        return {
            "memory_type": self.MEMORY_TYPE,
            "count": db_stats.get("count", 0),
            "cached": len(self._cache),
            "sessions_count": db_stats.get("sessions_count", 0),
            "avg_importance": db_stats.get("avg_importance", 0.0),
            "vector_ready": self._vector_ready,
            "vector_store": vec_stats,
            "document_store": {
                "store_type": "sqlite",
                "db_path": db_stats.get("db_path"),
            },
        }

    # ---------- 缓存内部 ----------
    def _cache_append(self, item: MemoryItem) -> None:
        self._cache.append(item)
        self._cache_index[item.id] = item
        self._sessions.setdefault(self.session_id, []).append(item.id)
        # FIFO 驱逐
        while len(self._cache) > self.max_cache:
            old = self._cache.pop(0)
            self._cache_index.pop(old.id, None)
            lst = self._sessions.get(old.session_id)
            if lst:
                try:
                    lst.remove(old.id)
                except ValueError:
                    pass
                if not lst:
                    self._sessions.pop(old.session_id, None)

    def _cache_remove(self, memory_id: str) -> bool:
        item = self._cache_index.pop(memory_id, None)
        if item is None:
            return False
        try:
            self._cache.remove(item)
        except ValueError:
            pass
        lst = self._sessions.get(item.session_id)
        if lst:
            try:
                lst.remove(memory_id)
            except ValueError:
                pass
            if not lst:
                self._sessions.pop(item.session_id, None)
        return True

    # ---------- 工具 ----------
    @staticmethod
    def _row_to_item(row: Dict[str, Any]) -> MemoryItem:
        return MemoryItem(
            content=row["content"],
            id=row["memory_id"],
            timestamp=datetime.fromtimestamp(int(row["timestamp"])),
            importance=float(row.get("importance", 0.5)),
            metadata=row.get("properties") or {},
            user_id=row.get("user_id", "default"),
            session_id=row.get("session_id", "default"),
        )

    @staticmethod
    def _touch(item: MemoryItem) -> MemoryItem:
        item.touch()
        return item
