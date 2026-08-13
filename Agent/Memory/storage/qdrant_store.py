"""Qdrant 向量存储（可选依赖，未安装/未配置自动降级）。

情景记忆的语义检索层。不可用时所有方法安全降级：
- add_vectors → False
- search_similar → []
- delete_by_memory_ids → 无操作
EpisodicMemory 据此回退到 SQLite 关键词检索。

配置（环境变量）：
- QDRANT_URL / QDRANT_API_KEY
- QDRANT_COLLECTION（默认 hello_agents_vectors）
- QDRANT_VECTOR_SIZE（默认 384）
- QDRANT_DISTANCE（cosine/dot/euclidean，默认 cosine）
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 可选依赖
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None


_DISTANCE_MAP = {
    "cosine": "COSINE",
    "dot": "DOT",
    "euclid": "EUCLID",
    "euclidean": "EUCLID",
}


class QdrantStore:
    """Qdrant 向量存储封装。不可用时 available=False，方法全部安全降级。"""

    _instances: Dict[tuple, "QdrantStore"] = {}
    _lock = threading.Lock()

    def __new__(cls, url: Optional[str] = None, api_key: Optional[str] = None,
               collection_name: str = "hello_agents_vectors",
               vector_size: int = 1024, distance: str = "cosine",
               timeout: int = 30):
        key = (url or "local", collection_name)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    inst = super().__new__(cls)
                    cls._instances[key] = inst
        return cls._instances[key]

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None,
                 collection_name: str = "hello_agents_vectors",
                 vector_size: int = 1024, distance: str = "cosine",
                 timeout: int = 30):
        if hasattr(self, "_initialized"):
            return

        self.collection_name = collection_name
        self.vector_size = int(vector_size)
        self.distance = (distance or "cosine").lower()
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.available = False
        self.client = None

        if not _QDRANT_AVAILABLE:
            logger.info("qdrant-client 未安装，向量检索降级关闭。pip install qdrant-client 可启用。")
        elif not url:
            logger.info("未配置 QDRANT_URL，向量检索降级关闭。")
        else:
            try:
                self.client = QdrantClient(
                    url=url, api_key=api_key, timeout=timeout
                ) if api_key else QdrantClient(url=url, timeout=timeout)
                self._ensure_collection()
                self.available = True
                logger.info(f"✅ Qdrant 连接成功: {collection_name}")
            except Exception as e:
                logger.warning(f"Qdrant 连接失败，向量检索降级关闭: {e}")
                self.client = None

        self._initialized = True

    # ---------- 集合 ----------
    def _ensure_collection(self) -> None:
        if not self.client:
            return
        names = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in names:
            dist = getattr(Distance, _DISTANCE_MAP.get(self.distance, "COSINE"), Distance.COSINE)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=dist),
            )
            logger.info(f"✅ 创建 Qdrant 集合: {self.collection_name}")
        # payload 索引（失败忽略，可能已存在）
        for field in ("memory_type", "user_id", "session_id", "memory_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass

    # ---------- 增 ----------
    def add_vectors(self, vectors: List[List[float]],
                   payloads: List[Dict[str, Any]],
                   ids: List[str]) -> bool:
        if not self.available or not self.client:
            return False
        if not vectors:
            return False
        points = []
        for vec, payload, pid in zip(vectors, payloads, ids):
            if len(vec) != self.vector_size:
                logger.warning(f"向量维度不匹配: 期望 {self.vector_size}，实际 {len(vec)}，跳过")
                continue
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
        if not points:
            return False
        try:
            self.client.upsert(collection_name=self.collection_name,
                               points=points, wait=True)
            return True
        except Exception as e:
            logger.warning(f"Qdrant upsert 失败: {e}")
            return False

    # ---------- 查 ----------
    def search_similar(self, query_vector: List[float], limit: int = 10,
                       where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """语义检索，返回 [{score, payload}]。"""
        if not self.available or not self.client:
            return []
        if len(query_vector) != self.vector_size:
            logger.warning("查询向量维度不匹配，跳过向量检索")
            return []
        qfilter = None
        if where:
            conds = [FieldCondition(key=k, match=MatchValue(value=v))
                     for k, v in where.items()
                     if isinstance(v, (str, int, float, bool))]
            if conds:
                qfilter = Filter(must=conds)
        try:
            # qdrant-client >= 1.10 移除了 .search，改用 .query_points；
            # 兼容旧版本：有 query_points 用 query_points，否则回退 search。
            if hasattr(self.client, "query_points"):
                resp = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector, query_filter=qfilter,
                    limit=limit, with_payload=True, with_vectors=False,
                )
                points = resp.points if hasattr(resp, "points") else resp
            else:
                points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector, query_filter=qfilter,
                    limit=limit, with_payload=True, with_vectors=False,
                )
            return [{"score": getattr(h, "score", 0.0),
                     "payload": (h.payload if hasattr(h, "payload") else (h.get("payload") or {}))}
                    for h in points]
        except Exception as e:
            logger.warning(f"Qdrant search 失败: {e}")
            return []

    # ---------- 删 ----------
    def delete_by_memory_ids(self, memory_ids: List[str]) -> None:
        """按 payload.memory_id 过滤删除（不依赖点 id）。"""
        if not self.available or not self.client or not memory_ids:
            return
        try:
            conds = [FieldCondition(key="memory_id", match=MatchValue(value=mid))
                     for mid in memory_ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(should=conds),
                wait=True,
            )
        except Exception as e:
            logger.warning(f"Qdrant delete 失败: {e}")

    # ---------- 统计 ----------
    def get_stats(self) -> Dict[str, Any]:
        if not self.available or not self.client:
            return {"enabled": False}
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "enabled": True,
                "name": self.collection_name,
                "points_count": info.points_count,
                "vector_size": self.vector_size,
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}


def qdrant_from_env() -> QdrantStore:
    """从环境变量构造（单例）。未配置则返回 available=False 的占位实例。"""
    return QdrantStore(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
        vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "384")),
        distance=os.getenv("QDRANT_DISTANCE", "cosine"),
        timeout=int(os.getenv("QDRANT_TIMEOUT", "30")),
    )
