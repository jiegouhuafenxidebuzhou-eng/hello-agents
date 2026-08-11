"""语义记忆：本地缓存 + SQLite 权威 + Qdrant 向量 + Neo4j 图谱（四层）。

适配 Agent/Memory/manager.py 的 MemoryManager 插口接口：
    add(content, importance, metadata) -> id
    retrieve(query, limit) -> List[MemoryItem]
    forget(memory_id) -> bool
    get_stats() -> dict
    clear()

四层职责：
1. 本地缓存（内存，有界 FIFO）：热路径单条命中 + stats；写时同步写 SQLite。
2. SQLite 权威存储：持久化、跨会话、结构化过滤；memory_type='semantic' 与 episodic 分区共用表。
3. Qdrant 向量存储（可选）：语义召回；未装/未配时降级关闭。
4. Neo4j 图谱存储（可选）：实体-关系知识图谱，图遍历召回；未装/未配时降级关闭。

三元组抽取：LLM 优先（需 llm_client + 图谱可用），失败/无 LLM 回退规则
（CJK/Latin 分词取实体，两两 CO_OCCURS）。

身份隔离（方案 A）：user_id/session_id 在构造时烙入，所有读写都带 user_id 过滤。
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from Agent.Memory.memory_item import MemoryItem, make_item
from Agent.Memory.embedding import get_embedder
from Agent.Memory.storage.sqlite_store import EpisodicSQLiteStore
from Agent.Memory.storage.qdrant_store import QdrantStore, qdrant_from_env
from Agent.Memory.storage.neo4j_store import Neo4jStore, neo4j_from_env

logger = logging.getLogger(__name__)

# LLM 三元组抽取的系统提示
_TRIPLE_SYSTEM_PROMPT = (
    "你是一个信息抽取助手。从给定文本中抽取实体与三元组，只返回严格的 JSON，"
    "不要输出任何解释、markdown 代码围栏或额外文本。"
    "JSON schema：\n"
    '{"entities":[{"name":string,"type":string}],'
    '"triples":[{"subject":string,"predicate":string,"object":string}]}\n'
    "要求：\n"
    "1. name/subject/predicate/object 必须是文本中出现的原词或其规范化形式；\n"
    "2. entity type 从 PERSON/ORG/PRODUCT/LOCATION/CONCEPT/EVENT/MISC 中选；\n"
    "3. predicate 用大写英文动词或动词短语（如 WORKS_AT, LOCATED_IN, CO_OCCURS, BELONGS_TO）；\n"
    "4. subject 与 object 必须在 entities 列表中出现；\n"
    "5. 文本无明确关系时返回空 triples 数组，但 entities 仍尽量抽取；\n"
    "6. 实体数上限 10，三元组数上限 15。"
)


class SemanticMemory:
    """语义记忆，四层存储。"""

    MEMORY_TYPE = "semantic"

    def __init__(self,
                 user_id: str = "default",
                 session_id: str = "default",
                 db_path: Optional[str] = None,
                 vector_store: Optional[QdrantStore] = None,
                 use_embedder: bool = True,
                 max_cache: int = 200,
                 llm_client: Optional[Any] = None,
                 graph_store: Optional[Neo4jStore] = None,
                 enable_graph: bool = True):
        self.user_id = user_id
        self.session_id = session_id
        self.max_cache = max_cache

        # 1) 权威存储（单例，身份烙入本实例用于过滤；与 episodic 共用表，靠 memory_type 分区）
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

        # 4) 图谱存储（可选）：传入优先，否则按 env 构造
        if graph_store is None and enable_graph:
            graph_store = neo4j_from_env()
        self.graph_store = graph_store
        self.graph_ready = bool(
            enable_graph
            and self.graph_store is not None
            and getattr(self.graph_store, "available", False)
        )

        # LLM 仅在图谱可写时才有意义（否则抽取了也没地方存）；无图谱时跳过抽取
        self.llm_client = llm_client
        self._llm_ready = bool(self.graph_ready and self.llm_client is not None)

        # 本地缓存（有界 FIFO）+ 会话索引
        self._cache: List[MemoryItem] = []
        self._cache_index: Dict[str, MemoryItem] = {}
        self._sessions: Dict[str, List[str]] = {}

    # ---------- 添加 ----------
    def add(self, content: str, importance: float = 0.5,
            metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加一条语义记忆，返回 id。

        写本地缓存 + SQLite 权威层；向量层/图谱层可用时同步写入（失败不影响权威存储）。
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
                logger.warning(f"语义记忆向量入库失败 (id={item.id[:8]}): {e}")

        # 4) 图谱索引（可选）：三元组抽取 → 实体/关系入库；失败不阻断权威存储
        if self.graph_ready:
            self._index_to_graph(item)

        return item.id

    def _index_to_graph(self, item: MemoryItem) -> None:
        """抽取三元组并写入 Neo4j。整体 try/except，图谱失败不影响权威存储。"""
        try:
            entities, triples = self._extract_triples(item.content)
            # 1) 写实体（MERGE by entity_id）
            ent_name_to_id: Dict[str, str] = {}
            for e in entities:
                name = str(e.get("name", "")).strip()
                if not name:
                    continue
                eid = self._entity_id(name, self.user_id)
                ent_name_to_id[name] = eid
                etype = str(e.get("type", "MISC")).upper() or "MISC"
                self.graph_store.add_entity(
                    entity_id=eid, name=name, entity_type=etype,
                    user_id=self.user_id, memory_id=item.id,
                    importance=item.importance,
                )
            # 2) 写关系
            for t in triples:
                s_name = str(t.get("subject", "")).strip()
                o_name = str(t.get("object", "")).strip()
                if not s_name or not o_name:
                    continue
                s_id = ent_name_to_id.get(s_name) or self._entity_id(s_name, self.user_id)
                o_id = ent_name_to_id.get(o_name) or self._entity_id(o_name, self.user_id)
                if s_id == o_id:
                    continue
                self.graph_store.add_relationship(
                    from_entity_id=s_id, to_entity_id=o_id,
                    relationship_type=str(t.get("predicate", "CO_OCCURS")),
                    user_id=self.user_id, memory_id=item.id,
                    strength=item.importance, evidence=item.content[:100],
                )
        except Exception as e:
            logger.warning(f"语义记忆图谱入库失败 (id={item.id[:8]}): {e}")

    # ---------- 三元组抽取 ----------
    def _extract_triples(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """LLM 优先，失败/无 LLM 回退规则。返回 (entities, triples)。"""
        if self._llm_ready:
            try:
                return self._extract_triples_llm(content)
            except Exception as e:
                logger.warning(f"LLM 三元组抽取失败，回退规则: {e}")
        return self._extract_triples_rule(content)

    def _extract_triples_llm(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """LLM 抽取。返回 (entities, triples)，失败抛异常由上层捕获回退。"""
        messages = [
            {"role": "system", "content": _TRIPLE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        text = self.llm_client.complete(messages, temperature=0)
        if not text:
            raise RuntimeError("LLM 返回空")
        return self._parse_triple_json(text)

    @staticmethod
    def _parse_triple_json(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """解析 LLM 输出。剥离 markdown 围栏，容错；失败抛 ValueError。"""
        s = text.strip()
        # 剥离 ```json ... ``` 围栏
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s).strip()
        # 截取第一个 { 到最后一个 }（容错多余文本）
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未找到 JSON 对象")
        obj = json.loads(s[start:end + 1])
        entities = obj.get("entities") or []
        triples = obj.get("triples") or []
        # 字段校验，丢弃缺字段项
        entities = [e for e in entities
                    if isinstance(e, dict) and e.get("name") and e.get("type")]
        triples = [t for t in triples
                   if isinstance(t, dict) and t.get("subject")
                   and t.get("predicate") and t.get("object")]
        return entities, triples

    def _extract_triples_rule(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """规则兜底：CJK/Latin 分词取名词性 token 为实体，两两 CO_OCCURS。

        复用 HashingEmbedder 的分词逻辑（self._tokenize），零依赖、无 stdout 噪声。
        """
        toks = self._tokenize(content)
        ents: List[str] = []
        seen = set()
        for t in toks:
            if t.isdigit():
                continue
            # 单字符 Latin 跳过；CJK 单字保留
            if len(t) < 2 and ord(t[0]) <= 0x2E80:
                continue
            if t in seen:
                continue
            seen.add(t)
            ents.append(t)
            if len(ents) >= 8:  # 限幅，避免 O(n^2) 爆炸
                break
        entities = [{"name": n, "type": "MISC"} for n in ents]
        triples = []
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                triples.append({"subject": a, "predicate": "CO_OCCURS", "object": b})
        return entities, triples

    # ---------- 检索 ----------
    def retrieve(self, query: Optional[str] = None,
                 limit: int = 5) -> List[MemoryItem]:
        """检索语义记忆。

        - query 为空：按时间倒序取该用户最近 limit 条。
        - 有真语义向量层：向量召回 → 写 candidates。
        - 有图谱层：图遍历召回（query 分词取实体名 → 搜实体 → 找相关 → 收集 memory_id）→ 写 candidates。
        - 向量+图谱都无结果：回退 SQLite 关键词（content LIKE）。
        - 融合：combined = (vec*0.6 + graph*0.3 + recency*0.1) * (0.8 + importance*0.4)。
        """
        if not query or not query.strip():
            rows = self.store.search_memories(
                user_id=self.user_id, memory_type=self.MEMORY_TYPE, limit=limit,
            )
            return [self._touch(self._row_to_item(r)) for r in rows]

        # 收集两类候选，按 memory_id 去重
        candidates: Dict[str, Dict[str, Any]] = {}

        # (a) 向量召回
        if self._vector_ready:
            self._retrieve_vector(query, limit, candidates)

        # (b) 图谱召回
        if self.graph_ready:
            self._retrieve_graph(query, limit, candidates)

        # (c) 兜底：两类都空 → SQLite 关键词
        if not candidates:
            rows = self.store.search_memories(
                user_id=self.user_id, memory_type=self.MEMORY_TYPE,
                keyword=query, limit=limit,
            )
            return [self._touch(self._row_to_item(r)) for r in rows]

        # 融合排序：回 SQLite 取原文 + 综合评分
        now_ts = int(datetime.now().timestamp())
        scored: List[tuple] = []
        for mid, parts in candidates.items():
            row = self.store.get_memory(mid)
            if not row:
                continue
            vec = float(parts.get("vector_score", 0.0))
            graph = float(parts.get("graph_score", 0.0))
            age_days = max(0.0, (now_ts - int(row["timestamp"])) / 86400.0)
            recency = 1.0 / (1.0 + age_days)
            imp = float(row.get("importance", 0.5))
            base = vec * 0.6 + graph * 0.3 + recency * 0.1
            combined = base * (0.8 + imp * 0.4)

            item = self._row_to_item(row)
            item.metadata = {**(item.metadata or {}),
                             "relevance_score": round(combined, 4),
                             "vector_score": round(vec, 4),
                             "graph_score": round(graph, 4),
                             "recency_score": round(recency, 4)}
            scored.append((combined, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._touch(it) for _, it in scored[:limit]]

    def _retrieve_vector(self, query: str, limit: int,
                         candidates: Dict[str, Dict[str, Any]]) -> None:
        """向量召回 + 子评分，写入共享 candidates dict（不返回，交由融合决定）。"""
        try:
            qvec = self.embedder.encode(query)
        except Exception as e:
            logger.warning(f"查询嵌入失败，跳过向量召回: {e}")
            return

        hits = self.vector_store.search_similar(
            query_vector=qvec,
            limit=max(limit * 5, 20),
            where={"memory_type": self.MEMORY_TYPE, "user_id": self.user_id},
        )
        if not hits:
            return

        now_ts = int(datetime.now().timestamp())
        seen = set()
        for hit in hits:
            payload = hit.get("payload") or {}
            mid = payload.get("memory_id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            vec_score = float(hit.get("score", 0.0))
            # 子评分与 episodic 一致：(vec*0.8+recency*0.2)*(0.8+imp*0.4)，作为融合前的向量分
            ts = int(payload.get("timestamp") or now_ts)
            age_days = max(0.0, (now_ts - ts) / 86400.0)
            recency = 1.0 / (1.0 + age_days)
            imp = float(payload.get("importance", 0.5))
            sub = (vec_score * 0.8 + recency * 0.2) * (0.8 + imp * 0.4)
            candidates.setdefault(mid, {})["vector_score"] = round(sub, 4)

    def _retrieve_graph(self, query: str, limit: int,
                        candidates: Dict[str, Dict[str, Any]]) -> None:
        """图谱召回：query 分词取实体名 → 搜实体 → 找相关 → 收集 memory_id → 线性衰减打分。"""
        # 用规则分词从 query 抽实体名（快速、无 stdout 噪声；不走 LLM）
        q_tokens = self._tokenize(query)
        q_tokens = [t for t in q_tokens
                    if not t.isdigit() and (len(t) >= 2 or ord(t[0]) > 0x2E80)][:8]
        if not q_tokens:
            return

        memory_ids: List[str] = []
        seen_ids = set()
        for name in q_tokens:
            ents = self.graph_store.search_entities_by_name(
                name, user_id=self.user_id, limit=10,
            )
            for e in ents:
                eid = e.get("entity_id")
                if not eid:
                    continue
                related = self.graph_store.find_related_entities(
                    entity_id=eid, max_depth=2, limit=20, user_id=self.user_id,
                )
                for r in related:
                    for mid in r.get("rel_memory_ids", []):
                        if mid and mid not in seen_ids:
                            seen_ids.add(mid)
                            memory_ids.append(mid)
                    mid = r.get("memory_id")
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        memory_ids.append(mid)
            # 命中实体自身也带 memory_id（最新）
            mid = e.get("memory_id") if isinstance(e, dict) else None
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                memory_ids.append(mid)

        if not memory_ids:
            return
        hit_count = len(memory_ids)
        max_graph = max(limit * 5, 20)
        for i, mid in enumerate(memory_ids[:max_graph]):
            # 命中越靠前/越多 → 分数越高；简单线性衰减
            graph_score = max(0.0, 1.0 - i / max(hit_count, 1))
            candidates.setdefault(mid, {})["graph_score"] = round(graph_score, 4)

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
        """主动遗忘：缓存 + SQLite + Qdrant + Neo4j 四处清除。任一成功即返回 True。"""
        removed_cache = self._cache_remove(memory_id)
        removed_db = self.store.delete_memory(memory_id)
        if self._vector_ready:
            try:
                self.vector_store.delete_by_memory_ids([memory_id])
            except Exception as e:
                logger.warning(f"向量删除失败 (id={memory_id[:8]}): {e}")
        if self.graph_ready:
            try:
                self.graph_store.delete_by_memory_id(memory_id)
            except Exception as e:
                logger.warning(f"图谱删除失败 (id={memory_id[:8]}): {e}")
        return removed_cache or removed_db

    # ---------- 清空 ----------
    def clear(self) -> None:
        """清空本 user_id 的全部语义记忆（隔离语义，不清全局）。"""
        # 先取本用户 semantic 的全部 id（用于 Qdrant 按需删除）
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
        if self.graph_ready:
            try:
                self.graph_store.delete_by_user(self.user_id)
            except Exception as e:
                logger.warning(f"清空图谱层失败: {e}")
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
        graph_stats = (self.graph_store.get_stats()
                       if self.graph_ready else {"enabled": False})
        return {
            "memory_type": self.MEMORY_TYPE,
            "count": db_stats.get("count", 0),
            "cached": len(self._cache),
            "sessions_count": db_stats.get("sessions_count", 0),
            "avg_importance": db_stats.get("avg_importance", 0.0),
            "vector_ready": self._vector_ready,
            "graph_ready": self.graph_ready,
            "llm_ready": self._llm_ready,
            "vector_store": vec_stats,
            "graph_store": graph_stats,
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
    def _tokenize(text: str) -> List[str]:
        """CJK 按字切，Latin 按非字母数字切；小写化。

        复刻 HashingEmbedder._tokenize 的逻辑，保证与嵌入兜底分词一致。
        """
        tokens: List[str] = []
        buf = []
        for ch in text:
            if ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf).lower())
                    buf = []
                if ch.strip() and ord(ch) > 0x2E80:
                    # CJK 等非拉丁字符按单字成词
                    tokens.append(ch.lower())
        if buf:
            tokens.append("".join(buf).lower())
        return tokens

    @staticmethod
    def _entity_id(name: str, user_id: str) -> str:
        """稳定实体 id：md5(name:user_id) 的前 16 hex 位。

        跨进程稳定（不依赖 PYTHONHASHSEED），同名同用户 MERGE；不同用户隔离。
        """
        raw = f"{name.lower().strip()}:{user_id}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:16]

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
