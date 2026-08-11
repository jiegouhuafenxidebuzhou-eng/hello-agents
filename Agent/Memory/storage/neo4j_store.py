"""Neo4j 图存储（可选依赖，未安装/未配置自动降级）。

语义记忆的实体-关系图谱层。不可用时所有方法安全降级：
- add_entity / add_relationship → False
- find_related_entities / search_entities_by_name → []
- delete_by_memory_id / delete_by_user → 0
SemanticMemory 据此回退到 SQLite 关键词 + Qdrant 向量检索。

配置（环境变量，见 Agent/Memory/.env.example）：
- NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
- NEO4J_MAX_CONNECTION_LIFETIME / NEO4J_MAX_CONNECTION_POOL_SIZE / NEO4J_CONNECTION_TIMEOUT
"""

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 可选依赖
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False
    GraphDatabase = None
    ServiceUnavailable = None
    AuthError = None


# 关系类型白名单：大写字母开头，仅允许大写字母/数字/下划线
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _sanitize_rel_type(relationship_type: str) -> Optional[str]:
    """把任意字符串规整为合法的 Neo4j 关系类型；非法输入返回 None。

    Neo4j 关系类型必须匹配 [A-Z][A-Z0-9_]*，且无法参数化（必须拼入 Cypher）。
    规则：大写化 → 非字母数字字符替换为 _ → 校验。校验失败返回 None（调用方跳过）。
    """
    if not relationship_type:
        return None
    t = re.sub(r"[^A-Za-z0-9]+", "_", str(relationship_type)).upper()
    if t and _REL_TYPE_RE.match(t):
        return t
    return None


class Neo4jStore:
    """Neo4j 图存储封装。不可用时 available=False，方法全部安全降级。"""

    _instances: Dict[tuple, "Neo4jStore"] = {}
    _lock = threading.Lock()

    def __new__(cls, uri: Optional[str] = None, username: Optional[str] = None,
               password: Optional[str] = None, database: str = "neo4j",
               max_connection_lifetime: int = 3600,
               max_connection_pool_size: int = 50,
               connection_timeout: int = 60):
        key = (uri or "none", database or "neo4j")
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    inst = super().__new__(cls)
                    cls._instances[key] = inst
        return cls._instances[key]

    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None,
                 password: Optional[str] = None, database: str = "neo4j",
                 max_connection_lifetime: int = 3600,
                 max_connection_pool_size: int = 50,
                 connection_timeout: int = 60):
        if hasattr(self, "_initialized"):
            return

        self.uri = uri
        self.username = username
        self.password = password
        self.database = database or "neo4j"
        self.available = False
        self.driver = None

        if not _NEO4J_AVAILABLE:
            logger.info("neo4j 未安装，图谱降级关闭。pip install neo4j 可启用。")
        elif not uri:
            logger.info("未配置 NEO4J_URI，图谱降级关闭。")
        else:
            try:
                self.driver = GraphDatabase.driver(
                    uri, auth=(username or "neo4j", password or ""),
                    max_connection_lifetime=max_connection_lifetime,
                    max_connection_pool_size=max_connection_pool_size,
                    connection_acquisition_timeout=connection_timeout,
                )
                self.driver.verify_connectivity()
                self._create_constraints_indexes()
                self.available = True
                logger.info(f"✅ Neo4j 连接成功: {uri} (database={self.database})")
            except Exception as e:
                logger.warning(f"Neo4j 连接失败，图谱降级关闭: {e}")
                self.driver = None

        self._initialized = True

    # ---------- 约束 / 索引 ----------
    def _create_constraints_indexes(self) -> None:
        """建约束 + 索引。每条独立 try/except（已存在则忽略）。"""
        if not self.driver:
            return
        cypher_list = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_user_index IF NOT EXISTS FOR (e:Entity) ON (e.user_id)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        with self.driver.session(database=self.database) as session:
            for cypher in cypher_list:
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.debug(f"索引/约束可能已存在，忽略: {e}")

    # ---------- 增：实体 ----------
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   user_id: str, memory_id: str, importance: float = 0.5,
                   properties: Optional[Dict[str, Any]] = None) -> bool:
        """MERGE 实体（按 entity_id 唯一）。覆盖 name/type/user_id/memory_id/importance。"""
        if not self.available or not self.driver:
            return False
        extra = properties or {}
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MERGE (e:Entity {entity_id: $entity_id})
                    SET e.name = $name,
                        e.type = $entity_type,
                        e.user_id = $user_id,
                        e.memory_id = $memory_id,
                        e.importance = $importance,
                        e += $extra,
                        e.updated_at = datetime()
                    ON CREATE SET e.created_at = datetime()
                    RETURN e
                """, entity_id=entity_id, name=name, entity_type=entity_type,
                    user_id=user_id, memory_id=memory_id,
                    importance=float(importance), extra=extra)
                return result.peek() is not None
        except Exception as e:
            logger.warning(f"Neo4j add_entity 失败 (id={entity_id[:8]}): {e}")
            return False

    # ---------- 增：关系 ----------
    def add_relationship(self, from_entity_id: str, to_entity_id: str,
                         relationship_type: str, user_id: str, memory_id: str,
                         strength: float = 1.0, evidence: str = "",
                         properties: Optional[Dict[str, Any]] = None) -> bool:
        """MERGE 关系。关系类型经 _sanitize_rel_type 规整后拼入 Cypher（安全）。"""
        if not self.available or not self.driver:
            return False
        t = _sanitize_rel_type(relationship_type)
        if t is None:
            logger.warning(f"非法关系类型，跳过: {relationship_type}")
            return False
        extra = properties or {}
        # t 已通过白名单校验，可安全拼入；其余值全走参数
        cypher = (
            "MATCH (from:Entity {entity_id: $from_id}), (to:Entity {entity_id: $to_id}) "
            f"MERGE (from)-[r:{t}]->(to) "
            "SET r.memory_id = $memory_id, "
            "    r.user_id = $user_id, "
            "    r.strength = $strength, "
            "    r.evidence = $evidence, "
            "    r += $extra, "
            "    r.updated_at = datetime() "
            "ON CREATE SET r.created_at = datetime() "
            "RETURN r"
        )
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, from_id=from_entity_id, to_id=to_entity_id,
                                     memory_id=memory_id, user_id=user_id,
                                     strength=float(strength), evidence=evidence, extra=extra)
                return result.peek() is not None
        except Exception as e:
            logger.warning(f"Neo4j add_relationship 失败 ({t}): {e}")
            return False

    # ---------- 查：相关实体 ----------
    def find_related_entities(self, entity_id: str, max_depth: int = 2,
                              limit: int = 50, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """变长路径 (start)-[*1..D]-(related)，user 作用域。返回 [{entity_id,name,...,distance,rel_memory_ids,rel_types}]。"""
        if not self.available or not self.driver:
            return []
        depth = int(max(1, min(5, max_depth)))  # clamp 到 [1,5]，防路径爆炸 + 安全拼入
        try:
            with self.driver.session(database=self.database) as session:
                cypher = (
                    "MATCH path = (start:Entity {entity_id: $entity_id})"
                    f"-[*1..{depth}]-(related:Entity) "
                    "WHERE start.user_id = $user_id AND related.user_id = $user_id "
                    "  AND start.entity_id <> related.entity_id "
                    "RETURN DISTINCT related, "
                    "       length(path) AS distance, "
                    "       [rel IN relationships(path) | rel.memory_id] AS rel_memory_ids, "
                    "       [rel IN relationships(path) | type(rel)] AS rel_types "
                    "ORDER BY distance, related.name "
                    "LIMIT $limit"
                )
                records = session.run(cypher, entity_id=entity_id,
                                       user_id=user_id, limit=int(limit))
                results: List[Dict[str, Any]] = []
                for r in records:
                    node = r["related"]
                    results.append({
                        "entity_id": node.get("entity_id"),
                        "name": node.get("name"),
                        "type": node.get("type"),
                        "user_id": node.get("user_id"),
                        "memory_id": node.get("memory_id"),
                        "importance": node.get("importance"),
                        "distance": r["distance"],
                        "rel_memory_ids": [m for m in (r["rel_memory_ids"] or []) if m],
                        "rel_types": r["rel_types"] or [],
                    })
                return results
        except Exception as e:
            logger.warning(f"Neo4j find_related_entities 失败 (id={entity_id[:8]}): {e}")
            return []

    # ---------- 查：按名搜实体 ----------
    def search_entities_by_name(self, name_pattern: str, user_id: str,
                                limit: int = 20) -> List[Dict[str, Any]]:
        """CONTAINS 模糊匹配实体名（user 作用域）。pattern 走参数，无注入风险。"""
        if not self.available or not self.driver or not name_pattern:
            return []
        try:
            with self.driver.session(database=self.database) as session:
                records = session.run("""
                    MATCH (e:Entity)
                    WHERE e.user_id = $user_id
                      AND toLower(e.name) CONTAINS toLower($pattern)
                    RETURN e
                    ORDER BY e.name
                    LIMIT $limit
                """, user_id=user_id, pattern=name_pattern, limit=int(limit))
                return [dict(r["e"]) for r in records]
        except Exception as e:
            logger.warning(f"Neo4j search_entities_by_name 失败: {e}")
            return []

    # ---------- 删：按 memory_id ----------
    def delete_by_memory_id(self, memory_id: str) -> int:
        """删除该 memory 产生的关系，并清理由此产生的孤儿实体。返回删除关系数。"""
        if not self.available or not self.driver or not memory_id:
            return 0
        deleted = 0
        try:
            with self.driver.session(database=self.database) as session:
                # 1) 删该 memory 的所有关系
                summary = session.run(
                    "MATCH ()-[r]->() WHERE r.memory_id = $memory_id DELETE r",
                    memory_id=memory_id,
                ).consume()
                deleted = summary.counters.relationships_deleted
                # 2) 删由该 memory 标记且已无任何关系的孤儿实体
                session.run(
                    "MATCH (e:Entity) WHERE e.memory_id = $memory_id "
                    "AND NOT (e)--() DETACH DELETE e",
                    memory_id=memory_id,
                ).consume()
            return deleted
        except Exception as e:
            logger.warning(f"Neo4j delete_by_memory_id 失败 (id={memory_id[:8]}): {e}")
            return 0

    # ---------- 删：按 user ----------
    def delete_by_user(self, user_id: str) -> int:
        """删除某用户全部实体（含依附关系）。返回删除节点数。"""
        if not self.available or not self.driver or not user_id:
            return 0
        try:
            with self.driver.session(database=self.database) as session:
                summary = session.run(
                    "MATCH (e:Entity) WHERE e.user_id = $user_id DETACH DELETE e",
                    user_id=user_id,
                ).consume()
                return summary.counters.nodes_deleted
        except Exception as e:
            logger.warning(f"Neo4j delete_by_user 失败: {e}")
            return 0

    # ---------- 统计 ----------
    def get_stats(self) -> Dict[str, Any]:
        if not self.available or not self.driver:
            return {"enabled": False}
        try:
            with self.driver.session(database=self.database) as session:
                nodes = session.run(
                    "MATCH (e:Entity) RETURN count(e) AS c"
                ).single()["c"]
                rels = session.run(
                    "MATCH ()-[r]->() RETURN count(r) AS c"
                ).single()["c"]
            return {
                "enabled": True,
                "uri": self.uri,
                "database": self.database,
                "entity_nodes": nodes,
                "total_relationships": rels,
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    # ---------- 健康检查 ----------
    def health_check(self) -> bool:
        if not self.available or not self.driver:
            return False
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 AS health").consume()
            return True
        except Exception:
            return False

    def __del__(self):
        try:
            if self.driver is not None:
                self.driver.close()
        except Exception:
            pass


def neo4j_from_env() -> Neo4jStore:
    """从环境变量构造（单例）。未配置则返回 available=False 的占位实例。"""
    return Neo4jStore(
        uri=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        max_connection_lifetime=int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600")),
        max_connection_pool_size=int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")),
        connection_timeout=int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "60")),
    )
