"""SQLite 权威文档存储（纯标准库实现）。

情景记忆 / 语义记忆的持久化权威层。职责：
- 进程重启后记忆仍在（持久化）
- 按 user_id / session_id / memory_type / 时间 / 重要性 做结构化过滤
- 向量召回后回查完整原文

设计：
- 单例 per db_path：同一路径只初始化一次表结构。
- 线程本地连接（threading.local）：每线程独立 connection，避免跨线程报错。
- 通用 memories 表带 memory_type 字段，episodic/semantic 共用，为未来留接口。
"""

import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional


class EpisodicSQLiteStore:
    """SQLite 权威存储，单例 + 线程本地连接。"""

    _instances: Dict[str, "EpisodicSQLiteStore"] = {}
    _lock = threading.Lock()
    _initialized_paths: set = set()

    def __new__(cls, db_path: str = None):
        # 默认放在 Agent/Memory/memory_data/episodic.db
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data", "episodic.db"
            )
        abs_path = os.path.abspath(db_path)
        if abs_path not in cls._instances:
            with cls._lock:
                if abs_path not in cls._instances:
                    inst = super().__new__(cls)
                    cls._instances[abs_path] = inst
        return cls._instances[abs_path]

    def __init__(self, db_path: str = None):
        if hasattr(self, "_initialized"):
            return
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "memory_data", "episodic.db"
            )
        self.db_path = os.path.abspath(db_path)
        self.local = threading.local()

        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # 初始化表（每个路径只初始化一次）
        if self.db_path not in self._initialized_paths:
            self._init_database()
            self._initialized_paths.add(self.db_path)
            print(f"[OK] SQLite 权威存储初始化完成: {self.db_path}")

        self._initialized = True

    # ---------- 连接 ----------
    def _get_connection(self) -> sqlite3.Connection:
        """获取本线程的连接（惰性建立）。"""
        conn = getattr(self.local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            self.local.connection = conn
        return conn

    def _init_database(self) -> None:
        """建表 + 建索引。"""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        for idx_sql in (
            "CREATE INDEX IF NOT EXISTS idx_mem_user ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_mem_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_mem_ts ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_mem_imp ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_mem_session ON memories (session_id)",
        ):
            cur.execute(idx_sql)
        conn.commit()

    # ---------- 增 ----------
    def add_memory(self, memory_id: str, user_id: str, content: str,
                   memory_type: str, timestamp: int, importance: float,
                   session_id: Optional[str] = None,
                   properties: Optional[Dict[str, Any]] = None) -> str:
        """插入或覆盖一条记忆（INSERT OR REPLACE）。"""
        conn = self._get_connection()
        conn.execute("""
            INSERT OR REPLACE INTO memories
            (id, user_id, session_id, content, memory_type, timestamp,
             importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            memory_id, user_id, session_id, content, memory_type,
            int(timestamp), float(importance),
            json.dumps(properties, ensure_ascii=False) if properties else None,
        ))
        conn.commit()
        return memory_id

    # ---------- 查 ----------
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """按 id 取单条，返回字典或 None。"""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT id, user_id, session_id, content, memory_type, timestamp, "
            "importance, properties, created_at FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def search_memories(self,
                       user_id: Optional[str] = None,
                       memory_type: Optional[str] = None,
                       session_id: Optional[str] = None,
                       start_time: Optional[int] = None,
                       end_time: Optional[int] = None,
                       importance_threshold: Optional[float] = None,
                       keyword: Optional[str] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """结构化过滤检索。keyword 走 content LIKE（大小写不敏感）。"""
        where: List[str] = []
        params: List[Any] = []
        if user_id:
            where.append("user_id = ?"); params.append(user_id)
        if memory_type:
            where.append("memory_type = ?"); params.append(memory_type)
        if session_id:
            where.append("session_id = ?"); params.append(session_id)
        if start_time is not None:
            where.append("timestamp >= ?"); params.append(int(start_time))
        if end_time is not None:
            where.append("timestamp <= ?"); params.append(int(end_time))
        if importance_threshold is not None:
            where.append("importance >= ?"); params.append(float(importance_threshold))
        if keyword:
            where.append("LOWER(content) LIKE ?"); params.append(f"%{keyword.lower()}%")

        where_clause = ("WHERE " + " AND ".join(where)) if where else ""
        sql = (
            "SELECT id, user_id, session_id, content, memory_type, timestamp, "
            "importance, properties, created_at FROM memories "
            f"{where_clause} ORDER BY importance DESC, timestamp DESC LIMIT ?"
        )
        params.append(int(limit))
        rows = self._get_connection().execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ---------- 改 ----------
    def update_memory(self, memory_id: str, content: Optional[str] = None,
                      importance: Optional[float] = None,
                      properties: Optional[Dict[str, Any]] = None) -> bool:
        """部分更新。"""
        fields: List[str] = []
        params: List[Any] = []
        if content is not None:
            fields.append("content = ?"); params.append(content)
        if importance is not None:
            fields.append("importance = ?"); params.append(float(importance))
        if properties is not None:
            fields.append("properties = ?")
            params.append(json.dumps(properties, ensure_ascii=False))
        if not fields:
            return False
        fields.append("updated_at = datetime('now')")
        params.append(memory_id)
        cur = self._get_connection().execute(
            f"UPDATE memories SET {', '.join(fields)} WHERE id = ?", params,
        )
        self._get_connection().commit()
        return cur.rowcount > 0

    # ---------- 删 ----------
    def delete_memory(self, memory_id: str) -> bool:
        cur = self._get_connection().execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,),
        )
        self._get_connection().commit()
        return cur.rowcount > 0

    def delete_by_user(self, user_id: str, memory_type: Optional[str] = None) -> int:
        """删除某用户（可限类型）的全部记忆，返回删除条数。"""
        where = ["user_id = ?"]
        params: List[Any] = [user_id]
        if memory_type:
            where.append("memory_type = ?"); params.append(memory_type)
        cur = self._get_connection().execute(
            f"DELETE FROM memories WHERE {' AND '.join(where)}", params,
        )
        self._get_connection().commit()
        return cur.rowcount

    # ---------- 统计 ----------
    def get_stats(self, memory_type: Optional[str] = None,
                  user_id: Optional[str] = None) -> Dict[str, Any]:
        where: List[str] = []
        params: List[Any] = []
        if memory_type:
            where.append("memory_type = ?"); params.append(memory_type)
        if user_id:
            where.append("user_id = ?"); params.append(user_id)
        where_clause = ("WHERE " + " AND ".join(where)) if where else ""

        conn = self._get_connection()
        count = conn.execute(
            f"SELECT COUNT(*) AS c FROM memories {where_clause}", params,
        ).fetchone()["c"]
        avg = conn.execute(
            f"SELECT COALESCE(AVG(importance), 0) AS a FROM memories {where_clause}",
            params,
        ).fetchone()["a"]
        sessions = conn.execute(
            f"SELECT COUNT(DISTINCT session_id) AS c FROM memories {where_clause}",
            params,
        ).fetchone()["c"]
        return {
            "store_type": "sqlite",
            "db_path": self.db_path,
            "count": count,
            "sessions_count": sessions,
            "avg_importance": round(float(avg), 4),
        }

    # ---------- 工具 ----------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "timestamp": row["timestamp"],
            "importance": row["importance"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
            "created_at": row["created_at"],
        }

    def close(self) -> None:
        conn = getattr(self.local, "connection", None)
        if conn is not None:
            conn.close()
            delattr(self.local, "connection")
