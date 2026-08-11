"""RAG 管道实现：文档解析 → 分块 → 向量化 → 存储 → 检索。

设计要点（对齐 Agent 标准库优先 + 优雅降级哲学）：
- 嵌入：复用 `Agent.Memory.embedding.get_embedder()`，hashing 兜底也照常入库。
- 向量库：优先 `Agent.Memory.storage.qdrant_store.QdrantStore`（独立 RAG 集合，
  与记忆集合物理隔离）；Qdrant 不可用时落到本模块 `InMemoryVectorStore`（纯 stdlib）。
- 富格式解析：可选 `markitdown`，未装仅处理文本类格式，不崩溃。
- 查询扩展（MQE/HyDE）：可选 LLM，未配/失败回退原 query。
"""

import hashlib
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

from Agent.Memory.embedding import get_embedder
from Agent.Memory.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

RAG_MEMORY_TYPE = "rag_chunk"
DEFAULT_COLLECTION = "hello_agents_rag_vectors"


# ============================================================
# 纯标准库兜底向量库（Qdrant 不可用时启用）
# ============================================================

class InMemoryVectorStore:
    """纯标准库内存向量库：list 存储 + 线性余弦扫描。

    签名对齐 `QdrantStore`：`available` 标志、`add_vectors(vectors, payloads, ids)`、
    `search_similar(query_vector, limit, where)->[{"score","payload"}]`、
    `delete_by_memory_ids(ids)`、`get_stats()`。小知识库够用，进程重启即失
    （与 WorkingMemory 同为内存态，设计如此）。
    """

    def __init__(self, vector_size: int = 384):
        self.vector_size = int(vector_size)
        self.available = True
        self._points: List[Dict[str, Any]] = []  # {"id","vector","payload"}

    def add_vectors(self, vectors: List[List[float]],
                    payloads: List[Dict[str, Any]],
                    ids: List[str]) -> bool:
        if not vectors:
            return False
        for vec, payload, pid in zip(vectors, payloads, ids):
            if len(vec) != self.vector_size:
                logger.warning(f"向量维度不匹配: 期望 {self.vector_size}，实际 {len(vec)}，跳过")
                continue
            # 同 id 覆盖
            self._points = [p for p in self._points if p["id"] != pid]
            self._points.append({"id": pid, "vector": list(vec), "payload": payload or {}})
        return True

    def search_similar(self, query_vector: List[float], limit: int = 10,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if len(query_vector) != self.vector_size:
            logger.warning("查询向量维度不匹配，跳过向量检索")
            return []
        scored = []
        for p in self._points:
            if where and not _match_where(p["payload"], where):
                continue
            score = _cosine(query_vector, p["vector"])
            scored.append({"score": score, "payload": p["payload"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def delete_by_memory_ids(self, memory_ids: List[str]) -> None:
        if not memory_ids:
            return
        idset = set(memory_ids)
        self._points = [p for p in self._points
                        if p["payload"].get("memory_id") not in idset]

    def clear(self) -> bool:
        self._points.clear()
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "store_type": "in_memory",
            "name": "in_memory_rag",
            "points_count": len(self._points),
            "vector_size": self.vector_size,
        }


def _match_where(payload: Dict[str, Any], where: Dict[str, Any]) -> bool:
    """payload 需包含 where 中所有键值对（AND 语义）。"""
    for k, v in where.items():
        if payload.get(k) != v:
            return False
    return True


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ============================================================
# 文档解析 + 分块（纯标准库，markdown 感知）
# ============================================================

# 文本类格式：纯 stdlib 直读
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml",
    ".html", ".htm", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".css", ".scss", ".log", ".conf", ".ini", ".cfg", ".yaml", ".yml", ".toml",
    ".rst", ".org", ".sql", ".sh", ".bat", ".ps1",
}
# 富格式：需 markitdown
_RICH_EXTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
}

_MARKITDOWN = None  # 惰性单例
_MARKITDOWN_TRIED = False


def _get_markitdown():
    """惰性加载 markitdown（可选依赖）。未装返回 None。"""
    global _MARKITDOWN, _MARKITDOWN_TRIED
    if _MARKITDOWN_TRIED:
        return _MARKITDOWN
    _MARKITDOWN_TRIED = True
    try:
        from markitdown import MarkItDown  # noqa: F401
        _MARKITDOWN = MarkItDown()
        logger.info("[RAG] markitdown 已加载，富格式文档可用")
    except ImportError:
        logger.info("[RAG] markitdown 未安装，富格式文档将跳过；pip install markitdown 可启用")
    except Exception as e:
        logger.warning(f"[RAG] markitdown 初始化失败: {e}")
    return _MARKITDOWN


def _convert_to_markdown(path: str) -> str:
    """把任意支持格式转为 markdown 文本。纯 stdlib 优先，富格式走 markitdown。"""
    if not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower()

    # 文本类：直接读
    if ext in _TEXT_EXTS or ext == "":
        return _read_text(path)

    # 富格式：markitdown（可选）
    if ext in _RICH_EXTS:
        md = _get_markitdown()
        if md is None:
            logger.warning(f"[RAG] 跳过富格式文件（未装 markitdown）: {path}")
            return ""
        try:
            result = md.convert(path)
            text = getattr(result, "text_content", None)
            return text if isinstance(text, str) else ""
        except Exception as e:
            logger.warning(f"[RAG] markitdown 解析失败 {path}: {e}")
            return ""

    # 未知扩展名：尝试当文本读
    return _read_text(path)


def _read_text(path: str) -> str:
    """按 utf-8 读，失败回退 latin-1。"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
            or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x2A6DF)


def _approx_token_len(text: str) -> int:
    """近似 token 估计：CJK 按 1，其余按空白分词。"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk = len([t for t in text.split() if t])
    return cjk + non_cjk


def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """按 markdown 标题层级切段落，保留 heading_path。"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0

    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })

    for raw in lines:
        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip("#"))
            title = raw.lstrip("#").strip()
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            char_pos += len(raw) + 1
            continue
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1
    flush_buf(char_pos)
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None,
                       "start": 0, "end": len(text)}]
    return paragraphs


def _chunk_paragraphs(paragraphs: List[Dict],
                       chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    """把段落按 token 上限合并成 chunk，带 overlap。"""
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1
        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            chunks.append(_emit_chunk(cur))
            # overlap：保留尾部若干段落
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0
    if cur:
        chunks.append(_emit_chunk(cur))
    return chunks


def _emit_chunk(cur: List[Dict]) -> Dict:
    content = "\n\n".join(x["content"] for x in cur)
    heading_path = next((x["heading_path"] for x in reversed(cur)
                         if x.get("heading_path")), None)
    return {
        "content": content,
        "start": cur[0]["start"],
        "end": cur[-1]["end"],
        "heading_path": heading_path,
    }


def load_and_chunk_texts(paths: List[str],
                        chunk_size: int = 800,
                        chunk_overlap: int = 100,
                        namespace: Optional[str] = None,
                        source_label: str = "rag") -> List[Dict]:
    """加载多格式文档并分块。返回 chunk 列表，每项含 id/content/metadata。

    chunk_id = md5(doc_id|start|end|content_hash)，跨进程稳定，同内容去重。
    """
    chunks: List[Dict] = []
    seen_hashes = set()

    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"[RAG] 文件不存在: {path}")
            continue
        ext = os.path.splitext(path)[1].lower()
        markdown_text = _convert_to_markdown(path)
        if not markdown_text.strip():
            logger.warning(f"[RAG] 未提取到内容: {path}")
            continue

        doc_id = hashlib.md5(f"{path}|{len(markdown_text)}".encode("utf-8")).hexdigest()
        paras = _split_paragraphs_with_headings(markdown_text)
        token_chunks = _chunk_paragraphs(
            paras, chunk_tokens=max(1, chunk_size),
            overlap_tokens=max(0, chunk_overlap))

        for ch in token_chunks:
            content = ch["content"]
            norm = content.strip()
            if not norm:
                continue
            content_hash = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunk_id = hashlib.md5(
                f"{doc_id}|{ch['start']}|{ch['end']}|{content_hash}".encode("utf-8")
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source_path": path,
                    "file_ext": ext,
                    "doc_id": doc_id,
                    "start": ch["start"],
                    "end": ch["end"],
                    "content_hash": content_hash,
                    "namespace": namespace or "default",
                    "source": source_label,
                    "heading_path": ch.get("heading_path"),
                    "memory_type": RAG_MEMORY_TYPE,
                },
            })
    logger.info(f"[RAG] 分块完成: total_chunks={len(chunks)}")
    return chunks


# ============================================================
# 向量化 + 入库
# ============================================================

def _get_dimension(fallback: int = 384) -> int:
    dim = get_embedder().dimension
    if not dim or dim <= 0:
        return int(os.getenv("EMBED_DIM", str(fallback)))
    return int(dim)


def _coerce_vec(vec, dimension: int) -> List[float]:
    """把 embedder 返回值规整为 dimension 维 float 列表。"""
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    # 单条向量被包成嵌套
    if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
        vec = vec[0]
    out = [float(x) for x in vec]
    if len(out) < dimension:
        out.extend([0.0] * (dimension - len(out)))
    elif len(out) > dimension:
        out = out[:dimension]
    return out


def index_chunks(store, chunks: List[Dict], embedder,
                 rag_namespace: str = "default", batch_size: int = 32) -> int:
    """批量 embed + upsert。返回成功入库的向量数。"""
    if not chunks:
        return 0
    dimension = _get_dimension()
    vecs: List[List[float]] = []
    metas: List[Dict[str, Any]] = []
    ids: List[str] = []

    texts = [_preprocess_for_embedding(c["content"]) for c in chunks]
    for i in range(0, len(texts), batch_size):
        part = texts[i: i + batch_size]
        try:
            part_vecs = embedder.encode_batch(part) if hasattr(embedder, "encode_batch") \
                else [embedder.encode(t) for t in part]
        except Exception as e:
            logger.warning(f"[RAG] 批次 {i} 编码失败: {e}，用零向量兜底")
            part_vecs = [[0.0] * dimension for _ in part]
        for v in part_vecs:
            vecs.append(_coerce_vec(v, dimension))

    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "memory_type": RAG_MEMORY_TYPE,
            "content": ch["content"],
            "rag_namespace": rag_namespace,
            "is_rag_data": True,
            "data_source": "rag_pipeline",
        }
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])

    ok = store.add_vectors(vectors=vecs, payloads=metas, ids=ids)
    if not ok:
        logger.warning("[RAG] 向量入库失败（store.add_vectors 返回 False）")
        return 0
    logger.info(f"[RAG] 入库完成: {len(vecs)} 个向量")
    return len(vecs)


def _preprocess_for_embedding(text: str) -> str:
    """剥离 markdown 标记，提升嵌入质量。"""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def embed_query(query: str) -> List[float]:
    """单条 query embed，维度对齐 + 失败零向量兜底。"""
    dimension = _get_dimension()
    if not query:
        return [0.0] * dimension
    try:
        vec = get_embedder().encode(query)
        return _coerce_vec(vec, dimension)
    except Exception as e:
        logger.warning(f"[RAG] query 嵌入失败: {e}")
        return [0.0] * dimension


# ============================================================
# 检索
# ============================================================

def _build_filter(rag_namespace: Optional[str]) -> Dict[str, Any]:
    where = {"memory_type": RAG_MEMORY_TYPE, "is_rag_data": True}
    if rag_namespace:
        where["rag_namespace"] = rag_namespace
    return where


def search_vectors(store, query: str, top_k: int = 8,
                   rag_namespace: Optional[str] = None,
                   score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """基础向量检索。返回 [{"score","payload"}]。"""
    if not query:
        return []
    qv = embed_query(query)
    if all(x == 0.0 for x in qv):
        return []
    hits = store.search_similar(
        query_vector=qv, limit=top_k * 2, where=_build_filter(rag_namespace))
    if score_threshold is not None:
        hits = [h for h in hits if float(h.get("score", 0.0)) >= score_threshold]
    return hits[:top_k]


def _prompt_mqe(llm_client, query: str, n: int) -> List[str]:
    """多查询扩展：LLM 生成 n 个语义等价查询。失败回退 [query]。"""
    try:
        text = llm_client.complete([
            {"role": "system",
             "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。"
                        "用中文，简短，避免标点。"},
            {"role": "user",
             "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"},
        ], temperature=0.3)
        if not text:
            return [query]
        lines = [ln.strip("- \t") for ln in text.splitlines()]
        outs = [ln for ln in lines if ln]
        return outs[:n] or [query]
    except Exception:
        return [query]


def _prompt_hyde(llm_client, query: str) -> Optional[str]:
    """假设文档嵌入：LLM 生成一段可能的答案性段落用于检索。失败返 None。"""
    try:
        text = llm_client.complete([
            {"role": "system",
             "content": "根据用户问题，写一段可能的答案性段落，"
                        "用于向量检索的查询文档（不要分析过程）。"},
            {"role": "user",
             "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"},
        ], temperature=0.3)
        return text.strip() if text else None
    except Exception:
        return None


def search_vectors_expanded(store, query: str, top_k: int = 8,
                            rag_namespace: Optional[str] = None,
                            score_threshold: Optional[float] = None,
                            enable_mqe: bool = False,
                            enable_hyde: bool = False,
                            llm_client=None,
                            candidate_pool_multiplier: int = 4) -> List[Dict[str, Any]]:
    """MQE + HyDE 扩展检索：多查询召回 → memory_id 去重 → 取最高分。

    无 LLM 或 LLM 失败时自动回退为基础检索（仅原 query）。
    """
    if not query:
        return []

    expansions = [query]
    if enable_mqe and llm_client is not None:
        expansions.extend(_prompt_mqe(llm_client, query, 2))
    if enable_hyde and llm_client is not None:
        hyde = _prompt_hyde(llm_client, query)
        if hyde:
            expansions.append(hyde)

    # 去重保序
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq or [query]

    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))
    where = _build_filter(rag_namespace)

    agg: Dict[str, Dict[str, Any]] = {}
    for q in expansions:
        qv = embed_query(q)
        if all(x == 0.0 for x in qv):
            continue
        try:
            hits = store.search_similar(query_vector=qv, limit=per, where=where)
        except Exception as e:
            logger.warning(f"[RAG] 扩展检索失败: {e}")
            continue
        for h in hits:
            mid = h.get("payload", {}).get("memory_id", h.get("id"))
            s = float(h.get("score", 0.0))
            if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                agg[mid] = h

    merged = list(agg.values())
    if score_threshold is not None:
        merged = [h for h in merged if float(h.get("score", 0.0)) >= score_threshold]
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


# ============================================================
# 高层工厂
# ============================================================

def create_rag_pipeline(qdrant_url: Optional[str] = None,
                        qdrant_api_key: Optional[str] = None,
                        collection_name: str = DEFAULT_COLLECTION,
                        rag_namespace: str = "default",
                        llm_client=None) -> Dict[str, Any]:
    """构造完整 RAG 管道。

    Qdrant 可用（装了 qdrant-client 且配了 url）→ 用 QdrantStore（独立 RAG 集合）；
    否则 → 用 InMemoryVectorStore 兜底。嵌入复用全局 get_embedder()。

    返回与参考实现兼容的 dict：
        {store, namespace, add_documents, search, search_advanced, get_stats}
    """
    embedder = get_embedder()
    dimension = _get_dimension()

    # 选择向量库：优先 Qdrant，不可用则内存兜底
    url = qdrant_url or os.getenv("QDRANT_URL")
    api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    collection = collection_name or os.getenv("QDRANT_RAG_COLLECTION", DEFAULT_COLLECTION)

    store = None
    if url:
        try:
            store = QdrantStore(url=url, api_key=api_key,
                               collection_name=collection,
                               vector_size=dimension, distance="cosine")
            if not getattr(store, "available", False):
                logger.warning("[RAG] Qdrant 连接失败，回退内存向量库")
                store = None
        except Exception as e:
            logger.warning(f"[RAG] Qdrant 初始化失败，回退内存向量库: {e}")
            store = None
    if store is None:
        store = InMemoryVectorStore(vector_size=dimension)
        logger.info(f"[RAG] 使用内存向量库（dimension={dimension}），进程重启即失")

    def add_documents(file_paths: List[str], chunk_size: int = 800,
                      chunk_overlap: int = 100) -> int:
        chunks = load_and_chunk_texts(
            paths=file_paths, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, namespace=rag_namespace, source_label="rag")
        return index_chunks(store=store, chunks=chunks, embedder=embedder,
                            rag_namespace=rag_namespace)

    def add_text_chunks(text: str, chunk_size: int = 800,
                        chunk_overlap: int = 100, document_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> int:
        """直接把一段文本分块入库（不经文件）。"""
        import tempfile
        doc_id = document_id or f"text_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"
        tmp = os.path.join(tempfile.gettempdir(), f"rag_{doc_id}.md")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            chunks = load_and_chunk_texts(
                paths=[tmp], chunk_size=chunk_size,
                chunk_overlap=chunk_overlap, namespace=rag_namespace, source_label="rag")
            # 覆盖 doc_id 与 source_path，便于引用定位
            for ch in chunks:
                ch["metadata"]["source_path"] = f"text://{doc_id}"
                ch["metadata"]["doc_id"] = doc_id
                if metadata:
                    ch["metadata"].update(metadata)
            return index_chunks(store=store, chunks=chunks, embedder=embedder,
                                rag_namespace=rag_namespace)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def search(query: str, top_k: int = 8,
               score_threshold: Optional[float] = None):
        return search_vectors(store=store, query=query, top_k=top_k,
                              rag_namespace=rag_namespace,
                              score_threshold=score_threshold)

    def search_advanced(query: str, top_k: int = 8,
                        enable_mqe: bool = False, enable_hyde: bool = False,
                        score_threshold: Optional[float] = None):
        return search_vectors_expanded(store=store, query=query, top_k=top_k,
                                       rag_namespace=rag_namespace,
                                       enable_mqe=enable_mqe,
                                       enable_hyde=enable_hyde,
                                       llm_client=llm_client,
                                       score_threshold=score_threshold)

    def get_stats():
        return store.get_stats()

    return {
        "store": store,
        "namespace": rag_namespace,
        "collection_name": collection,
        "embedder": embedder,
        "llm_client": llm_client,
        "vector_backend": "qdrant" if url and getattr(store, "available", False) and not isinstance(store, InMemoryVectorStore) else "in_memory",
        "add_documents": add_documents,
        "add_text_chunks": add_text_chunks,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats,
    }
