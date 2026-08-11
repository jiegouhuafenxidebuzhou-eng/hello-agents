"""Agent 的 RAG 管道：文档解析 → 分块 → 向量化 → 存储 → 检索。

标准库优先 + 可选依赖优雅降级：
- 嵌入复用 `Agent.Memory.embedding.get_embedder()`（DashScope/Local/Hashing 三级降级）。
- 向量库优先用 `Agent.Memory.storage.qdrant_store.QdrantStore`；Qdrant 不可用时
  自动落到本模块的 `InMemoryVectorStore`（纯标准库 list + 余弦扫描），小知识库零依赖可跑。
- 富格式文档（PDF/Office/图片/音频）解析为可选依赖 `markitdown`，未装时仅处理文本类格式。
- LLM 查询扩展（MQE/HyDE）为可选，未配 LLM 时回退原 query。
"""

from Agent.Memory.rag.pipeline import (
    InMemoryVectorStore,
    create_rag_pipeline,
    load_and_chunk_texts,
)

__all__ = ["InMemoryVectorStore", "create_rag_pipeline", "load_and_chunk_texts"]
