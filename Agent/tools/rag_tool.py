"""RAG 工具：把检索增强生成作为 Tool 暴露给 LLM 自主调用。

LLM 通过一个统一工具对知识库执行 add_document / add_text / ask / search /
stats / clear。与 memory 工具的分工（参考第八章）：
- memory：管「我和用户经历过什么、学到了什么」（个人历史/偏好）。
- rag：管「外部文档里有什么可用知识」（知识库问答/引用）。

身份设计（方案 A，与 MemoryTool 一致）：rag_namespace / collection 在构造时
烙入，不出现在 function 参数 schema 里，LLM 无法切换到别人的命名空间。

降级链（对齐 Agent 标准库优先哲学）：
- 无 Qdrant → 内存向量库兜底（小知识库零依赖可跑）。
- 无真嵌入模型 → hashing 兜底（关键词级召回）。
- 无 LLM → ask 降级为返回检索到的原文片段（不生成答案）。
"""

import os
import time
from typing import Any, Dict, List, Optional

from Agent.LLMClient.tool import Tool
from Agent.Memory.rag.pipeline import create_rag_pipeline


class RAGTool(Tool):
    """统一 RAG 工具，供 LLM 调用。"""

    def __init__(self,
                 knowledge_base_path: str = "./knowledge_base",
                 collection_name: str = "hello_agents_rag_vectors",
                 rag_namespace: str = "default",
                 llm_client=None,
                 enable_llm: bool = True,
                 qdrant_url: Optional[str] = None,
                 qdrant_api_key: Optional[str] = None):
        # 身份烙入：namespace / collection 不进 function schema，LLM 无法伪造
        self.rag_namespace = rag_namespace
        self.collection_name = collection_name
        self.knowledge_base_path = knowledge_base_path
        os.makedirs(knowledge_base_path, exist_ok=True)

        # LLM 惰性构造：显式传入优先；未传且 enable_llm=True 时尝试从 env 构造，
        # 未配置则置 None（降级关闭 ask 的答案生成，改为返回原文片段）。
        # enable_llm=False 可强制关闭 LLM（仅检索，不生成答案；测试降级路径用）。
        self.llm_client = llm_client
        if self.llm_client is None and enable_llm:
            try:
                from Agent.LLMClient.llm_client import HelloAgentsLLM
                self.llm_client = HelloAgentsLLM()
            except Exception as e:
                print(f"[rag] LLM 未启用（ask 将降级为返回原文片段）: {e}")
                self.llm_client = None

        self._pipelines: Dict[str, Dict[str, Any]] = {}
        self._init_default_pipeline(qdrant_url, qdrant_api_key)

    def _init_default_pipeline(self, qdrant_url, qdrant_api_key) -> None:
        try:
            self._pipelines[self.rag_namespace] = create_rag_pipeline(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                collection_name=self.collection_name,
                rag_namespace=self.rag_namespace,
                llm_client=self.llm_client,
            )
            backend = self._pipelines[self.rag_namespace].get("vector_backend")
            print(f"✅ RAG 工具初始化: namespace={self.rag_namespace}, "
                  f"collection={self.collection_name}, backend={backend}")
        except Exception as e:
            print(f"❌ RAG 工具初始化失败: {e}")
            self._pipelines[self.rag_namespace] = None

    def _get_pipeline(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """获取指定命名空间管道，不存在则按构造参数自动创建。"""
        ns = namespace or self.rag_namespace
        if ns in self._pipelines and self._pipelines[ns] is not None:
            return self._pipelines[ns]
        self._pipelines[ns] = create_rag_pipeline(
            collection_name=self.collection_name,
            rag_namespace=ns,
            llm_client=self.llm_client,
        )
        return self._pipelines[ns]

    # ---------- Tool 接口 ----------
    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return (
            "检索增强生成(RAG)工具。对外部知识库执行操作。"
            "action 可选: add_document(添加文件到知识库)、add_text(添加文本到知识库)、"
            "ask(基于知识库智能问答)、search(检索相关片段)、stats(查看知识库统计)、"
            "clear(清空知识库,需confirm=true)。"
            "add_document 需要 file_path；add_text 需要 text；ask 需要 question；"
            "search 需要 query。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_document", "add_text", "ask", "search",
                             "stats", "clear"],
                    "description": "要执行的 RAG 操作",
                },
                "file_path": {"type": "string", "description": "add_document 时的文件路径"},
                "text": {"type": "string", "description": "add_text 时的文本内容"},
                "question": {"type": "string", "description": "ask 时的用户问题"},
                "query": {"type": "string", "description": "search 时的检索词"},
                "limit": {"type": "integer", "description": "返回数量，默认5"},
                "include_citations": {"type": "boolean",
                                     "description": "ask/search 是否带引用来源，默认true"},
                "confirm": {"type": "boolean",
                            "description": "clear 时需传 true 确认清空"},
            },
            "required": ["action"],
        }

    def run(self, **kwargs) -> str:
        action = kwargs.get("action")
        try:
            if action == "add_document":
                return self._add_document(**kwargs)
            if action == "add_text":
                return self._add_text(**kwargs)
            if action == "ask":
                return self._ask(**kwargs)
            if action == "search":
                return self._search(**kwargs)
            if action == "stats":
                return self._get_stats()
            if action == "clear":
                return self._clear(**kwargs)
            return ("❌ 不支持的操作: {0}。支持: "
                    "add_document/add_text/ask/search/stats/clear").format(action)
        except Exception as e:
            return f"❌ RAG 操作 '{action}' 失败: {e}"

    # ---------- 操作实现 ----------
    def _add_document(self, file_path: str = None,
                      chunk_size: int = 800, chunk_overlap: int = 100,
                      namespace: Optional[str] = None, **kwargs) -> str:
        if not file_path or not os.path.exists(file_path):
            return f"❌ 文件不存在: {file_path}"
        pipeline = self._get_pipeline(namespace)
        t0 = time.time()
        n = pipeline["add_documents"]([file_path], chunk_size=chunk_size,
                                      chunk_overlap=chunk_overlap)
        ms = int((time.time() - t0) * 1000)
        if n == 0:
            return f"⚠️ 未能从文件解析到内容: {os.path.basename(file_path)}"
        return (f"✅ 文档已添加: {os.path.basename(file_path)}\n"
                f"📊 分块数: {n}\n⏱️ {ms}ms\n"
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}")

    def _add_text(self, text: str = None, document_id: str = None,
                  chunk_size: int = 800, chunk_overlap: int = 100,
                  namespace: Optional[str] = None, **kwargs) -> str:
        if not text or not text.strip():
            return "❌ add_text 需要 text 参数"
        pipeline = self._get_pipeline(namespace)
        t0 = time.time()
        n = pipeline["add_text_chunks"](text, chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap,
                                        document_id=document_id)
        ms = int((time.time() - t0) * 1000)
        if n == 0:
            return "⚠️ 未能从文本生成有效分块"
        return (f"✅ 文本已添加\n📊 分块数: {n}\n⏱️ {ms}ms\n"
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}")

    def _search(self, query: str = None, limit: int = 5,
                include_citations: bool = True,
                namespace: Optional[str] = None, **kwargs) -> str:
        if not query or not query.strip():
            return "❌ search 需要 query 参数"
        pipeline = self._get_pipeline(namespace)
        results = pipeline["search"](query=query, top_k=limit)
        if not results:
            return f"🔍 未找到与 '{query}' 相关的内容"
        lines = [f"🔍 找到 {len(results)} 条相关片段:"]
        for i, r in enumerate(results, 1):
            payload = r.get("payload", {})
            score = float(r.get("score", 0.0))
            content = (payload.get("content") or "")[:200]
            source = payload.get("source_path", "unknown")
            lines.append(f"{i}. **{source}** (相似度: {score:.3f})")
            lines.append(f"   {content}{'...' if len(payload.get('content', '')) > 200 else ''}")
            if include_citations and payload.get("heading_path"):
                lines.append(f"   章节: {payload['heading_path']}")
        return "\n".join(lines)

    def _ask(self, question: str = None, query: str = None,
             limit: int = 5, include_citations: bool = True,
             max_chars: int = 1200, namespace: Optional[str] = None, **kwargs) -> str:
        user_q = (question or query or "").strip()
        if not user_q:
            return "❌ ask 需要 question 参数"
        pipeline = self._get_pipeline(namespace)

        # 1) 检索（LLM 可用时启用 MQE+HyDE 增强召回）
        t0 = time.time()
        if self.llm_client is not None:
            results = pipeline["search_advanced"](
                query=user_q, top_k=limit, enable_mqe=True, enable_hyde=True)
        else:
            results = pipeline["search"](query=user_q, top_k=limit)
        search_ms = int((time.time() - t0) * 1000)

        if not results:
            return (f"🤔 知识库中未找到与「{user_q}」相关的信息。\n"
                    f"💡 可先 add_document/add_text 添加相关内容，或用 stats 查看知识库状态。")

        # 2) 拼上下文
        context_parts: List[str] = []
        citations: List[Dict[str, Any]] = []
        for i, r in enumerate(results):
            payload = r.get("payload", {})
            content = (payload.get("content") or "").strip()
            if not content:
                continue
            context_parts.append(f"片段{i + 1}：{self._clean(content)}")
            if include_citations:
                citations.append({
                    "index": i + 1,
                    "source": os.path.basename(str(payload.get("source_path", "unknown"))),
                    "score": float(r.get("score", 0.0)),
                })
        context = "\n\n".join(context_parts)
        if len(context) > max_chars:
            context = self._smart_truncate(context, max_chars)

        # 3) LLM 生成答案；无 LLM 降级为返回原文片段
        if self.llm_client is None:
            return self._format_no_llm_answer(user_q, context, citations,
                                              include_citations, search_ms)

        llm_start = time.time()
        answer = self.llm_client.complete([
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",
             "content": f"请基于以下上下文回答问题。\n\n【问题】{user_q}\n\n"
                        f"【相关上下文】\n{context}\n\n"
                        f"【要求】严格基于上下文回答，不编造；信息不足请说明。"},
        ], temperature=0.3)
        llm_ms = int((time.time() - llm_start) * 1000)

        if not answer or not answer.strip():
            return self._format_no_llm_answer(user_q, context, citations,
                                              include_citations, search_ms)

        return self._format_answer(user_q, answer.strip(), citations,
                                    include_citations, search_ms, llm_ms)

    def _get_stats(self) -> str:
        pipeline = self._get_pipeline()
        stats = pipeline["get_stats"]()
        embedder = pipeline.get("embedder")
        lines = [
            "📊 RAG 知识库统计",
            f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}",
            f"📋 集合: {self.collection_name}",
            f"📦 向量后端: {pipeline.get('vector_backend', 'unknown')}",
            f"🔢 分块数: {stats.get('points_count', 0)}",
            f"🧠 嵌入模型: {getattr(embedder, 'name', 'unknown')} "
            f"(真语义: {getattr(embedder, 'available', False)})",
            f"🤖 LLM: {'启用' if self.llm_client else '未启用(ask降级)'}",
        ]
        return "\n".join(lines)

    def _clear(self, confirm: bool = False, namespace: Optional[str] = None,
               **kwargs) -> str:
        if not confirm:
            return ("⚠️ 危险操作：清空知识库将删除该命名空间所有数据！\n"
                    "请用 confirm=true 确认。")
        pipeline = self._get_pipeline(namespace)
        store = pipeline.get("store")
        ns = pipeline.get("namespace", self.rag_namespace)
        ok = False
        if store is not None:
            # InMemoryVectorStore 有 clear()；QdrantStore 走 delete_by_memory_ids 不便，
            # 这里用重建 pipeline 的方式清空（参考实现的策略）
            if hasattr(store, "clear"):
                ok = store.clear()
            else:
                ok = True  # Qdrant 侧：重建集合即清空
        # 重建该命名空间管道
        self._pipelines[ns] = create_rag_pipeline(
            collection_name=self.collection_name, rag_namespace=ns,
            llm_client=self.llm_client)
        return f"✅ 知识库已清空（命名空间: {ns}）" if ok else f"✅ 已重建（命名空间: {ns}）"

    # ---------- 便捷接口（供 Agent 在 ReAct 循环外直接调用） ----------
    def get_relevant_context(self, query: str, limit: int = 3,
                            max_chars: int = 1200,
                            namespace: Optional[str] = None) -> str:
        """为查询获取相关知识库上下文（不调 LLM，仅返回拼接片段）。"""
        if not query:
            return ""
        pipeline = self._get_pipeline(namespace)
        results = pipeline["search"](query=query, top_k=limit)
        if not results:
            return ""
        parts = [r.get("payload", {}).get("content", "") for r in results]
        ctx = "\n\n".join(p for p in parts if p)
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars] + "..."
        return ctx

    # ---------- 内部工具 ----------
    @staticmethod
    def _clean(content: str) -> str:
        content = " ".join(content.split())
        return content[:300] + "..." if len(content) > 300 else content

    @staticmethod
    def _smart_truncate(context: str, max_chars: int) -> str:
        if len(context) <= max_chars:
            return context
        truncated = context[:max_chars]
        last_break = truncated.rfind("\n\n")
        if last_break > max_chars * 0.7:
            return truncated[:last_break] + "\n\n[...更多内容被截断]"
        return truncated[:max_chars - 20] + "...[内容被截断]"

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是专业的知识助手：严格基于提供的上下文回答，不编造；"
            "信息不足时坦诚说明；回答简洁，必要时用要点。"
        )

    def _format_answer(self, question, answer, citations,
                       include_citations, search_ms, llm_ms) -> str:
        out = [f"🤖 **智能问答结果**\n", answer]
        if include_citations and citations:
            out.append("\n\n📚 **参考来源**")
            for c in citations:
                emoji = "🟢" if c["score"] > 0.8 else "🟡" if c["score"] > 0.6 else "🔵"
                out.append(f"{emoji} [{c['index']}] {c['source']} (相似度: {c['score']:.3f})")
        out.append(f"\n⚡ 检索: {search_ms}ms | 生成: {llm_ms}ms")
        return "\n".join(out)

    def _format_no_llm_answer(self, question, context, citations,
                              include_citations, search_ms) -> str:
        out = [f"📄 **检索结果**（未配置 LLM，返回原文片段）\n", context]
        if include_citations and citations:
            out.append("\n\n📚 **参考来源**")
            for c in citations:
                out.append(f"- [{c['index']}] {c['source']} (相似度: {c['score']:.3f})")
        out.append(f"\n💡 配置 LLM 后 ask 可生成自然语言答案")
        out.append(f"\n⚡ 检索: {search_ms}ms")
        return "\n".join(out)
