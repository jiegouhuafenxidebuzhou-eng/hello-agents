# Hello-Agents / Agent 项目现状梳理

> 生成日期：2026-08-11　｜　最近更新：2026-08-11（RAG 工具落地）
> 范围：仓库根下的 `Agent/` 目录（不含 `MyAgent/`、`Co-creation-projects/` 等）
> 目的：盘点当前已实现的能力、架构与设计决策，标出缺口，为下一步开发提供基线。

---

## 1. 一句话定位

`Agent/` 是一个**轻量、纯标准库优先**的 LLM Agent 脚手架：把「LLM 客户端 + 工具调用（ReAct 循环）+ 记忆系统 + RAG 知识库」拼到一起。记忆层已落地**工作记忆 + 情景记忆 + 语义记忆**三类，覆盖短期/长期/知识图谱三个维度；RAG 层以 `rag_tool` 为入口提供外部文档检索增强生成，与 Memory 分工互补。`MemoryManager` 的三类后端插口全部填满。

设计取向：**能用标准库就不用重依赖**（`MemoryItem` 用 `dataclass` 而非 `pydantic`，token 估算用字符数而非 `tiktoken`，SQLite 用标准库 `sqlite3`，嵌入模型为可选依赖并带零依赖兜底），与同仓 `MyAgent/Memory/` 那套 pydantic + Qdrant/Neo4j 的重实现形成对照。

---

## 2. 目录结构

```
Agent/
├── LLMClient/                 # LLM 交互层
│   ├── llm_client.py          #   HelloAgentsLLM：OpenAI 兼容客户端（think 流式 / chat function calling）
│   ├── tool.py                #   Tool / FunctionTool / ListToolsTool / ToolRegistry
│   ├── my_agent.py            #   MyAgent：LLM + 工具的 ReAct 循环封装
│   ├── .env                   #   ⚠️ 含真实 API key（已被 .gitignore 忽略，未入库；见 §9）
│   └── __init__.py            #   空文件
├── Memory/                    # 记忆层
│   ├── memory_item.py         #   MemoryItem dataclass + make_item 工厂
│   ├── working.py             #   WorkingMemory：纯内存工作记忆（增/检索/遗忘）
│   ├── episodic.py            #   EpisodicMemory：三层情景记忆（缓存+SQLite+Qdrant）
│   ├── semantic.py            #   SemanticMemory：四层语义记忆（缓存+SQLite+Qdrant+Neo4j 图谱）
│   ├── embedding.py           #   Embedder 协议 + DashScope/Local/Hashing 三层降级
│   ├── storage/
│   │   ├── sqlite_store.py    #     EpisodicSQLiteStore：纯标准库 sqlite3 权威存储
│   │   ├── qdrant_store.py    #     QdrantStore：可选 qdrant-client 向量存储
│   │   └── neo4j_store.py     #     Neo4jStore：可选 neo4j 图存储
│   ├── rag/
│   │   └── pipeline.py        #     RAG 管道：文档解析→分块→向量化→检索（含内存兜底向量库）
│   ├── manager.py             #   MemoryManager：统一管理 working/episodic/semantic
│   ├── memory_tool.py         #   MemoryTool：把记忆系统作为 Tool 暴露给 LLM
│   ├── rag_tool.py            #   RAGTool：把外部知识库检索增强生成作为 Tool 暴露给 LLM
│   ├── .env.example           #   统一配置模板（Qdrant/Embedding/Neo4j/LLM，已消费）
│   └── __init__.py            #   空文件
├── session.py                 # 会话级 Agent 工厂（方案 A：身份烙入 + 物理隔离）
└── test/
    ├── working_memory_test.py #   工作记忆 7 项用例
    ├── isolation_test.py      #   多用户/多会话/多线程隔离 5 项用例
    ├── episodic_test.py       #   情景记忆 6 项用例（纯 SQLite 路径，零依赖）
    ├── semantic_test.py       #   语义记忆 7 项用例（纯 SQLite 路径，零依赖）
    └── rag_test.py            #   RAG 工具 6 项用例（内存向量库 + hashing embedder，零依赖）
```

---

## 3. 核心模块逐一说明

### 3.1 LLMClient 层

#### `llm_client.py` — `HelloAgentsLLM`
- 基于 `openai` SDK 的 OpenAI 兼容封装，配置从 `LLMClient/.env` 读取（`LLM_MODEL_ID` / `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_TIMEOUT`）。
- **`think(messages, temperature=0)`**：流式调用，边收边 `print`，返回拼接后的完整文本。用于「纯对话」场景。
- **`chat(messages, tools=None, temperature=0)`**：非流式，支持原生 function calling；返回 `assistant.message` 对象（含 `tool_calls`）。用于让 LLM 自主决定调工具。
- **`complete(messages, temperature=0)`**：非流式、**静默**调用，返回完整文本，失败返 `None`（不打印 stdout）。供记忆系统等不需要流式输出的内部任务使用（如语义记忆的三元组抽取），避免 `think()` 的流式噪声掩盖失败。
- 异常被 catch 后返回 `None`（不抛出），调用方需判空。

#### `tool.py` — 工具体系
- `Tool`（基类）：抽象 `name` / `description` / `parameters`（JSON Schema）/ `run(**kwargs)`；`spec()` 生成 OpenAI function calling 描述。
- `FunctionTool`：把普通可调用对象快速包成工具，参数不匹配时退化为按位置传参。
- `ListToolsTool`（内置）：让 LLM 自主查询当前注册了哪些工具——LLM 不必在 prompt 里硬记工具清单。
- `ToolRegistry`：注册 / 查询 / 执行 / 输出 specs；构造时可选挂入 `ListToolsTool`。

#### `my_agent.py` — `MyAgent`
ReAct 循环的封装：
1. system + user 消息开局，带上 `registry.specs()`。
2. 每步调 `llm_client.chat`：
   - 无 `tool_calls` → 直接返回 `content` 作为最终答案。
   - 有 `tool_calls` → 把 assistant 消息追加进历史，依次执行工具，把结果以 `role=tool` 回填，进入下一步。
3. 达 `max_steps` 仍未收敛则终止返回 `None`。

### 3.2 Memory 层

#### `memory_item.py` — `MemoryItem`
- 纯 `dataclass`，字段：`content / id(uuid) / timestamp / importance(0-1) / metadata / last_accessed / user_id / session_id`。
- `touch(now)` 更新最近访问时间，供遗忘策略参考。
- `make_item()` 快捷工厂。

#### `working.py` — `WorkingMemory`（短期记忆后端）
三大能力：
- **添加 `add(content, importance, metadata)`**：先惰性清过期 → 建项 → 进列表/索引 → 按上限驱逐。
- **检索 `retrieve(query, limit)`**：时间近度倒序 + 可选关键词过滤（大小写不敏感）；命中项 `touch` 更新访问时间。
- **遗忘**，混合三策略：
  1. **TTL 过期**：超 `ttl_minutes` 的项在 `add`/`retrieve`/`get_stats` 时惰性清除（`_expire_old`）。
  2. **容量 / Token 驱逐**：超 `max_capacity` 或 `max_tokens` 时，按 `优先级 = 重要性 × 时间衰减` 驱逐最低者（`_evict` + `_priority` + `_time_decay`）。
  3. **主动遗忘 `forget(id)`**：显式删一条；另有 `cleanup()` 主动清过期、`clear()` 全清。

配置项：`max_capacity=20 / max_tokens=2000 / ttl_minutes=120 / decay_hours=6.0 / min_decay=0.1`，token 估算默认按字符数（`_default_token_counter`，可换 `tiktoken`）。

#### `episodic.py` — `EpisodicMemory`（长期记忆后端，三层架构）
适配 `MemoryManager` 插口（`add/retrieve/forget/get_stats/clear`），三层职责：
1. **本地缓存**（内存，有界 FIFO `max_cache=200`）：热路径单条命中 + stats；写时同步写 SQLite。修了参考实现里无界缓存的内存泄漏。
2. **SQLite 权威存储**（`EpisodicSQLiteStore`）：持久化、跨会话、结构化过滤；向量召回后回查完整原文。
3. **Qdrant 向量存储**（`QdrantStore`，可选）：语义召回。

检索逻辑：
- query 为空 → SQLite 按时间倒序取该用户最近 limit 条。
- 装了真语义嵌入（非 hashing 兜底）且 Qdrant 可用 → 向量召回 + 回 SQLite 取原文 + 综合评分 `(向量*0.8 + 近因*0.2) × (0.8 + 重要性*0.4)`。
- 否则 / 向量无结果 → 回退 SQLite `content LIKE` 关键词检索。
- `clear()` 只清本 `user_id` 的 episodic（隔离语义，不清全局）。
- `vector_ready` 标志：仅当 Qdrant 可用 **且** embedder 是真语义（非 hashing）时为 True——hashing 兜底不走向量路径（与关键词检索同质，无意义）。

#### `semantic.py` — `SemanticMemory`（知识图谱后端，四层架构）
适配 `MemoryManager` 插口（`add/retrieve/forget/get_stats/clear`），四层职责：
1. **本地缓存**（内存，有界 FIFO `max_cache=200`）：热路径单条命中 + stats；写时同步写 SQLite。
2. **SQLite 权威存储**（`EpisodicSQLiteStore`，`memory_type="semantic"`）：与 episodic **共用同一张 `memories` 表靠 `memory_type` 分区**，不新建表。
3. **Qdrant 向量存储**（`QdrantStore`，可选）：语义召回，与 episodic 共用集合，靠 payload `memory_type` 过滤。
4. **Neo4j 图谱存储**（`Neo4jStore`，可选）：实体-关系知识图谱，图遍历召回——这是语义记忆区别于情景记忆的核心能力。

**三元组抽取**（`_extract_triples`，LLM 优先 + 规则兜底）：
- **LLM 路径**（`_llm_ready = graph_ready AND llm_client`）：用 `_TRIPLE_SYSTEM_PROMPT` 让 LLM 返回严格 JSON `{entities:[{name,type}], triples:[{subject,predicate,object}]}`，predicate 限大写动词、entity type 限枚举。调 `llm_client.complete()`（静默）。`_parse_triple_json` 剥 markdown 围栏 + 截最外层 `{...}` + 字段校验丢残项。
- **规则兜底**：复刻 `HashingEmbedder._tokenize` 的 CJK/Latin 分词，取 ≤8 实体两两 `CO_OCCURS`，零依赖、无 stdout 噪声。任何 LLM 失败/异常/无 LLM 时自动回退。
- 实体 id 用 `md5(name:user_id)[:16]`——**跨进程稳定**（修正参考实现 `hash()` 被 `PYTHONHASHSEED` 盐化导致 MERGE 失效的 bug）+ 跨用户隔离。

检索逻辑：
- query 为空 → SQLite 按时间倒序取该用户最近 limit 条。
- 有向量层 → 向量召回写 candidates（子评分 `(向量*0.8+近因*0.2)×(0.8+重要性*0.4)`）。
- 有图谱层 → 图谱召回：query 分词取实体名 → `search_entities_by_name` → `find_related_entities` 收集路径上的 `memory_id` → 线性衰减打 `graph_score` 写 candidates。
- 两类都空 → 回退 SQLite `content LIKE` 关键词检索。
- **融合评分** `combined = (向量*0.6 + 图谱*0.3 + 近因*0.1) × (0.8 + 重要性*0.4)`，回 SQLite 取原文，metadata 带 `relevance/vector/graph/recency` 分数。
- 图谱写入/删除失败只 log，不阻断权威存储（`_index_to_graph` 整体 try/except）。
- `graph_ready`/`llm_ready` 标志：无 Neo4j 或无 LLM 时相应关闭，自动降级。

#### `embedding.py` — 嵌入抽象 + 降级链
`Embedder` 协议（`encode/dimension/available`）+ 三实现按优先级回退：
1. `DashScopeEmbedder`：OpenAI 兼容 REST（需 `requests` + `EMBED_API_KEY/EMBED_BASE_URL`）——真语义。
2. `LocalEmbedder`：`sentence-transformers`——真语义。
3. `HashingEmbedder`：**纯标准库零依赖**，hashing trick 投影到固定维度（`EMBED_DIM`，默认 384），L2 归一化，效果≈关键词重叠余弦——仅作兜底。
`get_embedder()` 线程安全单例；`embedding_available()` 判断是否真语义。未装任何重依赖时自动落到 hashing，情景记忆仍可用。

#### `storage/sqlite_store.py` — `EpisodicSQLiteStore`
纯标准库 `sqlite3`，**单例 per db_path + 线程本地连接**。通用 `memories` 表（带 `memory_type`，episodic/semantic 共用，为未来留接口），方法：`add_memory / get_memory / search_memories（结构化+keyword LIKE）/ update_memory / delete_memory / delete_by_user / get_stats`。默认 `Agent/Memory/memory_data/episodic.db`。

#### `storage/qdrant_store.py` — `QdrantStore`
可选依赖 `qdrant-client`，**单例 per (url,collection)**。`available=False` 时所有方法安全降级（`add_vectors→False`、`search_similar→[]`、`delete_by_memory_ids→无操作`）。`qdrant_from_env()` 从 `QDRANT_*` 环境变量构造。按 payload `memory_id` 删除（不依赖点 id）。

#### `storage/neo4j_store.py` — `Neo4jStore`
可选依赖 `neo4j`，**单例 per (uri,database)**。`available=False` 时所有方法安全降级（`add_entity/add_relationship→False`、`find_related_entities/search_entities_by_name→[]`、`delete_by_*→0`），**构造不抛异常**（与参考实现 `MyAgent/Memory/storage/neo4j_store.py` 的硬失败不同）。`neo4j_from_env()` 从 `NEO4J_*` 环境变量构造。

图 Schema：`:Entity` 节点（`entity_id` 唯一 PK + `name/type/user_id/memory_id/importance`），动态类型关系（带 `memory_id/user_id/strength/evidence`）。方法：`add_entity`（MERGE）、`add_relationship`（MERGE）、`find_related_entities`（变长路径 `[*1..D]`，user 作用域，返回 `rel_memory_ids` 收集候选记忆）、`search_entities_by_name`（`CONTAINS`）、`delete_by_memory_id`（先删关系再删孤儿）、`delete_by_user`、`get_stats`、`health_check`。

**安全要点**：Neo4j 关系类型与变长深度无法参数化，必须拼入 Cypher。`_sanitize_rel_type()` 把任意谓词规整为合法 `[A-Z][A-Z0-9_]*`（大写化 + 非字母数字→`_` + 校验，失败返 None 跳过），`max_depth` clamp 到 `[1,5]`，其余值全走参数——规避参考实现里 f-string 拼接的 Cypher 注入风险。

#### `manager.py` — `MemoryManager`
- 统一接口：`add / retrieve / forget / stats / clear`，参数带 `memory_type`。
- `backends = {"working": WorkingMemory, "episodic": <可启用>, "semantic": <可启用>}`。
  - **`episodic` 默认 `None`**，显式传实例或 `enable_episodic=True` 时按本会话身份构造 `EpisodicMemory`。
  - **`semantic` 默认 `None`**，显式传实例或 `enable_semantic=True` 时按本会话身份构造 `SemanticMemory`；`enable_llm=True` 时顺带构造 `HelloAgentsLLM()`（未配 env 抛 `ValueError`→try/except 置 `None`，不阻断语义记忆，仅降级为规则抽取）；`db_path` 默认复用 `episodic_db_path`（共享 SQLite 单例 + `memory_type` 分区），需物理隔离则传 `semantic_db_path`。
  - `retrieve(memory_type="all")` 可跨已启用类型检索并按时间倒序合并。
- 三类后端插口全部填满；任何后端只要实现 `add/retrieve/forget/get_stats` 即可挂上，`MemoryTool` 无需改动。

#### `memory_tool.py` — `MemoryTool`
- 把 `MemoryManager` 包成单一 Tool 暴露给 LLM，**一个工具搞定增/检索/遗忘/统计**（不为每类记忆单独建工具）。
- 参数：`action(add/retrieve/forget/stats)` + `memory_type(working/episodic/semantic/all)` + `content/query/memory_id/limit/importance`。
- **安全设计（方案 A）**：`user_id` / `session_id` 在 `MemoryTool` 构造时烙入，**不出现在 function 参数 schema 里**，因此 LLM 无法伪造身份——即使调用时硬塞 `user_id="bob"` 也会被忽略。

#### `rag_tool.py` + `rag/pipeline.py` — RAG 工具（外部知识库检索增强生成）
与 Memory 的分工（参考第八章 §11）：**Memory 管「我和用户经历过什么」**（个人历史/偏好，按 user_id 隔离）；**RAG 管「外部文档里有什么可用知识」**（知识库问答/引用，按 `rag_namespace` 隔离，全局共享不绑 user_id）。

`RAGTool(Tool)` 把知识库作为单一 Tool 暴露给 LLM，参数 `action(add_document/add_text/ask/search/stats/clear)` + `file_path/text/question/query/limit/include_citations/confirm`。**身份防伪造（方案 A）**：`rag_namespace`/collection 在构造时烙入，不进 function schema，LLM 无法切换到别人的命名空间。

`rag/pipeline.py` 实现 RAG 全流程，复用 Agent 已有基础设施（不跨目录 import MyAgent）：
- **文档解析（标准库优先）**：文本类格式（txt/md/csv/json/py/yaml 等）纯 stdlib `open().read()`；富格式（PDF/Office/图片/音频）走**可选依赖** `markitdown`，未装则跳过该文件并提示，不崩溃。分块移植参考的 markdown 感知分块（`_split_paragraphs_with_headings` + `_chunk_paragraphs` + `_approx_token_len`），纯 stdlib，保留 `heading_path`/`start`/`end`/`doc_id`/`content_hash` 元数据 + 跨文件内容去重。
- **向量化**：复用 `Agent.Memory.embedding.get_embedder()`（DashScope/Local/Hashing 三级降级）。RAG 不做 `vector_ready` 门控——hashing embedder 也照常入库（关键词级召回也是有效 RAG）。
- **向量库**：优先 `QdrantStore`（独立集合 `hello_agents_rag_vectors`，env `QDRANT_RAG_COLLECTION`，与记忆集合物理隔离避免污染 HNSW）；**Qdrant 不可用时自动落到 `InMemoryVectorStore`**（纯 stdlib list + 余弦扫描），小知识库零依赖可跑，进程重启即失。签名对齐 `QdrantStore`（`add_vectors(vectors,payloads,ids)` / `search_similar→[{"score","payload"}]`）。
- **检索**：基础 `search_vectors`（where 过滤 `memory_type`+`rag_namespace`，score_threshold 后置过滤）；高级 `search_vectors_expanded` 支持 **MQE 多查询扩展 + HyDE 假设文档嵌入**（LLM 可用时走 `complete()`，失败回退原 query），跨扩展按 `memory_id` 去重取最高分。
- **ask 流程**：检索→拼上下文（智能截断 `max_chars`）→`HelloAgentsLLM.complete()` 生成答案→带引用来源格式化。**无 LLM 时降级为返回检索到的原文片段**（不编造）。
- **裁剪**：未移植参考里的 cross-encoder rerank / graph signals / neighbor expansion（Agent 无 RAG 侧 Neo4j，且这些是重依赖），rerank 留作后续扩展点。

降级链（对齐 Agent 哲学，逐层叠加无需改代码）：无 Qdrant→内存向量库；无真嵌入→hashing 关键词级；无 LLM→ask 返回原文片段。`pip install openai python-dotenv` 即可跑通；配齐 Qdrant+真 embedder 升级语义召回，再加 LLM 升级自然语言答案。

### 3.3 `session.py` — 会话级工厂

`build_session_agent(llm_client, user_id, session_id, extra_tools, max_steps, enable_episodic=False, episodic_db_path=None, enable_rag=False, rag_namespace="default")`：
- 每个会话构造**全新的** `WorkingMemory` → `MemoryManager` → `MemoryTool` → `MyAgent`，身份烙入。
- `enable_episodic=True` 时额外构造一个 `EpisodicMemory`（身份烙入）挂进 manager；SQLite/Qdrant 虽全局共享单例，但所有读写带 `user_id` 过滤，跨会话可召回长期记忆而互不串读。
- `enable_rag=True` 时额外构造 `RAGTool`（`rag_namespace` 烙入，llm_client 复用全局实例）挂进 tools；LLM 可在 ReAct 循环里自主调 `rag` 工具检索外部知识库。无 Qdrant 时降级内存向量库，无 LLM 时 ask 降级返回原文片段。
- `HelloAgentsLLM` 是无状态 HTTP 客户端，全局共享、不重复建。
- **隔离方式**：不同会话 = 不同 Python 对象 → 物理隔离，无需过滤、无需加锁。

---

## 4. 数据流

### 4.1 一次带记忆的对话
```
用户问题
   │
   ▼
MyAgent.run(question)
   │  messages = [system, user]
   │  tool_specs = registry.specs()  ← 含 memory 工具 + list_tools + 业务工具
   ▼
循环（最多 max_steps 步）：
   HelloAgentsLLM.chat(messages, tools)  ──► assistant.message
        │
        ├─ 无 tool_calls ──► 返回 content（最终答案）
        │
        └─ 有 tool_calls ──► 逐个执行：
              registry.execute(name, args)
                └─ 若是 memory 工具 ──► MemoryManager.add/retrieve/forget/stats
                                          ├─ WorkingMemory（纯内存）
                                          ├─ EpisodicMemory（缓存+SQLite+Qdrant，可选）
                                          └─ SemanticMemory（缓存+SQLite+Qdrant+Neo4j，可选）
              结果以 role=tool 回填 messages ──► 进入下一步
```

### 4.2 工作记忆的生命周期
```
add ──► _expire_old（清 TTL 过期）──► 入列表/索引 ──► _evict（超容量按优先级驱逐）
                                                                            │
retrieve ──► _expire_old ──► 关键词过滤 ──► 时间倒序 ──► touch(更新访问) ──► 返回
                                                                            │
forget(id) ──► 从列表/索引删除 ──► 扣减 current_tokens
```

### 4.3 情景记忆的生命周期（三层）
```
add ──► 本地缓存(FIFO 有界) ──► SQLite(权威,INSERT OR REPLACE) ──► [向量层可用?] encode+upsert Qdrant
                                                                                      (失败仅 log,不影响权威)
retrieve(query):
  ├─ query 空 ─────► SQLite 按时间倒序取本用户最近 limit
  ├─ vector_ready ─► Qdrant 语义召回 ──► 回 SQLite 取原文 ──► 综合评分(向量.8+近因.2)×重要性权重 ──► 取 limit
  └─ 回退 ─────────► SQLite content LIKE 关键词检索
forget(id) ──► 缓存移除 ──► SQLite delete ──► [向量层可用?] Qdrant 按 memory_id 删
clear() ──► SQLite delete by (user_id, episodic) ──► Qdrant 删本用户 episodic 向量 ──► 清缓存
```

### 4.4 语义记忆的生命周期（四层）
```
add ──► 本地缓存(FIFO 有界) ──► SQLite(权威, memory_type='semantic') ──► [向量层可用?] encode+upsert Qdrant
                                                                              (失败仅 log,不影响权威)
                                          └─► [图谱层可用?] _index_to_graph:
                                                 _extract_triples ──► [LLM 可用?] LLM 抽 JSON 三元组
                                                                                       (失败/无 LLM ↓)
                                                                  回退规则: CJK 分词取实体, 两两 CO_OCCURS
                                                 → add_entity (MERGE by entity_id) + add_relationship
                                                 (整步 try/except, 图谱失败不阻断权威存储)
retrieve(query):
  ├─ query 空 ─────────────► SQLite 按时间倒序取本用户最近 limit
  ├─ vector_ready ────────► Qdrant 向量召回 ─► 写 candidates (子评分 向量.8+近因.2 ×重要性权重)
  ├─ graph_ready ─────────► query 分词取实体名 → 搜实体 → find_related_entities 收集 memory_id
  │                          → 线性衰减打 graph_score ─► 写 candidates
  ├─ 两类都空 ─────────────► 回退 SQLite content LIKE 关键词
  └─ 融合: combined=(向量.6+图谱.3+近因.1)×(0.8+重要性.4) ─► 回 SQLite 取原文 ─► 取 limit
forget(id) ──► 缓存移除 ──► SQLite delete ──► [向量层] Qdrant 删 ──► [图谱层] delete_by_memory_id (先删关系再删孤儿)
clear() ──► SQLite delete by (user_id, semantic) ──► Qdrant 删本用户 semantic 向量 ──► Neo4j delete_by_user ──► 清缓存
```

### 4.5 RAG 知识库的生命周期（与 Memory 分工：外部文档 vs 个人历史）
```
add_document(file_path) ──► _convert_to_markdown [文本类:stdlib / 富格式:可选markitdown]
                          ──► _split_paragraphs_with_headings ──► _chunk_paragraphs(overlap)
                          ──► 内容去重(content_hash) ──► index_chunks:
                                 embedder.encode_batch ──► store.add_vectors(vectors,payloads,ids)
                                 payload: memory_type=rag_chunk / rag_namespace / source_path / doc_id / content / is_rag_data
                          (Qdrant 不可用 → InMemoryVectorStore 兜底; hashing embedder 也照常入库)

ask(question):
  ├─ [LLM 可用] search_advanced: MQE 多查询 + HyDE 假设文档 ──► 多路向量召回 ──► memory_id 去重取最高分
  └─ [无 LLM ] search: 单 query 向量召回
  ──► 拼上下文(智能截断 max_chars) ──► [LLM 可用] complete() 生成自然语言答案 + 引用来源
                                       [无 LLM ] 降级返回检索到的原文片段(不编造)
search(query) ──► 向量召回 ──► where 过滤(memory_type+rag_namespace) ──► score_threshold 后置 ──► 返回片段+相似度
clear(confirm=true) ──► 重建该 namespace 管道(内存库清空 / Qdrant 侧重建集合)
```

> RAG 与 Memory 检索路由（参考第八章 §15）：涉及用户历史/偏好 → memory 工具；涉及外部文档/手册/知识库 → rag 工具；复杂助手常同时调用二者。

---

## 5. 配置

`LLMClient/.env`（实际运行读取）：
```
LLM_MODEL_ID="deepseek-v4-pro"
LLM_API_KEY="sk-..."          # 真实密钥（被 .gitignore 忽略，未入库；见 §9）
LLM_BASE_URL="https://api.deepseek.com"
SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY"
```

`Memory/.env.example`（情景/语义记忆已消费其中 Qdrant/Embedding/Neo4j/LLM 变量；复制为 `Agent/Memory/.env` 后生效）：
- LLM 四件套、Tavily/SerpApi。
- Qdrant：`QDRANT_URL/QDRANT_API_KEY/QDRANT_COLLECTION/QDRANT_VECTOR_SIZE/QDRANT_DISTANCE`（不配则向量层关闭，降级关键词）。
- Embedding：`EMBED_MODEL_TYPE(dashscope|local|hashing，默认 hashing)/EMBED_MODEL_NAME/EMBED_API_KEY/EMBED_BASE_URL/EMBED_DIM`（不配真模型则 hashing 兜底，仍可用）。
- Neo4j：`NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD/NEO4J_DATABASE` 等（不配则图谱层关闭，语义记忆降级为向量+关键词检索）。语义记忆的三元组抽取默认走规则兜底；配齐 LLM 后自动升级为 LLM 抽取。

> 情景/语义记忆的「最小可跑」配置：**零额外环境变量**——默认 hashing embedder + 无 Qdrant + 无 Neo4j + 无 LLM → 纯 SQLite + 关键词检索即可工作。配齐 Qdrant + 真 embedder 后升级语义召回，再加 Neo4j 升级图遍历召回，再加 LLM 升级三元组抽取——逐层叠加，无需改代码。

---

## 6. 测试

| 文件 | 覆盖 | 运行方式（仓库根目录下） |
|---|---|---|
| `test/working_memory_test.py` | 增/检索/关键词过滤/主动遗忘/TTL 过期/容量驱逐/Token 驱逐/limit | `python -m Agent.test.working_memory_test` |
| `test/isolation_test.py` | 跨用户隔离/同用户跨会话隔离/身份烙入/LLM 无法伪造身份/8 用户并发无串读 | `python -m Agent.test.isolation_test` |
| `test/episodic_test.py` | 增/关键词检索/遗忘/跨用户隔离/跨会话长期召回/降级可用（纯 SQLite，零依赖） | `python -m Agent.test.episodic_test` |
| `test/semantic_test.py` | 增/关键词检索/遗忘/跨用户隔离/跨会话长期召回/降级可用/与情景记忆分区隔离（纯 SQLite，零依赖） | `python -m Agent.test.semantic_test` |
| `test/rag_test.py` | 添加文本/搜索/ask降级/分块去重/命名空间隔离/统计+清空/降级可用（内存向量库+hashing，零依赖） | `python -m Agent.test.rag_test` |

注：测试通过 `__main__` 直接跑函数，非 `pytest` 断言框架；`isolation_test` / `episodic_test` / `semantic_test` 里用 `assert`，可用 `python -m` 执行。导入路径要求从仓库根运行。`episodic_test` / `semantic_test` 用临时 db 跑完自动清理。LLM + Neo4j 真路径需配 `.env`，无法单测，手动验证步骤见测试文件 docstring。

---

## 7. 设计决策清单（为什么这么做）

1. **标准库优先**：`MemoryItem` 用 `dataclass`，token 用字符数——工作记忆只是粗略预算，避免引入 `pydantic`/`tiktoken` 重依赖。
2. **混合遗忘策略**：TTL（时间维）+ 容量/Token 驱逐（资源维）+ 主动遗忘（语义维），三层叠加，覆盖短期记忆的典型失效场景。
3. **惰性清理**：过期项不在后台定时扫，而是在 `add`/`retrieve`/`get_stats` 入口顺带清，省线程、省锁。
4. **统一记忆工具**：不为 working/episodic/semantic 各建工具，一个 `memory` 工具 + `action/memory_type` 参数即可——记忆本质就是「增/检索/遗忘」三件事。
5. **方案 A 身份烙入**：`user_id/session_id` 在服务端构造时烙入 MemoryTool/Manager/WorkingMemory，function schema 里不暴露这两个字段 → LLM 物理上无法伪造身份。
6. **会话级物理隔离**：每会话一套独立 Python 对象，不同会话天然不串读，比「全局表 + 按 user_id 过滤」更简单也更安全。
7. **情景记忆三层 + 降级**（新增）：本地缓存 + SQLite 权威 + Qdrant 向量，后两层均为可选依赖；未装 `qdrant-client`/未配真嵌入时自动退化为「SQLite + 关键词」，`pip install openai python-dotenv` 即可跑通。配齐后无缝升级语义召回。
8. **真语义才走向量**（新增）：`vector_ready` 要求 Qdrant 可用 **且** embedder 非 hashing 兜底——hashing 与关键词检索同质，走向量无意义。避免「装了 Qdrant 但没真模型」时做无效计算。
9. **SQLite 自建而非复用**（新增）：参考 `MyAgent/Memory/storage/document_store.py` 但在 `Agent/` 内用标准库重写精简版，避免跨目录耦合与 pydantic 重依赖，贴合本项目标准库优先哲学。规避了参考实现里的 bug（`from venv import logger`、`timedelta.hours`、无界缓存等）。
10. **语义记忆四层 + 全栈降级**（新增）：缓存 + SQLite 权威 + Qdrant 向量 + Neo4j 图谱，后三层均为可选依赖；未装 `neo4j`/`qdrant-client`/未配真嵌入/未配 LLM 时逐层自动退化，最终退到「SQLite + 关键词」仍可用。与情景记忆降级哲学一致。
11. **LLM 三元组抽取 + 规则兜底**（新增）：LLM 抽真语义三元组（`WORKS_AT`/`LOCATED_IN` 等），比参考实现的 spaCy NER + 两两 `CO_OCCURS` 更强；但 spaCy 重依赖且语言模型大，本项目改用「LLM 优先 + 复刻 HashingEmbedder 分词的规则兜底」，零依赖仍能建图。
12. **实体 id 跨进程稳定**（新增）：`md5(name:user_id)` 而非参考实现的 `hash(name)`——后者被 `PYTHONHASHSEED` 盐化，跨进程/重启后同实体 MERGE 失效、图谱碎片化。
13. **Cypher 注入防御**（新增）：关系类型无法参数化，`_sanitize_rel_type` 白名单校验 + `max_depth` clamp + 其余值全参数化，规避参考实现 f-string 拼 Cypher 的注入风险。
14. **语义与情景共享 SQLite**（新增）：同一张 `memories` 表靠 `memory_type` 分区（semantic_test 7 验证不串读），单例 store 复用，不新建表——贴合标准库优先与极简取向。需物理隔离可传 `semantic_db_path`。

---

## 8. 现状能力清单 & 缺口

### ✅ 已实现
- LLM 客户端（流式 + function calling + 静默 `complete()`）
- 工具体系（注册/查询/执行 + 让 LLM 自查工具清单的内置工具）
- ReAct Agent 循环（LLM 自主调工具 → 回填 → 收敛）
- 工作记忆全功能（增/检索/遗忘，TTL + 驱逐 + 主动遗忘）
- **情景记忆全功能**（三层：本地缓存 + SQLite 权威 + Qdrant 向量；增/检索/遗忘/清空；跨会话召回；可选依赖+降级）
- **语义记忆全功能**（四层：本地缓存 + SQLite 权威 + Qdrant 向量 + Neo4j 图谱；LLM 三元组抽取 + 规则兜底；向量+图谱融合检索；增/检索/遗忘/清空；跨会话召回；全栈降级）
- **Neo4j 图存储**（可选依赖 neo4j，优雅降级，Cypher 注入防御）
- **Embedding 抽象 + 三层降级链**（DashScope/Local/Hashing 零依赖兜底）
- 记忆作为工具暴露给 LLM（统一 memory 工具，working/episodic/semantic 同一个工具切换）
- **RAG 工具**（外部知识库检索增强生成；rag_tool 统一入口，add_document/add_text/ask/search/stats/clear；markdown 感知分块；MQE+HyDE 扩展检索；内存向量库兜底；无 LLM 时 ask 降级返回原文片段；与 Memory 分工互补）
- 多用户/多会话/多线程隔离（含身份防伪造，三类记忆均隔离 + RAG 命名空间隔离）
- 工作记忆 7 项 + 隔离 5 项 + 情景记忆 6 项 + 语义记忆 7 项 + RAG 6 项测试

### ❌ 缺口（按优先级）
| 优先级 | 缺口 | 说明 | 可复用资源 |
|---|---|---|---|
| 中 | **记忆整合（consolidation）** | 工作记忆高重要性项 → 情景/语义记忆的沉淀机制缺失（参考 MyAgent 的 `consolidate_memories`） | `MyAgent/Memory/manager.py` |
| 中 | **业务工具示例** | `MyAgent` 已能挂任意 Tool，但仓库内 `Agent/` 没有现成业务工具（搜索/计算等） | 同仓多项目有实现可借鉴 |
| 中 | **语义记忆自动沉淀** | 当前需 LLM/Agent 显式 `add(memory_type="semantic")`；缺从对话/情景记忆自动抽取事实沉淀的机制 | `MyAgent/Memory/manager.py` 的 auto-classify |
| 低 | **token 精确化** | 工作记忆 token 用字符数估算偏粗，中文场景偏差更大 | 接 `tiktoken` 即可 |
| 低 | **RAG 重排序** | 当前 RAG 仅向量召回 + MQE/HyDE 扩展，未接 cross-encoder 重排序，专业领域召回精度可提升 | 参考实现 `rerank_with_cross_encoder` |
| 低 | **工作记忆持久化** | 工作记忆纯内存，进程重启即失（设计如此，非缺陷） | — |

### 🔁 与 `MyAgent/Memory/` 的关系
同仓库 `MyAgent/Memory/` 已有一套**更完整但更重**的记忆系统：pydantic `MemoryItem`、`BaseMemory` 抽象基类、working/episodic/semantic/perceptual 四类、auto-classify、importance 计算、consolidation、多种 forget 策略、Qdrant/Neo4j 存储、RAG pipeline、Embedding。两条线接口风格不同（`Agent/` 用 `add/retrieve/forget`，`MyAgent/` 用 `add_memory/retrieve_memories/...`）。

情景记忆实现时，**参考了 `MyAgent/Memory/type/episodic.py` 的三层思路，但在 `Agent/` 内用标准库自建精简版**（SQLite/Qdrant/Embedding 均重写），未跨目录 import，规避了参考实现里的若干 bug。语义记忆同样**参考了 `MyAgent/Memory/type/semantic.py` + `storage/neo4j_store.py` 的图谱思路，但在 `Agent/` 内自建**（`Neo4jStore` 优雅降级而非硬失败、`md5` 实体 id 而非 `hash()`、LLM 三元组抽取而非 spaCy NER、Cypher 注入防御），并复用了情景记忆已建好的 SQLite/Qdrant/Embedding 基础设施。RAG 工具同样**参考了 `MyAgent/Memory/tools/builtin/rag_tool.py` + `rag/document.py` 的全流程，但在 `Agent/` 内自建**：Tool 基类改用 Agent 的 `Tool`（`name/description/parameters` 属性 + `run(**kwargs)`）而非 MyAgent 的 `Tool/ToolParameter`；向量库复用 Agent 的 `QdrantStore`（签名 `payloads=`/返回 `payload`）而非 MyAgent 的 `QdrantVectorStore`，且 Qdrant 不可用时落到自建的 `InMemoryVectorStore` 兜底（参考实现是硬失败）；文档解析改 stdlib 优先 + 可选 markitdown（参考硬依赖 markitdown）；LLM 调用走 Agent 的 `complete()`（静默降级）；裁剪了 cross-encoder rerank / graph signals / neighbor expansion 等重逻辑，保留 markdown 感知分块 + MQE/HyDE 扩展检索。

---

## 9. 需要处理的隐患（`.gitignore` 相关，已实测）

用 `git check-ignore` 实测后，发现两类问题，一个是好消息一个是真坑：

1. ✅ **真实密钥未被提交**：`Agent/LLMClient/.env` 被 `.gitignore:123` 的 `.env` 规则忽略，`git ls-files` 确认它**未进入版本库**。只要不 `git add -f` 强加，密钥不会泄露。但仍建议在该 key 的服务商后台定期轮换，并保留 `.env` 在忽略列表。

2. ✅ **`memory/` 规则误伤 `Agent/Memory/` 源码目录**（已修复）：`.gitignore` 原有一行 `memory/`，会忽略任意层级下名为 `memory/` 的目录，在 Windows（大小写不敏感）上**也命中了 `Agent/Memory/`**——执行 `git add Agent/` 时整个 Memory 模块会被静默跳过。**已修复**：改为 `/memory/`（仅匹配仓库根下的 memory 目录），`git check-ignore` 实测确认 `Agent/Memory/` 及 `storage/` 子目录不再被忽略，可正常提交；同时 `.env` 仍被忽略、密钥安全。

3. ℹ️ 旁注：`.gitignore:166` 的 `test_*.py` 只匹配「以 `test_` 开头」的文件，不影响 `Agent/test/working_memory_test.py`、`isolation_test.py`、`episodic_test.py`（它们是 `_test.py` 结尾），这些测试文件可正常提交。

---

## 10. 下一步候选方向

1. **记忆整合（consolidation）**（最高价值）：工作记忆里高重要性项按阈值沉淀到情景/语义记忆，参考 `MyAgent/Memory/manager.py` 的 `consolidate_memories`。语义记忆尤其可借此从对话/情景记忆里自动抽取事实沉淀为知识图谱，无需 LLM 显式 `add`。
2. **语义记忆自动沉淀**：在 ReAct 循环或会话结束时，自动跑一次三元组抽取把本轮关键事实写入语义记忆——让图谱随对话增长。
3. **扩展业务工具**：在 `MyAgent` 上挂搜索/计算/数据库/MCP 等工具，补全 Agent 的「手」（RAG 工具已就绪，可与之协同）。
4. **真语义嵌入 + 图谱接入**：配齐 `EMBED_MODEL_TYPE=dashscope` 或 `local` + Qdrant + Neo4j + LLM，让语义记忆从「SQLite 关键词」逐层升级为「向量召回 + 图遍历召回 + LLM 抽取」全能力（代码已就绪，仅需配置）。
5. **感知记忆（perceptual）**：`MyAgent/Memory/type/perceptual.py` 仍是空文件，若有多模态输入需求可补第四类记忆。
