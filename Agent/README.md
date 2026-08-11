# Hello-Agents / Agent 项目现状梳理

> 生成日期：2026-08-11　｜　最近更新：2026-08-11（情景记忆落地）
> 范围：仓库根下的 `Agent/` 目录（不含 `MyAgent/`、`Co-creation-projects/` 等）
> 目的：盘点当前已实现的能力、架构与设计决策，标出缺口，为下一步开发提供基线。

---

## 1. 一句话定位

`Agent/` 是一个**轻量、纯标准库优先**的 LLM Agent 脚手架：把「LLM 客户端 + 工具调用（ReAct 循环）+ 记忆系统」拼到一起。记忆层已落地**工作记忆 + 情景记忆**两类，语义记忆仍是空占位。

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
│   ├── embedding.py           #   Embedder 协议 + DashScope/Local/Hashing 三层降级
│   ├── storage/
│   │   ├── sqlite_store.py    #     EpisodicSQLiteStore：纯标准库 sqlite3 权威存储
│   │   └── qdrant_store.py    #     QdrantStore：可选 qdrant-client 向量存储
│   ├── manager.py             #   MemoryManager：统一管理 working/episodic/semantic
│   ├── memory_tool.py         #   MemoryTool：把记忆系统作为 Tool 暴露给 LLM
│   ├── .env.example           #   统一配置模板（Qdrant/Embedding，情景记忆已消费）
│   └── __init__.py            #   空文件
├── session.py                 # 会话级 Agent 工厂（方案 A：身份烙入 + 物理隔离）
└── test/
    ├── working_memory_test.py #   工作记忆 7 项用例
    ├── isolation_test.py      #   多用户/多会话/多线程隔离 5 项用例
    └── episodic_test.py       #   情景记忆 6 项用例（纯 SQLite 路径，零依赖）
```

---

## 3. 核心模块逐一说明

### 3.1 LLMClient 层

#### `llm_client.py` — `HelloAgentsLLM`
- 基于 `openai` SDK 的 OpenAI 兼容封装，配置从 `LLMClient/.env` 读取（`LLM_MODEL_ID` / `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_TIMEOUT`）。
- **`think(messages, temperature=0)`**：流式调用，边收边 `print`，返回拼接后的完整文本。用于「纯对话」场景。
- **`chat(messages, tools=None, temperature=0)`**：非流式，支持原生 function calling；返回 `assistant.message` 对象（含 `tool_calls`）。用于让 LLM 自主决定调工具。
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

#### `manager.py` — `MemoryManager`
- 统一接口：`add / retrieve / forget / stats / clear`，参数带 `memory_type`。
- `backends = {"working": WorkingMemory, "episodic": <可启用>, "semantic": None}`。
  - **`episodic` 默认 `None`**，显式传实例或 `enable_episodic=True` 时按本会话身份构造 `EpisodicMemory`。
  - **`semantic` 仍是 `None` 占位**，调用抛 `ValueError("记忆类型 '...' 未启用")`。
  - `retrieve(memory_type="all")` 可跨已启用类型检索并按时间倒序合并。
- 接口约定（注释里写明）：任何后端只要实现 `add/retrieve/forget/get_stats` 即可挂上，`MemoryTool` 无需改动。

#### `memory_tool.py` — `MemoryTool`
- 把 `MemoryManager` 包成单一 Tool 暴露给 LLM，**一个工具搞定增/检索/遗忘/统计**（不为每类记忆单独建工具）。
- 参数：`action(add/retrieve/forget/stats)` + `memory_type(working/episodic/semantic/all)` + `content/query/memory_id/limit/importance`。
- **安全设计（方案 A）**：`user_id` / `session_id` 在 `MemoryTool` 构造时烙入，**不出现在 function 参数 schema 里**，因此 LLM 无法伪造身份——即使调用时硬塞 `user_id="bob"` 也会被忽略。

### 3.3 `session.py` — 会话级工厂

`build_session_agent(llm_client, user_id, session_id, extra_tools, max_steps, enable_episodic=False, episodic_db_path=None)`：
- 每个会话构造**全新的** `WorkingMemory` → `MemoryManager` → `MemoryTool` → `MyAgent`，身份烙入。
- `enable_episodic=True` 时额外构造一个 `EpisodicMemory`（身份烙入）挂进 manager；SQLite/Qdrant 虽全局共享单例，但所有读写带 `user_id` 过滤，跨会话可召回长期记忆而互不串读。
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
                                          └─ EpisodicMemory（缓存+SQLite+Qdrant，可选）
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

---

## 5. 配置

`LLMClient/.env`（实际运行读取）：
```
LLM_MODEL_ID="deepseek-v4-pro"
LLM_API_KEY="sk-..."          # 真实密钥（被 .gitignore 忽略，未入库；见 §9）
LLM_BASE_URL="https://api.deepseek.com"
SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY"
```

`Memory/.env.example`（情景记忆已消费其中 Qdrant/Embedding 变量；复制为 `Agent/Memory/.env` 后生效）：
- LLM 四件套、Tavily/SerpApi。
- Qdrant：`QDRANT_URL/QDRANT_API_KEY/QDRANT_COLLECTION/QDRANT_VECTOR_SIZE/QDRANT_DISTANCE`（不配则向量层关闭，降级关键词）。
- Embedding：`EMBED_MODEL_TYPE(dashscope|local|hashing，默认 hashing)/EMBED_MODEL_NAME/EMBED_API_KEY/EMBED_BASE_URL/EMBED_DIM`（不配真模型则 hashing 兜底，仍可用）。
- Neo4j：仅模板预留，当前 Agent/ 实现未消费（为未来 semantic 留）。

> 情景记忆的「最小可跑」配置：**零额外环境变量**——默认 hashing embedder + 无 Qdrant → 纯 SQLite + 关键词检索即可工作。配齐 Qdrant + 真 embedder 后自动升级为语义召回，无需改代码。

---

## 6. 测试

| 文件 | 覆盖 | 运行方式（仓库根目录下） |
|---|---|---|
| `test/working_memory_test.py` | 增/检索/关键词过滤/主动遗忘/TTL 过期/容量驱逐/Token 驱逐/limit | `python -m Agent.test.working_memory_test` |
| `test/isolation_test.py` | 跨用户隔离/同用户跨会话隔离/身份烙入/LLM 无法伪造身份/8 用户并发无串读 | `python -m Agent.test.isolation_test` |
| `test/episodic_test.py` | 增/关键词检索/遗忘/跨用户隔离/跨会话长期召回/降级可用（纯 SQLite，零依赖） | `python -m Agent.test.episodic_test` |

注：测试通过 `__main__` 直接跑函数，非 `pytest` 断言框架；`isolation_test` / `episodic_test` 里用 `assert`，可用 `python -m` 执行。导入路径要求从仓库根运行。`episodic_test` 用临时 db 跑完自动清理。

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

---

## 8. 现状能力清单 & 缺口

### ✅ 已实现
- LLM 客户端（流式 + function calling）
- 工具体系（注册/查询/执行 + 让 LLM 自查工具清单的内置工具）
- ReAct Agent 循环（LLM 自主调工具 → 回填 → 收敛）
- 工作记忆全功能（增/检索/遗忘，TTL + 驱逐 + 主动遗忘）
- **情景记忆全功能**（三层：本地缓存 + SQLite 权威 + Qdrant 向量；增/检索/遗忘/清空；跨会话召回；可选依赖+降级）
- **Embedding 抽象 + 三层降级链**（DashScope/Local/Hashing 零依赖兜底）
- 记忆作为工具暴露给 LLM（统一 memory 工具，working/episodic 同一个工具切换）
- 多用户/多会话/多线程隔离（含身份防伪造，工作记忆 + 情景记忆均隔离）
- 工作记忆 7 项 + 隔离 5 项 + 情景记忆 6 项测试

### ❌ 缺口（按优先级）
| 优先级 | 缺口 | 说明 | 可复用资源 |
|---|---|---|---|
| 高 | **语义记忆 `semantic` 后端** | `MemoryManager.backends["semantic"]` 仍是 `None`，调用抛错 | `MyAgent/Memory/type/semantic.py`、`storage/neo4j_store.py` |
| 中 | **记忆整合（consolidation）** | 工作记忆高重要性项 → 情景记忆的沉淀机制缺失（参考 MyAgent 的 `consolidate_memories`） | `MyAgent/Memory/manager.py` |
| 中 | **业务工具示例** | `MyAgent` 已能挂任意 Tool，但仓库内 `Agent/` 没有现成业务工具（搜索/计算等） | 同仓多项目有实现可借鉴 |
| 低 | **token 精确化** | 工作记忆 token 用字符数估算偏粗，中文场景偏差更大 | 接 `tiktoken` 即可 |
| 低 | **工作记忆持久化** | 工作记忆纯内存，进程重启即失（设计如此，非缺陷） | — |

### 🔁 与 `MyAgent/Memory/` 的关系
同仓库 `MyAgent/Memory/` 已有一套**更完整但更重**的记忆系统：pydantic `MemoryItem`、`BaseMemory` 抽象基类、working/episodic/semantic/perceptual 四类、auto-classify、importance 计算、consolidation、多种 forget 策略、Qdrant/Neo4j 存储、RAG pipeline、Embedding。两条线接口风格不同（`Agent/` 用 `add/retrieve/forget`，`MyAgent/` 用 `add_memory/retrieve_memories/...`）。

情景记忆实现时，**参考了 `MyAgent/Memory/type/episodic.py` 的三层思路，但在 `Agent/` 内用标准库自建精简版**（SQLite/Qdrant/Embedding 均重写），未跨目录 import，规避了参考实现里的若干 bug。下一步补 semantic 时可同样参考 `MyAgent/Memory/type/semantic.py` + Neo4j 存储。

---

## 9. 需要处理的隐患（`.gitignore` 相关，已实测）

用 `git check-ignore` 实测后，发现两类问题，一个是好消息一个是真坑：

1. ✅ **真实密钥未被提交**：`Agent/LLMClient/.env` 被 `.gitignore:123` 的 `.env` 规则忽略，`git ls-files` 确认它**未进入版本库**。只要不 `git add -f` 强加，密钥不会泄露。但仍建议在该 key 的服务商后台定期轮换，并保留 `.env` 在忽略列表。

2. ✅ **`memory/` 规则误伤 `Agent/Memory/` 源码目录**（已修复）：`.gitignore` 原有一行 `memory/`，会忽略任意层级下名为 `memory/` 的目录，在 Windows（大小写不敏感）上**也命中了 `Agent/Memory/`**——执行 `git add Agent/` 时整个 Memory 模块会被静默跳过。**已修复**：改为 `/memory/`（仅匹配仓库根下的 memory 目录），`git check-ignore` 实测确认 `Agent/Memory/` 及 `storage/` 子目录不再被忽略，可正常提交；同时 `.env` 仍被忽略、密钥安全。

3. ℹ️ 旁注：`.gitignore:166` 的 `test_*.py` 只匹配「以 `test_` 开头」的文件，不影响 `Agent/test/working_memory_test.py`、`isolation_test.py`、`episodic_test.py`（它们是 `_test.py` 结尾），这些测试文件可正常提交。

---

## 10. 下一步候选方向

1. **补齐 semantic 后端**（最高价值）：填上 `MemoryManager` 最后一个空插口。可参考 `MyAgent/Memory/type/semantic.py` + `storage/neo4j_store.py`，在本项目内自建轻量版（图存储或复用 SQLite 的 concepts 表）。
2. **记忆整合（consolidation）**：工作记忆里高重要性项按阈值沉淀到情景记忆，参考 `MyAgent/Memory/manager.py` 的 `consolidate_memories`。
3. **扩展业务工具**：在 `MyAgent` 上挂搜索/计算/数据库/MCP 等工具，补全 Agent 的「手」。
4. **真语义嵌入接入**：配齐 `EMBED_MODEL_TYPE=dashscope` 或 `local` + Qdrant，让情景记忆从关键词检索升级为语义召回（代码已就绪，仅需配置）。
