"""情景记忆「写入却检索不到」诊断脚本。

复现 trip_planner_with_memory 里第二轮 retrieve 查不到第一轮记忆的问题，
逐层拆开 EpisodicMemory 的写入/检索链路，定位是：
  (A) Qdrant 没连上；
  (B) 维度不匹配 → 向量写不进 + 查不动；
  (C) 向量层正常但语义召回为空，回退关键词又用整句 LIKE 必然落空；
  (D) 以上都不是（需要进一步看）。

运行：
    cd D:\\develop\\Project\\Hello-Agent\\hello-agents
    python -m Agent.test.episodic_memory_test
"""

import os
from pathlib import Path

# ---------- 先加载环境变量，再 import 记忆模块（embedder/qdrant 在 import 期就读 env） ----------
_ENV_DIR = Path(__file__).resolve().parent.parent / "Memory"
_env_path = _ENV_DIR / ".env"
if not _env_path.exists():
    _env_path = _ENV_DIR / ".env.example"  # 回退：示例文件里已填配置
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=True)
    print(f"[env] 已加载 {_env_path.name}")
except ImportError:
    print("[env] 未安装 python-dotenv，改用系统环境变量")


from Agent.Memory.embedding import get_embedder, refresh_embedder  # noqa: E402
from Agent.Memory.storage.qdrant_store import qdrant_from_env      # noqa: E402
from Agent.Memory.storage.sqlite_store import EpisodicSQLiteStore  # noqa: E402
from Agent.Memory.types.episodic import EpisodicMemory             # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ============================================================================
# Step 1: embedder 真实维度 vs Qdrant 集合配置维度
# ============================================================================
def step1_check_dims():
    section("Step 1  embedder 维度 vs Qdrant vector_size")
    emb = refresh_embedder()
    # 用真实 encode 取维度，不迷信 .dimension（DashScopeEmbedder 的 _dimension 可能 None）
    real_dim = len(emb.encode("维度探测"))
    qdrant_size = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))

    print(f"  embedder          = {emb.name}")
    print(f"  embedder.available= {emb.available}  (False=hashing 兜底，非真语义)")
    print(f"  真实 encode 维度  = {real_dim}")
    print(f"  QDRANT_VECTOR_SIZE= {qdrant_size}")
    print(f"  EMBED_DIM         = {os.getenv('EMBED_DIM', '(默认 384)')}  "
          "(仅 hashing 生效；dashscope/local 忽略此值)")

    if real_dim != qdrant_size:
        print(f"\n  ❌❌ 维度不匹配！embedder 产 {real_dim} 维，"
              f"Qdrant 集合按 {qdrant_size} 维建。")
        print("     → add 时 add_vectors 会「向量维度不匹配，跳过」，向量进不了 Qdrant；")
        print("     → retrieve 时 search_similar 会「查询向量维度不匹配，跳过」直接返回 []。")
        print("     → 最终回退 SQLite 关键词 LIKE，用整句 query 匹配不到原文 → 查不到。")
        print("     修复：把 QDRANT_VECTOR_SIZE 改成 embedder 的真实维度，"
              "并删除旧集合让其按新维度重建（见 Step 5）。")
    else:
        print("  ✅ 维度一致")
    return emb, real_dim, qdrant_size


# ============================================================================
# Step 2: Qdrant 连通性 + 集合实际维度/点数
# ============================================================================
def step2_check_qdrant():
    section("Step 2  Qdrant 连通性")
    vs = qdrant_from_env()
    print(f"  QdrantStore.available = {vs.available}")
    print(f"  QDRANT_URL            = {os.getenv('QDRANT_URL', '(未设置)')}")
    print(f"  collection_name       = {vs.collection_name}")
    print(f"  vector_size(配置)    = {vs.vector_size}")
    if not vs.available:
        print("  ⚠️  Qdrant 不可用（未装 qdrant-client / 未配 URL / 连接失败）")
        print("     此时 _vector_ready=False，retrieve 直接走 SQLite 关键词回退。")
        return vs, None, None

    info = vs.get_stats()
    print(f"  集合 stats            = {info}")
    # 取集合真实配置维度
    real_vec_size = None
    points = None
    try:
        col = vs.client.get_collection(vs.collection_name)
        real_vec_size = col.config.params.vectors.size
        points = col.points_count
        print(f"  集合真实 vector_size = {real_vec_size}")
        print(f"  集合 points_count    = {points}")
    except Exception as e:
        print(f"  读取集合配置失败: {e}")
    if real_vec_size is not None and real_vec_size != vs.vector_size:
        print(f"  ⚠️  集合真实维度 {real_vec_size} != 配置维度 {vs.vector_size}"
              "：旧集合维度残留，需删除重建（见 Step 5）。")
    return vs, real_vec_size, points


# ============================================================================
# Step 3: 构造 EpisodicMemory，写入一条记忆，分别看 SQLite / Qdrant 是否落库
# ============================================================================
def step3_add_and_inspect(emb, real_dim):
    section("Step 3  写入一条记忆，分别检查 SQLite / Qdrant 落库情况")
    mem = EpisodicMemory(user_id="diag_user", session_id="diag")
    print(f"  EpisodicMemory._vector_ready = {mem._vector_ready}")

    content = "用户偏好：历史文化类景点，曾计划前往北京旅行，预算有限想省钱"
    mid = mem.add(content, importance=0.9, metadata={"kind": "preference", "source": "diag"})
    print(f"  写入 memory_id = {mid[:12]}...")
    print(f"  content        = {content}")

    # --- SQLite 层 ---
    store: EpisodicSQLiteStore = mem.store
    row = store.get_memory(mid)
    print(f"\n  [SQLite] get_memory 命中 = {bool(row)}")
    if row:
        print(f"           content = {row['content'][:40]}...")
    s = store.get_stats(memory_type="episodic", user_id="diag_user")
    print(f"           stats   = count={s.get('count')}")

    # --- Qdrant 层：直接调底层，绕过 retrieve 的回退逻辑 ---
    print("\n  [Qdrant] 直接 add_vectors 测试（绕过 retrieve）")
    if not mem._vector_ready:
        print("           _vector_ready=False，跳过（无向量层）")
        return mem, mid, content

    try:
        vec = emb.encode(content)
        print(f"           encode 维度 = {len(vec)}")
        ok = mem.vector_store.add_vectors(
            vectors=[vec],
            payloads=[{"memory_id": mid, "user_id": "diag_user",
                       "memory_type": "episodic", "content": content}],
            ids=[mid],
        )
        print(f"           add_vectors 返回 = {ok}")
        col = mem.vector_store.client.get_collection(mem.vector_store.collection_name)
        print(f"           集合 points_count = {col.points_count}")
        if not ok or col.points_count == 0:
            print("           ❌ 向量没进 Qdrant（多半是维度不匹配被跳过，看上方 warning）")
        else:
            print("           ✅ 向量已入 Qdrant")
    except Exception as e:
        print(f"           ❌ 直接写向量失败: {type(e).__name__}: {e}")
    return mem, mid, content


# ============================================================================
# Step 4: 检索 —— 复现「第二轮 query 查不到第一轮记忆」
# ============================================================================
def step4_retrieve(mem, content):
    section("Step 4  检索（复现第二轮查不到的问题）")
    # 这是 trip_planner 第二轮的实际 query
    query = "再帮我规划一次 3 天的旅行"
    print(f"  query = {query!r}")
    print(f"  存储的 content = {content!r}")
    print(f"  query 与 content 无关键词重叠（北京/历史/省钱 都没出现在 query 里）")

    results = mem.retrieve(query=query, limit=3)
    print(f"\n  retrieve() 返回 {len(results)} 条")
    for m in results:
        print(f"    - {m.content}")

    if results:
        print("\n  ✅ 检索到记忆，问题已不复现（可能维度已修好）")
    else:
        print("\n  ❌ 查不到 —— 复现成功。逐层定位：")
        # 单测向量召回
        if mem._vector_ready:
            try:
                qvec = mem.embedder.encode(query)
                hits = mem.vector_store.search_similar(
                    query_vector=qvec, limit=10,
                    where={"memory_type": "episodic", "user_id": "diag_user"},
                )
                print(f"    [向量] search_similar 返回 {len(hits)} 条 "
                      f"(查询向量维度={len(qvec)}, 集合维度={mem.vector_store.vector_size})")
                if len(qvec) != mem.vector_store.vector_size:
                    print("    [向量] ❌ 查询向量维度 != 集合维度 → 被跳过，返回 []")
            except Exception as e:
                print(f"    [向量] 异常: {e}")
        # 单测关键词回退
        rows = mem.store.search_memories(
            user_id="diag_user", memory_type="episodic",
            keyword=query, limit=3,
        )
        print(f"    [关键词] LIKE '%{query}%' 命中 {len(rows)} 条 "
              "(整句 LIKE，自然句几乎不可能子串匹配)")
        # 对照：用能命中的关键词
        rows2 = mem.store.search_memories(
            user_id="diag_user", memory_type="episodic",
            keyword="北京", limit=3,
        )
        print(f"    [关键词] LIKE '%北京%' 命中 {len(rows2)} 条 "
              "(证明 SQLite 原文确实在，只是整句 query 匹配不到)")


# ============================================================================
# Step 5: 修复建议 / 可选重建集合
# ============================================================================
def step5_fix_hint(emb, real_dim, qdrant_size, vs):
    section("Step 5  修复建议")
    print("  根因：向量维度与 Qdrant 集合维度不一致，导致向量写不进、查不动，")
    print("        最终回退到「整句 LIKE」关键词检索，匹配不到原文。")
    print("\n  方案一（推荐）：让 QDRANT_VECTOR_SIZE 等于 embedder 真实维度")
    print(f"     把 .env 里 QDRANT_VECTOR_SIZE={qdrant_size} 改为 "
          f"QDRANT_VECTOR_SIZE={real_dim}")
    print("     然后删除旧集合让其按新维度重建：")
    print("       from Agent.Memory.storage.qdrant_store import qdrant_from_env")
    print("       vs = qdrant_from_env()")
    print("       vs.client.delete_collection(vs.collection_name)")
    print("\n  方案二：不想用向量，就让 retrieve 走得到关键词回退")
    print("     把 EMBED_MODEL_TYPE 设为 hashing（或不配 QDRANT_URL），")
    print("     _vector_ready=False，retrieve 直接走 SQLite。")
    print("     但注意：此时 keyword 用整句 query 仍会落空，")
    print("     需把 trip_planner 里的 retrieve query 换成「北京 历史 省钱」这类关键词。")
    if vs is not None and vs.available:
        print("\n  是否现在帮你删除旧集合并按正确维度重建？(y/n) ", end="")
        try:
            ans = input().strip().lower()
        except EOFError:
            ans = "n"
        if ans == "y":
            try:
                vs.client.delete_collection(vs.collection_name)
                print("  已删除旧集合。")
                # 改完 .env 后重新构造会按新维度建
                print("  请先把 .env 的 QDRANT_VECTOR_SIZE 改对，再重跑本脚本验证。")
            except Exception as e:
                print(f"  删除失败: {e}")


if __name__ == "__main__":
    emb, real_dim, qdrant_size = step1_check_dims()
    vs, real_vec_size, points = step2_check_qdrant()
    mem, mid, content = step3_add_and_inspect(emb, real_dim)
    step4_retrieve(mem, content)
    step5_fix_hint(emb, real_dim, qdrant_size, vs)

    # 清理本次诊断写入的脏数据
    section("清理")
    try:
        mem.store.delete_by_user("diag_user", memory_type="episodic")
        print("  已清理 diag_user 的 SQLite 诊断数据")
    except Exception as e:
        print(f"  清理 SQLite 失败: {e}")
    if mem._vector_ready:
        try:
            mem.vector_store.delete_by_memory_ids([mid])
            print("  已清理 Qdrant 诊断向量")
        except Exception as e:
            print(f"  清理 Qdrant 失败: {e}")
