"""语义记忆测试用例（纯 SQLite 路径，不依赖 Qdrant/Neo4j/LLM）。

覆盖：增/检索（关键词回退）/遗忘/跨用户隔离/跨会话长期召回/降级可用/
      与情景记忆分区隔离（共享 db 靠 memory_type 分区）。
运行：
    cd D:\\develop\\Project\\Hello-Agent\\hello-agents
    python -m Agent.test.semantic_test

LLM 三元组抽取 + Neo4j 图谱路径需要 LLM_API_KEY 与 NEO4J_*，无法单测；
手动验证：配置 .env 后构造
    SemanticMemory(enable_graph=True, llm_client=HelloAgentsLLM())
add 一条事实后查 graph_store.get_stats()['entity_nodes'] > 0。
"""

import os
import tempfile

from Agent.Memory.types.semantic import SemanticMemory
from Agent.Memory.types.episodic import EpisodicMemory


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def _new(user_id: str, session_id: str, db_path: str) -> SemanticMemory:
    """构造纯 SQLite 路径实例（关闭向量/嵌入/图谱，确保走关键词回退）。"""
    return SemanticMemory(user_id=user_id, session_id=session_id,
                          db_path=db_path, use_embedder=False,
                          vector_store=None, graph_store=None,
                          enable_graph=False)


def test_basic_add_stats(tmp_db: str):
    section("1. 基础：添加 + 统计")
    sm = _new("alice", "s1", tmp_db)
    sm.add("用户偏好日语学习", importance=0.9)
    sm.add("张三在阿里工作", importance=0.6)
    stats = sm.get_stats()
    print("stats:", stats)
    assert stats["count"] >= 2, "计数应 >=2"
    assert stats["memory_type"] == "semantic"
    assert stats["vector_ready"] is False, "纯 SQLite 路径应关闭向量"
    assert stats["graph_ready"] is False, "纯 SQLite 路径应关闭图谱"
    assert stats["llm_ready"] is False, "无 LLM 客户端应关闭 LLM 抽取"
    print("✅ 添加 + 统计正确")


def test_retrieve_keyword(tmp_db: str):
    section("2. 关键词检索（无向量/无图谱 → SQLite LIKE 回退）")
    sm = _new("alice", "s1", tmp_db)
    sm.add("用户偏好日语学习", importance=0.9)
    sm.add("今天天气很好", importance=0.2)
    hits = sm.retrieve(query="日语")
    print("命中:", [m.content for m in hits])
    assert any("日语" in m.content for m in hits), "应命中日语记忆"
    # 空查询：取最近
    recent = sm.retrieve(query=None, limit=5)
    print("最近:", [m.content for m in recent])
    assert len(recent) >= 1
    print("✅ 关键词检索 + 空查询正确")


def test_forget(tmp_db: str):
    section("3. 主动遗忘")
    sm = _new("alice", "s1", tmp_db)
    mid = sm.add("一条会被遗忘的语义记忆", importance=0.5)
    assert sm.get(mid) is not None, "遗忘前应能查到"
    ok = sm.forget(mid)
    assert ok, "forget 应返回 True"
    assert sm.get(mid) is None, "遗忘后应查不到"
    assert sm.forget(mid) is False, "重复遗忘返回 False"
    print("✅ 遗忘正确（缓存+SQLite 同步清除）")


def test_cross_user_isolation(tmp_db: str):
    section("4. 跨用户隔离")
    smA = _new("alice", "s1", tmp_db)
    smB = _new("bob", "s1", tmp_db)
    smA.add("alice 的偏好：喜欢黑咖啡", importance=0.9)
    resA = smA.retrieve(query="咖啡")
    resB = smB.retrieve(query="咖啡")
    print("alice:", [m.content for m in resA])
    print("bob  :", [m.content for m in resB])
    assert resA and "黑咖啡" in resA[0].content
    assert all("黑咖啡" not in m.content for m in resB), "bob 不应看到 alice 的记忆"
    print("✅ 跨用户不串读")


def test_cross_session_recall(tmp_db: str):
    section("5. 跨会话长期召回（SQLite 是跨会话权威）")
    sm_s1 = _new("alice", "s1", tmp_db)
    sm_s1.add("上周确立了日语学习计划", importance=0.8)

    # 新会话实例，同 user，应能从 SQLite 召回上一会话的语义记忆
    sm_s2 = _new("alice", "s2", tmp_db)
    hits = sm_s2.retrieve(query="日语")
    print("s2 召回:", [m.content for m in hits])
    assert any("日语" in m.content for m in hits), "新会话应召回历史语义记忆"
    print("✅ 跨会话召回正确（SQLite 权威层）")


def test_degradation(tmp_db: str):
    section("6. 降级：无 Neo4j / 无 Qdrant / 无 LLM 仍可用")
    sm = _new("alice", "s1", tmp_db)
    assert not sm._vector_ready
    assert not sm.graph_ready
    sm.add("降级场景下的语义记忆", importance=0.5)
    hits = sm.retrieve(query="降级")
    assert hits, "降级路径应仍能检索"
    print("✅ 纯 SQLite 关键词路径可用")


def test_partition_from_episodic(tmp_db: str):
    section("7. 与情景记忆分区隔离（共享 db 靠 memory_type 分区）")
    ep = EpisodicMemory(user_id="alice", session_id="s1", db_path=tmp_db,
                       use_embedder=False, vector_store=None)
    ep.add("episodic 事件：用户咨询退货", importance=0.7)

    sm = _new("alice", "s1", tmp_db)
    # 语义记忆的检索不应跨 memory_type 读到情景记忆
    hits = sm.retrieve(query="退货")
    print("semantic 检索 '退货':", [m.content for m in hits])
    assert all("episodic" not in m.content for m in hits), "语义检索不应跨区读到情景记忆"
    # 反向：情景检索也不应读到语义记忆
    sm.add("语义事实：退货需 7 天内申请", importance=0.9)
    ep_hits = ep.retrieve(query="7 天")
    print("episodic 检索 '7 天':", [m.content for m in ep_hits])
    assert all("语义事实" not in m.content for m in ep_hits), "情景检索不应跨区读到语义记忆"
    print("✅ memory_type 分区隔离正确")


if __name__ == "__main__":
    # 用临时 db 跑，跑完清理
    tmp = os.path.join(tempfile.gettempdir(), "hello_agents_semantic_test.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        test_basic_add_stats(tmp)
        test_retrieve_keyword(tmp)
        test_forget(tmp)
        # 跨用户/跨会话测试共享同一 db 以验证过滤与权威性
        test_cross_user_isolation(tmp)
        test_cross_session_recall(tmp)
        test_degradation(tmp)
        test_partition_from_episodic(tmp)
        print("\n✅ 全部语义记忆测试通过")
    finally:
        for f in (tmp,):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
