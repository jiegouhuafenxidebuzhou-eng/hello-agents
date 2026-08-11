"""情景记忆测试用例（纯 SQLite 路径，不依赖 Qdrant/嵌入模型）。

覆盖：增/检索（关键词回退）/遗忘/跨用户隔离/跨会话长期召回/降级可用。
运行：
    cd D:\\develop\\Project\\Hello-Agent\\hello-agents
    python -m Agent.test.episodic_test
"""

import os
import tempfile

from Agent.Memory.episodic import EpisodicMemory


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def _new(user_id: str, session_id: str, db_path: str) -> EpisodicMemory:
    """构造纯 SQLite 路径实例（关闭向量/嵌入，确保走关键词回退）。"""
    return EpisodicMemory(user_id=user_id, session_id=session_id,
                          db_path=db_path, use_embedder=False,
                          vector_store=None)


def test_basic_add_stats(tmp_db: str):
    section("1. 基础：添加 + 统计")
    ep = _new("alice", "s1", tmp_db)
    ep.add("用户想退货，订单A12345", importance=0.9)
    ep.add("用户提到明天要出差", importance=0.6)
    stats = ep.get_stats()
    print("stats:", stats)
    assert stats["count"] >= 2, "计数应 >=2"
    assert stats["memory_type"] == "episodic"
    assert stats["vector_ready"] is False, "纯 SQLite 路径应关闭向量"
    print("✅ 添加 + 统计正确")


def test_retrieve_keyword(tmp_db: str):
    section("2. 关键词检索（无向量，走 SQLite LIKE）")
    ep = _new("alice", "s1", tmp_db)
    ep.add("订单A12345需要退货", importance=0.9)
    ep.add("今天天气很好", importance=0.2)
    hits = ep.retrieve(query="订单")
    print("命中:", [m.content for m in hits])
    assert any("订单" in m.content for m in hits), "应命中订单记忆"
    # 空查询：取最近
    recent = ep.retrieve(query=None, limit=5)
    print("最近:", [m.content for m in recent])
    assert len(recent) >= 1
    print("✅ 关键词检索 + 空查询正确")


def test_forget(tmp_db: str):
    section("3. 主动遗忘")
    ep = _new("alice", "s1", tmp_db)
    mid = ep.add("一条会被遗忘的记忆", importance=0.5)
    assert ep.get(mid) is not None, "遗忘前应能查到"
    ok = ep.forget(mid)
    assert ok, "forget 应返回 True"
    assert ep.get(mid) is None, "遗忘后应查不到"
    assert ep.forget(mid) is False, "重复遗忘返回 False"
    print("✅ 遗忘正确（缓存+SQLite 同步清除）")


def test_cross_user_isolation(tmp_db: str):
    section("4. 跨用户隔离")
    epA = _new("alice", "s1", tmp_db)
    epB = _new("bob", "s1", tmp_db)
    epA.add("alice 的秘密：密码是 999", importance=0.9)
    resA = epA.retrieve(query="密码")
    resB = epB.retrieve(query="密码")
    print("alice:", [m.content for m in resA])
    print("bob  :", [m.content for m in resB])
    assert resA and "999" in resA[0].content
    assert all("999" not in m.content for m in resB), "bob 不应看到 alice 的记忆"
    print("✅ 跨用户不串读")


def test_cross_session_recall(tmp_db: str):
    section("5. 跨会话长期召回（SQLite 是跨会话权威）")
    ep_s1 = _new("alice", "s1", tmp_db)
    ep_s1.add("上周和用户聊过退货政策", importance=0.8)

    # 新会话实例，同 user，应能从 SQLite 召回上一会话的情景
    ep_s2 = _new("alice", "s2", tmp_db)
    hits = ep_s2.retrieve(query="退货")
    print("s2 召回:", [m.content for m in hits])
    assert any("退货" in m.content for m in hits), "新会话应召回历史情景记忆"
    print("✅ 跨会话召回正确（SQLite 权威层）")


def test_degradation(tmp_db: str):
    section("6. 降级：无 Qdrant / 无嵌入模型仍可用")
    ep = _new("alice", "s1", tmp_db)
    assert not ep._vector_ready
    ep.add("降级场景下的记忆", importance=0.5)
    hits = ep.retrieve(query="降级")
    assert hits, "降级路径应仍能检索"
    print("✅ 纯 SQLite 关键词路径可用")


if __name__ == "__main__":
    # 用临时 db 跑，跑完清理
    tmp = os.path.join(tempfile.gettempdir(), "hello_agents_episodic_test.db")
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
        print("\n✅ 全部情景记忆测试通过")
    finally:
        for f in (tmp,):
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
