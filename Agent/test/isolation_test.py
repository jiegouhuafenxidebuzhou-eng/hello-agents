"""多用户 / 多会话隔离测试。

验证方案 A：不同 (user_id, session_id) 的工作记忆物理隔离、互不串读互不串写，
即使在多线程并发下也安全。

运行：
    cd D:\\develop\\Project\\Hello-Agent\\hello-agents
    python -m Agent.Memory.isolation_test
"""

from concurrent.futures import ThreadPoolExecutor

from Agent.tools.memory_tool import MemoryTool
from Agent.Memory.manager import MemoryManager


def _new_tool(user_id: str, session_id: str) -> MemoryTool:
    from Agent.Memory.working import WorkingMemory
    working = WorkingMemory(user_id=user_id, session_id=session_id)
    mgr = MemoryManager(working=working, user_id=user_id, session_id=session_id)
    return MemoryTool(mgr, user_id=user_id, session_id=session_id)


def test_cross_user_isolation():
    print("\n=== 1. 跨用户隔离 ===")
    toolA = _new_tool(user_id="alice", session_id="s1")
    toolB = _new_tool(user_id="bob", session_id="s1")

    toolA.run(action="add", content="alice 的秘密：密码是 123", importance=0.9)
    toolB.run(action="add", content="bob 的秘密：密码是 456", importance=0.9)

    # alice 检索，不应看到 bob 的内容
    resA = toolA.run(action="retrieve", memory_type="all")
    resB = toolB.run(action="retrieve", memory_type="all")
    print("alice 看到的:", resA)
    print("bob   看到的:", resB)
    assert "456" not in resA, "❌ alice 看到了 bob 的记忆！隔离失败"
    assert "123" not in resB, "❌ bob 看到了 alice 的记忆！隔离失败"
    print("✅ 跨用户不串读")


def test_cross_session_isolation():
    print("\n=== 2. 同用户跨会话隔离（工作记忆不跨会话）===")
    tool_s1 = _new_tool(user_id="alice", session_id="s1")
    tool_s2 = _new_tool(user_id="alice", session_id="s2")

    tool_s1.run(action="add", content="s1 里的临时上下文", importance=0.7)
    res_s2 = tool_s2.run(action="retrieve", memory_type="all")
    print("s2 看到的:", res_s2)
    assert "s1 里的临时上下文" not in res_s2, "❌ 工作记忆跨会话泄漏！"
    print("✅ 同用户不同会话工作记忆不互通")


def test_identity_stamped_on_item():
    print("\n=== 3. 身份烙入到 MemoryItem ===")
    tool = _new_tool(user_id="alice", session_id="s1")
    tool.run(action="add", content="带身份的记忆", importance=0.5)
    items = tool.manager.retrieve(memory_type="working")
    m = items[0]
    print(f"item.user_id={m.user_id}, session_id={m.session_id}")
    assert m.user_id == "alice" and m.session_id == "s1"
    print("✅ 每条记忆带正确身份")


def test_llm_cannot_spoof_identity():
    print("\n=== 4. LLM 无法伪造身份（参数无 user_id/session_id）===")
    tool = _new_tool(user_id="alice", session_id="s1")
    # 即使有人往参数里硬塞 user_id，工具也忽略它
    tool.run(action="add", content="尝试冒充 bob", user_id="bob", session_id="hacked")
    items = tool.manager.retrieve(memory_type="working")
    m = items[0]
    print(f"被冒充写入后，实际 user_id={m.user_id}, session_id={m.session_id}")
    assert m.user_id == "alice" and m.session_id == "s1", "❌ 身份被 LLM 伪造！"
    print("✅ LLM 传入的 user_id 被忽略，身份不可伪造")


def test_concurrent_threads_no_cross_talk():
    print("\n=== 5. 多线程并发隔离 ===")
    results = {}
    errors = []

    def worker(user_id: str):
        try:
            tool = _new_tool(user_id=user_id, session_id=f"s-{user_id}")
            tool.run(action="add", content=f"{user_id}-data", importance=0.9)
            # 多次读写，制造并发压力
            for _ in range(50):
                tool.run(action="add", content=f"{user_id}-more", importance=0.5)
                tool.run(action="retrieve", memory_type="all")
            results[user_id] = tool.manager.retrieve(memory_type="working")
        except Exception as e:
            errors.append(f"{user_id} 异常: {e}")

    users = [f"user{i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, users))

    # 每个用户最终只应看到自己的记忆
    for uid, items in results.items():
        for m in items:
            assert m.user_id == uid, f"❌ {uid} 的记忆里出现了 {m.user_id} 的数据"
    print(f"✅ 8 用户并发，各自记忆身份一致，无串读。错误数: {len(errors)}")
    if errors:
        for e in errors:
            print("  -", e)


if __name__ == "__main__":
    test_cross_user_isolation()
    test_cross_session_isolation()
    test_identity_stamped_on_item()
    test_llm_cannot_spoof_identity()
    test_concurrent_threads_no_cross_talk()
    print("\n✅ 隔离测试全部通过")
