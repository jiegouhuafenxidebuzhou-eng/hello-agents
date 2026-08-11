"""工作记忆测试用例。

覆盖：增/检索/关键词过滤/主动遗忘/TTL过期/容量驱逐/Token上限驱逐/limit。
运行：
    cd D:\\develop\\Project\\Hello-Agent\\hello-agents
    python -m Agent.Memory.working_memory_test
"""

from datetime import datetime, timedelta

from Agent.Memory.types.working import WorkingMemory
from Agent.Memory.memory_item import make_item


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def test_basic_add_retrieve():
    section("1. 基础：添加 + 检索（按时间近度倒序）")
    wm = WorkingMemory(max_capacity=10, max_tokens=500, ttl_minutes=60)
    wm.add("用户叫张三", importance=0.9)
    wm.add("在咨询退货流程", importance=0.7)
    wm.add("订单号是 A12345", importance=0.8)
    print("stats:", wm.get_stats())
    print("全部记忆（最近在前）：")
    for m in wm.retrieve():
        print(f"  {m} -> {m.content}")


def test_keyword_filter():
    section("2. 关键词过滤：检索 query='订单'")
    wm = WorkingMemory(max_capacity=10, max_tokens=500, ttl_minutes=60)
    wm.add("用户叫张三", importance=0.9)
    wm.add("订单号是 A12345", importance=0.8)
    wm.add("今天天气不错", importance=0.2)
    print("命中结果：")
    for m in wm.retrieve(query="订单"):
        print(f"  {m.content} (重要性 {m.importance})")


def test_forget():
    section("3. 主动遗忘：forget(id)")
    wm = WorkingMemory(max_capacity=10, max_tokens=500, ttl_minutes=60)
    id_zhang = wm.add("用户叫张三", importance=0.9)
    wm.add("订单号是 A12345", importance=0.8)
    print(f"遗忘前数量: {len(wm.retrieve())}")
    print(f"forget 返回: {wm.forget(id_zhang)}")
    print(f"遗忘后数量: {len(wm.retrieve())}")
    print(f"再次 forget 同一条: {wm.forget(id_zhang)}  (应返回 False)")


def test_ttl_expiry():
    section("4. TTL 过期：超过 ttl_minutes 的记忆被惰性清理")
    wm = WorkingMemory(max_capacity=10, max_tokens=500, ttl_minutes=60)
    wm.add("这是一条新记忆", importance=0.5)
    old = make_item("这是一条2小时前的旧记忆", importance=0.95)
    old.timestamp = datetime.now() - timedelta(hours=2)
    wm._items.append(old); wm._index[old.id] = old
    wm.current_tokens += wm._counter(old.content)
    print(f"清理前数量: {len(wm._items)}  (应为2)  # 直接读内部列表，不触发惰性清理")
    removed = wm.cleanup()
    print(f"cleanup 返回: {removed}  (应为1)")
    print(f"清理后数量: {len(wm._items)}  (应为1)")
    print("剩余:", [m.content for m in wm.retrieve()])


def test_capacity_eviction():
    section("5. 容量驱逐：超出 max_capacity 时驱逐最低优先级")
    wm = WorkingMemory(max_capacity=3, max_tokens=500, ttl_minutes=60)
    wm.add("记忆A", importance=0.9)
    wm.add("记忆B", importance=0.2)
    wm.add("记忆C", importance=0.7)
    print("容量上限=3，再加入第4条（重要性0.8）会驱逐最低者：")
    wm.add("记忆D", importance=0.8)
    print("当前记忆:", sorted(m.content for m in wm.retrieve()))
    print("应保留: A, C, D（B 被遗忘）")


def test_token_eviction():
    section("6. Token 上限驱逐：超出 max_tokens 时驱逐最低优先级")
    wm = WorkingMemory(max_capacity=10, max_tokens=10, ttl_minutes=60)
    wm.add("短记忆甲", importance=0.9)
    wm.add("短记忆乙", importance=0.1)
    print(f"当前 tokens: {wm.current_tokens} / {wm.max_tokens}")
    print("再加入较长内容，应驱逐低重要性者：")
    wm.add("这是一条比较长的记忆内容", importance=0.5)
    print("当前记忆:", [m.content for m in wm.retrieve()])
    print("stats:", wm.get_stats())


def test_retrieve_limit():
    section("7. limit 限制：retrieve(limit=2)")
    wm = WorkingMemory(max_capacity=10, max_tokens=500, ttl_minutes=60)
    for i in range(5):
        wm.add(f"记忆{i}", importance=0.5)
    print(f"总数: {len(wm.retrieve())}")
    print(f"limit=2 返回: {[m.content for m in wm.retrieve(limit=2)]}")


if __name__ == '__main__':
    test_basic_add_retrieve()
    test_keyword_filter()
    test_forget()
    test_ttl_expiry()
    test_capacity_eviction()
    test_token_eviction()
    test_retrieve_limit()
    print("\n✅ 全部测试用例运行完毕")
