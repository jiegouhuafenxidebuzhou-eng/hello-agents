"""RAG 工具测试用例（零依赖路径：hashing embedder + 内存向量库，不依赖 Qdrant/真嵌入/LLM）。

覆盖：添加文本/搜索/ask降级/统计/清空、分块去重、命名空间隔离、未配 Qdrant 不崩溃。
运行：
    cd D:\\develop\\Project\\Hello-Agent\\hello-agents
    python -m Agent.test.rag_test

LLM + Qdrant 真路径手动验证：
    1. 在 Agent/Memory/.env 配 QDRANT_URL / QDRANT_API_KEY /
       EMBED_MODEL_TYPE=dashscope / EMBED_API_KEY / EMBED_BASE_URL /
       LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL。
    2. 把本文件里 _new() 的 qdrant_url 改为读 env（去掉固定 None），
       并将 use_real_llm 改为 True。
    3. add_document 一个 PDF → ask 提问，验证语义召回 + 自然语言答案 + 引用来源。
"""

import os
import shutil
import tempfile

from Agent.tools.rag_tool import RAGTool


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def _new(rag_namespace: str = "default", kb_path: str = None,
         use_llm: bool = False, llm=None) -> RAGTool:
    """构造零依赖实例：不传 qdrant_url → 内存向量库；无 LLM → 降级。

    用临时目录做 knowledge_base_path，避免污染工作区。
    - use_llm=False（默认）：强制关闭 LLM，走降级路径，测试稳定可复现。
    - llm=<stub>：注入桩 LLM，用于触发并断言 MQE/HyDE 而不依赖网络/外部 LLM。
    """
    kb = kb_path or os.path.join(tempfile.gettempdir(), f"rag_kb_{rag_namespace}")
    os.makedirs(kb, exist_ok=True)
    if llm is None and use_llm:
        try:
            from Agent.LLMClient.llm_client import HelloAgentsLLM
            llm = HelloAgentsLLM()
        except Exception as e:
            print(f"[test] LLM 未配置，降级: {e}")
    # enable_llm=False 强制不自动从 env 构造 LLM；llm 非空时直接使用
    return RAGTool(knowledge_base_path=kb, rag_namespace=rag_namespace,
                  llm_client=llm, enable_llm=False,
                  qdrant_url=None, qdrant_api_key=None)


class _StubLLM:
    """桩 LLM：按 system prompt 内容返回 MQE/HyDE 的 canned 文本，并计数调用次数。"""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, temperature=0):
        self.calls += 1
        sys_content = messages[0]["content"] if messages else ""
        if "查询扩展" in sys_content or "多样化查询" in sys_content:
            # MQE：返回 2 个扩展查询
            return "机器学习技术\nML 原理\n"
        if "答案性段落" in sys_content:
            # HyDE：返回一段假设答案
            return "机器学习是让计算机从数据中学习规律并做出预测的技术。"
        return ""


def test_add_search():
    section("1. 添加文本 + 搜索")
    rag = _new()
    rag.run(action="add_text", text="机器学习是让计算机从数据中学习规律的技术。")
    rag.run(action="add_text", text="深度学习是机器学习的一个分支，使用多层神经网络。")
    rag.run(action="add_text", text="今天天气很好，适合出去散步。")  # 无关项
    res = rag.run(action="search", query="什么是机器学习", limit=3)
    print(res)
    assert "机器学习" in res, "搜索应命中机器学习相关片段"
    print("✅ 添加 + 搜索正确")


def test_ask_degradation():
    section("2. ask 降级（无 LLM，返回原文片段）")
    rag = _new()
    rag.run(action="add_text",
            text="RAG 是检索增强生成，先从知识库检索相关片段，再交给 LLM 生成答案。")
    ans = rag.run(action="ask", question="RAG 是什么？")
    print(ans)
    assert rag.llm_client is None, "降级路径应强制无 LLM"
    assert "未配置 LLM" in ans or "检索结果" in ans, "无 LLM 时应降级返回原文片段"
    print("✅ ask 降级正确")


def test_chunk_dedup():
    section("3. 分块去重（相同内容重复添加不翻倍）")
    rag = _new()
    text = "这是一段用于测试去重的文本内容，内容足够长以触发分块。" * 3
    r1 = rag.run(action="add_text", text=text)
    n1 = int(r1.split("分块数: ")[1].split("\n")[0])
    # 再加一次相同内容 → chunk 去重应让总数不增加
    r2 = rag.run(action="add_text", text=text)
    n2 = int(r2.split("分块数: ")[1].split("\n")[0])
    print(f"首次分块数: {n1}, 重复添加后: {n2}")
    assert n2 <= n1, "重复内容应被去重，分块数不应增加"
    print("✅ 分块去重正确")


def test_namespace_isolation():
    section("4. 命名空间隔离")
    ragA = _new(rag_namespace="nsA")
    ragB = _new(rag_namespace="nsB")
    ragA.run(action="add_text", text="项目A的专属知识：部署在阿里云上。")
    ragB.run(action="add_text", text="项目B的专属知识：部署在腾讯云上。")
    resA = ragA.run(action="search", query="部署在哪里")
    resB = ragB.run(action="search", query="部署在哪里")
    print("nsA:", resA)
    print("nsB:", resB)
    assert "阿里云" in resA and "腾讯云" not in resA, "nsA 不应看到 nsB 的内容"
    assert "腾讯云" in resB and "阿里云" not in resB, "nsB 不应看到 nsA 的内容"
    print("✅ 命名空间隔离正确")


def test_stats_and_clear():
    section("5. 统计 + 清空")
    rag = _new()
    rag.run(action="add_text", text="用于统计和清空测试的内容。")
    stats = rag.run(action="stats")
    print(stats)
    assert "分块数" in stats and "in_memory" in stats
    # 不带 confirm 应拒绝
    denied = rag.run(action="clear")
    assert "confirm=true" in denied, "未确认应拒绝清空"
    ok = rag.run(action="clear", confirm=True)
    print(ok)
    assert "清空" in ok or "重建" in ok
    stats2 = rag.run(action="stats")
    print("清空后:", stats2)
    print("✅ 统计 + 清空正确")


def test_no_qdrant_no_crash():
    section("6. 降级：无 Qdrant / 无真嵌入 / 无 LLM 仍可用")
    rag = _new()
    assert rag.llm_client is None, "降级路径应强制无 LLM"
    rag.run(action="add_text", text="降级场景下的知识库内容。")
    res = rag.run(action="search", query="降级")
    assert res and "降级" in res, "降级路径应仍能检索"
    print("✅ 零依赖路径可用（内存向量库 + hashing embedder）")


def test_search_advanced_mqe_hyde():
    section("7. search 高级检索：MQE 多查询扩展 + HyDE 假设文档嵌入（stub LLM）")
    stub = _StubLLM()
    rag = _new(llm=stub)
    assert rag.llm_client is stub, "应注入桩 LLM"
    rag.run(action="add_text", text="机器学习是研究如何让计算机从数据中学习规律并做出预测的技术。")
    rag.run(action="add_text", text="深度学习使用多层神经网络进行特征表示的学习。")
    rag.run(action="add_text", text="今天天气晴朗，适合户外运动。")  # 无关项

    # search 默认 enable_advanced=True，LLM 可用 → 触发 MQE + HyDE
    res = rag.run(action="search", query="什么是机器学习", limit=3)
    print(res)
    print(f"[stub] LLM complete 调用次数: {stub.calls}")
    assert stub.calls >= 2, "MQE + HyDE 应至少调用 LLM complete 2 次"
    assert "机器学习" in res, "应命中机器学习相关片段"
    print("✅ MQE + HyDE 扩展检索已触发（search 动作）")


def test_heading_follows_body_no_leak():
    section("9. 标题紧接正文：内容不泄漏到下一节")
    from Agent.Memory.rag.pipeline import _split_paragraphs_with_headings
    # H1 下两行正文，紧接着 H2，中间无空行 —— 旧逻辑会把 AAA/BBB 泄漏进 H2 段落
    md = "# H1\nAAA\nBBB\n# H2\nCCC\n"
    paras = _split_paragraphs_with_headings(md)
    print("\n".join(f"{p['heading_path']}: {p['content']!r}" for p in paras))
    h1 = [p for p in paras if p["heading_path"] == "H1"]
    h2 = [p for p in paras if p["heading_path"] == "H2"]
    assert h1 and "AAA" in h1[0]["content"], "H1 段落应含 AAA"
    assert h2 and h2[0]["content"] == "CCC", "H2 段落应只含 CCC，不混入 AAA/BBB"
    print("✅ 标题紧接正文不再泄漏内容")


def test_code_fence_hash_not_heading():
    section("10. 代码围栏内的 # 不被当成标题")
    from Agent.Memory.rag.pipeline import _split_paragraphs_with_headings
    md = "# 标题\n\n```python\n# 这是注释\n#include <stdio.h>\nx = 1\n```\n\n正文。\n"
    paras = _split_paragraphs_with_headings(md)
    heading_paths = [p["heading_path"] for p in paras]
    print(heading_paths)
    # 围栏内的 `# 这是注释` 不应产生新标题路径；只应出现 "标题"（或 None）
    assert all(hp in (None, "标题") for hp in heading_paths), \
        f"代码围栏内的 # 被误判为标题: {heading_paths}"
    print("✅ 代码围栏内的 # 不被当成标题")


def test_ask_advanced_mqe_hyde():
    section("8. ask 高级检索：MQE + HyDE（stub LLM，答案生成降级）")
    stub = _StubLLM()
    rag = _new(llm=stub)
    rag.run(action="add_text", text="RAG 即检索增强生成，先检索知识库片段再交 LLM 生成答案。")
    ans = rag.run(action="ask", question="RAG 是什么？")
    print(ans)
    # ask 检索阶段也应触发 MQE + HyDE（≥2 次）；答案生成阶段 stub 返回 "" → 降级返回原文
    assert stub.calls >= 2, "ask 检索阶段应触发 MQE + HyDE"
    assert "检索结果" in ans or "RAG" in ans, "stub 答案为空时应降级返回原文片段"
    print("✅ ask 的 MQE + HyDE 已触发，答案生成降级正确")


if __name__ == "__main__":
    # 每个用例独立 kb 目录，跑完统一清理
    kb_dirs = []
    try:
        # 实例化前先建好目录，便于 _new 复用
        test_add_search()
        test_ask_degradation()
        test_chunk_dedup()
        test_namespace_isolation()
        test_stats_and_clear()
        test_no_qdrant_no_crash()
        test_search_advanced_mqe_hyde()
        test_heading_follows_body_no_leak()
        test_code_fence_hash_not_heading()
        test_ask_advanced_mqe_hyde()
        print("\n✅ 全部 RAG 测试通过")
    finally:
        kb_root = tempfile.gettempdir()
        for d in os.listdir(kb_root):
            if d.startswith("rag_kb_"):
                p = os.path.join(kb_root, d)
                try:
                    shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
