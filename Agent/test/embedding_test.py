"""嵌入模型连通性测试（DashScope / 本地 / hashing 均可）。

验证：环境变量加载 → embedder 构造 → 单条/批量 encode → 维度一致 →
      语义相似度合理（相似句余弦 > 不相关句）→ 连通性判定。
运行：
    cd D:\\develop\\Project\\Hello-Agent\\hello-agents
    python -m Agent.test.embedding_test

依赖：python-dotenv（已用于 Agent/LLMClient/llm_client.py）；
      DashScope 路径另需 requests。
配置：Agent/Memory/.env；不存在时回退读取 .env.example（示例里已填 key）。
"""

import math
import os
from pathlib import Path

# 先加载环境变量，再 import embedding（后者 __init__ 期就会读 env 做连通探测）
_ENV_DIR = Path(__file__).resolve().parent.parent / "Memory"
_env_path = _ENV_DIR / ".env"
if not _env_path.exists():
    _env_path = _ENV_DIR / ".env.example"  # 回退：示例文件里已填配置
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=True)
    print(f"[env] 已加载 {_env_path.name}")
except ImportError:
    print("[env] 未安装 python-dotenv，改用系统环境变量（EMBED_* 需自行注入）")

from Agent.Memory.embedding import (  # noqa: E402
    get_embedder, refresh_embedder, embedding_available,
)


def section(title: str) -> None:
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def _cosine(a, b) -> float:
    """L2 归一化向量的余弦相似度（embedder 已归一化；hashing 也归一化）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def test_config_and_build():
    section("1. 环境变量 + embedder 构造")
    print(f"  EMBED_MODEL_TYPE = {os.getenv('EMBED_MODEL_TYPE', '(未设置→默认 hashing)')}")
    print(f"  EMBED_MODEL_NAME = {os.getenv('EMBED_MODEL_NAME', '(未设置)')}")
    print(f"  EMBED_BASE_URL   = {os.getenv('EMBED_BASE_URL', '(未设置)')}")
    key = os.getenv("EMBED_API_KEY", "")
    print(f"  EMBED_API_KEY    = {key[:6]}{'***' if key else '(未设置)'}")
    print(f"  EMBED_DIM        = {os.getenv('EMBED_DIM', '(默认 384)')}")

    emb = refresh_embedder()  # 按当前 env 强制重建单例
    print(f"  -> 实例: {emb.name}")
    print(f"  -> available(真语义): {emb.available}")
    assert emb is not None
    print("✅ embedder 构造成功")
    return emb


def test_single_encode(emb):
    section("2. 单条 encode（连通性核心）")
    try:
        vec = emb.encode("今天天气真好，我想去公园散步")
    except Exception as e:
        print(f"❌ encode 失败：{e}")
        raise
    print(f"  返回维度: {len(vec)} (embedder.dimension={emb.dimension})")
    assert len(vec) > 0, "向量不应为空"
    assert len(vec) == emb.dimension, "返回维度与 embedder.dimension 不一致"
    # 抽样打印前 5 维，确认是数值
    print(f"  前5维: {[round(v, 6) for v in vec[:5]]}")
    print("✅ 单条 encode 连通正常")
    return vec


def test_batch_encode(emb):
    section("3. 批量 encode")
    texts = ["苹果是一种水果", "香蕉也是水果", "今天讲了个笑话"]
    vecs = emb.encode_batch(texts)
    assert len(vecs) == len(texts)
    assert all(len(v) == emb.dimension for v in vecs)
    print(f"  批量 {len(texts)} 条，每条维度 {emb.dimension}")
    print("✅ 批量 encode 正常")
    return vecs


def test_semantic_sanity(emb):
    section("4. 语义相似度合理性")
    # 相似语义应比不相关语义更接近
    v_a = emb.encode("我喜欢吃苹果")
    v_b = emb.encode("我爱吃香蕉")        # 同属吃水果，应较近
    v_c = emb.encode("今天股市大跌")      # 无关，应较远
    sim_close = _cosine(v_a, v_b)
    sim_far = _cosine(v_a, v_c)
    print(f"  相似句余弦: {sim_close:.4f}")
    print(f"  无关句余弦: {sim_far:.4f}")
    # hashing 兜底是关键词级，可能误判；仅对真语义嵌入断言
    if emb.available:
        assert sim_close > sim_far, (
            f"真语义嵌入下相似句应更接近（{sim_close:.4f} > {sim_far:.4f}）"
        )
        print("✅ 语义相似度排序正确")
    else:
        print("⚠️  当前为 hashing 兜底（非真语义），跳过相似度断言")
    return sim_close, sim_far


def test_embedding_available_flag():
    section("5. embedding_available() 开关")
    avail = embedding_available()
    print(f"  embedding_available() = {avail}")
    # 只要构造没炸，这里仅打印，不强制 True（hashing 时为 False）
    print("✅ 状态可读")


if __name__ == "__main__":
    try:
        emb = test_config_and_build()
        test_single_encode(emb)
        test_batch_encode(emb)
        test_semantic_sanity(emb)
        test_embedding_available_flag()
        print("\n✅ 嵌入模型连通性测试通过")
        if not emb.available:
            print("   注：当前走 hashing 兜底，未连上真语义嵌入模型。"
                  "请检查 EMBED_MODEL_TYPE / EMBED_API_KEY / EMBED_BASE_URL。")
    except Exception as e:
        print(f"\n❌ 测试失败：{type(e).__name__}: {e}")
        raise
