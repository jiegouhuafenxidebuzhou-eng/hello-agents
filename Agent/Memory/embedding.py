"""嵌入（Embedding）抽象 + 降级链。

按优先级回退：
1. DashScopeEmbedder：OpenAI 兼容 REST（需 requests + EMBED_API_KEY）——真语义。
2. LocalEmbedder：sentence-transformers（需装包 + 模型）——真语义。
3. HashingEmbedder：纯标准库 hashing trick——零依赖兜底，效果≈关键词重叠余弦。

未装任何重依赖时自动落到 HashingEmbedder，`available` 仍为 True（但只是关键词级）。
EpisodicMemory 用 `semantic_ready` 判断是否走真语义向量路径。

环境变量：
- EMBED_MODEL_TYPE: dashscope | local | hashing（默认 hashing，零依赖）
- EMBED_MODEL_NAME / EMBED_API_KEY / EMBED_BASE_URL
- EMBED_DIM: hashing 维度（默认 384）
"""

import hashlib
import math
import os
import threading
from typing import List, Optional, Union


# ============== 抽象 ==============

class Embedder:
    """嵌入模型基类（最小接口）。"""

    def encode(self, text: str) -> List[float]:
        raise NotImplementedError

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        """是否真语义（非 hashing 兜底）。"""
        return True

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ============== DashScope / OpenAI 兼容 REST ==============

class DashScopeEmbedder(Embedder):
    """OpenAI 兼容的 Embedding REST 调用（{base_url}/embeddings）。"""

    def __init__(self, model_name: str = "text-embedding-v3",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        if not self.base_url or not self.api_key:
            raise ValueError("DashScopeEmbedder 需 EMBED_BASE_URL 和 EMBED_API_KEY")
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError("DashScopeEmbedder 需 requests: pip install requests")
        self._dimension = None
        try:
            self._dimension = len(self.encode("health_check"))
        except Exception:
            self._dimension = None

    def encode(self, text: str) -> List[float]:
        import requests
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers,
                             json={"model": self.model_name, "input": [text]},
                             timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"Embedding REST 失败: {resp.status_code} {resp.text}")
        items = resp.json().get("data") or []
        if not items:
            raise RuntimeError("Embedding 返回为空")
        return list(items[0].get("embedding") or [])

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)


# ============== 本地 sentence-transformers ==============

class LocalEmbedder(Embedder):
    """本地 Transformer 嵌入（sentence-transformers）。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("LocalEmbedder 需 sentence-transformers: "
                              "pip install sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self._dimension = len(self._model.encode("test"))

    def encode(self, text: str) -> List[float]:
        vec = self._model.encode(text)
        if hasattr(vec, "tolist"):
            return list(vec.tolist())
        return list(vec)

    @property
    def dimension(self) -> int:
        return int(self._dimension)


# ============== 纯标准库 hashing 兜底 ==============

class HashingEmbedder(Embedder):
    """零依赖兜底嵌入：hashing trick 投影到固定维度。

    - 分词：CJK 字符按字切，Latin 按非字母数字切。
    - 每个词经 md5 哈希 → 桶 = h % dim，符号 = +1/-1（由 (h // dim) % 2 定）。
    - 累加后 L2 归一化。
    确定性、跨进程一致，效果≈词重叠的余弦相似度——足以支撑降级时的向量召回。
    """

    def __init__(self, dim: int = 384):
        self._dim = int(dim)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def available(self) -> bool:
        return False  # 非真语义，调用方据此降级

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        buf = []
        for ch in text:
            if ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    tokens.append("".join(buf).lower())
                    buf = []
                if ch.strip() and ord(ch) > 0x2E80:
                    # CJK 等非拉丁字符按单字成词
                    tokens.append(ch.lower())
        if buf:
            tokens.append("".join(buf).lower())
        return tokens

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        for tok in self._tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            bucket = h % self._dim
            sign = 1.0 if ((h // self._dim) % 2 == 0) else -1.0
            vec[bucket] += sign
        # L2 归一化，零向量保持原样（余弦=0）
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ============== 工厂 + 单例 ==============

_lock = threading.RLock()
_embedder: Optional[Embedder] = None


def _build_embedder() -> Embedder:
    preferred = os.getenv("EMBED_MODEL_TYPE", "hashing").strip().lower()
    dim = int(os.getenv("EMBED_DIM", "384"))

    def _try_dashscope():
        if preferred != "dashscope":
            return None
        try:
            return DashScopeEmbedder(
                model_name=os.getenv("EMBED_MODEL_NAME", "text-embedding-v3"),
                api_key=os.getenv("EMBED_API_KEY"),
                base_url=os.getenv("EMBED_BASE_URL"),
            )
        except Exception as e:
            print(f"[embedder] DashScope 不可用: {e}")
            return None

    def _try_local():
        if preferred != "local":
            return None
        try:
            return LocalEmbedder(os.getenv("EMBED_MODEL_NAME",
                                           "sentence-transformers/all-MiniLM-L6-v2"))
        except Exception as e:
            print(f"[embedder] Local 不可用: {e}")
            return None

    # 按首选优先，否则按降级链
    order = {
        "dashscope": [_try_dashscope, _try_local],
        "local": [_try_local, _try_dashscope],
        "hashing": [],
    }.get(preferred, [])

    for factory in order:
        inst = factory()
        if inst is not None:
            return inst

    # 兜底
    print(f"[embedder] 无可用语义模型，使用 HashingEmbedder(dim={dim}) 兜底")
    return HashingEmbedder(dim=dim)


def get_embedder() -> Embedder:
    """获取全局共享嵌入实例（线程安全单例）。"""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def refresh_embedder() -> Embedder:
    """强制重建（切换环境变量后调用）。"""
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder


def embedding_available() -> bool:
    """是否装了真语义嵌入模型（非 hashing 兜底）。"""
    return get_embedder().available
