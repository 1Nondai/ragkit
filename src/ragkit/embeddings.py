"""Embedding 抽象与两种实现。"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embeddings(Protocol):
    """最小接口：够用即可，不绑定任何具体 SDK。"""

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

@runtime_checkable
class ImageEmbeddings(Protocol):
    """支持图片输入的 embedding（图搜图/图搜文用）。

    单独定义协议而不是塞进 Embeddings：绝大多数文本 embedding 模型
    不支持图片输入，用 isinstance 判能力、不支持就跳过图片这一路。
    """

    def embed_image(self, image_bytes: bytes) -> list[float]: ...

class HashingEmbedding:
    """把文本哈希成固定维度的稀疏-稠密混合向量（确定性、零依赖）。"""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self._dim = dim

    @property
    def dim(self) -> int:
        """向量维度（建集合时必须与模型一致）。"""
        return self._dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        lowered = (text or "").lower()
        tokens: list[str] = []
        for tok in re.findall(r"[a-z0-9]+", lowered):
            tokens.append(tok)
        for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
            tokens.extend(run)
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        return tokens

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self._dim
            sign = 1.0 if digest[-1] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档（入库用）。"""
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """编码单条查询（检索用；与入库必须是同一个模型）。"""
        return self._embed_one(text)

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """图片字节 → 确定性向量（哈希填充，**没有语义能力**）。

        同样的图得到同样的向量，所以离线能验证"图搜图"的管道；
        但两张语义相似、像素不同的图在它眼里毫不相关。
        """
        vec = [0.0] * self._dim
        data = image_bytes or b""
        for i in range(self._dim):
            digest = hashlib.blake2b(data + i.to_bytes(4, "big"), digest_size=8).digest()
            vec[i] = 1.0 if digest[-1] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

class OpenAICompatEmbedding:
    """通过 OpenAI 兼容接口取稠密向量（DeepSeek/通义/智谱/OpenAI 均可用）。

    惰性导入：构造时不 import SDK，避免"只想算个哈希"也被迫装依赖。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        dim: int = 1024,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._dim = dim
        self.batch_size = batch_size
        self._client = None

    @property
    def dim(self) -> int:
        """向量维度（建集合时必须与模型一致）。"""
        return self._dim

    def _ensure_client(self):
        if self._client is None:
            from langchain_openai import OpenAIEmbeddings

            kwargs = {"model": self.model}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = OpenAIEmbeddings(**kwargs)
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档（入库用）。"""
        client = self._ensure_client()
        out: list[list[float]] = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            out.extend(client.embed_documents(texts[i : i + self.batch_size]))
        if out and len(out[0]) != self._dim:
            self._dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        """编码单条查询（检索用；与入库必须是同一个模型）。"""
        return self._ensure_client().embed_query(text)

class DashScopeMultimodalEmbedding:
    """阿里云多模态 embedding（multimodal-embedding-v1，图文对齐 1024 维）。"""

    def __init__(self, model: str = "multimodal-embedding-v1", *,
                 api_key: str | None = None, dim: int = 1024) -> None:
        self.model = model
        self.api_key = api_key
        self._dim = dim

    @property
    def dim(self) -> int:
        """向量维度（建集合时必须与模型一致）。"""
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档（入库用）。"""
        return [self._call([{"text": t}]) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """编码单条查询（检索用；与入库必须是同一个模型）。"""
        return self._call([{"text": text}])

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """图片字节 → 向量（图搜图用；仅多模态模型支持）。"""
        import base64
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return self._call([{"image": b64}])

    def _call(self, input_data: list[dict]) -> list[float]:
        try:
            from dashscope import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("需要安装 dashscope：pip install dashscope") from exc
        kwargs = {"model": self.model, "input": input_data}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        resp = TextEmbedding.call(**kwargs)
        if resp.status_code != 200:
            raise RuntimeError(f"multimodal embedding 失败: {resp.code} {resp.message}")
        vec = resp.output["embeddings"][0]["embedding"]
        return [float(v) for v in vec]

def build_embeddings():
    """按配置构造 embedding（工厂方法，避免调用方关心具体类）。"""
    from ragkit.config import get_settings

    s = get_settings().embedding
    if s.provider == "hashing":
        return HashingEmbedding(dim=256)
    if s.provider == "openai":
        from ragkit.config import get_settings as _gs

        dim = _gs().milvus.dense_dim
        return OpenAICompatEmbedding(
            model=s.model, base_url=s.base_url, api_key=s.api_key,
            dim=dim, batch_size=s.batch_size,
        )
    if s.provider == "dashscope-mm":
        return DashScopeMultimodalEmbedding(model=s.model, api_key=s.api_key,
                                            dim=s.dim_override or 1024)
    raise ValueError(f"未知的 embedding provider: {s.provider}")
